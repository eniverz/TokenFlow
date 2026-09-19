import torch
import torch.nn as nn
from torch.distributions import Beta
from transformers import PreTrainedModel, AutoModel

from .configuration_resvla import ResVLAConfig
from .modular_resvla import (
    ActionStateEncoder,
    AdaRMSNorm,
    AnchorMLP,
    BlockDCT,
    DiT,
    EmbodimentEncoder,
    EmbodimentEmbeddings,
    FlowRefiner,
    LearnableCutoffProjector,
    TimestepEncoder
)


class ResVLA(PreTrainedModel):
    config_class = ResVLAConfig
    base_model_prefix = "resvla"

    def __init__(self, config: ResVLAConfig):
        super().__init__(config)
        self.config = config

        self.vlm: PreTrainedModel = AutoModel.from_config(config.vlm_config, attn_implementation=config.attn_implementation)
        self.anchor_dit = DiT(
            hidden_dim=config.hidden_size,
            num_heads=config.dit_q_heads,
            num_kv_heads=config.dit_kv_heads,
            cross_attention_dim=config.hidden_size,
            attn_implementation=config.attn_implementation,
            num_layers=config.num_anchor_layers,
            dropout=config.dit_dropout,
        )
        self.flow = FlowRefiner(
            hidden_dim=config.hidden_size,
            flow_dim=config.action_dim,
            num_heads=config.dit_q_heads,
            num_kv_heads=config.dit_kv_heads,
            cross_attention_dim=config.hidden_size,
            attn_implementation=config.attn_implementation,
            num_layers=config.num_refiner_layers,
            dropout=config.dit_dropout
        )

        self.dct_layer = BlockDCT(config.action_horizon, dct_type='dct', norm='ortho')
        self.idct_layer = BlockDCT(config.action_horizon, dct_type='idct', norm='ortho')

        self.future_tokens = nn.Parameter(torch.empty(1, config.action_horizon, config.hidden_size))
        self.learnable_cutoff_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))

        self.learnable_cutoff_projector = LearnableCutoffProjector(
            num_freq_components=config.action_horizon,
            hidden_dim=config.hidden_size
        )

        self.anchor_mlp = AnchorMLP(
            action_horizon=config.action_horizon,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_size
        )

        self.embodiment_embedding = EmbodimentEmbeddings(config.embodiment_config, hidden_dim=config.hidden_size)
        self.state_encoder = ActionStateEncoder(input_dim=config.action_dim, output_dim=config.hidden_size)
        self.embodiment_encoder = EmbodimentEncoder(input_dim=config.hidden_size, output_dim=config.hidden_size)
        self.timestep_encoder = TimestepEncoder(hidden_dim=config.hidden_size)

        self.beta_dist = None

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)

        if module is self:
            nn.init.trunc_normal_(module.future_tokens, mean=0.0, std=0.02)
            nn.init.trunc_normal_(module.learnable_cutoff_token, mean=0.0, std=0.02)
        elif isinstance(module, TimestepEncoder):
            freqs = module._init_exponent()
            module.freqs.copy_(freqs)
        elif isinstance(module, AdaRMSNorm):
            nn.init.zeros_(module.linear.weight)
            nn.init.zeros_(module.linear.bias)
        elif isinstance(module, BlockDCT):
            mat = module.compute_matrix()
            module.mat.copy_(mat)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Sample time steps for diffusion process.

        :param batch_size: number of samples to generate
        :param device: device to place the tensor on
        :param dtype: data type of the tensor
        :return: [B, 1] tensor of sampled time steps
        """
        if self.beta_dist is None:
            self.beta_dist = Beta(
                torch.as_tensor(self.config.noise_beta_alpha, dtype=dtype, device=device),
                torch.as_tensor(self.config.noise_beta_beta, dtype=dtype, device=device)
            )
        sample = self.beta_dist.sample([batch_size])
        return self.config.noise_s * (1 - sample)

    def forward(self, examples: dict):
        """
        :param examples: a dict containing the following keys:

            - vlm_inputs: input languages and images after processor
            - states: [B, 1, action_dim] tensor of states
            - actions: [B, action_horizon, action_dim] tensor of actions
            - embodiment: [B, 1] list of embodiment names
        
        :return: metrics and loss dict
        """
        inputs = examples["vlm_inputs"]
        actions = examples["actions"]
        states = examples["states"]
        embodiment_names = examples["embodiment"]

        bs = actions.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.vlm(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        vl_embs = list(vlm_outputs.hidden_states[1:])

        with torch.autocast("cuda", dtype=torch.float32):
            # prepare embodiment embeddings based on embodiment_tag
            embodiment_embeddings = self.embodiment_embedding(embodiment_names)
            embodiment_cond = self.embodiment_encoder(embodiment_embeddings)
            state_features = self.state_encoder(states, embodiment_cond)

            # Anchor DiT for coarse actions
            attention_mask = inputs["attention_mask"].bool()
            anchor_output = self.compute_anchor(state_features, vl_embs, attention_mask)

            # 2. loss with DCT
            # 2.1 extract features
            future_token_outputs = anchor_output[:, :self.config.action_horizon, :] # (bs, action_horizon, hidden_size)
            cutoff_token_output = anchor_output[:, self.config.action_horizon:, :] # (bs, 1, hidden_size)

            # 2.2 get frequency components mask
            freq_mask = self.learnable_cutoff_projector(cutoff_token_output).unsqueeze(-1) # (bs, action_horizon, 1)
            full_freq_action = self.anchor_mlp(future_token_outputs) # (bs, action_horizon, action_dim)
            masked_freq_action = full_freq_action * freq_mask # (bs, action_horizon, action_dim)

            # 2.3 anchor loss
            anchor_loss = (((full_freq_action - self.dct_layer(actions.transpose(1, 2)).transpose(1, 2)) ** 2) * freq_mask).sum() / (freq_mask.sum() + 1e-8)

            # 3. refiner with Flow Matching
            # 3.1 convert actions to time-domain
            anchor_time = self.idct_layer(masked_freq_action.transpose(1, 2)).transpose(1, 2) # (bs, action_horizon, action_dim)

            # 3.2 prepare flow matching
            x_0 = anchor_time.repeat_interleave(self.config.num_repeat_diffusion_steps, dim=0) 
            x_1 = actions.repeat_interleave(self.config.num_repeat_diffusion_steps, dim=0)
            embodiment_cond = embodiment_cond.repeat_interleave(self.config.num_repeat_diffusion_steps, dim=0)
            res_bridge_vl_embs = [
                torch.cat((state_features, emb), dim=1).repeat_interleave(
                    self.config.num_repeat_diffusion_steps, dim=0
                )
                for emb in vl_embs[-self.config.num_refiner_layers:]
            ]
            flow_context_mask = torch.cat((
                torch.ones((bs, 1), dtype=torch.bool, device=actions.device),
                attention_mask,
            ), dim=1).repeat_interleave(self.config.num_repeat_diffusion_steps, dim=0)

            noise = self.config.fm_noise_std * torch.randn(
                x_1.shape, device=x_1.device, dtype=x_1.dtype
            )
            x_0 = x_0 + noise

            # 3.3 interpolate trajectory
            t = self.sample_time(bs * self.config.num_repeat_diffusion_steps, x_0.device, x_0.dtype) # (bs * num_repeat_diffusion_steps)
            t = t.unsqueeze(-1).unsqueeze(-1) # (bs * num_repeat_diffusion_steps, 1, 1)
            temb = self.timestep_encoder(t) # (bs * num_repeat_diffusion_steps, hidden_size)
            noisy_traj = (1 - t) * x_0 + t * x_1
            velocity = x_1 - x_0

            # 3.4 flow matching
            pred = self.flow(
                x=noisy_traj,
                context=res_bridge_vl_embs,
                context_mask=flow_context_mask,
                temb=temb,
                embodiment_cond=embodiment_cond
            )

            # 3.5 flow loss
            flow_loss = ((pred - velocity).float() ** 2).mean()

        total_loss = flow_loss + (flow_loss.detach() / (anchor_loss.detach() + 1e-8)) * self.config.anchor_loss_weight * anchor_loss

        return {
            "loss": total_loss,
            "losses/total_loss": total_loss.detach(),
            "losses/flow_loss": flow_loss.detach(),
            "losses/anchor_loss": anchor_loss.detach(),
        }

    @torch.inference_mode()
    def predict_action(self, examples: dict) -> torch.Tensor:
        """
        :param examples: a dict containing the following keys:

            - vlm_inputs: input languages and images after processor
            - states: [B, 1, action_dim] tensor of states
            - embodiment: [B, 1] list of embodiment names
        
        :return: [B, action_horizon, action_dim] tensor of predicted actions
        """
        inputs = examples["vlm_inputs"]
        states = examples["states"]
        embodiment_names = examples["embodiment"]

        bs = states.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.vlm(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        vl_embs = list(vlm_outputs.hidden_states[1:])

        with torch.autocast("cuda", dtype=torch.float32):
            # prepare embodiment embeddings based on embodiment_tag
            embodiment_embeddings = self.embodiment_embedding(embodiment_names)
            embodiment_cond = self.embodiment_encoder(embodiment_embeddings)
            state_features = self.state_encoder(states, embodiment_cond)

            # Anchor DiT for coarse actions
            attention_mask = inputs["attention_mask"].bool()
            anchor_output = self.compute_anchor(state_features, vl_embs, attention_mask)

            # 2. loss with DCT
            # 2.1 extract features
            future_token_outputs = anchor_output[:, :self.config.action_horizon, :] # (bs, action_horizon, hidden_size)
            cutoff_token_output = anchor_output[:, self.config.action_horizon:, :] # (bs, 1, hidden_size)

            # 2.2 get frequency components mask
            freq_mask = self.learnable_cutoff_projector(cutoff_token_output, mode="eval").unsqueeze(-1) # (bs, action_horizon, 1)
            full_freq_action = self.anchor_mlp(future_token_outputs) # (bs, action_horizon, action_dim)
            masked_freq_action = full_freq_action * freq_mask # (bs, action_horizon, action_dim)

            # 3. refiner with Flow Matching
            # 3.1 convert actions to time-domain
            sample = self.idct_layer(masked_freq_action.transpose(1, 2)).transpose(1, 2) # (bs, action_horizon, action_dim)

            # 3.2 prepare flow matching
            res_bridge_vl_embs = [
                torch.cat((state_features, emb), dim=1)
                for emb in vl_embs[-self.config.num_refiner_layers:]
            ]
            flow_context_mask = torch.cat((
                torch.ones((bs, 1), dtype=torch.bool, device=states.device),
                attention_mask,
            ), dim=1)

            noise = self.config.inference_fm_noise_std * torch.randn(
                sample.shape, device=sample.device, dtype=sample.dtype
            )
            sample = sample + noise

            # 3.3 denoise with Euler integration
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                t = torch.full((bs, 1, 1), fill_value=step * dt, device=sample.device)
                temb = self.timestep_encoder(t)
                pred = self.flow(
                    x=sample,
                    context=res_bridge_vl_embs,
                    context_mask=flow_context_mask,
                    temb=temb,
                    embodiment_cond=embodiment_cond
                )
                sample = sample + pred * dt

            return sample

    def compute_anchor(
        self,
        state_features: torch.Tensor,
        vl_embs: list[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bs = state_features.shape[0]
        # 1. Anchor DiT
        # 1.1 prepare time conditions
        dummy_timesteps = torch.ones((bs, 1), device=state_features.device)
        time_cond = self.timestep_encoder(dummy_timesteps)

        # 1.3 prepare anchor input and context for cross attention
        future_embs = self.future_tokens.expand(bs, -1, -1) # (bs, action_horizon, hidden_size)
        cutoff_embs = self.learnable_cutoff_token.expand(bs, -1, -1) # (bs, 1, hidden_size)
        anchor_input = torch.cat((future_embs, cutoff_embs), dim=1)
        context = [torch.cat((state_features, vl), dim=1) for vl in vl_embs]
        context_mask = torch.cat((
            torch.ones((bs, 1), dtype=torch.bool, device=state_features.device),
            attention_mask,
        ), dim=1)

        # 1.4 cross attention for anchor: coarse actions
        anchor_output = self.anchor_dit(
                x=anchor_input,
                context=context,
                context_mask=context_mask,
                temb=time_cond
            )
        
        return anchor_output
