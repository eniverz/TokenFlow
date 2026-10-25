import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_oat import OATConfig
from .modular_oat import AdaRMSNorm, EmbodimentEmbeddings, FiniteScalarQuantization, RegisterQueryEncoder, SinglePassDecoder, SinusoidalPositionalEmbedding


class OAT(PreTrainedModel):
    config_class = OATConfig

    def __init__(self, config: OATConfig):
        super().__init__(config)
        self.encoder_embeddings = EmbodimentEmbeddings(config, config.num_tokens)
        self.decoder_embeddings = EmbodimentEmbeddings(config, config.action_horizon)
        self.encoder = RegisterQueryEncoder(config)
        self.decoder = SinglePassDecoder(config)
        self.quantizer = FiniteScalarQuantization(config.quantizer_levels)
        self.post_init()
        # Apply parent-module overrides after HF initializes the child layers.
        for module in self.modules():
            if isinstance(module, (AdaRMSNorm, RegisterQueryEncoder, SinglePassDecoder)):
                self._init_weights(module)

    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, AdaRMSNorm):
            nn.init.zeros_(module.linear.weight)
            nn.init.zeros_(module.linear.bias)
        elif isinstance(module, RegisterQueryEncoder):
            nn.init.zeros_(module.head.weight)
            nn.init.zeros_(module.head.bias)
        elif isinstance(module, SinglePassDecoder):
            nn.init.zeros_(module.head.weight)
            nn.init.zeros_(module.head.bias)
            nn.init.trunc_normal_(module.mask_token, std=0.02)
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pos_emb = module._init_pos_emb()
            module.pos_emb.copy_(pos_emb)

    def forward(self, samples):
        actions = samples["actions"]
        encoder_cond = self.encoder_embeddings(samples["embodiment"])
        decoder_cond = self.decoder_embeddings(samples["embodiment"])
        latents = self.encoder(actions, encoder_cond)
        quantized, _ = self.quantizer(latents)
        quantized_actions = self.decoder(quantized, decoder_cond)

        # Train the shared decoder on continuous latents, as in v3.
        continuous_latents = latents.detach()
        if self.training:
            ratio = self.config.continuous_noise_std * torch.rand_like(latents[:, :1, :1])
            continuous_latents = continuous_latents + torch.randn_like(continuous_latents) * ratio
        continuous_actions = self.decoder(continuous_latents, decoder_cond)

        action_mask = samples["action_mask"].unsqueeze(1)
        num_valid_elements = action_mask.sum() * actions.shape[1]
        quantized_error = F.mse_loss(quantized_actions.float(), actions.float(), reduction="none")
        continuous_error = F.mse_loss(continuous_actions.float(), actions.float(), reduction="none")
        quantized_loss = quantized_error.masked_fill(~action_mask, 0).sum() / num_valid_elements
        continuous_loss = continuous_error.masked_fill(~action_mask, 0).sum() / num_valid_elements
        return {
            "loss": (
                self.config.quantized_recons_loss_weight * quantized_loss
                + self.config.continuous_recons_loss_weight * continuous_loss
            ),
            "loss/quantized_recons_loss": quantized_loss.detach(),
            "loss/continuous_recons_loss": continuous_loss.detach(),
        }

    @torch.no_grad()
    def encode(self, actions, embodiments):
        cond = self.encoder_embeddings(embodiments)
        _, tokens = self.quantizer(self.encoder(actions, cond))
        return tokens

    @torch.no_grad()
    def decode(self, token_ids, embodiments):
        cond = self.decoder_embeddings(embodiments)
        return self.decoder(self.quantizer.from_codes(token_ids), cond)
