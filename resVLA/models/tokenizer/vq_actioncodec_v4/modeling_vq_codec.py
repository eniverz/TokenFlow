from transformers import PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_vq_codec import ActionCodecConfig
from .consistency import ConsistencyAugmenter, compute_consistency_loss
from .modular_vq_codec import (
    AdaRMSNorm,
    LayerDCT,
    CNNAdapter,
    CNNDecoder,
    ConditionEncoder,
    ConditionalFiLM,
    EmbodimentEmbeddings,
    EMAVectorQuantize,
    PartEmbeddings,
    PerceiverEncoder,
    PerceiverDecoder,
    ResidualVectorQuantize,
    RuleBasedCodec,
    SinusoidalPositionalEmbedding,
)


class ActionCodec(PreTrainedModel):
    config_class = ActionCodecConfig

    def __init__(self, config: ActionCodecConfig):
        super().__init__(config)

        self.cnn_adapter = CNNAdapter(config)
        self.encoder = PerceiverEncoder(config)
        self.decoder = PerceiverDecoder(config)
        self.cnn_decoder = CNNDecoder(config)
        self.rvq = ResidualVectorQuantize(config)
        self.consistency = ConsistencyAugmenter(config)
        self.rule_codec = RuleBasedCodec(config)

        self.embodiment_embeddings = EmbodimentEmbeddings(config)
        self.part_embeddings = PartEmbeddings(config)
        self.cond_encoder = ConditionEncoder(config)

        self.block_dct = LayerDCT(config, 'dct')
        self.block_idct = LayerDCT(config, 'idct')

        self.post_init()

        token_ids = {token: i for i, token in enumerate(config.vocabulary)}
        self.rule_parts = [
            part for part in config.part_config
            if any(pattern in part for pattern in config.rule_based_key_patterns)
        ]
        nn_parts = [part for part in config.part_config if part not in self.rule_parts]
        self.templates, self.code_slices, self.rule_steps = {}, {}, {}
        for embodiment, dims in config.part_dims.items():
            template = [token_ids["<action_start>"]]
            slices = {part: [] for part in config.part_config if dims[part]}
            rule_steps = {}
            for level in range(config.num_quantizers):
                for part in nn_parts:
                    if dims[part]:
                        template.append(token_ids[f"<{part}_{level}>"])
                        start = len(template)
                        template.extend([-1] * config.num_tokens_per_quantizer)
                        slices[part].append(slice(start, len(template)))
            for part in self.rule_parts:
                if dims[part]:
                    template.append(token_ids[f"<{part}>"])
                    start = len(template)
                    length = self.rule_codec.num_tokens
                    template.extend([-1] * length)
                    slices[part].append(slice(start, len(template)))
                    for i in range(length):
                        rule_steps[start + i] = (start, length - i)
            template.append(token_ids["<action_end>"])
            self.templates[embodiment] = template
            self.code_slices[embodiment] = slices
            self.rule_steps[embodiment] = rule_steps

    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, AdaRMSNorm):
            nn.init.zeros_(module.linear.weight)
            nn.init.zeros_(module.linear.bias)
        elif isinstance(module, ConditionalFiLM):
            nn.init.zeros_(module.linear.weight)
            nn.init.zeros_(module.linear.bias)
        elif isinstance(module, LayerDCT):
            mat = module.compute_matrix()
            module.mat.copy_(mat)
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pos_emb = module._init_pos_emb()
            module.pos_emb.copy_(pos_emb)
        elif isinstance(module, EMAVectorQuantize):
            nn.init.xavier_uniform_(module.in_proj.weight)
            nn.init.zeros_(module.out_proj.weight)

    def _part_encoder(self, part_embs, part_actions, part_mask, embodiment_embs):
        """
        Encode a batch of actions from different parts.

        :param part_embs: Tensor of shape (valid_bs, hidden_size) identifying each sample's part.
        :param part_actions: Tensor of shape (valid_bs, action_horizon, max_part_dim) representing the actions for the part.
        :param part_mask: Boolean tensor of shape (valid_bs, max_part_dim) indicating valid actions for the part.
        :param embodiment_embs: Tensor of shape (valid_bs, num_tokens_per_quantizer, hidden_size) representing the embeddings for the embodiments.
        :return latents: Tensor of shape (valid_bs, num_tokens_per_quantizer, hidden_size) representing the encoded latents for the part.
        """
        B, H, D = part_actions.shape
        freq_parts = self.block_dct(part_actions.transpose(1, 2)).transpose(1, 2)

        # Mixture condition
        cond = self.cond_encoder(embodiment_embs, part_embs)  # (valid_bs, hidden_size)

        # CNN Adapter
        part_features = self.cnn_adapter(freq_parts, cond)  # (valid_bs, (horizon - 1) * max_part_dim, cross_attention_dim)

        # Prepare Perceiver Encoder
        patch_mask = part_mask.unsqueeze(1).expand(-1, H - 1, -1).reshape(B, (H - 1) * D)

        latents = self.encoder(
            embodiment_embs,
            part_features,
            patch_mask,
            cond=cond,
        )  # (valid_bs, num_tokens, hidden_size)

        return latents

    def _quantize(self, latents: torch.Tensor, return_level_data: bool = False):
        """Quantize (B,T,D) latents and preserve the encoder/decoder layout.

        Returns quantized latents (B,T,D), codes (B,K,T), and scalar metrics.
        With return_level_data, also returns per-level consistency residuals and codes.
        """
        return self.rvq(latents, return_level_data=return_level_data)

    def _part_decoder(self, part_embs, latents, part_mask, embodiment_embs):
        """
        Decode a batch of latents from different parts back to actions.

        :param part_embs: Tensor of shape (valid_bs, hidden_size) identifying each sample's part.
        :param latents: Tensor of shape (valid_bs, num_tokens_per_quantizer, hidden_size) representing the encoded latents for the part.
        :param part_mask: Boolean tensor of shape (valid_bs, max_part_dim) indicating valid actions for the part.
        :param embodiment_embs: Tensor of shape (valid_bs, num_tokens_per_quantizer, hidden_size) representing the embeddings for the embodiments.
        :return reconstructed_actions: Tensor of shape (valid_bs, action_horizon, max_part_dim) representing the reconstructed actions for the part.
        """
        B, D = part_mask.shape
        H = self.config.action_horizon

        # Mixture condition
        cond = self.cond_encoder(embodiment_embs, part_embs)

        # Perceiver Decoder
        part_features = self.decoder(latents, cond)
        patch_mask = part_mask.unsqueeze(1).expand(-1, H - 1, -1).reshape(B, (H - 1) * D)
        part_features = part_features.masked_fill(~patch_mask.unsqueeze(-1), 0)

        # CNN Decoder and inverse DCT
        freq_parts = self.cnn_decoder(part_features, cond)
        reconstructed_actions = self.block_idct(freq_parts.transpose(1, 2)).transpose(1, 2)

        return reconstructed_actions.masked_fill(~part_mask.unsqueeze(1), 0)

    def _split_actions(self, samples):
        actions_dict, actions_mask = {}, {}
        for part in self.config.part_config:
            actions, masks = [], []
            for i, embodiment in enumerate(samples["embodiment"]):
                dims = self.config.part_dims[embodiment][part]
                padding = (0, self.config.max_part_dim - len(dims))
                actions.append(F.pad(samples["actions"][i, :, dims], padding))
                masks.append(F.pad(samples["action_mask"][i, dims], padding))
            actions_dict[part] = torch.stack(actions)
            actions_mask[part] = torch.stack(masks)
        return {**samples, "actions_dict": actions_dict, "actions_mask": actions_mask}

    def forward(self, samples):
        """
        Forward pass for the ActionCodec.

        :param samples: A dictionary containing the input samples. Expected keys are:

            - actions: Tensor of shape (batch_size, action_horizon, action_dim) representing the input actions.
            - action_mask: Boolean tensor of shape (batch_size, action_dim) indicating valid action dimensions.
            - embodiment: List of str (batch_size) representing the embodiment types for each sample.

        Returns scalar loss metrics for NN parts only. Rule-based parts bypass
        augmentation, reconstruction, consistency, and commitment losses.
        """
        samples = self._split_actions(samples)
        actions_dict = {
            name: actions for name, actions in samples["actions_dict"].items()
            if not any(pattern in name for pattern in self.config.rule_based_key_patterns)
        }
        actions_mask = samples["actions_mask"]
        embodiment_names = samples["embodiment"]

        all_embodiment_embs = self.embodiment_embeddings(embodiment_names)

        parts, masks, embodiment_embs, part_embs = {}, [], [], []

        for part_name, part_actions in actions_dict.items():
            part_mask = actions_mask[part_name]
            present = part_mask.any(dim=-1)
            if not present.any():
                # all embodiments don't have this part
                continue
            parts[part_name] = part_actions[present]
            masks.append(part_mask[present])
            embodiment_embs.append(all_embodiment_embs[present])
            part_embs.append(self.part_embeddings(part_name).expand(parts[part_name].shape[0], -1))

        valid_parts = torch.cat(list(parts.values()), dim=0)
        valid_mask = torch.cat(masks, dim=0)
        embodiment_embs = torch.cat(embodiment_embs, dim=0)
        part_embs = torch.cat(part_embs, dim=0)

        if self.config.consistency_loss_weight > 0:
            augmented_actions, layer_weights = self.consistency(actions_dict)
            augmented_parts = torch.cat([
                augmented_actions[name][actions_mask[name].any(dim=-1)] for name in parts
            ], dim=0)
            batch_size = valid_parts.shape[0]
            latents = self._part_encoder(
                torch.cat([part_embs, part_embs]),
                torch.cat([valid_parts, augmented_parts]),
                torch.cat([valid_mask, valid_mask]),
                torch.cat([embodiment_embs, embodiment_embs]),
            )
            quantized, _, metrics, residuals, codes = self._quantize(latents, return_level_data=True)
            metrics.update(compute_consistency_loss(residuals, codes, batch_size, layer_weights))
            consistency_loss = self.config.consistency_loss_weight * metrics["loss/consistency_loss"]
            latents, quantized = latents[:batch_size], quantized[:batch_size]
        else:
            latents = self._part_encoder(part_embs, valid_parts, valid_mask, embodiment_embs)
            quantized, _, metrics = self._quantize(latents)
            consistency_loss = 0
        quantized_actions = self._part_decoder(part_embs, quantized, valid_mask, embodiment_embs)

        # For stage3 flow matching, the decoder needs to be able to handle continuous latents
        continuous_latents = latents.detach()
        # Add noise to the continuous latents
        ratio = 0.03 * torch.rand((latents.shape[0], 1, 1), device=latents.device)
        continuous_latents = continuous_latents + torch.randn_like(continuous_latents) * ratio.to(latents.dtype)
        continuous_actions = self._part_decoder(part_embs, continuous_latents, valid_mask, embodiment_embs)

        action_mask = valid_mask.unsqueeze(1)
        num_valid_elements = valid_mask.sum() * valid_parts.shape[1]
        quantized_error = F.mse_loss(quantized_actions.float(), valid_parts.float(), reduction="none")
        continuous_error = F.mse_loss(continuous_actions.float(), valid_parts.float(), reduction="none")
        metrics["loss/quantized_recons_loss"] = quantized_error.masked_fill(~action_mask, 0).sum() / num_valid_elements
        metrics["loss/continuous_recons_loss"] = continuous_error.masked_fill(~action_mask, 0).sum() / num_valid_elements

        offset = 0
        for (part_name, part), mask in zip(parts.items(), masks):
            end = offset + part.shape[0]
            num_elements = mask.sum() * part.shape[1]
            metrics[f"mse/quantized/{part_name}"] = quantized_error[offset:end].detach().masked_fill(~mask.unsqueeze(1), 0).sum() / num_elements
            metrics[f"mse/continuous/{part_name}"] = continuous_error[offset:end].detach().masked_fill(~mask.unsqueeze(1), 0).sum() / num_elements
            offset = end

        metrics["loss"] = (
            self.config.quantized_recons_loss_weight * metrics["loss/quantized_recons_loss"]
            + self.config.continuous_recons_loss_weight * metrics["loss/continuous_recons_loss"]
            + consistency_loss
            + self.config.commit_loss_weight * metrics["loss/commit_loss"]
        )
        return metrics

    def encode(self, samples):
        """Return per-sample {part: latents[T,D]}; rule parts keep float [H,1] trajectories."""
        samples = self._split_actions(samples)
        embodiments = samples["embodiment"]
        latents = [{} for _ in embodiments]
        embodiment_embs = self.embodiment_embeddings(embodiments)
        for part in self.config.part_config:
            mask = samples["actions_mask"][part]
            present = mask.any(dim=-1)
            indices = present.nonzero().flatten().tolist()
            if not indices:
                continue
            actions = samples["actions_dict"][part][present].float()
            if part in self.rule_parts:
                part_latents = actions[:, :, :1]
            else:
                part_embs = self.part_embeddings(part).expand(len(indices), -1)
                part_latents = self._part_encoder(part_embs, actions, mask[present], embodiment_embs[present])
            for row, index in enumerate(indices):
                latents[index][part] = part_latents[row]
        return latents

    @torch.no_grad()
    def quantize(self, latents):
        """Return per-sample {part: code_ids[K,T]} without structural tokens."""
        codes = [{} for _ in latents]
        for part in self.config.part_config:
            indices = [i for i, sample in enumerate(latents) if part in sample]
            if not indices:
                continue
            part_latents = torch.stack([latents[i][part] for i in indices])
            if part in self.rule_parts:
                part_codes = torch.tensor([
                    self.rule_codec.encode(action[:, 0].tolist()) for action in part_latents
                ], device=part_latents.device).unsqueeze(1)
            else:
                _, part_codes, _ = self._quantize(part_latents)
            for row, index in enumerate(indices):
                codes[index][part] = part_codes[row]
        return codes

    def dequantize(self, codes):
        """Look up code IDs as per-sample float latents and rule trajectories."""
        latents = [{} for _ in codes]
        for part in self.config.part_config:
            indices = [i for i, sample in enumerate(codes) if part in sample]
            if not indices:
                continue
            part_codes = torch.stack([codes[i][part] for i in indices])
            if part in self.rule_parts:
                part_latents = torch.tensor([
                    self.rule_codec.decode(code[0].tolist()) for code in part_codes
                ], device=part_codes.device, dtype=torch.float32).unsqueeze(-1)
            else:
                part_latents = self.rvq.from_codes(part_codes)
            for row, index in enumerate(indices):
                latents[index][part] = part_latents[row]
        return latents

    def decode(self, latents, embodiments):
        """Reconstruct normalized [B,H,A] actions from continuous or quantized latents."""
        actions = torch.zeros(len(embodiments), self.config.action_horizon, self.config.action_dim, device=self.device)
        embodiment_embs = self.embodiment_embeddings(embodiments)
        for part in self.config.part_config:
            indices = [i for i, name in enumerate(embodiments) if self.config.part_dims[name][part]]
            if not indices:
                continue
            part_latents = torch.stack([latents[i][part] for i in indices])
            if part in self.rule_parts:
                decoded = part_latents
            else:
                mask = torch.tensor([
                    [d < len(self.config.part_dims[embodiments[i]][part]) for d in range(self.config.max_part_dim)]
                    for i in indices
                ], device=self.device)
                decoded = self._part_decoder(
                    self.part_embeddings(part).expand(len(indices), -1),
                    part_latents, mask, embodiment_embs[indices],
                )
            for row, index in enumerate(indices):
                dims = self.config.part_dims[embodiments[index]][part]
                actions[index, :, dims] = decoded[row, :, :len(dims)]
        return actions

    def serialize(self, codes, embodiments):
        """Pack code IDs with part and boundary tokens into local ID sequences."""
        sequences = [torch.tensor(self.templates[name], device=self.device) for name in embodiments]
        for i, embodiment in enumerate(embodiments):
            for part, sections in self.code_slices[embodiment].items():
                for level, section in enumerate(sections):
                    sequences[i][section] = codes[i][part][level]
        return sequences

    def deserialize(self, sequences, embodiments):
        """Extract per-part code IDs from local ID sequences."""
        return [
            {
                part: torch.stack([sequences[i][section] for section in sections])
                for part, sections in self.code_slices[name].items()
            }
            for i, name in enumerate(embodiments)
        ]

    def sequence_lengths(self, embodiments):
        return [len(self.templates[name]) for name in embodiments]

    def token_range(self, prefix, embodiment):
        """Return the allowed local ID interval [lower, upper) after prefix."""
        step = len(prefix)
        token = self.templates[embodiment][step]
        if token != -1:
            return token, token + 1
        size = self.config.codebook_size
        upper = size
        if step in self.rule_steps[embodiment]:
            start, remaining = self.rule_steps[embodiment][step]
            rank = 0
            for value in prefix[start:].tolist():
                rank = rank * size + value
            upper = min(size, (
                self.rule_codec.num_valid_sequences - 1 - rank * size ** remaining
            ) // size ** (remaining - 1) + 1)
        return 0, upper
