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

    def forward(self, samples):
        """
        Forward pass for the ActionCodec.

        :param samples: A dictionary containing the input samples. Expected keys are:

            - actions: Tensor of shape (batch_size, action_horizon, action_dim) representing the input actions.
            - actions_dict: Dict[part_name, actions]
            - actions_mask: Dict[part_name, mask] where mask is a boolean tensor of shape (batch_size, max_part_dim) indicating valid actions.
            - embodiment: List of str (batch_size) representing the embodiment types for each sample.

        Returns scalar loss metrics for NN parts only. Rule-based parts bypass
        augmentation, reconstruction, consistency, and commitment losses.
        """
        actions_dict = {
            name: actions for name, actions in samples["actions_dict"].items()
            if not any(pattern in name for pattern in self.config.rule_based_key_patterns)
        }
        actions_mask = samples["actions_mask"]
        embodiment_names = samples["embodiment"]

        all_embodiment_embs = self.embodiment_embeddings(embodiment_names)

        augmented_actions, layer_weights = self.consistency(actions_dict)

        parts, masks, embodiment_embs, part_embs, augmented_parts = {}, [], [], [], []

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
            augmented_parts.append(augmented_actions[part_name][present])

        valid_parts = torch.cat(list(parts.values()), dim=0)
        valid_mask = torch.cat(masks, dim=0)
        embodiment_embs = torch.cat(embodiment_embs, dim=0)
        part_embs = torch.cat(part_embs, dim=0)
        augmented_parts = torch.cat(augmented_parts, dim=0)

        batch_size = valid_parts.shape[0]
        latents = self._part_encoder(
            torch.cat([part_embs, part_embs]),
            torch.cat([valid_parts, augmented_parts]),
            torch.cat([valid_mask, valid_mask]),
            torch.cat([embodiment_embs, embodiment_embs]),
        )
        quantized, _, metrics, residuals, codes = self._quantize(latents, return_level_data=True)
        metrics.update(compute_consistency_loss(residuals, codes, batch_size, layer_weights))
        latents, quantized = latents[:batch_size], quantized[:batch_size]
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
            + self.config.consistency_loss_weight * metrics["loss/consistency_loss"]
            + self.config.commit_loss_weight * metrics["loss/commit_loss"]
        )
        return metrics
