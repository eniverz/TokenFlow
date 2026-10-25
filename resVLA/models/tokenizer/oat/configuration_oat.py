import math
from dataclasses import field
from typing import List, Literal

from transformers import PretrainedConfig


class OATConfig(PretrainedConfig):
    model_type = "oat"

    embodiment_config: List[str] = field(default_factory=lambda: ["franka"])
    action_dim: int = 32
    action_horizon: int = 16
    hidden_size: int = 256
    num_heads: int = 8
    num_kv_heads: int = 8
    attention_bias: bool = False
    encoder_depth: int = 6
    decoder_depth: int = 6
    dropout: float = 0.1
    norm_eps: float = 1e-6
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"
    num_tokens: int = 16
    quantizer_levels: List[int] = field(default_factory=lambda: [8, 8, 6, 5])

    quantized_recons_loss_weight: float = 1.5
    continuous_recons_loss_weight: float = 1.0
    continuous_noise_std: float = 0.03

    @property
    def latent_dim(self):
        return len(self.quantizer_levels)

    @property
    def cross_attention_dim(self):
        return self.hidden_size

    @property
    def codebook_size(self):
        return math.prod(self.quantizer_levels)
