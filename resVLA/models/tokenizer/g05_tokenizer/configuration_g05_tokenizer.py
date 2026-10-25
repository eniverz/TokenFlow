import math
from dataclasses import field
from typing import List, Optional

from transformers import PretrainedConfig


class G05TokenizerConfig(PretrainedConfig):
    model_type = "g05_tokenizer"

    embodiment_config: List[str] = field(default_factory=lambda: ["franka"])
    part_config: List[str] = field(default_factory=lambda: ["left_arm", "right_arm", "left_gripper", "right_gripper", "torso"])
    action_dim: int = 32
    part_dims: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    action_horizon: int = 32
    max_part_dim: int = 9
    rule_based_key_patterns: List[str] = field(default_factory=lambda: ["gripper"])
    rule_based_min_block_len: int = 1
    rule_based_binarize_threshold: float = 0.0
    attn_implementation: str = "sdpa"

    horizon_patch_size: int = 8
    conv_in_action_kernel: int = 2
    encoder_channels: int = 256
    latent_dim: int = 128
    c_mults: List[int] = field(default_factory=lambda: [1, 2, 2])
    strides: List[List[int]] = field(default_factory=lambda: [[1, 1], [2, 1], [2, 1]])
    transformer_depths: List[int] = field(default_factory=lambda: [2, 2, 2])
    num_heads: int = 8
    dim_heads: int = 64
    ffn_mult: float = 4.0
    use_layer_scale: bool = True
    layer_scale_init: float = 0.01
    use_qk_norm: bool = True
    rope_base: int = 10000
    use_block_dct: bool = True
    block_dct_block_size: int = 8

    num_quantizers: int = 2
    n_codebooks: int = 4
    codebook_size: int = 4096
    codebook_dim: int = 8
    quantizer_dropout: float = 0.5
    ema_decay: float = 0.95
    threshold_ema_dead: float = 2.0
    use_rotation_trick: bool = False
    use_fsq: bool = False
    fsq_levels: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 4, 4])
    use_lipschitz_encoder: bool = False

    commitment_loss_weight: float = 0.25
    reconstruction_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.0
    consistency_delta_max_begin: int = 1
    consistency_delta_max_end: int = 3
    consistency_eps_begin: float = 0.02
    consistency_eps_end: float = 0.1
    consistency_layer_weights_begin: Optional[List[float]] = None
    consistency_layer_weights_end: Optional[List[float]] = None
    consistency_schedule_begin_fraction: float = 0.2
    consistency_schedule_end_fraction: float = 0.8

    @property
    def horizon(self):
        return self.action_horizon

    @property
    def max_component_dim(self):
        return self.max_part_dim

    @property
    def code_a(self):
        return self.max_part_dim - self.conv_in_action_kernel + 1

    @property
    def code_h(self):
        h = self.action_horizon // self.horizon_patch_size
        for stride_h, _ in self.strides:
            h //= stride_h
        return h

    @property
    def num_tokens_per_quantizer(self):
        return self.code_h * self.code_a

    @property
    def num_tokens(self):
        return self.num_tokens_per_quantizer * self.num_quantizers

    @property
    def tokens_per_key(self):
        return self.num_tokens_per_quantizer * self.n_codebooks

    @property
    def vocab_size_with_specials(self):
        return math.prod(self.fsq_levels) if self.use_fsq else self.codebook_size

    @property
    def vocabulary(self):
        tokens = [f"<action_{i:04d}>" for i in range(self.codebook_size)]
        for part in self.part_config:
            if any(pattern in part for pattern in self.rule_based_key_patterns):
                tokens.append(f"<{part}>")
            else:
                tokens.extend(f"<{part}_{level}>" for level in range(self.num_quantizers))
        return tokens + ["<action_start>", "<action_end>"]
