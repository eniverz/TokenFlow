from dataclasses import field
from typing import List, Literal, Optional

from transformers import PretrainedConfig

class ActionCodecConfig(PretrainedConfig):
    model_type = "vq_actioncodec_v4"

    embodiment_config: List[str] = field(default_factory=lambda: ["franka"])
    part_config: List[str] = field(default_factory=lambda: ["left_arm", "right_arm", "left_gripper", "right_gripper", "torso"])

    action_dim: int = 32
    part_dims: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    action_horizon: int = 16
    max_part_dim: int = 6
    dct_norm_type: Optional[str] = 'ortho'
    rule_based_key_patterns: List[str] = field(default_factory=lambda: ["gripper"])
    rule_based_min_block_len: int = 1
    rule_based_binarize_threshold: float = 0.5

    # ————————————————————————————————
    # Attention Settings
    # ————————————————————————————————
    hidden_size: int = 512
    cross_attention_dim: int = 32
    num_heads: int = 4
    num_kv_heads: int = 4
    num_layers: int = 3
    norm_eps: float = 1e-6
    attn_dropout: float = 0.0
    attention_bias: bool = False
    use_layer_scale: bool = True
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"

    # ————————————————————————————————
    # Codebook Settings
    # ————————————————————————————————
    num_tokens: int = 16
    num_quantizers: int = 4
    codebook_size: int = 2048
    codebook_dim: int = 512
    quantizer_dropout: float = 0.2
    threshold_ema_dead: float = 2.0

    # ————————————————————————————————
    # Training Loss Weights
    # ————————————————————————————————
    commit_loss_weight: float = 0.25
    consistency_loss_weight: float = 0.1
    consistency_start_step: int = 20000
    consistency_end_step: int = 80000
    consistency_delta_max_start: int = 1
    consistency_delta_max_end: int = 3
    consistency_eps_begin: float = 0.02
    consistency_eps_end: float = 0.1
    consistency_layer_weights_end: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.3, 0.1])
    quantized_recons_loss_weight: float = 1.5
    continuous_recons_loss_weight: float = 1.0
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    ema_update_interval: int = 1 # need to be same as accumulation_steps in config

    @property
    def num_tokens_per_quantizer(self):
        return self.num_tokens // self.num_quantizers

    @property
    def vocabulary(self):
        tokens = [f"<action_{i:04d}>" for i in range(self.codebook_size)]
        for part in self.part_config:
            if any(pattern in part for pattern in self.rule_based_key_patterns):
                tokens.append(f"<{part}>")
            else:
                tokens.extend(f"<{part}_{level}>" for level in range(self.num_quantizers))
        return tokens + ["<action_start>", "<action_end>"]
