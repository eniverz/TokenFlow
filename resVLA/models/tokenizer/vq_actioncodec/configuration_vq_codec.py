from dataclasses import field
from typing import List, Literal, Optional

from transformers import PretrainedConfig

class ActionCodecConfig(PretrainedConfig):
    model_type = "vq_actioncodec"

    embodiment_config: List[str] = field(default_factory=lambda: ["franka"])
    # TODO: assume that each embodiment has full set of parts in `action_dict`.
    part_config: List[str] = field(default_factory=lambda: ["left_arm", "right_arm", "left_gripper", "right_gripper", "torso"])

    action_horizon: int = 16
    max_part_dim: int = 6
    dct_norm_type: Optional[str] = 'ortho'

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
    consistency_delta_max: int = 1
    quantized_recons_loss_weight: float = 1.5
    continuous_recons_loss_weight: float = 1.0
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    ema_update_interval: int = 1 # need to be same as accumulation_steps in config

    def __post_init__(self, **kwargs):
        if self.num_quantizers < 1:
            raise ValueError(f"num_quantizers ({self.num_quantizers}) must be >= 1")

        if self.codebook_size < 1:
            raise ValueError(f"codebook_size ({self.codebook_size}) must be >= 1")

        if self.codebook_dim < 1:
            raise ValueError(f"codebook_dim ({self.codebook_dim}) must be >= 1")
        
        if self.num_tokens % self.num_quantizers != 0:
            raise ValueError(f"num_tokens ({self.num_tokens}) must be divisible by num_quantizers ({self.num_quantizers})")

        super().__post_init__(**kwargs)

    @property
    def num_tokens_per_quantizer(self):
        return self.num_tokens // self.num_quantizers
