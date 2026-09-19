from collections.abc import Mapping
from dataclasses import field
from typing import Literal, Optional, List

from transformers import AutoConfig, PretrainedConfig, Qwen3VLConfig

class ResVLAConfig(PretrainedConfig):
    model_type = "resvla"

    sub_configs = {"vlm_config": AutoConfig}

    # ————————————————————————————————
    # vlm backend config
    # ————————————————————————————————
    attn_implementation: Literal[
        "eager", "sdpa", "flash_attention_2", "flash_attention_3"
    ] = "flash_attention_2"
    vlm_config: Optional[PretrainedConfig] = None

    # ————————————————————————————————
    # time sampler noise parameters
    # ————————————————————————————————
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999

    # ————————————————————————————————
    # Anchor DiT parameters
    # ————————————————————————————————
    dit_q_heads: int = 16
    dit_kv_heads: int = 16
    dit_dropout: float = 0.0
    num_anchor_layers: int = 19

    # ————————————————————————————————
    # Refiner parameters
    # ————————————————————————————————
    num_refiner_layers: int = 9
    num_repeat_diffusion_steps: int = 2
    num_inference_steps: int = 4
    fm_noise_std: float = 1
    inference_fm_noise_std: float = 0.5

    # ————————————————————————————————
    # Action config
    # ————————————————————————————————
    action_horizon: int = 16
    action_dim: int = 32
    embodiment_config: List[str] = field(default_factory=lambda: ["franka"])

    # ————————————————————————————————
    # relative loss weight
    # ————————————————————————————————
    anchor_loss_weight: float = 0.1

    def __post_init__(self, **kwargs):
        if self.vlm_config is None:
            self.vlm_config = Qwen3VLConfig()
        elif isinstance(self.vlm_config, Mapping):
            vlm_cfg = dict(self.vlm_config)
            model_type = vlm_cfg.pop("model_type")
            self.vlm_config = AutoConfig.for_model(model_type, **vlm_cfg)

        super().__post_init__(**kwargs)

    @property
    def hidden_size(self):
        return self.vlm_config.text_config.hidden_size
