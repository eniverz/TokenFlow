from collections.abc import Mapping
from dataclasses import field

from transformers import AutoConfig, PretrainedConfig


class TokenFlowConfig(PretrainedConfig):
    model_type = "tokenflow_stage3"
    sub_configs = {"stage2_config": AutoConfig}
    has_no_defaults_at_init = True

    stage2_config: PretrainedConfig
    attn_implementation: str = "flash_attention_2"
    action_dim: int = 32
    action_dims: dict[str, list[int]] = field(default_factory=dict)
    initializer_range: float = 0.02
    refiner_hidden_size: int = 512
    refiner_num_heads: int = 8
    refiner_num_kv_heads: int = 8
    attention_bias: bool = False
    rope_base: int = 10000
    num_refiner_layers: int = 4
    num_inference_steps: int = 4
    fm_noise_std: float = 0.1

    def __post_init__(self, **kwargs):
        if isinstance(self.stage2_config, Mapping):
            config = dict(self.stage2_config)
            model_type = config.pop("model_type")
            self.stage2_config = AutoConfig.for_model(model_type, **config)
        super().__post_init__(**kwargs)
