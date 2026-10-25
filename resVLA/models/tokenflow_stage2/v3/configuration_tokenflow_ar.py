from collections.abc import Mapping
from dataclasses import field

from transformers import AutoConfig, PretrainedConfig


class TokenFlowARConfig(PretrainedConfig):
    model_type = "tokenflow_ar_v3"
    sub_configs = {"vlm_config": AutoConfig, "action_tokenizer_config": AutoConfig}
    has_no_defaults_at_init = True

    vlm_config: PretrainedConfig
    action_tokenizer_config: PretrainedConfig
    attn_implementation: str = "flash_attention_2"
    action_dim: int = 32
    initializer_range: float = 0.02
    token_ids: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        for name in self.sub_configs:
            config = getattr(self, name)
            if isinstance(config, Mapping):
                config = dict(config)
                model_type = config.pop("model_type")
                setattr(self, name, AutoConfig.for_model(model_type, **config))
        self.action_tokenizer_config.attn_implementation = self.attn_implementation
        if not self.token_ids:
            start = self.vlm_config.text_config.vocab_size
            self.token_ids = {token: start + i for i, token in enumerate(self.action_tokens)}
        self.vocab_size = max(self.token_ids.values()) + 1
        self.pad_token_id = self.vlm_config.text_config.eos_token_id
        super().__post_init__(**kwargs)

    @property
    def hidden_size(self):
        return self.vlm_config.text_config.hidden_size

    @property
    def action_tokens(self):
        cfg = self.action_tokenizer_config
        tokens = list(cfg.vocabulary)
        tokens.extend(f"<embodiment_{name}>" for name in cfg.embodiment_config)
        return tokens + ["<state>"]
