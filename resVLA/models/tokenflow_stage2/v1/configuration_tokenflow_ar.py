from dataclasses import field

from transformers import AutoTokenizer, PretrainedConfig, Qwen3VLConfig


class TokenFlowARConfig(Qwen3VLConfig):
    model_type = "tokenflow_ar"

    action_tokenizer_config: dict = field(default_factory=dict)
    action_tokenizer_path: str = ""
    action_dim: int = 32
    initializer_range: float = 0.02
    token_ids: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0
    part_dims: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    @property
    def action_tokens(self):
        cfg = self.action_tokenizer_config
        tokens = [f"<action_{i:04d}>" for i in range(cfg["codebook_size"])]
        for part in cfg["part_config"]:
            if any(pattern in part for pattern in cfg["rule_based_key_patterns"]):
                tokens.append(f"<{part}>")
            else:
                tokens.extend(f"<{part}_{level}>" for level in range(cfg["num_quantizers"]))
        tokens.extend(f"<embodiment_{name}>" for name in cfg["embodiment_config"])
        return tokens + ["<state>", "<action_start>", "<action_end>"]

    @classmethod
    def from_backbones(cls, vlm_path, action_tokenizer_path, **kwargs):
        codec_config, _ = PretrainedConfig.get_config_dict(action_tokenizer_path)
        config = cls.from_dict(
            Qwen3VLConfig.from_pretrained(vlm_path).to_dict(),
            model_type=cls.model_type,
            action_tokenizer_config=codec_config,
            action_tokenizer_path=action_tokenizer_path,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(vlm_path)
        tokenizer.add_tokens(config.action_tokens)
        config.token_ids = dict(zip(config.action_tokens, tokenizer.convert_tokens_to_ids(config.action_tokens)))
        config.vocab_size = max(len(tokenizer), config.text_config.vocab_size)
        config.pad_token_id = tokenizer.pad_token_id
        return config
