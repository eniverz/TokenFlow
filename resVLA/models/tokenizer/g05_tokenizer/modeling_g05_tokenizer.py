import torch
from transformers import PreTrainedModel

from .configuration_g05_tokenizer import G05TokenizerConfig
from .native.constrained_tokenizer import ConstrainedSequenceTokenizer
from .native.modeling_actioncodec2v2 import ActionCodecV2Model


class G05Tokenizer(PreTrainedModel):
    config_class = G05TokenizerConfig

    def __init__(self, config: G05TokenizerConfig):
        super().__init__(config)
        self.codec = ActionCodecV2Model(config)
        self.rule_codec = ConstrainedSequenceTokenizer(
            seq_len=config.action_horizon,
            min_block_len=config.rule_based_min_block_len,
            vocab_size=config.codebook_size,
            binarize_threshold=config.rule_based_binarize_threshold,
        )
        self.rule_parts = [
            part for part in config.part_config
            if any(pattern in part for pattern in config.rule_based_key_patterns)
        ]
        self.post_init()

    @torch.no_grad()
    def encode(self, samples):
        codes = [{} for _ in samples["embodiment"]]
        for part in self.config.part_config:
            mask = samples["actions_mask"][part]
            present = mask.any(dim=-1)
            indices = present.nonzero().flatten().tolist()
            if not indices:
                continue
            actions = samples["actions_dict"][part][present].float()
            actions = actions * mask[present].unsqueeze(1)
            if part in self.rule_parts:
                part_codes = torch.tensor([
                    self.rule_codec.encode(self.rule_codec.binarize(action[:, 0].tolist()))
                    for action in actions
                ], device=actions.device).unsqueeze(1)
            else:
                part_codes = self.codec.encode({part: actions})[part][:, :self.config.num_quantizers]
            for row, index in enumerate(indices):
                codes[index][part] = part_codes[row]
        return codes

    @torch.no_grad()
    def decode(self, codes, embodiments, part_dims, action_dim):
        actions = torch.zeros(len(embodiments), self.config.action_horizon, action_dim, device=self.device)
        for part in self.config.part_config:
            indices = [i for i, name in enumerate(embodiments) if part_dims[name][part]]
            if not indices:
                continue
            part_codes = torch.stack([codes[i][part] for i in indices])
            if part in self.rule_parts:
                decoded = torch.tensor([
                    self.rule_codec.decode(code[0].tolist(), safe=False) for code in part_codes
                ], device=actions.device, dtype=actions.dtype).unsqueeze(-1)
                decoded = decoded * 2 - 1
            else:
                decoded = self.codec.decode({part: part_codes})[part]
            for row, index in enumerate(indices):
                dims = part_dims[embodiments[index]][part]
                actions[index, :, dims] = decoded[row, :, :len(dims)]
        return actions
