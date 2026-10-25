import torch
import torch.nn.functional as F
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

        token_ids = {token: i for i, token in enumerate(config.vocabulary)}
        nn_parts = [part for part in config.part_config if part not in self.rule_parts]
        self.templates, self.code_slices, self.rule_steps = {}, {}, {}
        for embodiment, dims in config.part_dims.items():
            template = [token_ids["<action_start>"]]
            slices = {part: [] for part in config.part_config if dims[part]}
            rule_steps = {}
            for level in range(config.num_quantizers):
                for part in nn_parts:
                    if dims[part]:
                        template.append(token_ids[f"<{part}_{level}>"])
                        start = len(template)
                        template.extend([-1] * config.num_tokens_per_quantizer)
                        slices[part].append(slice(start, len(template)))
            for part in self.rule_parts:
                if dims[part]:
                    template.append(token_ids[f"<{part}>"])
                    start = len(template)
                    length = self.rule_codec.num_tokens
                    template.extend([-1] * length)
                    slices[part].append(slice(start, len(template)))
                    for i in range(length):
                        rule_steps[start + i] = (start, length - i)
            template.append(token_ids["<action_end>"])
            self.templates[embodiment] = template
            self.code_slices[embodiment] = slices
            self.rule_steps[embodiment] = rule_steps

    def encode(self, samples):
        """Return per-sample {part: latents[T,D]}; rule parts keep float [H,1] trajectories."""
        latents = [{} for _ in samples["embodiment"]]
        for part in self.config.part_config:
            actions, masks = [], []
            for i, embodiment in enumerate(samples["embodiment"]):
                dims = self.config.part_dims[embodiment][part]
                padding = (0, self.config.max_part_dim - len(dims))
                actions.append(F.pad(samples["actions"][i, :, dims], padding))
                masks.append(F.pad(samples["action_mask"][i, dims], padding))
            actions, mask = torch.stack(actions).float(), torch.stack(masks)
            present = mask.any(dim=-1)
            indices = present.nonzero().flatten().tolist()
            if not indices:
                continue
            actions = actions[present] * mask[present].unsqueeze(1)
            if part in self.rule_parts:
                part_latents = actions[:, :, :1]
            else:
                if self.codec.block_dct is not None:
                    actions = self.codec.block_dct.dct(actions)
                features = self.codec._pretransform(actions, self.config.horizon_patch_size)
                features = self.codec.encoder(self.codec.conv_in(features))
                part_latents = features.flatten(2).transpose(1, 2)
            for row, index in enumerate(indices):
                latents[index][part] = part_latents[row]
        return latents

    @torch.no_grad()
    def quantize(self, latents):
        """Return per-sample {part: code_ids[K,T]} without structural tokens."""
        codes = [{} for _ in latents]
        for part in self.config.part_config:
            indices = [i for i, sample in enumerate(latents) if part in sample]
            if not indices:
                continue
            part_latents = torch.stack([latents[i][part] for i in indices])
            if part in self.rule_parts:
                part_codes = torch.tensor([
                    self.rule_codec.encode(self.rule_codec.binarize(action[:, 0].tolist()))
                    for action in part_latents
                ], device=part_latents.device).unsqueeze(1)
            else:
                _, part_codes, _ = self.codec.rvq(part_latents.transpose(1, 2).contiguous())
                part_codes = part_codes[:, :self.config.num_quantizers]
            for row, index in enumerate(indices):
                codes[index][part] = part_codes[row]
        return codes

    def dequantize(self, codes):
        """Look up code IDs as per-sample float latents and -1/+1 rule trajectories."""
        latents = [{} for _ in codes]
        for part in self.config.part_config:
            indices = [i for i, sample in enumerate(codes) if part in sample]
            if not indices:
                continue
            part_codes = torch.stack([codes[i][part] for i in indices])
            if part in self.rule_parts:
                part_latents = torch.tensor([
                    self.rule_codec.decode(code[0].tolist(), safe=False) for code in part_codes
                ], device=part_codes.device, dtype=torch.float32).unsqueeze(-1)
                part_latents = part_latents * 2 - 1
            else:
                part_latents = self.codec.rvq.from_codes(part_codes).transpose(1, 2)
            for row, index in enumerate(indices):
                latents[index][part] = part_latents[row]
        return latents

    def decode(self, latents, embodiments):
        """Reconstruct normalized [B,H,A] actions from continuous or quantized latents."""
        actions = torch.zeros(len(embodiments), self.config.action_horizon, self.config.action_dim, device=self.device)
        for part in self.config.part_config:
            indices = [i for i, name in enumerate(embodiments) if self.config.part_dims[name][part]]
            if not indices:
                continue
            part_latents = torch.stack([latents[i][part] for i in indices])
            if part in self.rule_parts:
                decoded = part_latents
            else:
                decoded = self.codec._decode_zq(part_latents.transpose(1, 2).contiguous())
                if self.codec.block_dct is not None:
                    decoded = self.codec.block_dct.idct(decoded, self.config.action_horizon)
                decoded = decoded[:, :self.config.action_horizon]
            for row, index in enumerate(indices):
                dims = self.config.part_dims[embodiments[index]][part]
                actions[index, :, dims] = decoded[row, :, :len(dims)]
        return actions

    def serialize(self, codes, embodiments):
        """Pack code IDs with part and boundary tokens into local ID sequences."""
        sequences = [torch.tensor(self.templates[name], device=self.device) for name in embodiments]
        for i, embodiment in enumerate(embodiments):
            for part, sections in self.code_slices[embodiment].items():
                for level, section in enumerate(sections):
                    sequences[i][section] = codes[i][part][level]
        return sequences

    def deserialize(self, sequences, embodiments):
        """Extract per-part code IDs from local ID sequences."""
        return [
            {
                part: torch.stack([sequences[i][section] for section in sections])
                for part, sections in self.code_slices[name].items()
            }
            for i, name in enumerate(embodiments)
        ]

    def sequence_lengths(self, embodiments):
        return [len(self.templates[name]) for name in embodiments]

    def token_range(self, prefix, embodiment):
        """Return the allowed local ID interval [lower, upper) after prefix."""
        step = len(prefix)
        token = self.templates[embodiment][step]
        if token != -1:
            return token, token + 1
        size = self.config.codebook_size
        upper = size
        if step in self.rule_steps[embodiment]:
            start, remaining = self.rule_steps[embodiment][step]
            rank = 0
            for value in prefix[start:].tolist():
                rank = rank * size + value
            upper = min(size, (
                self.rule_codec.num_valid_sequences - 1 - rank * size ** remaining
            ) // size ** (remaining - 1) + 1)
        return 0, upper
