import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, Qwen3VLForConditionalGeneration

from .action_codec_wrapper import ActionCodecWrapper
from .configuration_tokenflow_ar import TokenFlowARConfig
from .modular_tokenflow_ar import FusedLinearCE


class TokenFlowAR(Qwen3VLForConditionalGeneration):
    config_class = TokenFlowARConfig
    _keep_in_fp32_modules_strict = ["action_tokenizer"]

    def __init__(self, config: TokenFlowARConfig):
        super().__init__(config)

        actioncodec_config = AutoConfig.for_model(**{
            **config.action_tokenizer_config,
            "dtype": torch.float32,
            "attn_implementation": config._attn_implementation
        })
        self.config.action_tokenizer_config = actioncodec_config.to_dict()
        action_tokenizer = AutoModel.from_config(
            actioncodec_config,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        self.action_tokenizer = ActionCodecWrapper(action_tokenizer)
        hidden_size = config.text_config.hidden_size
        self.state_encoder = nn.Sequential(
            nn.Linear(config.action_dim, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        cfg = self.action_tokenizer.config
        self.action_token_begin = config.token_ids["<action_0000>"]
        self.rule_parts = [
            part for part in cfg.part_config
            if any(pattern in part for pattern in cfg.rule_based_key_patterns)
        ]
        self.nn_parts = [part for part in cfg.part_config if part not in self.rule_parts]
        self.templates, self.code_slices, self.rule_steps = {}, {}, {}
        for embodiment, dims in config.part_dims.items():
            template = [config.token_ids["<action_start>"]]
            slices = {part: [] for part in cfg.part_config if dims[part]}
            rule_steps = {}
            for level in range(cfg.num_quantizers):
                for part in self.nn_parts:
                    if dims[part]:
                        template.append(config.token_ids[f"<{part}_{level}>"])
                        start = len(template)
                        template.extend([-1] * (cfg.num_tokens // cfg.num_quantizers))
                        slices[part].append(slice(start, len(template)))
            for part in self.rule_parts:
                if dims[part]:
                    template.append(config.token_ids[f"<{part}>"])
                    start = len(template)
                    length = self.action_tokenizer.rule_codec.num_tokens
                    template.extend([-1] * length)
                    slices[part].append(slice(start, len(template)))
                    for i in range(length):
                        rule_steps[start + i] = (start, length - i)
            template.append(config.token_ids["<action_end>"])
            self.templates[embodiment] = template
            self.code_slices[embodiment] = slices
            self.rule_steps[embodiment] = rule_steps

        self.action_tokenizer.requires_grad_(False)
        self.action_tokenizer.eval()
        self.post_init()

    def train(self, mode=True):
        super().train(mode)
        self.action_tokenizer.eval()
        return self

    @torch.no_grad()
    def _encode_actions(self, samples):
        codec = self.action_tokenizer
        device = samples["states"].device
        embodiments = samples["embodiment"]
        sequences = [torch.tensor(self.templates[name], device=device) for name in embodiments]
        with torch.autocast(device_type=device.type, enabled=False):
            codes = codec.encode(samples)
        for i, embodiment in enumerate(embodiments):
            for part, sections in self.code_slices[embodiment].items():
                for level, section in enumerate(sections):
                    sequences[i][section] = codes[i][part][level] + self.action_token_begin
        return sequences

    def _prepare_inputs(self, samples, action_ids):
        inputs = samples["vlm_inputs"]
        device = inputs["input_ids"].device
        sequences, token_types, labels = [], [], []
        for i, embodiment in enumerate(samples["embodiment"]):
            valid = inputs["attention_mask"][i].bool()
            condition = torch.tensor([
                self.config.token_ids[f"<embodiment_{embodiment}>"],
                self.config.token_ids["<state>"],
            ], device=device)
            prefix = torch.cat([inputs["input_ids"][i, valid], condition])
            sequences.append(torch.cat([prefix, action_ids[i]]))
            token_types.append(torch.cat([
                inputs["mm_token_type_ids"][i, valid],
                torch.zeros(len(condition) + len(action_ids[i]), device=device, dtype=torch.long),
            ]))
            labels.append(torch.cat([torch.full_like(prefix, -100), action_ids[i]]))

        input_ids = pad_sequence(sequences, batch_first=True, padding_value=self.config.pad_token_id, padding_side="left")
        attention_mask = pad_sequence([torch.ones_like(seq) for seq in sequences], batch_first=True, padding_side="left")
        mm_token_type_ids = pad_sequence(token_types, batch_first=True, padding_side="left")
        labels = pad_sequence(labels, batch_first=True, padding_value=-100, padding_side="left")
        position_ids, _ = self.model.get_rope_index(
            input_ids, mm_token_type_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
        )
        inputs_embeds = self.get_input_embeddings()(input_ids)
        states = samples["states"] * samples["state_mask"].unsqueeze(1)
        state_embeds = self.state_encoder(states.to(self.state_encoder[0].weight.dtype))
        inputs_embeds = inputs_embeds.masked_scatter(
            (input_ids == self.config.token_ids["<state>"]).unsqueeze(-1),
            state_embeds.to(inputs_embeds.dtype),
        )
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "pixel_values_videos": inputs.get("pixel_values_videos"),
            "video_grid_thw": inputs.get("video_grid_thw"),
        }, labels

    def forward(self, samples):
        action_ids = self._encode_actions(samples)
        inputs, labels = self._prepare_inputs(samples, action_ids)
        outputs = self.model(**inputs, use_cache=False, return_dict=True)
        hidden_states, labels = outputs.last_hidden_state[:, :-1], labels[:, 1:]
        valid = labels != -100
        labels = labels[valid]
        loss, correct = FusedLinearCE.apply(self.lm_head.weight, hidden_states[valid], labels)
        action_mask = (labels >= self.action_token_begin) & (
            labels < self.action_token_begin + self.action_tokenizer.config.codebook_size
        )
        accuracy = (correct * action_mask).sum() / action_mask.sum()
        return {"loss": loss, "action_token": accuracy.detach()}

    @torch.inference_mode()
    def predict_action(self, samples):
        """Generate and decode normalized actions without ground-truth action fields."""
        device = samples["states"].device
        embodiments = samples["embodiment"]
        batch_size = len(embodiments)
        codec = self.action_tokenizer
        codebook_size = codec.config.codebook_size
        empty = [torch.empty(0, device=device, dtype=torch.long) for _ in embodiments]
        inputs, _ = self._prepare_inputs(samples, empty)
        attention_mask = inputs["attention_mask"]
        next_position = inputs["position_ids"].amax(dim=(0, 2)) + 1
        lengths = torch.tensor([len(self.templates[name]) for name in embodiments], device=device)
        max_length = int(lengths.max())
        generated = torch.full((batch_size, max_length), self.config.pad_token_id, device=device, dtype=torch.long)
        outputs = super().forward(**inputs, logits_to_keep=1, use_cache=True, return_dict=True)

        for step in range(max_length):
            for i, embodiment in enumerate(embodiments):
                template = self.templates[embodiment]
                if step >= len(template):
                    continue
                token = template[step]
                if token == -1:
                    upper = codebook_size
                    if step in self.rule_steps[embodiment]:
                        start, remaining = self.rule_steps[embodiment][step]
                        rank = 0
                        for value in generated[i, start:step].tolist():
                            rank = rank * codebook_size + value - self.action_token_begin
                        upper = min(codebook_size, (
                            codec.rule_codec.num_valid_sequences - 1 - rank * codebook_size ** remaining
                        ) // codebook_size ** (remaining - 1) + 1)
                    token = outputs.logits[i, -1, self.action_token_begin:self.action_token_begin + upper].argmax() + self.action_token_begin
                generated[i, step] = token
            if step + 1 < max_length:
                attention_mask = torch.cat([attention_mask, (step < lengths).long().unsqueeze(1)], dim=1)
                outputs = super().forward(
                    input_ids=generated[:, step:step + 1],
                    attention_mask=attention_mask,
                    position_ids=(next_position + step).view(1, batch_size, 1).expand(3, -1, -1),
                    past_key_values=outputs.past_key_values,
                    logits_to_keep=1,
                    use_cache=True,
                    return_dict=True,
                )

        codes = [
            {
                part: torch.stack([generated[i, section] for section in sections]) - self.action_token_begin
                for part, sections in self.code_slices[name].items()
            }
            for i, name in enumerate(embodiments)
        ]
        with torch.autocast(device_type=device.type, enabled=False):
            return codec.decode(codes, embodiments, self.config.part_dims, self.config.action_dim)
