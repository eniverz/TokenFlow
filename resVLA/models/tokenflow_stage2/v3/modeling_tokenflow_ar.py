import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoModelForImageTextToText, PreTrainedModel

from .configuration_tokenflow_ar import TokenFlowARConfig
from .modular_tokenflow_ar import FusedLinearCE


class TokenFlowAR(PreTrainedModel):
    config_class = TokenFlowARConfig
    base_model_prefix = "vlm"
    _keep_in_fp32_modules_strict = ["action_tokenizer"]

    def __init__(self, config: TokenFlowARConfig):
        super().__init__(config)

        self.vlm = AutoModelForImageTextToText.from_config( # use AutoModelForImageTextToText to support lm_head
            config.vlm_config,
            attn_implementation=config.attn_implementation,
            dtype=config.vlm_config.dtype,
        )
        self.action_tokenizer = AutoModel.from_config(
            config.action_tokenizer_config,
            dtype=torch.float32,
        )
        hidden_size = config.vlm_config.text_config.hidden_size
        self.state_encoder = nn.Sequential(
            nn.Linear(config.action_dim, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        self.action_token_begin = config.token_ids["<action_0000>"]

        self.action_tokenizer.requires_grad_(False)
        self.action_tokenizer.eval()
        self.post_init()

    def train(self, mode=True):
        super().train(mode)
        self.action_tokenizer.eval()
        return self

    @torch.no_grad()
    def _encode_actions(self, samples):
        device = samples["states"].device
        with torch.autocast(device_type=device.type, enabled=False):
            latents = self.action_tokenizer.encode(samples)
            codes = self.action_tokenizer.quantize(latents)
            sequences = self.action_tokenizer.serialize(codes, samples["embodiment"])
        return [sequence + self.action_token_begin for sequence in sequences]

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
        position_ids, _ = self.vlm.model.get_rope_index(
            input_ids, mm_token_type_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
        )
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
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
        outputs = self.vlm.model(**inputs, use_cache=False, return_dict=True)
        hidden_states, labels = outputs.last_hidden_state[:, :-1], labels[:, 1:]
        valid = labels != -100
        labels = labels[valid]
        loss, correct = FusedLinearCE.apply(self.vlm.lm_head.weight, hidden_states[valid], labels)
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
        empty = [torch.empty(0, device=device, dtype=torch.long) for _ in embodiments]
        inputs, _ = self._prepare_inputs(samples, empty)
        attention_mask = inputs["attention_mask"]
        next_position = inputs["position_ids"].amax(dim=(0, 2)) + 1
        sequence_lengths = self.action_tokenizer.sequence_lengths(embodiments)
        lengths = torch.tensor(sequence_lengths, device=device)
        max_length = int(lengths.max())
        generated = torch.full((batch_size, max_length), self.config.pad_token_id, device=device, dtype=torch.long)
        outputs = self.vlm(**inputs, logits_to_keep=1, use_cache=True, return_dict=True)

        for step in range(max_length):
            for i, embodiment in enumerate(embodiments):
                if step >= sequence_lengths[i]:
                    continue
                prefix = generated[i, :step] - self.action_token_begin
                lower, upper = self.action_tokenizer.token_range(prefix, embodiment)
                if upper == lower + 1:
                    token = lower + self.action_token_begin
                else:
                    start = self.action_token_begin + lower
                    end = self.action_token_begin + upper
                    token = outputs.logits[i, -1, start:end].argmax() + start
                generated[i, step] = token
            if step + 1 < max_length:
                attention_mask = torch.cat([attention_mask, (step < lengths).long().unsqueeze(1)], dim=1)
                outputs = self.vlm(
                    input_ids=generated[:, step:step + 1],
                    attention_mask=attention_mask,
                    position_ids=(next_position + step).view(1, batch_size, 1).expand(3, -1, -1),
                    past_key_values=outputs.past_key_values,
                    logits_to_keep=1,
                    use_cache=True,
                    return_dict=True,
                )

        sequences = [
            generated[i, :length] - self.action_token_begin
            for i, length in enumerate(sequence_lengths)
        ]
        with torch.autocast(device_type=device.type, enabled=False):
            codes = self.action_tokenizer.deserialize(sequences, embodiments)
            latents = self.action_tokenizer.dequantize(codes)
            return self.action_tokenizer.decode(latents, embodiments)
