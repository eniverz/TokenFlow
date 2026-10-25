import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoModelForImageTextToText, PreTrainedModel

from .configuration_tokenflow_ar import TokenFlowPDConfig
from .modular_tokenflow_ar import FusedLinearCE


class TokenFlowPD(PreTrainedModel):
    config_class = TokenFlowPDConfig
    base_model_prefix = "vlm"
    _keep_in_fp32_modules_strict = ["action_tokenizer"]

    def __init__(self, config: TokenFlowPDConfig):
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
        max_length = max(self.action_tokenizer.sequence_lengths(config.action_tokenizer_config.embodiment_config))
        self.action_queries = nn.Parameter(torch.randn(max_length, hidden_size) * config.initializer_range)

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
            sequences.append(torch.cat([prefix, torch.full_like(action_ids[i], self.action_token_begin)]))
            token_types.append(torch.cat([
                inputs["mm_token_type_ids"][i, valid],
                torch.zeros(len(condition) + len(action_ids[i]), device=device, dtype=torch.long),
            ]))
            labels.append(torch.cat([torch.full_like(prefix, -100), action_ids[i]]))

        input_ids = pad_sequence(sequences, batch_first=True, padding_value=self.config.pad_token_id, padding_side="left")
        attention_mask = pad_sequence([torch.ones_like(seq) for seq in sequences], batch_first=True, padding_side="left")
        mm_token_type_ids = pad_sequence(token_types, batch_first=True, padding_side="left")
        labels = pad_sequence(labels, batch_first=True, padding_value=-100, padding_side="left")
        labels = labels.masked_fill(
            (labels < self.action_token_begin)
            | (labels >= self.action_token_begin + self.action_tokenizer.config.codebook_size), -100,
        )
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
        for i, sequence in enumerate(action_ids):
            inputs_embeds[i, -len(sequence):] = self.action_queries[:len(sequence)].to(inputs_embeds.dtype)
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
        hidden_states = outputs.last_hidden_state
        valid = labels != -100
        labels = labels[valid]
        loss, correct = FusedLinearCE.apply(self.vlm.lm_head.weight, hidden_states[valid], labels)
        accuracy = correct.float().mean()
        return {"loss": loss, "action_token": accuracy.detach()}

    @torch.no_grad()
    def predict_action(self, samples, num_context_layers=0):
        """Decode all query logits in one pass, optionally returning masked prefix contexts."""
        device = samples["states"].device
        embodiments = samples["embodiment"]
        sequence_lengths = self.action_tokenizer.sequence_lengths(embodiments)
        placeholders = [torch.full((length,), -100, device=device, dtype=torch.long) for length in sequence_lengths]
        inputs, _ = self._prepare_inputs(samples, placeholders)
        outputs = self.vlm.model(
            **inputs, use_cache=False,
            output_hidden_states=num_context_layers > 0, return_dict=True,
        )
        logits = F.linear(
            outputs.last_hidden_state[:, -max(sequence_lengths):],
            self.vlm.lm_head.weight[
                self.action_token_begin:self.action_token_begin + self.action_tokenizer.config.codebook_size
            ],
        )
        if num_context_layers:
            contexts = outputs.hidden_states[1:][-num_context_layers:]
            context_mask = inputs["attention_mask"].bool()
            for i, length in enumerate(sequence_lengths):
                context_mask[i, -length:] = False

        sequences = []
        for i, (embodiment, length) in enumerate(zip(embodiments, sequence_lengths)):
            sequence = logits[i, -length:].argmax(dim=-1)
            # Codec constraints select from fixed logits; no predicted token is fed back to the VLM.
            for step in range(length):
                lower, upper = self.action_tokenizer.token_range(sequence[:step], embodiment)
                if upper == lower + 1:
                    sequence[step] = lower
                elif lower != 0 or upper != self.action_tokenizer.config.codebook_size:
                    sequence[step] = logits[i, step - length, lower:upper].argmax() + lower
            sequences.append(sequence)
        with torch.autocast(device_type=device.type, enabled=False):
            codes = self.action_tokenizer.deserialize(sequences, embodiments)
            latents = self.action_tokenizer.dequantize(codes)
            actions = self.action_tokenizer.decode(latents, embodiments)
        if num_context_layers:
            return actions, contexts, context_mask
        return actions
