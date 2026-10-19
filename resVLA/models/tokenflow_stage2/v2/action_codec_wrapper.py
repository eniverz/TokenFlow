import torch
import torch.nn as nn
from transformers import PreTrainedModel


class ActionCodecWrapper(nn.Module):
    """Wrap a loaded codec; encode returns one {part: codes[K,T]} dict per sample.

    Native decode receives (codes, embodiments, part_dims, action_dim) and returns
    normalized [B,H,D] actions. Codecs without these methods use the v3 modules.
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.config = model.config
        self.rule_codec = model.rule_codec
        self.rule_parts = [
            part for part in self.config.part_config
            if any(pattern in part for pattern in self.config.rule_based_key_patterns)
        ]
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def encode(self, samples):
        model = self.model
        if hasattr(model, "encode"):
            return model.encode(samples)

        embodiments = samples["embodiment"]
        codes = [{} for _ in embodiments]
        embodiment_embs = model.embodiment_embeddings(embodiments)
        for part in self.config.part_config:
            mask = samples["actions_mask"][part]
            present = mask.any(dim=-1)
            indices = present.nonzero().flatten().tolist()
            if not indices:
                continue
            actions = samples["actions_dict"][part][present].float()
            if part in self.rule_parts:
                part_codes = torch.tensor(
                    [self.rule_codec.encode(action[:, 0].tolist()) for action in actions],
                    device=actions.device,
                ).unsqueeze(1)
            else:
                part_embs = model.part_embeddings(part).expand(len(indices), -1)
                latents = model._part_encoder(part_embs, actions, mask[present], embodiment_embs[present])
                _, part_codes, _ = model._quantize(latents)
            for row, index in enumerate(indices):
                codes[index][part] = part_codes[row]
        return codes

    @torch.no_grad()
    def decode(self, codes, embodiments, part_dims, action_dim):
        model = self.model
        if hasattr(model, "decode"):
            return model.decode(codes, embodiments, part_dims, action_dim)

        device = model.device
        actions = torch.zeros(len(embodiments), self.config.action_horizon, action_dim, device=device)
        embodiment_embs = model.embodiment_embeddings(embodiments)
        for part in self.config.part_config:
            indices = [i for i, name in enumerate(embodiments) if part_dims[name][part]]
            if not indices:
                continue
            part_codes = torch.stack([codes[i][part] for i in indices])
            if part in self.rule_parts:
                decoded = torch.tensor([
                    self.rule_codec.decode(code[0].tolist()) for code in part_codes
                ], device=device, dtype=actions.dtype).unsqueeze(-1)
            else:
                mask = torch.tensor([
                    [d < len(part_dims[embodiments[i]][part]) for d in range(self.config.max_part_dim)]
                    for i in indices
                ], device=device)
                latents = model.rvq.from_codes(part_codes)
                decoded = model._part_decoder(
                    model.part_embeddings(part).expand(len(indices), -1),
                    latents, mask, embodiment_embs[indices],
                )
            for row, index in enumerate(indices):
                dims = part_dims[embodiments[index]][part]
                actions[index, :, dims] = decoded[row, :, :len(dims)]
        return actions
