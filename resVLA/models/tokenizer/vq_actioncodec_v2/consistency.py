import torch
import torch.nn as nn

from .configuration_vq_codec import ActionCodecConfig


def time_shift(actions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Shift (B,H,D) actions by per-sample offsets, repeating boundary frames."""
    positions = torch.arange(actions.shape[1], device=actions.device)
    indices = (positions[None, :] + deltas[:, None]).clamp(0, actions.shape[1] - 1)
    return actions.gather(1, indices[:, :, None].expand_as(actions))


class ConsistencyAugmenter(nn.Module):
    """Schedule paired-view augmentation using a checkpointed micro-batch counter."""

    def __init__(self, config: ActionCodecConfig):
        super().__init__()
        self.config = config

        self.num_forwards = nn.Buffer(torch.tensor(0, dtype=torch.long), persistent=True)

    def forward(self, actions_dict):
        cfg = self.config
        layer_weights = [0.0] * cfg.num_quantizers
        if not self.training or not torch.is_grad_enabled():
            return actions_dict, layer_weights

        step = self.num_forwards.item() // cfg.ema_update_interval
        self.num_forwards.add_(1)
        if step < cfg.consistency_start_step:
            return actions_dict, layer_weights

        progress = min((step - cfg.consistency_start_step) / (cfg.consistency_end_step - cfg.consistency_start_step), 1.0)
        delta_max = round(cfg.consistency_delta_max_start + progress * (cfg.consistency_delta_max_end - cfg.consistency_delta_max_start))
        eps = cfg.consistency_eps_begin + progress * (cfg.consistency_eps_end - cfg.consistency_eps_begin)
        layer_weights = [progress * weight for weight in cfg.consistency_layer_weights_end]

        actions = next(iter(actions_dict.values()))
        batch_size, device = actions.shape[0], actions.device
        # Shared across parts: 50% shift, 40% scale, 10% both.
        kind = torch.rand(batch_size, device=device)
        deltas = torch.randint(1, delta_max + 1, (batch_size,), device=device)
        deltas = deltas * (2 * torch.randint(0, 2, deltas.shape, device=device) - 1)
        deltas = deltas * ((kind < 0.5) | (kind >= 0.9))
        scales = 1 + (2 * torch.rand(batch_size, device=device) - 1) * eps * (kind >= 0.5)

        augmented = {}
        for part_name, part_actions in actions_dict.items():
            shifted = time_shift(part_actions, deltas)
            # Scale normalized arm actions; preserve gripper values and padded zeros.
            if part_name.endswith("_arm"):
                shifted = shifted * scales.to(shifted.dtype)[:, None, None]
            augmented[part_name] = shifted
        return augmented, layer_weights


def compute_consistency_loss(consist_residuals, level_codes, batch_size, layer_weights):
    """Penalize the first differing RVQ level at each token position.

    Residuals have shape (2B,T,D), ordered as original then augmented samples.
    Only the augmented residual receives gradients; levels use scheduled weights.
    """
    metrics = {}
    loss = consist_residuals[0].new_zeros((), dtype=torch.float32)
    prefix_match = torch.ones_like(level_codes[0][:batch_size], dtype=torch.bool)
    for i, (residual, codes) in enumerate(zip(consist_residuals, level_codes)):
        changed = codes[:batch_size] != codes[batch_size:]
        active = prefix_match & changed
        distance = (residual[batch_size:].float() - residual[:batch_size].detach().float()).norm(dim=-1)
        loss = loss + layer_weights[i] * (active * distance).mean()
        metrics[f"consistency/token_change_rate/quantizer_{i}"] = changed.float().mean()
        metrics[f"consistency/active_fraction/quantizer_{i}"] = active.float().mean()
        prefix_match = prefix_match & ~changed

    metrics["loss/consistency_loss"] = loss
    return metrics
