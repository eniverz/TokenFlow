import torch


def time_shift(actions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Shift (B,H,D) actions by per-sample offsets, repeating boundary frames."""
    positions = torch.arange(actions.shape[1], device=actions.device)
    indices = (positions[None, :] + deltas[:, None]).clamp(0, actions.shape[1] - 1)
    return actions.gather(1, indices[:, :, None].expand_as(actions))


def compute_consistency_loss(consist_residuals, level_codes, batch_size):
    """Penalize the first differing RVQ level at each token position.

    Residuals have shape (2B,T,D), ordered as original then augmented samples.
    Only the augmented residual receives gradients; all levels have weight 1.
    """
    metrics = {}
    loss = consist_residuals[0].new_zeros((), dtype=torch.float32)
    prefix_match = torch.ones_like(level_codes[0][:batch_size], dtype=torch.bool)
    for i, (residual, codes) in enumerate(zip(consist_residuals, level_codes)):
        changed = codes[:batch_size] != codes[batch_size:]
        active = prefix_match & changed
        distance = (residual[batch_size:].float() - residual[:batch_size].detach().float()).norm(dim=-1)
        loss = loss + (active * distance).mean()
        metrics[f"consistency/token_change_rate/quantizer_{i}"] = changed.float().mean()
        metrics[f"consistency/active_fraction/quantizer_{i}"] = active.float().mean()
        prefix_match = prefix_match & ~changed

    metrics["loss/consistency_loss"] = loss
    return metrics
