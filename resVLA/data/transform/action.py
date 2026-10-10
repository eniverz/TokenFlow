import json
from pathlib import Path
from typing import Dict, Literal, Sequence

import torch
import torch.nn.functional as F

from . import BaseTransform


# MARK: Pad action/state

class PadStateAction(BaseTransform):
    def __init__(self, action_dim: int, pad_value: float = 0.0):
        super().__init__()
        self.pad_value = pad_value
        self.action_dim = action_dim

    def __call__(self, data):
        if "actions" in data and data["actions"] is not None:
            actions = data["actions"]
            actions, action_mask = self.pad(actions)
            data["actions"] = actions
            data["action_mask"] = action_mask

        states = data["states"]
        states, state_mask = self.pad(states)
        data["states"] = states
        data["state_mask"] = state_mask

        return data

    def pad(self, value):
        dim = value.shape[-1]
        if dim > self.action_dim:
            raise ValueError(
                f"Value dimension {dim} is greater than the specified action_dim "
                f"{self.action_dim}."
            )

        pad_width = self.action_dim - dim
        ret = F.pad(value, (0, pad_width), mode="constant", value=self.pad_value)
        mask = F.pad(
            torch.ones(dim, dtype=torch.bool, device=value.device),
            (0, pad_width),
            mode="constant",
            value=False,
        )
        return ret, mask


class PadActionToDict(BaseTransform):
    def __init__(
        self,
        dim_config: Dict[str, Sequence[int]],
        pad_value: int = 0,
        max_part_dim: int = 6
    ):
        super().__init__()
        self.dim_config = dim_config
        self.pad_value = pad_value
        self.max_part_dim = max_part_dim

    def __call__(self, data):
        if "actions" not in data or data["actions"] is None:
            return data

        actions = data["actions"]
        action_dict = {}
        action_mask = {}
        # FIXME: use full set of parts.
        for key, dims in self.dim_config.items():
            vals, mask = self.pad(actions[..., dims])
            action_dict[key] = vals
            action_mask[key] = mask

        data["actions_dict"] = action_dict
        data["actions_mask"] = action_mask
        return data

    def pad(self, value):
        dim = value.shape[-1]
        if dim > self.max_part_dim:
            raise ValueError(
                f"Value dimension {dim} is greater than the specified max_part_dim "
                f"{self.max_part_dim}."
            )

        pad_width = self.max_part_dim - dim
        ret = F.pad(value, (0, pad_width), mode="constant", value=self.pad_value)
        mask = F.pad(
            torch.ones(dim, dtype=torch.bool, device=value.device),
            (0, pad_width),
            mode="constant",
            value=False,
        )
        return ret, mask


# MARK: Normalizer

class NormStats:
    min: Sequence[float]
    max: Sequence[float]
    q01: Sequence[float]
    q99: Sequence[float]
    mean: Sequence[float]
    std: Sequence[float]
    mask: Sequence[bool]

    def __init__(self, path: str, mask: Sequence[bool], stepwise: bool = False):
        self.stepwise = stepwise
        self.mask = mask
        self.load(path)

    def load(self, path: str):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path, "r") as f:
                stats = json.load(f)
            selected = stats["stepwise" if self.stepwise else "global"]
            self.min = selected["min"]
            self.max = selected["max"]
            self.q01 = selected["q01"]
            self.q99 = selected["q99"]
            self.mean = selected["mean"]
            self.std = selected["std"]
        else:
            raise FileNotFoundError(f"NormStats file not found at {self.path}")

class Normalizer(BaseTransform):
    """Normalize action/state using the norm bound to each sample embodiment."""

    def __init__(
        self,
        action_norm_stats: NormStats,
        state_norm_stats: NormStats,
        norm_type: Literal["min-max", "quantile", "mean-std"] = "quantile"
    ):
        super().__init__()
        self.action_norm_stats = action_norm_stats
        self.state_norm_stats = state_norm_stats
        self.norm_type = norm_type

    def __call__(self, data):
        if "actions" in data and data["actions"] is not None:
            actions = data["actions"]
            data["actions"] = self._apply(actions, self.action_norm_stats)
        
        states = data["states"]
        if states.ndim == 1:
            states = states.unsqueeze(0)
        data["states"] = self._apply(states, self.state_norm_stats)

        return data

    def _apply(self, data: torch.Tensor, stats: NormStats):
        if self.norm_type == "min-max":
            return self._apply_min_max(data, stats)
        if self.norm_type == "quantile":
            return self._apply_quantile(data, stats)
        if self.norm_type == "mean-std":
            return self._apply_mean_std(data, stats)
        raise ValueError(
            f"Unsupported norm type: {self.norm_type}. Supported types are 'min-max', 'quantile', and 'mean-std'."
        )

    def _masked_view(self, data: torch.Tensor, mask: torch.Tensor):
        dims = mask.shape[-1]
        return data[..., :dims], mask

    def _tensor(self, data: Sequence[float], to: torch.Tensor):
        return torch.as_tensor(data, dtype=to.dtype, device=to.device)

    def _apply_min_max(self, data: torch.Tensor, stats: NormStats):
        min_val = self._tensor(stats.min, data)
        max_val = self._tensor(stats.max, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)

        view, mask = self._masked_view(data, mask)
        view[..., mask] = 2 * (view[..., mask] - min_val[..., mask]) / (max_val[..., mask] - min_val[..., mask] + 1e-8) - 1
        return data

    def _apply_quantile(self, data: torch.Tensor, stats: NormStats):
        q01 = self._tensor(stats.q01, data)
        q99 = self._tensor(stats.q99, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)

        view, mask = self._masked_view(data, mask)
        view[..., mask] = 2 * (view[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask] + 1e-8) - 1
        return data

    def _apply_mean_std(self, data: torch.Tensor, stats: NormStats):
        mean = self._tensor(stats.mean, data)
        std = self._tensor(stats.std, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)

        view, mask = self._masked_view(data, mask)
        view[..., mask] = 2 * (view[..., mask] - mean[..., mask]) / (std[..., mask] + 1e-8) - 1
        return data

# MARK: Unnormalizer

class Unnormalizer(BaseTransform):
    """Invert the normalization performed by :class:`Normalizer`."""

    def __init__(
        self,
        action_norm_stats: NormStats,
        state_norm_stats: NormStats,
        norm_type: Literal["min-max", "quantile", "mean-std"] = "quantile",
    ):
        super().__init__()
        self.action_norm_stats = action_norm_stats
        self.state_norm_stats = state_norm_stats
        self.norm_type = norm_type

    def __call__(self, data):
        if "actions" in data and data["actions"] is not None:
            data["actions"] = self._apply(data["actions"], self.action_norm_stats)
        if "states" in data:
            data["states"] = self._apply(data["states"], self.state_norm_stats)
        return data

    def _apply(self, data: torch.Tensor, stats: NormStats):
        if self.norm_type == "min-max":
            return self._apply_min_max(data, stats)
        if self.norm_type == "quantile":
            return self._apply_quantile(data, stats)
        if self.norm_type == "mean-std":
            return self._apply_mean_std(data, stats)
        raise ValueError(
            f"Unsupported norm type: {self.norm_type}. Supported types are 'min-max', 'quantile', and 'mean-std'."
        )

    def _masked_view(self, data: torch.Tensor, mask: torch.Tensor):
        dims = mask.shape[-1]
        return data[..., :dims], mask

    def _tensor(self, data: Sequence[float], to: torch.Tensor):
        return torch.as_tensor(data, dtype=to.dtype, device=to.device)

    def _apply_min_max(self, data: torch.Tensor, stats: NormStats):
        min_val = self._tensor(stats.min, data)
        max_val = self._tensor(stats.max, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)

        view, mask = self._masked_view(data, mask)
        view[..., mask] = (view[..., mask] + 1) / 2 * (max_val[..., mask] - min_val[..., mask]) + min_val[..., mask]
        return data

    def _apply_quantile(self, data: torch.Tensor, stats: NormStats):
        q01 = self._tensor(stats.q01, data)
        q99 = self._tensor(stats.q99, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)

        view, mask = self._masked_view(data, mask)
        view[..., mask] = (view[..., mask] + 1) / 2 * (q99[..., mask] - q01[..., mask]) + q01[..., mask]
        return data

    def _apply_mean_std(self, data: torch.Tensor, stats: NormStats):
        mean = self._tensor(stats.mean, data)
        std = self._tensor(stats.std, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)

        view, mask = self._masked_view(data, mask)
        view[..., mask] = (view[..., mask] + 1) / 2 * std[..., mask] + mean[..., mask]
        return data

# MARK: Stepwise Normalizer/Unnormalizer

class StepwiseNormalizer(BaseTransform):
    """Normalize actions with per-timestep statistics."""

    def __init__(
        self,
        action_norm_stats: NormStats,
        state_norm_stats: NormStats,
        norm_type: Literal["min-max", "quantile", "mean-std"] = "quantile",
    ):
        super().__init__()
        self.action_norm_stats = action_norm_stats
        self.state_norm_stats = state_norm_stats
        self.norm_type = norm_type

    def __call__(self, data):
        if "actions" in data and data["actions"] is not None:
            data["actions"] = self._apply(data["actions"], self.action_norm_stats)
        states = data["states"]
        if states.ndim == 1:
            states = states.unsqueeze(0)
        data["states"] = self._apply(states, self.state_norm_stats)
        return data

    def _apply(self, data: torch.Tensor, stats: NormStats):
        if self.norm_type == "min-max":
            return self._apply_min_max(data, stats)
        if self.norm_type == "quantile":
            return self._apply_quantile(data, stats)
        if self.norm_type == "mean-std":
            return self._apply_mean_std(data, stats)
        raise ValueError(
            f"Unsupported norm type: {self.norm_type}. Supported types are 'min-max', 'quantile', and 'mean-std'."
        )

    def _masked_view(self, data: torch.Tensor, mask: torch.Tensor):
        dims = mask.shape[-1]
        return data[..., :dims], mask

    def _tensor(self, data: Sequence[float], to: torch.Tensor):
        return torch.as_tensor(data, dtype=to.dtype, device=to.device)

    def _apply_min_max(self, data: torch.Tensor, stats: NormStats):
        min_val = self._tensor(stats.min, data)
        max_val = self._tensor(stats.max, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)
        view, mask = self._masked_view(data, mask)
        transformed = 2 * (view - min_val) / (max_val - min_val + 1e-8) - 1
        view = torch.where(mask, transformed, view)
        return torch.cat([view, data[..., view.shape[-1]:]], dim=-1) if data.shape[-1] > view.shape[-1] else view

    def _apply_quantile(self, data: torch.Tensor, stats: NormStats):
        q01 = self._tensor(stats.q01, data)
        q99 = self._tensor(stats.q99, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)
        view, mask = self._masked_view(data, mask)
        transformed = 2 * (view - q01) / (q99 - q01 + 1e-8) - 1
        view = torch.where(mask, transformed, view)
        return torch.cat([view, data[..., view.shape[-1]:]], dim=-1) if data.shape[-1] > view.shape[-1] else view

    def _apply_mean_std(self, data: torch.Tensor, stats: NormStats):
        mean = self._tensor(stats.mean, data)
        std = self._tensor(stats.std, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)
        view, mask = self._masked_view(data, mask)
        transformed = 2 * (view - mean) / (std + 1e-8) - 1
        view = torch.where(mask, transformed, view)
        return torch.cat([view, data[..., view.shape[-1]:]], dim=-1) if data.shape[-1] > view.shape[-1] else view


class StepwiseUnnormalizer(BaseTransform):
    """Invert :class:`StepwiseNormalizer`."""

    def __init__(
        self,
        action_norm_stats: NormStats,
        state_norm_stats: NormStats,
        norm_type: Literal["min-max", "quantile", "mean-std"] = "quantile",
    ):
        super().__init__()
        self.action_norm_stats = action_norm_stats
        self.state_norm_stats = state_norm_stats
        self.norm_type = norm_type

    def __call__(self, data):
        if "actions" in data and data["actions"] is not None:
            data["actions"] = self._apply(data["actions"], self.action_norm_stats)
        if "states" in data:
            data["states"] = self._apply(data["states"], self.state_norm_stats)
        return data

    def _apply(self, data: torch.Tensor, stats: NormStats):
        if self.norm_type == "min-max":
            return self._apply_min_max(data, stats)
        if self.norm_type == "quantile":
            return self._apply_quantile(data, stats)
        if self.norm_type == "mean-std":
            return self._apply_mean_std(data, stats)
        raise ValueError(
            f"Unsupported norm type: {self.norm_type}. Supported types are 'min-max', 'quantile', and 'mean-std'."
        )

    def _masked_view(self, data: torch.Tensor, mask: torch.Tensor):
        dims = mask.shape[-1]
        return data[..., :dims], mask

    def _tensor(self, data: Sequence[float], to: torch.Tensor):
        return torch.as_tensor(data, dtype=to.dtype, device=to.device)

    def _apply_min_max(self, data: torch.Tensor, stats: NormStats):
        min_val = self._tensor(stats.min, data)
        max_val = self._tensor(stats.max, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)
        view, mask = self._masked_view(data, mask)
        transformed = (view + 1) / 2 * (max_val - min_val) + min_val
        view = torch.where(mask, transformed, view)
        return torch.cat([view, data[..., view.shape[-1]:]], dim=-1) if data.shape[-1] > view.shape[-1] else view

    def _apply_quantile(self, data: torch.Tensor, stats: NormStats):
        q01 = self._tensor(stats.q01, data)
        q99 = self._tensor(stats.q99, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)
        view, mask = self._masked_view(data, mask)
        transformed = (view + 1) / 2 * (q99 - q01) + q01
        view = torch.where(mask, transformed, view)
        return torch.cat([view, data[..., view.shape[-1]:]], dim=-1) if data.shape[-1] > view.shape[-1] else view

    def _apply_mean_std(self, data: torch.Tensor, stats: NormStats):
        mean = self._tensor(stats.mean, data)
        std = self._tensor(stats.std, data)
        mask = torch.as_tensor(stats.mask, dtype=torch.bool, device=data.device)
        view, mask = self._masked_view(data, mask)
        transformed = (view + 1) / 2 * std + mean
        view = torch.where(mask, transformed, view)
        return torch.cat([view, data[..., view.shape[-1]:]], dim=-1) if data.shape[-1] > view.shape[-1] else view

# MARK: Absolute/Delta action chunk

class ToDeltaAction(BaseTransform):
    """
    Chunk-wise delta action transform.
    """
    def __init__(self, mask: Sequence[bool]):
        super().__init__()
        self.mask = mask

    def __call__(self, data):
        if "actions" not in data:
            return data
        state, actions = data["states"], data["actions"]
        mask = torch.as_tensor(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= torch.expand_dims(torch.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        return data


class ToAbsoluteAction(BaseTransform):
    """
    Chunk-wise absolute state transform.
    """
    def __init__(self, mask: Sequence[bool]):
        super().__init__()
        self.mask = mask

    def __call__(self, data):
        if "actions" not in data:
            return data
        state, actions = data["states"], data["actions"]
        mask = torch.as_tensor(self.mask)
        dims = mask.shape[-1]
        state[..., :dims] += torch.expand_dims(torch.where(mask, actions[..., :dims], 0), axis=-2)
        data["states"] = state
        return data

# MARK: Inject Timestamp

class InjectActionTimestamp(BaseTransform):
    def __init__(self, action_horizon: int, action_freq: float):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_freq = action_freq

    def __call__(self, data):
        action_timestamps = torch.arange(0, self.action_horizon) / self.action_freq
        data["action_timestamps"] = action_timestamps
        return data
