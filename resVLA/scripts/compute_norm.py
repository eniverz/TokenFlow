#!/usr/bin/env python3
"""Compute action/state normalization statistics from a task config.

The statistics are computed independently for every ``TransformDataset`` in
the current task's ``MixtureDataset``.  Mixture weights are deliberately
ignored; every raw dataset listed in one transform dataset contributes its
samples equally to that transform dataset's statistics.

All transforms before the first ``Normalizer`` in the configured transform
chain are applied before statistics are collected.  This is important for
transforms such as ``ToDeltaAction`` and ``ToAbsoluteAction``.

Example:
    python -m resVLA.scripts.compute_norm --task resvla_libero --batch-size 128 --num-workers 8

The configured normalizer paths identify the transform dataset name.  Output
is written below ``norm_stats/<task>/<dataset>/``.  Each output file contains
``global`` and ``stepwise`` sections.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tqdm import tqdm
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from resVLA.tools.register_config_hook import register_all_config_hooks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFORM_DATASET_TARGET = "resVLA.data.transform_dataset.TransformDataset"
MIXTURE_DATASET_TARGET = "resVLA.data.mixture_dataset.MixtureDataset"
NORMALIZER_TARGETS = {
    "resVLA.data.transform.action.Normalizer",
    "resVLA.data.transform.action.StepwiseNormalizer",
}


def _plain(value: Any) -> Any:
    """Convert OmegaConf containers to ordinary Python containers."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def load_task_config(task: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config.yaml", overrides=[f"task={task}"])
    return cfg


def _normalizer_configs(unit_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    transforms = _plain(unit_cfg.get("transform", []))
    if not isinstance(transforms, list):
        raise TypeError("TransformDataset.transform must be a list")
    normalizers = []
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        if transform.get("_target_") in NORMALIZER_TARGETS:
            normalizers.append(transform)
    return normalizers


def _pre_normalizer_transforms(unit_cfg: Mapping[str, Any]) -> list[Any]:
    """Instantiate exactly the transforms before the first Normalizer."""
    transforms = _plain(unit_cfg.get("transform", []))
    pre_configs = []
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        if transform.get("_target_") in NORMALIZER_TARGETS:
            break
        pre_configs.append(transform)
    return [instantiate(config) for config in pre_configs]


def _resolve_output_path(path: str | Path, output_root: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = output_root / resolved
    return resolved


def _stats_path(normalizer_cfg: Mapping[str, Any], field: str) -> str | None:
    value = normalizer_cfg.get(f"{field}_norm_stats")
    if not isinstance(value, dict):
        return None
    path = value.get("path")
    return str(path) if path is not None else None


def _dataset_name(normalizer_cfg: Mapping[str, Any]) -> str:
    """Get the output directory name for one TransformDataset.

    Current configs use ``.../<name>/action_stats.json`` and
    ``.../<name>/state_stats.json``.  The parent directory is the unit name.
    """
    action_path = _stats_path(normalizer_cfg, "action")
    state_path = _stats_path(normalizer_cfg, "state")
    if action_path is None or state_path is None:
        raise ValueError(
            "Normalizer must specify action_norm_stats.path and state_norm_stats.path"
        )

    action = Path(action_path)
    state = Path(state_path)
    if action.name != "action_stats.json":
        raise ValueError(
            "action_norm_stats.path must end with <dataset>/action_stats.json"
        )
    name = action.parent.name

    if state.name != "state_stats.json":
        raise ValueError(
            "state_norm_stats.path must end with <dataset>/state_stats.json"
        )
    state_name = state.parent.name
    if name != state_name:
        raise ValueError(
            "action_norm_stats.path and state_norm_stats.path resolve to "
            f"different dataset names: {name!r} and {state_name!r}"
        )
    if not name or name in {".", ".."}:
        raise ValueError("Could not infer a dataset name from Normalizer paths")
    return name


def _as_batch_feature(value: Any, name: str) -> np.ndarray:
    """Convert a batch to a ``(batch, steps, features)`` float32 array."""
    tensor = torch.as_tensor(value).detach().cpu()
    if not tensor.is_floating_point():
        tensor = tensor.float()
    array = tensor.numpy().astype(np.float32, copy=False)
    if array.ndim == 0:
        array = array.reshape(1, 1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, 1, -1)
    elif array.ndim == 2:
        # State batches are (batch, features). Treat them as one-step data.
        array = array.reshape(array.shape[0], 1, array.shape[1])
    else:
        array = array.reshape(array.shape[0], array.shape[1], -1)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite value encountered in {name}")
    return array


def _collate_numeric(samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate only numerical fields needed by norm computation."""
    if not samples:
        raise ValueError("DataLoader returned an empty batch")
    try:
        actions = torch.stack([torch.as_tensor(sample["actions"]) for sample in samples])
        states = torch.stack([torch.as_tensor(sample["states"]) for sample in samples])
    except KeyError as exc:
        raise KeyError("Pre-Normalizer transform chain must produce 'actions' and 'states'") from exc
    except RuntimeError as exc:
        raise ValueError("All actions and states in a batch must have the same shape") from exc
    return {"actions": actions, "states": states}


def _iter_batches(dataset, batch_size: int, num_workers: int = 0):
    from torch.utils.data import DataLoader

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _collate_numeric,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    loader = DataLoader(**loader_kwargs)
    total_batches = math.ceil(len(dataset) / batch_size)
    for batch_index, batch in enumerate(
        tqdm(loader, total=total_batches, desc="Collecting batches", unit="batch")
    ):
        yield batch_index * batch_size, batch


def _order_quantile(data: np.ndarray, quantile: float) -> np.ndarray:
    """Match GalaxeaVLA's np.partition order-statistic quantile."""
    if data.shape[0] == 0:
        raise ValueError("Cannot compute statistics from an empty dataset")
    k = max(0, min(int(quantile * (data.shape[0] - 1)), data.shape[0] - 1))
    return np.partition(data, k, axis=0)[k].astype(np.float32, copy=True)


def _one_axis_stats(data: np.ndarray) -> dict[str, list[float]]:
    """Compute the requested statistics over the first axis."""
    if data.shape[0] == 0:
        raise ValueError("Cannot compute statistics from an empty dataset")
    std = np.std(data, axis=0, ddof=1 if data.shape[0] > 1 else 0)
    return {
        "min": np.min(data, axis=0).astype(np.float32).tolist(),
        "max": np.max(data, axis=0).astype(np.float32).tolist(),
        "q01": _order_quantile(data, 0.01).tolist(),
        "q99": _order_quantile(data, 0.99).tolist(),
        "mean": np.mean(data, axis=0, dtype=np.float64).astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }


def _compute_arrays(action: np.ndarray, state: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    action_stepwise = [_one_axis_stats(action[:, step, :]) for step in range(action.shape[1])]
    state_stepwise = [_one_axis_stats(state[:, step, :]) for step in range(state.shape[1])]

    state_flat = state.reshape(-1, state.shape[-1])
    state_global = _one_axis_stats(state_flat)

    # Match GalaxeaVLA's action aggregation: global min/max are envelopes,
    # global quantiles are the lower/upper envelopes across horizon steps, and
    # std uses the law of total variance over stepwise distributions.
    step_means = np.asarray([row["mean"] for row in action_stepwise], dtype=np.float64)
    step_stds = np.asarray([row["std"] for row in action_stepwise], dtype=np.float64)
    action_global = {
        "min": np.min(np.asarray([row["min"] for row in action_stepwise]), axis=0).tolist(),
        "max": np.max(np.asarray([row["max"] for row in action_stepwise]), axis=0).tolist(),
        "q01": np.min(np.asarray([row["q01"] for row in action_stepwise]), axis=0).tolist(),
        "q99": np.max(np.asarray([row["q99"] for row in action_stepwise]), axis=0).tolist(),
        "mean": np.mean(step_means, axis=0).astype(np.float32).tolist(),
        "std": np.sqrt(
            np.mean(step_stds**2, axis=0)
            + np.var(step_means, axis=0, ddof=1 if len(action_stepwise) > 1 else 0)
        ).astype(np.float32).tolist(),
    }

    def combine(stepwise: list[dict[str, list[float]]], global_stats: dict[str, list[float]]):
        result = {
            "global": dict(global_stats),
            "stepwise": {
                name: [row[name] for row in stepwise]
                for name in ("min", "max", "q01", "q99", "mean", "std")
            },
        }
        return result

    return (
        combine(action_stepwise, action_global),
        combine(state_stepwise, state_global),
    )



def collect_unit_stats(
    unit_cfg: Mapping[str, Any], batch_size: int = 128, num_workers: int = 16
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Instantiate one TransformDataset and collect transformed numerical data."""
    if unit_cfg.get("_target_") != TRANSFORM_DATASET_TARGET:
        raise ValueError(
            "The task dataset must target "
            "resVLA.data.transform_dataset.TransformDataset"
        )

    params = dict(unit_cfg)
    params.pop("_target_", None)
    params["action_only"] = True  # Images are irrelevant to numerical statistics
    params["transform"] = _pre_normalizer_transforms(unit_cfg)
    # Images are irrelevant to numerical statistics.  Keep the configured
    # value because a user-defined pre-normalizer transform may consume images.
    from resVLA.data.transform_dataset import TransformDataset

    dataset = TransformDataset(**params)
    if len(dataset) == 0:
        raise ValueError("TransformDataset is empty")

    with tempfile.TemporaryDirectory(prefix="resvla_norm_") as temp_dir:
        action_map = Path(temp_dir) / "actions.dat"
        state_map = Path(temp_dir) / "states.dat"

        batches = _iter_batches(dataset, batch_size, num_workers)
        first_index, first_batch = next(batches)
        first_action = _as_batch_feature(first_batch["actions"], "actions")
        first_state = _as_batch_feature(first_batch["states"], "states")
        first_count = first_action.shape[0]
        action_store = np.memmap(
            action_map,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), *first_action.shape[1:]),
        )
        state_store = np.memmap(
            state_map,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), *first_state.shape[1:]),
        )
        action_store[first_index : first_index + first_count] = first_action
        state_store[first_index : first_index + first_count] = first_state

        expected_index = first_index + first_count
        for index, batch in batches:
            action = _as_batch_feature(batch["actions"], "actions")
            state = _as_batch_feature(batch["states"], "states")
            if index != expected_index:
                raise RuntimeError(
                    f"DataLoader returned a non-contiguous batch index: "
                    f"{index} != {expected_index}"
                )
            if action.shape[1:] != first_action.shape[1:]:
                raise ValueError(
                    f"Action shape changed at batch {index}: "
                    f"{action.shape[1:]} != {first_action.shape[1:]}"
                )
            if state.shape[1:] != first_state.shape[1:]:
                raise ValueError(
                    f"State shape changed at batch {index}: "
                    f"{state.shape[1:]} != {first_state.shape[1:]}"
                )
            count = action.shape[0]
            if state.shape[0] != count:
                raise ValueError("Action and state batch sizes do not match")
            action_store[index : index + count] = action
            state_store[index : index + count] = state
            expected_index += count

        if expected_index != len(dataset):
            raise RuntimeError(
                f"DataLoader returned {expected_index} samples, expected {len(dataset)}"
            )

        action_store.flush()
        state_store.flush()
        action_stats, state_stats = _compute_arrays(
            np.asarray(action_store), np.asarray(state_store)
        )
    return action_stats, state_stats, len(dataset)


def _write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task name under configs/task, e.g. resvla_libero")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Samples per DataLoader batch (default: 128)",
    )
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers for data loading (default: 16)")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT),
        help="Base directory for relative norm paths (default: repository root)",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    cfg = load_task_config(args.task)
    # Resolve only the dataset subtree while it still has its root context
    # (e.g. ${action_horizon}). Trainer ${hydra:...} fields need Hydra's runtime
    # singleton, which the compose API does not initialize.
    dataset_cfg = _plain(cfg.get("dataset"))
    if not isinstance(dataset_cfg, dict) or dataset_cfg.get("_target_") != MIXTURE_DATASET_TARGET:
        raise ValueError(
            "The current task must resolve cfg.dataset to "
            "resVLA.data.mixture_dataset.MixtureDataset"
        )

    output_root = Path(args.output_root).resolve()
    units = dataset_cfg.get("datasets")
    if not isinstance(units, list) or not units:
        raise ValueError(
            "MixtureDataset.datasets must be a non-empty list of "
            "TransformDataset configurations"
        )

    task_name = str(_plain(cfg.get("task_name")))
    if not task_name or task_name == "None":
        raise ValueError("The task config must define task_name")

    for unit_index, unit_cfg in enumerate(units):
        if not isinstance(unit_cfg, dict) or unit_cfg.get("_target_") != TRANSFORM_DATASET_TARGET:
            raise ValueError(
                f"MixtureDataset.datasets[{unit_index}] must resolve to "
                "resVLA.data.transform_dataset.TransformDataset"
            )

        normalizers = _normalizer_configs(unit_cfg)
        if len(normalizers) != 1:
            raise ValueError(
                f"TransformDataset[{unit_index}] must contain exactly one "
                f"Normalizer; found {len(normalizers)}"
            )
        normalizer_cfg = normalizers[0]
        dataset_name = _dataset_name(normalizer_cfg)

        print(
            f"Computing statistics for TransformDataset[{unit_index}] "
            f"({dataset_name})"
        )
        action_stats, state_stats, sample_count = collect_unit_stats(
            unit_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        action_path = _stats_path(normalizer_cfg, "action")
        state_path = _stats_path(normalizer_cfg, "state")
        if action_path is None or state_path is None:
            raise ValueError(
                "Normalizer must specify action_norm_stats.path and "
                "state_norm_stats.path"
            )
        # Keep the configured paths authoritative.  In the current configs
        # these resolve to norm_stats/<task>/<dataset>/<field>_stats.json.
        action_output = _resolve_output_path(action_path, output_root)
        state_output = _resolve_output_path(state_path, output_root)
        _write_stats(action_output, action_stats)
        _write_stats(state_output, state_stats)
        print(f"  samples={sample_count:,}")
        print(f"  action stats: {action_output}")
        print(f"  state stats:  {state_output}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    register_all_config_hooks()
    main()
