"""Checkpoint-backed ResVLA policy used by the deployment server.

The training task configuration remains the source of truth for data transforms
and normalization.  The model and processor themselves are loaded from a
``save_pretrained`` checkpoint.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from accelerate.utils import send_to_device
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoModel, AutoProcessor

from resVLA.data.transform.process import flatten_dict, unflatten_dict
from resVLA.tools.register_models import register_all_models


NORMALIZER_TARGETS = {
    "resVLA.data.transform.action.Normalizer":
        "resVLA.data.transform.action.Unnormalizer",
    "resVLA.data.transform.action.StepwiseNormalizer":
        "resVLA.data.transform.action.StepwiseUnnormalizer",
}


def _target(value: Mapping[str, Any]) -> str:
    return str(value.get("_target_", ""))


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _walk_mappings(value: Any):
    value = _plain(value)
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _absolutize_transform_paths(value: Any) -> Any:
    """Make norm-stat paths independent of the server process cwd."""

    value = copy.deepcopy(_plain(value))
    for node in _walk_mappings(value):
        if "path" in node and isinstance(node["path"], str):
            node["path"] = to_absolute_path(node["path"])
    return value


def _find_transform_datasets(cfg: Any) -> list[Mapping[str, Any]]:
    datasets = []
    seen: set[int] = set()
    for node in _walk_mappings(cfg):
        target = _target(node)
        if target.endswith("TransformDataset") and id(node) not in seen:
            datasets.append(node)
            seen.add(id(node))
    return datasets


def _state_to_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _padding_mask(original_dim: int, model_dim: int) -> torch.Tensor:
    if not 0 < original_dim <= model_dim:
        raise ValueError(
            f"Original action dimension {original_dim} must be in [1, {model_dim}]."
        )
    mask = torch.zeros(model_dim, dtype=torch.bool)
    mask[:original_dim] = True
    return mask


def _unpad_actions(actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Remove model padding while retaining the original action dimensions."""

    mask = action_mask.to(device=actions.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != actions.shape[-1]:
        raise ValueError(
            f"Action mask has length {mask.numel()}, expected {actions.shape[-1]}."
        )
    return actions[..., mask]

class CheckpointPolicy:
    """Load a ResVLA checkpoint and derive deployment transforms from Hydra cfg."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        cfg: Any,
        device: str = "cuda",
        dtype: str = "bf16",
    ):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        if not self.checkpoint_dir.is_dir():
            raise ValueError(f"checkpoint_dir must be a directory: {self.checkpoint_dir}")

        register_all_models()
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("ResVLA deployment currently supports CUDA inference only.")
        self.dtype = self._resolve_dtype(dtype)

        model_kwargs = {} if self.dtype is None else {"dtype": self.dtype}
        self.model = AutoModel.from_pretrained(self.checkpoint_dir, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

        self.transforms_by_embodiment: dict[str, list[Any]] = {}
        self.inverse_plans_by_embodiment: dict[str, list[dict[str, Any]]] = {}
        self.metadata_by_embodiment: dict[str, dict[str, Any]] = {}
        self.action_dim = int(self.model.config.action_dim)
        action_horizon = getattr(self.model.config, "action_horizon", None)
        if action_horizon is None:
            codec_config = getattr(getattr(self.model, "action_tokenizer", None), "config", None)
            action_horizon = getattr(codec_config, "action_horizon", None)
        if action_horizon is None:
            raise ValueError("Neither model nor action tokenizer config defines action_horizon.")
        self.action_horizon = int(action_horizon)
        configured_embodiments = getattr(self.model.config, "embodiment_config", None)
        if configured_embodiments is None:
            configured_embodiments = getattr(self.model.config, "part_dims", {})
        task_cfg = _plain(cfg)
        if not isinstance(task_cfg, Mapping):
            raise TypeError("Resolved Hydra cfg must be a mapping.")
        if "action_dim" not in task_cfg or "action_horizon" not in task_cfg:
            raise ValueError("Hydra cfg must define top-level action_dim and action_horizon.")
        if int(task_cfg["action_dim"]) != self.action_dim:
            raise ValueError(
                "Top-level action_dim does not match the checkpoint model: "
                f"cfg={task_cfg['action_dim']}, model={self.action_dim}."
            )
        if int(task_cfg["action_horizon"]) != self.action_horizon:
            raise ValueError(
                "Top-level action_horizon does not match the checkpoint model: "
                f"cfg={task_cfg['action_horizon']}, model={self.action_horizon}."
            )
        collator_cfg = copy.deepcopy(task_cfg["train"]["dataloader"]["collate_fn"])
        # Reuse training collation settings with the checkpoint's processor.
        collator_cfg.pop("processor", None)
        self.collator = instantiate(collator_cfg, processor=self.processor)
        self._build_transform_registry(task_cfg)
        configured = set(configured_embodiments)
        missing = set(self.metadata_by_embodiment) - configured
        if missing:
            raise ValueError(
                "Transform config contains embodiments missing from model config: "
                f"{sorted(missing)}."
            )

    @staticmethod
    def _resolve_dtype(dtype: str | None):
        if dtype in (None, "none", "fp32", "float32"):
            return None
        if dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype in ("fp16", "float16"):
            return torch.float16
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _build_transform_registry(self, cfg: Any) -> None:
        datasets = _find_transform_datasets(cfg)
        if not datasets:
            raise ValueError("Hydra cfg does not contain a TransformDataset configuration.")

        for dataset_cfg in datasets:
            transform_cfgs = _plain(dataset_cfg.get("transform", []))
            if not isinstance(transform_cfgs, list):
                raise TypeError("TransformDataset.transform must be a list.")

            embodiment = None
            normalizer_cfg = None
            for transform_cfg in transform_cfgs:
                if not isinstance(transform_cfg, Mapping):
                    continue
                target = _target(transform_cfg)
                if target.endswith("InjectEmbodiment"):
                    embodiment = str(transform_cfg.get("embodiment"))
                if target in NORMALIZER_TARGETS:
                    if normalizer_cfg is not None:
                        raise ValueError(
                            f"Multiple normalizers found for embodiment {embodiment!r}."
                        )
                    normalizer_cfg = transform_cfg

            if not embodiment:
                raise ValueError("Each TransformDataset must inject an embodiment.")
            if normalizer_cfg is None:
                raise ValueError(f"No normalizer found for embodiment {embodiment!r}.")

            for transform_cfg in transform_cfgs:
                if _target(transform_cfg).endswith("PadStateAction"):
                    transform_action_dim = int(transform_cfg.get("action_dim"))
                    if transform_action_dim != self.action_dim:
                        raise ValueError(
                            "PadStateAction action_dim does not match the checkpoint model: "
                            f"transform={transform_action_dim}, model={self.action_dim}."
                        )

            transform_cfgs = _absolutize_transform_paths(transform_cfgs)
            transforms = [instantiate(transform_cfg) for transform_cfg in transform_cfgs]
            inverse_plan = self._build_inverse_plan(transform_cfgs)
            metadata = self._build_metadata(
                normalizer_cfg=_absolutize_transform_paths(normalizer_cfg),
                transform_cfgs=transform_cfgs,
                model_action_dim=self.action_dim,
            )

            if embodiment in self.metadata_by_embodiment:
                previous = self.metadata_by_embodiment[embodiment]
                if previous["norm_signature"] != metadata["norm_signature"]:
                    raise ValueError(
                        f"Embodiment {embodiment!r} maps to multiple incompatible norm configs."
                    )
                continue

            self.transforms_by_embodiment[embodiment] = transforms
            self.inverse_plans_by_embodiment[embodiment] = inverse_plan
            self.metadata_by_embodiment[embodiment] = metadata

    @staticmethod
    def _build_inverse_plan(transform_cfgs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        plan = []
        for transform_cfg in reversed(transform_cfgs):
            target = _target(transform_cfg)
            if target.endswith("PadStateAction"):
                plan.append({"kind": "unpad"})
            elif target in NORMALIZER_TARGETS:
                inverse_cfg = copy.deepcopy(dict(transform_cfg))
                inverse_cfg["_target_"] = NORMALIZER_TARGETS[target]
                plan.append({"kind": "transform", "transform": instantiate(inverse_cfg)})
            elif target.endswith("ToDeltaAction"):
                plan.append({"kind": "delta_to_absolute", "mask": transform_cfg.get("mask")})
            elif target.endswith("ToAbsoluteAction"):
                plan.append({"kind": "absolute_to_delta", "mask": transform_cfg.get("mask")})
        return plan

    @staticmethod
    def _build_metadata(
        normalizer_cfg: Mapping[str, Any],
        transform_cfgs: list[Mapping[str, Any]],
        model_action_dim: int,
    ) -> dict[str, Any]:
        action_stats_cfg = normalizer_cfg.get("action_norm_stats", {})
        state_stats_cfg = normalizer_cfg.get("state_norm_stats", {})
        action_mask = action_stats_cfg.get("mask")
        if action_mask is None:
            raise ValueError("Normalizer action_norm_stats.mask is required to recover the original action width.")
        if not any(_target(item).endswith("PadStateAction") for item in transform_cfgs):
            raise ValueError("Deployment requires a PadStateAction transform.")
        return {
            "action_mask": _padding_mask(len(action_mask), model_action_dim),
            "repack": next(
                (item.get("repack") for item in transform_cfgs
                 if _target(item).endswith("RepackTransform")),
                None,
            ),
            "norm_signature": (
                _target(normalizer_cfg),
                str(action_stats_cfg.get("path")),
                str(state_stats_cfg.get("path")),
                tuple(action_mask or []),
                tuple(state_stats_cfg.get("mask") or []),
                normalizer_cfg.get("norm_type", "quantile"),
            ),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_type": type(self.model).__name__,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "embodiments": sorted(self.metadata_by_embodiment),
        }

    def predict_action(
        self,
        data: dict | None = None,
        embodiment_name: str | None = None,
        execution_horizon: int | None = None,
        **legacy_payload: Any,
    ) -> dict[str, np.ndarray]:
        if data is None:
            data = dict(legacy_payload)
        else:
            data = copy.deepcopy(data)
            if legacy_payload:
                data.update(legacy_payload)
        if embodiment_name is None:
            embodiment_name = (
                data.get("embodiment")
                or data.get("embodiment_name")
                or data.get("embodiment_tag")
            )
        if embodiment_name is None:
            raise ValueError("embodiment_name is required for deployment inference.")
        embodiment_name = str(embodiment_name)
        if embodiment_name not in self.transforms_by_embodiment:
            raise ValueError(
                f"Unknown embodiment {embodiment_name!r}; available: "
                f"{sorted(self.transforms_by_embodiment)}"
            )

        transformed = self._apply_input_transforms(data, embodiment_name)
        state_for_inverse = transformed["states"].clone()
        # RepackTransform represents absent inference fields (e.g. actions) as
        # None, which default_collate cannot batch. Keep every available field.
        sample = {key: value for key, value in transformed.items() if value is not None}
        batch = send_to_device(self.collator([sample]), self.device)

        autocast_enabled = self.dtype is not None
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype or torch.float32,
            enabled=autocast_enabled,
        ):
            normalized = self.model.predict_action(batch)[0].detach().float().cpu()
            normalized = normalized.clamp(-1.0, 1.0)

        if normalized.shape[-1] != self.action_dim:
            raise ValueError(
                f"Model returned action dim {normalized.shape[-1]}, expected {self.action_dim}."
            )
        output = {
            "actions": normalized,
            "states": state_for_inverse,
        }
        output = self._apply_output_inverse(output, embodiment_name)
        actions = output["actions"]

        requested_horizon = actions.shape[0] if execution_horizon is None else int(execution_horizon)
        if not 1 <= requested_horizon <= actions.shape[0]:
            raise ValueError(
                f"execution_horizon must be in [1, {actions.shape[0]}], got {requested_horizon}."
            )
        return {"actions": actions[:requested_horizon].numpy()}

    def _apply_input_transforms(self, data: dict, embodiment_name: str) -> dict:
        transformed = copy.deepcopy(data)
        if {"state", "images", "prompt"} <= transformed.keys():
            repack = self.metadata_by_embodiment[embodiment_name]["repack"]
            if repack is None:
                raise ValueError("Flat observation requires a RepackTransform mapping.")
            source = flatten_dict({
                "states": transformed["state"],
                "images": transformed["images"],
                "prompt": transformed["prompt"],
            })
            mapping = flatten_dict(repack)
            transformed = unflatten_dict({
                raw_key: source[output_key]
                for output_key, raw_key in mapping.items()
                if output_key in source
            })
        transforms = self.transforms_by_embodiment[embodiment_name]
        for transform in transforms:
            transformed = transform(transformed)
            if "states" in transformed and not torch.is_tensor(transformed["states"]):
                transformed["states"] = _state_to_tensor(transformed["states"])
        if "states" not in transformed or "images" not in transformed or "prompt" not in transformed:
            raise ValueError("Input transforms must produce states, images, and prompt.")
        if transformed["states"].ndim != 2:
            raise ValueError(
                f"Expected transformed state shape [1, action_dim], got {tuple(transformed['states'].shape)}."
            )
        return transformed

    def _apply_output_inverse(self, data: dict, embodiment_name: str) -> dict:
        metadata = self.metadata_by_embodiment[embodiment_name]
        actions = data["actions"]
        states = data["states"]
        for operation in self.inverse_plans_by_embodiment[embodiment_name]:
            kind = operation["kind"]
            if kind == "unpad":
                actions = _unpad_actions(actions, metadata["action_mask"])
            elif kind == "transform":
                data = operation["transform"]({"actions": actions, "states": states})
                actions, states = data["actions"], data["states"]
            elif kind in {"delta_to_absolute", "absolute_to_delta"}:
                mask = torch.as_tensor(operation["mask"], dtype=torch.bool)
                dims = mask.numel()
                if states.shape[-1] < dims:
                    raise ValueError("Action transform mask exceeds state dimension.")
                if actions.shape[-1] < dims:
                    raise ValueError("Action transform mask exceeds action dimension.")
                state = states[..., :dims]
                while state.ndim > actions.ndim - 1:
                    state = state.squeeze(0)
                state = torch.where(mask, state, torch.zeros_like(state))
                if kind == "delta_to_absolute":
                    actions[..., :dims] += state.unsqueeze(-2)
                else:
                    actions[..., :dims] -= state.unsqueeze(-2)
        data["actions"], data["states"] = actions, states
        return data
