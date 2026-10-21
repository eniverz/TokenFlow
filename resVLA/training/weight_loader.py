from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def _safetensor_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]

    return sorted(path.glob("*.safetensors"))


def _stream_safetensors(path: str | Path):
    """Yield each tensor from each safetensors shard without retaining shards."""

    files = _safetensor_files(path)
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {path}")

    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as file:
            for key in file.keys():
                tensor = file.get_tensor(key)
                yield key, tensor
                del tensor


class BaseWeightLoader:
    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError


class MappedWeightLoader(BaseWeightLoader):
    """Load a safetensors checkpoint into a model using explicit prefix mapping.

    ``prefix_map`` is ordered from source prefixes to target prefixes.  The first
    matching source prefix is used.  There is intentionally no identity fallback:
    every source key must match a configured prefix and every mapped target key
    must already exist in the model.
    """

    def __init__(
        self,
        path: str,
        prefix_map: Mapping[str, str],
        exclude_keys: Sequence[str] = (),
    ) -> None:
        self.path = path
        self.prefix_map = dict(prefix_map)
        self.exclude_keys = tuple(exclude_keys)

    def _map_key(self, source_key: str) -> str:
        for source_prefix, target_prefix in self.prefix_map.items():
            if source_key.startswith(source_prefix):
                return target_prefix + source_key[len(source_prefix) :]

        raise KeyError(
            f"Source key {source_key!r} did not match any configured prefix in "
            f"prefix_map={self.prefix_map!r}"
        )

    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        target_state = model.state_dict()
        source_keys: set[str] = set()
        mapped_targets: set[str] = set()

        with torch.no_grad():
            for source_key, tensor in _stream_safetensors(self.path):
                if source_key in source_keys:
                    del tensor
                    raise KeyError(f"Duplicate safetensors key: {source_key}")
                source_keys.add(source_key)

                if source_key.startswith(self.exclude_keys):
                    del tensor
                    continue

                target_key = self._map_key(source_key)
                if target_key not in target_state:
                    raise KeyError(
                        f"Mapped key {source_key!r} -> {target_key!r}, but the target "
                        f"model has no such parameter or buffer"
                    )
                if target_key in mapped_targets:
                    raise KeyError(
                        f"Multiple source keys map to target key {target_key!r}"
                    )

                target = target_state[target_key]
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                mapped_targets.add(target_key)
                del tensor

        return model


class TorchCheckpointWeightLoader(BaseWeightLoader):
    """Load a PyTorch state dict into one submodule, stripping its source prefix."""

    def __init__(self, path: str, module: str, state_dict_key: str | None = None, prefix: str = ""):
        self.path = path
        self.module = module
        self.state_dict_key = state_dict_key
        self.prefix = prefix

    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        state_dict = torch.load(self.path, map_location="cpu", weights_only=True)
        if self.state_dict_key is not None:
            state_dict = state_dict[self.state_dict_key]
        state_dict = {key.removeprefix(self.prefix): value for key, value in state_dict.items()}
        model.get_submodule(self.module).load_state_dict(state_dict)
        return model


class CompositeWeightLoader(BaseWeightLoader):
    """Apply several loaders in declaration order."""

    def __init__(self, loaders: Sequence[Any]) -> None:
        self.loaders = tuple(loaders)

    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        for loader in self.loaders:
            model = loader.load(model)
        return model


class VocabResize(BaseWeightLoader):
    """Resize a backbone to the vocabulary size stored in the parent config."""

    def __init__(self, module: str) -> None:
        self.module = module

    def load(self, model: torch.nn.Module) -> torch.nn.Module:
        model.get_submodule(self.module).resize_token_embeddings(model.config.vocab_size)
        return model
