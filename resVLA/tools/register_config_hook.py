from collections.abc import Callable, Sequence, Mapping
from typing import Any, Optional

from omegaconf import OmegaConf
from pathlib import Path


def _register(name: str, func: Callable) -> None:
    """Idempotently register a resolver, replacing any existing one."""
    OmegaConf.register_new_resolver(name, func, replace=True)


def _oc_load(path: str, key: Optional[str] = None):
    try:
        from hydra.utils import to_absolute_path
    except ImportError:
        to_absolute_path = None

    path = Path(path)
    if not path.is_absolute() and to_absolute_path is not None:
        path = Path(to_absolute_path(path))

    cfg = OmegaConf.load(path)

    if key is not None and key != "":
        return OmegaConf.select(cfg, key)

    return OmegaConf.to_container(cfg, resolve=False)


def _oc_collate(config: Any) -> list[str]:
    """Collect embodiment names from InjectEmbodiment transforms."""
    config = OmegaConf.to_container(config, resolve=True)
    embodiments: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if node.get("_target_") == "resVLA.data.transform.embodiment.InjectEmbodiment":
                embodiment = node["embodiment"]
                if embodiment not in embodiments:
                    embodiments.append(embodiment)

            for value in node.values():
                visit(value)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for value in node:
                visit(value)

    visit(config)
    return embodiments


def register_all_config_hooks():
    _register("oc.load", _oc_load)
    _register("oc.collate", _oc_collate)
