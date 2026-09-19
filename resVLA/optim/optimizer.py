from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.optim import AdamW


class GroupedAdamW(AdamW):
    """AdamW with Hydra-configured module learning-rate groups.

    ``groups`` contains mappings with a ``name`` field.  The name is both the
    ``model.get_submodule`` path and the parameter-group name exposed to the
    trainer's learning-rate metrics.  Parameters not selected by a configured
    group are collected into one ``default`` group.

    The model is supplied at runtime when Hydra instantiates the optimizer;
    model parameters therefore do not need to appear in the YAML config.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        groups: Sequence[Mapping[str, Any]] = (),
        *,
        lr: float = 1.0e-3,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        **kwargs: Any,
    ) -> None:
        assigned_parameter_ids: set[int] = set()
        parameter_groups: list[dict[str, Any]] = []

        for group in groups:
            name = str(group["name"])
            module = model.get_submodule(name)
            module_parameters = []

            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id in assigned_parameter_ids:
                    continue
                assigned_parameter_ids.add(parameter_id)
                module_parameters.append(parameter)

            if not module_parameters:
                raise ValueError(
                    f"Optimizer group '{name}' has no unassigned trainable parameters."
                )

            parameter_groups.append(
                {
                    "params": module_parameters,
                    "lr": float(group.get("lr", lr)),
                    "weight_decay": weight_decay,
                    "name": name,
                }
            )

        unassigned_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in assigned_parameter_ids
        ]
        if unassigned_parameters:
            parameter_groups.insert(
                0,
                {
                    "params": unassigned_parameters,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "name": "default",
                },
            )

        optimizer_lr = lr if unassigned_parameters else (
            parameter_groups[0]["lr"] if parameter_groups else lr
        )

        super().__init__(
            parameter_groups,
            lr=optimizer_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs,
        )
