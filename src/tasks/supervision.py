from __future__ import annotations

from typing import Any

import torch


def infer_num_outputs(records: list[Any], task_type: str, configured: int | None = None) -> int:
    if configured is not None and configured > 0:
        return configured
    if not records:
        return 1
    first_label = getattr(records[0], "label", None)
    task_key = task_type.lower()
    if task_key == "ranking" and isinstance(first_label, (list, tuple)):
        return len(first_label)
    if task_key == "multiclass":
        return max(int(getattr(record, "label")) for record in records) + 1
    return 1


def tensorize_targets(records: list[Any], task_type: str, device: torch.device | str) -> torch.Tensor:
    labels = [getattr(record, "label", None) for record in records]
    task_key = task_type.lower()
    if task_key == "multiclass":
        return torch.tensor([int(label) for label in labels], dtype=torch.long, device=device)
    if task_key == "ranking":
        return torch.tensor(labels, dtype=torch.float32, device=device)
    if task_key in {"binary", "regression"}:
        return torch.tensor(labels, dtype=torch.float32, device=device)
    raise ValueError(f"Unsupported task type: {task_type!r}")


def tensorize_uplift_targets(
    records: list[Any],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    treatments: list[float] = []
    outcomes: list[float] = []
    propensities: list[float] = []
    has_propensity = True

    for record in records:
        label = getattr(record, "label", None)
        if isinstance(label, dict):
            treatments.append(float(label["treatment"]))
            outcomes.append(float(label["outcome"]))
            if "propensity" in label:
                propensities.append(float(label["propensity"]))
            else:
                has_propensity = False
        elif isinstance(label, (list, tuple)):
            if len(label) < 2:
                raise ValueError("Uplift labels must contain at least treatment and outcome.")
            treatments.append(float(label[0]))
            outcomes.append(float(label[1]))
            if len(label) >= 3:
                propensities.append(float(label[2]))
            else:
                has_propensity = False
        else:
            raise ValueError(
                "Uplift labels must be dicts with treatment/outcome keys or tuples "
                "of (treatment, outcome[, propensity])."
            )

    propensity_tensor = None
    if has_propensity and propensities:
        propensity_tensor = torch.tensor(propensities, dtype=torch.float32, device=device)
    return (
        torch.tensor(treatments, dtype=torch.float32, device=device),
        torch.tensor(outcomes, dtype=torch.float32, device=device),
        propensity_tensor,
    )
