from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import (
    binary_classification_metrics,
    ranking_metrics,
    regression_metrics,
    uplift_metrics,
)


@dataclass(slots=True)
class StandardScaler:
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None

    def fit(self, inputs: torch.Tensor) -> "StandardScaler":
        self.mean = inputs.mean(dim=0)
        self.std = inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
        return self

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fit before calling transform().")
        return (inputs - self.mean) / self.std

    def fit_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        self.fit(inputs)
        return self.transform(inputs)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits


def compute_embeddings(
    backbone,
    tokenizer,
    records,
    *,
    device: torch.device,
    pooling: str = "usr_last",
    batch_size: int = 32,
) -> torch.Tensor:
    backbone.eval()
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch = tokenizer.collate(batch_records, apply_mlm=False, device=device)
            encoded = backbone.encode(batch, use_masked_values=False)
            pooled = backbone.pooled_embedding(batch, encoded.history_embeddings, mode=pooling)
            embeddings.append(pooled.detach().cpu().to(torch.float32))
    if embeddings:
        return torch.cat(embeddings, dim=0)
    width = backbone.config.d_model if pooling in {"usr", "last_evt"} else 2 * backbone.config.d_model
    return torch.zeros((0, width), dtype=torch.float32)


def _loss_for_task(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> torch.Tensor:
    task_key = task_type.lower()
    if task_key == "binary":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if task_key == "regression":
        return F.mse_loss(logits, targets)
    if task_key == "ranking":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if task_key == "multiclass":
        return F.cross_entropy(logits, targets)
    raise ValueError(f"Unsupported probe task type: {task_type!r}")


def fit_lbfgs_probe(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    task_type: str,
    num_outputs: int = 1,
    max_iter: int = 128,
) -> LinearProbe:
    device = inputs.device
    probe = LinearProbe(inputs.shape[1], num_outputs).to(device)
    optimizer = torch.optim.LBFGS(
        probe.parameters(),
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = probe(inputs)
        loss = _loss_for_task(logits, targets, task_type)
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


@dataclass(slots=True)
class TLearnerProbe:
    treated_probe: LinearProbe
    control_probe: LinearProbe

    def predict_uplift(self, inputs: torch.Tensor) -> torch.Tensor:
        treated_scores = torch.sigmoid(self.treated_probe(inputs))
        control_scores = torch.sigmoid(self.control_probe(inputs))
        return treated_scores - control_scores


def fit_t_learner_probe(
    inputs: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    *,
    max_iter: int = 128,
) -> TLearnerProbe:
    treated_mask = treatment == 1
    control_mask = treatment == 0
    if not treated_mask.any() or not control_mask.any():
        raise ValueError("Uplift probe requires both treated and control examples.")
    treated_probe = fit_lbfgs_probe(
        inputs[treated_mask],
        outcome[treated_mask],
        task_type="binary",
        num_outputs=1,
        max_iter=max_iter,
    )
    control_probe = fit_lbfgs_probe(
        inputs[control_mask],
        outcome[control_mask],
        task_type="binary",
        num_outputs=1,
        max_iter=max_iter,
    )
    return TLearnerProbe(treated_probe=treated_probe, control_probe=control_probe)


def evaluate_probe(
    probe: LinearProbe,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    task_type: str,
) -> dict[str, float]:
    with torch.no_grad():
        logits = probe(inputs)
    task_key = task_type.lower()
    if task_key == "binary":
        return binary_classification_metrics(logits, targets)
    if task_key == "regression":
        return regression_metrics(logits, targets)
    if task_key == "ranking":
        return ranking_metrics(logits, targets)
    if task_key == "multiclass":
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).to(torch.float32).mean().item()
        return {"accuracy": float(accuracy)}
    raise ValueError(f"Unsupported probe task type: {task_type!r}")


def evaluate_t_learner(
    probe: TLearnerProbe,
    inputs: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    propensity: torch.Tensor | None = None,
) -> dict[str, float]:
    with torch.no_grad():
        uplift_scores = probe.predict_uplift(inputs)
    return uplift_metrics(uplift_scores, treatment, outcome, propensity)
