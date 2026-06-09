from __future__ import annotations

import math

import torch


def _binary_curves(scores: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = scores.detach().to(torch.float32).flatten()
    targets = targets.detach().to(torch.float32).flatten()
    order = torch.argsort(scores, descending=True)
    scores = scores[order]
    targets = targets[order]
    positives = targets.sum().clamp_min(1.0)
    negatives = (1.0 - targets).sum().clamp_min(1.0)
    tps = torch.cumsum(targets, dim=0)
    fps = torch.cumsum(1.0 - targets, dim=0)
    recall = tps / positives
    precision = tps / (tps + fps).clamp_min(1.0)
    fpr = fps / negatives
    return precision, recall, fpr


def binary_roc_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    _, recall, fpr = _binary_curves(scores, targets)
    if fpr.numel() == 0 or recall.numel() == 0:
        return float("nan")
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr, torch.ones(1, device=fpr.device)])
    recall = torch.cat([torch.zeros(1, device=recall.device), recall, torch.ones(1, device=recall.device)])
    return float(torch.trapz(recall, fpr).item())


def binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    precision, recall, _ = _binary_curves(scores, targets)
    if recall.numel() == 0:
        return float("nan")
    recall = torch.cat([torch.zeros(1, device=recall.device), recall])
    precision = torch.cat([precision[:1], precision])
    return float(torch.trapz(precision, recall).item())


def binary_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = torch.sigmoid(logits.detach().to(torch.float32))
    targets = targets.detach().to(torch.float32)
    predictions = (scores >= threshold).to(torch.float32)
    true_positive = ((predictions == 1) & (targets == 1)).sum().item()
    false_positive = ((predictions == 1) & (targets == 0)).sum().item()
    false_negative = ((predictions == 0) & (targets == 1)).sum().item()
    true_negative = ((predictions == 0) & (targets == 0)).sum().item()
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    accuracy = (true_positive + true_negative) / max(1.0, float(targets.numel()))
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": binary_average_precision(scores, targets),
        "roc_auc": binary_roc_auc(scores, targets),
    }


def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = predictions.detach().to(torch.float32).flatten()
    targets = targets.detach().to(torch.float32).flatten()
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = math.sqrt(mse)
    target_mean = targets.mean()
    total_var = torch.sum((targets - target_mean) ** 2).item()
    residual_var = torch.sum((targets - predictions) ** 2).item()
    r2 = 0.0 if total_var == 0.0 else 1.0 - residual_var / total_var
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def ranking_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    scores = torch.sigmoid(logits.detach().to(torch.float32))
    targets = targets.detach().to(torch.float32)
    ap_values: list[float] = []
    for column in range(scores.shape[1]):
        if targets[:, column].sum().item() <= 0:
            continue
        ap_values.append(binary_average_precision(scores[:, column], targets[:, column]))
    mean_ap = float(sum(ap_values) / len(ap_values)) if ap_values else float("nan")
    return {"map": mean_ap}


def uplift_metrics(
    uplift_scores: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    propensity: torch.Tensor | None = None,
) -> dict[str, float]:
    uplift_scores = uplift_scores.detach().to(torch.float32).flatten()
    treatment = treatment.detach().to(torch.float32).flatten()
    outcome = outcome.detach().to(torch.float32).flatten()
    order = torch.argsort(uplift_scores, descending=True)
    treatment = treatment[order]
    outcome = outcome[order]

    treated_count = torch.cumsum(treatment, dim=0).clamp_min(1.0)
    control_count = torch.cumsum(1.0 - treatment, dim=0).clamp_min(1.0)
    treated_gain = torch.cumsum(outcome * treatment, dim=0) / treated_count
    control_gain = torch.cumsum(outcome * (1.0 - treatment), dim=0) / control_count
    uplift_curve = treated_gain - control_gain
    auuc = float(torch.trapz(uplift_curve, dx=1.0).item() / max(1, uplift_curve.numel()))

    chosen_treatment = (uplift_scores > 0).to(torch.float32)
    if propensity is None:
        propensity = torch.full_like(chosen_treatment, fill_value=treatment.mean().clamp(0.05, 0.95).item())
    else:
        propensity = propensity.detach().to(torch.float32).flatten()
    matching = (chosen_treatment == treatment).to(torch.float32)
    denom = torch.where(chosen_treatment == 1.0, propensity, 1.0 - propensity).clamp_min(1e-6)
    weights = matching / denom
    weighted_reward = weights * outcome
    normalizer = weights.sum().clamp_min(1e-6)
    snips = float((weighted_reward.sum() / normalizer).item())
    return {"auuc": auuc, "snips": snips}
