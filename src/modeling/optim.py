from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


def _orthogonalize_update(update: torch.Tensor) -> torch.Tensor:
    matrix = update.detach().to(torch.float32)
    transpose = matrix.shape[0] < matrix.shape[1]
    if transpose:
        matrix = matrix.transpose(0, 1)
    try:
        u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        ortho = u @ vh
    except RuntimeError:
        ortho, _ = torch.linalg.qr(matrix, mode="reduced")
    if transpose:
        ortho = ortho.transpose(0, 1)
    ortho = ortho * math.sqrt(float(max(update.shape)))
    return ortho.to(update.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.ndim < 2:
                    parameter.add_(grad, alpha=-lr)
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(momentum).add_(grad)
                update = momentum_buffer
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(_orthogonalize_update(update), alpha=-lr)
        return loss


class CompositeOptimizer:
    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        self.optimizers = [optimizer for optimizer in optimizers if optimizer is not None]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, object]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        for optimizer, optimizer_state in zip(self.optimizers, state_dict.get("optimizers", [])):
            optimizer.load_state_dict(optimizer_state)


@dataclass(slots=True)
class OptimizerSplit:
    muon_params: list[torch.nn.Parameter]
    adamw_params: list[torch.nn.Parameter]


def split_muon_params(module: torch.nn.Module) -> OptimizerSplit:
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "embedding" not in name and "token" not in name:
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)
    return OptimizerSplit(muon_params=muon_params, adamw_params=adamw_params)


def build_muon_adamw_optimizer(
    module: torch.nn.Module,
    *,
    lr: float,
    adamw_lr: float | None = None,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
) -> CompositeOptimizer:
    split = split_muon_params(module)
    optimizers: list[torch.optim.Optimizer] = []
    if split.muon_params:
        optimizers.append(
            Muon(
                split.muon_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        )
    if split.adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                split.adamw_params,
                lr=adamw_lr if adamw_lr is not None else lr,
                betas=adamw_betas,
                weight_decay=weight_decay,
            )
        )
    return CompositeOptimizer(*optimizers)
