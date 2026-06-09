from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling
        return self.base(inputs) + update


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 8
    alpha: int = 8
    dropout: float = 0.0
    target_substrings: tuple[str, ...] = ("qkv", "fc1", "fc2")


def inject_lora(module: nn.Module, config: LoRAConfig | None = None) -> None:
    lora_config = config or LoRAConfig()
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear) and any(
            target in child_name for target in lora_config.target_substrings
        ):
            setattr(
                module,
                child_name,
                LoRALinear(
                    child_module,
                    rank=lora_config.rank,
                    alpha=lora_config.alpha,
                    dropout=lora_config.dropout,
                ),
            )
        else:
            inject_lora(child_module, lora_config)


def freeze_non_lora_parameters(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad = "lora_" in name


def lora_parameter_count(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return trainable, total
