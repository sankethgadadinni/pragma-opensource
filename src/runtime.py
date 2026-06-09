from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from config import RuntimeConfig, runtime_config_from_dict


@dataclass(slots=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def resolve_device(name: str, *, local_rank: int = 0) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", local_rank)
    return device


def initialize_runtime(payload: dict[str, Any] | None = None) -> tuple[RuntimeConfig, DistributedContext]:
    runtime = runtime_config_from_dict(payload)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed_enabled = world_size > 1
    device = resolve_device(runtime.device, local_rank=local_rank if distributed_enabled else 0)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision(runtime.matmul_precision)
    if runtime.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if distributed_enabled and not dist.is_initialized():
        backend = runtime.distributed.backend
        if device.type != "cuda" and backend == "nccl":
            backend = "gloo"
        dist.init_process_group(backend=backend)

    context = DistributedContext(
        enabled=distributed_enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )
    return runtime, context


def finalize_runtime(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()


def is_main_process(context: DistributedContext) -> bool:
    return context.rank == 0


def seed_everything(seed: int, *, rank: int = 0) -> None:
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    np.random.seed(actual_seed % (2**32))
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


def make_step_seed(base_seed: int, *, step: int, rank: int = 0) -> int:
    return int(base_seed) + int(step) * 1_009 + int(rank) * 104_729


def seed_step_generators(base_seed: int, *, step: int, rank: int = 0) -> int:
    actual_seed = make_step_seed(base_seed, step=step, rank=rank)
    random.seed(actual_seed)
    np.random.seed(actual_seed % (2**32))
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    return actual_seed


def make_cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def wrap_ddp(
    module: torch.nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = False,
) -> torch.nn.Module:
    if not context.enabled:
        return module
    if context.device.type == "cuda":
        return DistributedDataParallel(
            module,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=broadcast_buffers,
        )
    return DistributedDataParallel(
        module,
        device_ids=None,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    unwrapped = module
    while True:
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        return unwrapped


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in unwrap_module(module).state_dict().items()}


def resolve_resume_checkpoint(
    output_dir: Path,
    *,
    prefix: str,
    configured_path: str | None = None,
    override_path: str | Path | None = None,
) -> Path | None:
    if override_path is not None:
        candidate = Path(override_path)
    elif configured_path:
        candidate = Path(configured_path)
    else:
        candidate = output_dir / "checkpoints" / f"{prefix}_latest.pt"
    if not candidate.is_absolute():
        candidate = (output_dir.parent / candidate).resolve()
    if candidate.exists():
        return candidate
    if override_path is not None or configured_path:
        raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
    return None


def save_training_checkpoint(
    output_dir: Path,
    *,
    prefix: str,
    step: int,
    model: torch.nn.Module,
    optimizer: Any | None,
    iterator_state: dict[str, Any] | None = None,
    tokenizer_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    keep_last: int = 2,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{prefix}_step{step:06d}.pt"
    payload: dict[str, Any] = {
        "step": int(step),
        "state_dict": cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "iterator_state": dict(iterator_state or {}),
        "tokenizer_path": tokenizer_path,
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, checkpoint_path)

    latest_path = checkpoint_dir / f"{prefix}_latest.pt"
    shutil.copyfile(checkpoint_path, latest_path)

    if keep_last > 0:
        checkpoints = sorted(checkpoint_dir.glob(f"{prefix}_step*.pt"))
        stale = checkpoints[:-keep_last]
        for old_path in stale:
            old_path.unlink(missing_ok=True)
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Any | None = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    unwrap_module(model).load_state_dict(state_dict, strict=False)
    if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
