from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TokenizerConfig:
    categorical_threshold: int = 128
    numeric_bucket_count: int = 64
    text_vocab_size: int = 4096
    bpe_min_frequency: int = 2
    max_event_tokens: int = 24
    max_profile_tokens: int = 200
    max_events: int = 6500
    force_categorical_fields: tuple[str, ...] = field(
        default_factory=lambda: (
            "balance_quantile",
            "channel",
            "country",
            "currency",
            "direction",
            "merchant",
            "mcc",
            "plan",
            "service_region",
            "symbol",
            "type",
        )
    )


@dataclass(slots=True)
class MaskingConfig:
    token_mask_probability: float = 0.15
    event_mask_probability: float = 0.10
    key_mask_probability: float = 0.10
    unk_probability: float = 0.10
    label_smoothing: float = 0.1
    ignore_index: int = -100
    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"
    unk_token: str = "[UNK]"


@dataclass(slots=True)
class TextEncoderConfig:
    enabled: bool = False
    provider: str = "hash"
    model_name: str | None = None
    output_dim: int = 384
    max_length: int = 128
    local_files_only: bool = True
    placeholder_token: str = "[TEXT]"
    target_fields: tuple[str, ...] = field(default_factory=tuple)
    text_loss_weight: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    d_ffn: int
    profile_layers: int
    event_layers: int
    history_layers: int
    num_heads: int
    dropout: float = 0.1
    max_event_tokens: int = 24
    label_smoothing: float = 0.1
    text_encoder_dim: int = 0
    text_loss_weight: float = 1.0
    attention_backend: str = "auto"


@dataclass(slots=True)
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False


@dataclass(slots=True)
class CheckpointConfig:
    save_every: int = 0
    keep_last: int = 2
    resume_path: str | None = None


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 0
    device: str = "auto"
    deterministic: bool = False
    compile: bool = False
    matmul_precision: str = "high"
    log_every: int = 1
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass(slots=True)
class BenchmarkConfig:
    batch_size: int = 8
    steps: int = 10
    warmup_steps: int = 2
    num_records: int = 128
    backends: tuple[str, ...] = ("auto", "sdpa", "manual")
    pack_events: tuple[bool, ...] = (True, False)
    output_path: str = "artifacts/benchmark/results.json"


MODEL_VARIANTS: dict[str, dict[str, int]] = {
    "S": {
        "d_model": 192,
        "d_ffn": 768,
        "profile_layers": 1,
        "event_layers": 5,
        "history_layers": 2,
        "num_heads": 3,
    },
    "M": {
        "d_model": 512,
        "d_ffn": 2048,
        "profile_layers": 3,
        "event_layers": 16,
        "history_layers": 6,
        "num_heads": 8,
    },
    "L": {
        "d_model": 1024,
        "d_ffn": 4096,
        "profile_layers": 9,
        "event_layers": 45,
        "history_layers": 18,
        "num_heads": 16,
    },
}


def make_model_config(
    variant: str,
    vocab_size: int,
    *,
    dropout: float = 0.1,
    label_smoothing: float = 0.1,
    max_event_tokens: int = 24,
    text_encoder_dim: int = 0,
    text_loss_weight: float = 1.0,
    attention_backend: str = "auto",
) -> ModelConfig:
    variant_key = variant.upper()
    if variant_key not in MODEL_VARIANTS:
        known = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown PRAGMA variant {variant!r}. Expected one of: {known}")
    params = MODEL_VARIANTS[variant_key]
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        d_ffn=params["d_ffn"],
        profile_layers=params["profile_layers"],
        event_layers=params["event_layers"],
        history_layers=params["history_layers"],
        num_heads=params["num_heads"],
        dropout=dropout,
        max_event_tokens=max_event_tokens,
        label_smoothing=label_smoothing,
        text_encoder_dim=text_encoder_dim,
        text_loss_weight=text_loss_weight,
        attention_backend=attention_backend,
    )


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in {path!s}.")
    return payload


def tokenizer_config_from_dict(payload: dict[str, Any] | None = None) -> TokenizerConfig:
    data = dict(payload or {})
    if "force_categorical_fields" in data:
        data["force_categorical_fields"] = tuple(data["force_categorical_fields"])
    return TokenizerConfig(**data)


def masking_config_from_dict(payload: dict[str, Any] | None = None) -> MaskingConfig:
    return MaskingConfig(**dict(payload or {}))


def text_encoder_config_from_dict(payload: dict[str, Any] | None = None) -> TextEncoderConfig:
    data = dict(payload or {})
    if "target_fields" in data:
        data["target_fields"] = tuple(data["target_fields"])
    return TextEncoderConfig(**data)


def distributed_config_from_dict(payload: dict[str, Any] | None = None) -> DistributedConfig:
    return DistributedConfig(**dict(payload or {}))


def checkpoint_config_from_dict(payload: dict[str, Any] | None = None) -> CheckpointConfig:
    return CheckpointConfig(**dict(payload or {}))


def runtime_config_from_dict(payload: dict[str, Any] | None = None) -> RuntimeConfig:
    data = dict(payload or {})
    data["distributed"] = distributed_config_from_dict(data.get("distributed"))
    data["checkpointing"] = checkpoint_config_from_dict(data.get("checkpointing"))
    return RuntimeConfig(**data)


def benchmark_config_from_dict(payload: dict[str, Any] | None = None) -> BenchmarkConfig:
    data = dict(payload or {})
    if "backends" in data:
        data["backends"] = tuple(data["backends"])
    if "pack_events" in data:
        data["pack_events"] = tuple(bool(item) for item in data["pack_events"])
    return BenchmarkConfig(**data)
