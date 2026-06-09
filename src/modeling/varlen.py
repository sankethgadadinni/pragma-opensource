from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


SUPPORTED_ATTENTION_BACKENDS = ("auto", "flash", "sdpa", "manual")


@dataclass(slots=True)
class AttentionBackendInfo:
    requested: str
    resolved: str
    flash_available: bool
    sdpa_available: bool


def _flash_attention_available() -> bool:
    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except ImportError:
        return False
    return True


def _sdpa_available() -> bool:
    return hasattr(F, "scaled_dot_product_attention")


def resolve_attention_backend(
    requested: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    valid_mask: torch.Tensor | None = None,
) -> AttentionBackendInfo:
    key = requested.lower()
    if key not in SUPPORTED_ATTENTION_BACKENDS:
        known = ", ".join(SUPPORTED_ATTENTION_BACKENDS)
        raise ValueError(f"Unsupported attention backend {requested!r}. Expected one of: {known}")

    sdpa_available = _sdpa_available()
    flash_available = (
        _flash_attention_available()
        and device.type == "cuda"
        and dtype in {torch.float16, torch.bfloat16}
        and (valid_mask is None or bool(valid_mask.all().item()))
    )

    if key == "auto":
        resolved = "flash" if flash_available else "sdpa" if sdpa_available else "manual"
    elif key == "flash":
        resolved = "flash" if flash_available else "sdpa" if sdpa_available else "manual"
    elif key == "sdpa":
        resolved = "sdpa" if sdpa_available else "manual"
    else:
        resolved = "manual"

    return AttentionBackendInfo(
        requested=key,
        resolved=resolved,
        flash_available=flash_available,
        sdpa_available=sdpa_available,
    )


def _masked_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)
    if valid_mask is not None:
        key_mask = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    return scores


def manual_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    scores = _masked_scores(query, key, valid_mask=valid_mask)
    attention = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention = F.dropout(attention, p=dropout_p, training=training)
    return torch.matmul(attention, value)


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    attn_mask: torch.Tensor | None = None
    if valid_mask is not None:
        invalid = ~valid_mask[:, None, None, :]
        attn_mask = torch.zeros(invalid.shape, dtype=query.dtype, device=query.device)
        attn_mask = attn_mask.masked_fill(invalid, torch.finfo(query.dtype).min)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    from flash_attn import flash_attn_func

    packed_query = query.transpose(1, 2).contiguous()
    packed_key = key.transpose(1, 2).contiguous()
    packed_value = value.transpose(1, 2).contiguous()
    output = flash_attn_func(
        packed_query,
        packed_key,
        packed_value,
        dropout_p=dropout_p if training else 0.0,
        causal=False,
    )
    return output.transpose(1, 2).contiguous()


def apply_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    requested_backend: str,
    valid_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> tuple[torch.Tensor, AttentionBackendInfo]:
    backend_info = resolve_attention_backend(
        requested_backend,
        device=query.device,
        dtype=query.dtype,
        valid_mask=valid_mask,
    )
    if backend_info.resolved == "flash":
        output = flash_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            training=training,
        )
    elif backend_info.resolved == "sdpa":
        output = sdpa_attention(
            query,
            key,
            value,
            valid_mask=valid_mask,
            dropout_p=dropout_p,
            training=training,
        )
    else:
        output = manual_attention(
            query,
            key,
            value,
            valid_mask=valid_mask,
            dropout_p=dropout_p,
            training=training,
        )
    return output, backend_info
