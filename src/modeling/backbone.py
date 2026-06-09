from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig
from data.tokenizer import PragmaBatch


def sinusoidal_embedding(indices: torch.Tensor, dim: int) -> torch.Tensor:
    if indices.numel() == 0:
        return indices.new_zeros(indices.shape + (dim,), dtype=torch.float32)
    device = indices.device
    dtype = torch.float32
    values = indices.to(dtype)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    angles = values.unsqueeze(-1) * div_term
    output = torch.zeros(indices.shape + (dim,), device=device, dtype=dtype)
    output[..., 0::2] = torch.sin(angles)
    output[..., 1::2] = torch.cos(angles)
    return output


def apply_continuous_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    half_dim = head_dim // 2
    if half_dim == 0:
        return query, key

    device = query.device
    dtype = query.dtype
    inv_freq = torch.exp(
        torch.arange(0, half_dim, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(half_dim, 1))
    )
    angles = positions.to(dtype).unsqueeze(1).unsqueeze(-1) * inv_freq
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        left = x[..., :half_dim]
        right = x[..., half_dim : 2 * half_dim]
        rotated = torch.cat([left * cos - right * sin, left * sin + right * cos], dim=-1)
        if head_dim % 2 == 1:
            rotated = torch.cat([rotated, x[..., 2 * half_dim :]], dim=-1)
        return rotated

    return rotate(query), rotate(key)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape
        qkv = self.qkv(inputs).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        if rope_positions is not None:
            query, key = apply_continuous_rope(query, key, rope_positions)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if valid_mask is not None:
            key_mask = valid_mask[:, None, None, :]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.proj(output)
        output = self.dropout(output)
        if valid_mask is not None:
            output = output * valid_mask.unsqueeze(-1)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return self.dropout(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ffn=d_ffn, dropout=dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = inputs + self.attention(
            self.norm1(inputs),
            valid_mask=valid_mask,
            rope_positions=rope_positions,
        )
        return hidden + self.ffn(self.norm2(hidden))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        d_model: int,
        d_ffn: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, valid_mask=valid_mask, rope_positions=rope_positions)
        return hidden


class CalendarFeatureEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features.new_zeros(features.shape[:-1] + (self.net[-1].out_features,))
        return self.net(features)


@dataclass(slots=True)
class BackboneOutput:
    profile_sequence: torch.Tensor
    local_event_tokens: torch.Tensor
    event_embeddings: torch.Tensor
    history_embeddings: torch.Tensor


@dataclass(slots=True)
class PretrainOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    masked_targets: torch.Tensor
    backbone: BackboneOutput


class PragmaBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.usr_token = nn.Parameter(torch.randn(config.d_model) * 0.02)
        self.evt_token = nn.Parameter(torch.randn(config.d_model) * 0.02)

        self.profile_encoder = TransformerEncoder(
            depth=config.profile_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.event_encoder = TransformerEncoder(
            depth=config.event_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.history_encoder = TransformerEncoder(
            depth=config.history_layers,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.calendar_encoder = CalendarFeatureEncoder(config.d_model, config.dropout)
        self.mlm_projection = nn.Linear(3 * config.d_model, config.d_model)

    def embed_pairs(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        value_positions: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_embed = self.token_embedding(key_ids)
        value_embed = self.token_embedding(value_ids)
        position_embed = sinusoidal_embedding(value_positions, self.config.d_model).to(key_embed.dtype)
        hidden = key_embed + value_embed + position_embed
        hidden = self.embedding_dropout(hidden)
        if token_mask is not None:
            hidden = hidden * token_mask.unsqueeze(-1)
        return hidden

    def encode(self, batch: PragmaBatch, *, use_masked_values: bool = True) -> BackboneOutput:
        device = batch.profile_key_ids.device
        value_ids = batch.masked_event_value_ids if use_masked_values else batch.event_value_ids

        profile_tokens = self.embed_pairs(
            batch.profile_key_ids,
            batch.profile_value_ids,
            batch.profile_value_positions,
            token_mask=batch.profile_token_mask,
        )
        batch_size = profile_tokens.shape[0]
        usr_token = self.usr_token.view(1, 1, -1).expand(batch_size, 1, -1)
        profile_inputs = torch.cat([usr_token, profile_tokens], dim=1)
        profile_valid_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                batch.profile_token_mask,
            ],
            dim=1,
        )
        profile_positions = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.float32, device=device),
                batch.profile_times,
            ],
            dim=1,
        )
        profile_sequence = self.profile_encoder(
            profile_inputs,
            valid_mask=profile_valid_mask,
            rope_positions=profile_positions,
        )
        user_embedding = profile_sequence[:, :1, :]

        batch_size, event_count, event_token_count = value_ids.shape
        event_tokens = self.embed_pairs(
            batch.event_key_ids,
            value_ids,
            batch.event_value_positions,
            token_mask=batch.event_token_mask,
        )
        flat_event_tokens = event_tokens.view(batch_size * event_count, event_token_count, -1)
        flat_token_mask = batch.event_token_mask.view(batch_size * event_count, event_token_count)
        flat_event_mask = batch.event_mask.view(batch_size * event_count)

        local_event_tokens = event_tokens.new_zeros(batch_size * event_count, event_token_count, self.config.d_model)
        event_embeddings = event_tokens.new_zeros(batch_size * event_count, self.config.d_model)
        if flat_event_mask.any():
            valid_indices = flat_event_mask.nonzero(as_tuple=False).squeeze(-1)
            valid_tokens = flat_event_tokens.index_select(0, valid_indices)
            valid_token_mask = flat_token_mask.index_select(0, valid_indices)
            evt_token = self.evt_token.view(1, 1, -1).expand(valid_tokens.shape[0], 1, -1)
            event_inputs = torch.cat([evt_token, valid_tokens], dim=1)
            event_valid_mask = torch.cat(
                [
                    torch.ones((valid_tokens.shape[0], 1), dtype=torch.bool, device=device),
                    valid_token_mask,
                ],
                dim=1,
            )
            event_sequence = self.event_encoder(event_inputs, valid_mask=event_valid_mask)
            calendar_embeddings = self.calendar_encoder(
                batch.event_calendar_features.view(batch_size * event_count, 6).index_select(0, valid_indices)
            )
            local_event_tokens.index_copy_(0, valid_indices, event_sequence[:, 1:, :])
            event_embeddings.index_copy_(0, valid_indices, event_sequence[:, 0, :] + calendar_embeddings)

        local_event_tokens = local_event_tokens.view(
            batch_size, event_count, event_token_count, self.config.d_model
        )
        event_embeddings = event_embeddings.view(batch_size, event_count, self.config.d_model)

        history_inputs = torch.cat([user_embedding, event_embeddings], dim=1)
        history_valid_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                batch.event_mask,
            ],
            dim=1,
        )
        history_positions = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.float32, device=device),
                batch.event_history_times,
            ],
            dim=1,
        )
        history_embeddings = self.history_encoder(
            history_inputs,
            valid_mask=history_valid_mask,
            rope_positions=history_positions,
        )

        return BackboneOutput(
            profile_sequence=profile_sequence,
            local_event_tokens=local_event_tokens,
            event_embeddings=event_embeddings,
            history_embeddings=history_embeddings,
        )

    def forward_pretrain(self, batch: PragmaBatch) -> PretrainOutput:
        backbone = self.encode(batch, use_masked_values=True)
        mask = batch.mlm_labels >= 0
        targets = batch.mlm_labels[mask]
        if mask.any():
            history_event_embeddings = backbone.history_embeddings[:, 1:, :]
            user_context = backbone.history_embeddings[:, :1, :]
            event_context = history_event_embeddings.unsqueeze(2).expand_as(backbone.local_event_tokens)
            user_context = user_context.unsqueeze(2).expand(
                -1,
                backbone.local_event_tokens.shape[1],
                backbone.local_event_tokens.shape[2],
                -1,
            )
            combined = torch.cat(
                [backbone.local_event_tokens, event_context, user_context],
                dim=-1,
            )
            masked_hidden = combined[mask]
            projected = self.mlm_projection(masked_hidden)
            logits = projected @ self.token_embedding.weight.t()
            loss = F.cross_entropy(
                logits,
                targets,
                label_smoothing=self.config.label_smoothing,
            )
        else:
            logits = backbone.history_embeddings.new_zeros((0, self.config.vocab_size))
            targets = batch.mlm_labels.new_zeros((0,), dtype=torch.long)
            loss = backbone.history_embeddings.sum() * 0.0
        return PretrainOutput(loss=loss, logits=logits, masked_targets=targets, backbone=backbone)

    def pooled_embedding(self, batch: PragmaBatch, history_embeddings: torch.Tensor, mode: str = "usr_last") -> torch.Tensor:
        user_embedding = history_embeddings[:, 0, :]
        event_counts = batch.event_mask.long().sum(dim=1)
        batch_indices = torch.arange(history_embeddings.shape[0], device=history_embeddings.device)
        last_event_embedding = history_embeddings[batch_indices, event_counts.clamp(min=0)]
        if mode == "usr":
            return user_embedding
        if mode == "last_evt":
            return last_event_embedding
        if mode == "usr_last":
            return torch.cat([user_embedding, last_event_embedding], dim=-1)
        raise ValueError(f"Unknown pooling mode: {mode}")


class PragmaClassifier(nn.Module):
    def __init__(
        self,
        backbone: PragmaBackbone,
        *,
        num_outputs: int = 1,
        pooling: str = "usr_last",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        head_dim = backbone.config.d_model if pooling in {"usr", "last_evt"} else 2 * backbone.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(head_dim, num_outputs)

    def forward(self, batch: PragmaBatch) -> torch.Tensor:
        backbone = self.backbone.encode(batch, use_masked_values=False)
        pooled = self.backbone.pooled_embedding(batch, backbone.history_embeddings, mode=self.pooling)
        logits = self.head(self.dropout(pooled))
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits
