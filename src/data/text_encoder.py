from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from config import TextEncoderConfig

from .bpe import basic_word_tokenize


class FrozenTextEncoder:
    def __init__(self, config: TextEncoderConfig) -> None:
        self.config = config
        self.output_dim = config.output_dim

    def applies_to(self, field: str) -> bool:
        target_fields = set(self.config.target_fields)
        return not target_fields or field in target_fields

    def encode(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


class HashTextEncoder(FrozenTextEncoder):
    def encode(self, texts: list[str]) -> torch.Tensor:
        vectors = torch.zeros((len(texts), self.output_dim), dtype=torch.float32)
        if not texts:
            return vectors
        for row_index, text in enumerate(texts):
            tokens = basic_word_tokenize(text)[: self.config.max_length]
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                idx = int.from_bytes(digest[:8], byteorder="little", signed=False) % self.output_dim
                sign = 1.0 if digest[8] % 2 == 0 else -1.0
                vectors[row_index, idx] += sign
            vectors[row_index] = vectors[row_index] / vectors[row_index].norm(p=2).clamp_min(1.0)
        return vectors


class HuggingFaceTextEncoder(FrozenTextEncoder):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__(config)
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for text_encoder.provider='hf'. "
                "Install it in the conda environment or switch to provider='hash'."
            ) from exc
        if not config.model_name:
            raise ValueError("text_encoder.model_name must be set for provider='hf'.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from Hugging Face model config.")
        self.output_dim = int(hidden_size)

    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.output_dim), dtype=torch.float32)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
        return pooled.detach().to(torch.float32).cpu()


def build_text_encoder(config: TextEncoderConfig | None) -> FrozenTextEncoder | None:
    if config is None or not config.enabled:
        return None
    provider = config.provider.lower()
    if provider == "hash":
        return HashTextEncoder(config)
    if provider == "hf":
        return HuggingFaceTextEncoder(config)
    raise ValueError(f"Unsupported text encoder provider: {config.provider!r}")
