from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from .bpe import BPETokenizer
from .config import MaskingConfig, TokenizerConfig
from .masking import build_mlm_inputs
from .records import UserRecord, ensure_list, parse_timestamp


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Vocabulary:
    def __init__(self, tokens: list[str] | None = None) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        for token in tokens or []:
            self.add(token)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_id[token]

    def get(self, token: str, default: int | None = None) -> int | None:
        return self.token_to_id.get(token, default)

    def to_list(self) -> list[str]:
        return list(self.id_to_token)

    @classmethod
    def from_list(cls, tokens: list[str]) -> "Vocabulary":
        return cls(tokens=tokens)


@dataclass(slots=True)
class NumericBucketizer:
    bucket_count: int = 64
    edges: list[float] | None = None

    def fit(self, values: list[float]) -> None:
        non_zero = sorted(value for value in values if value != 0.0)
        if not non_zero:
            self.edges = []
            return
        if len(non_zero) == 1:
            self.edges = [non_zero[0]]
            return

        edge_count = max(1, min(self.bucket_count - 1, len(non_zero) - 1))
        edges: list[float] = []
        for edge_index in range(1, edge_count + 1):
            position = edge_index * (len(non_zero) - 1) / (edge_count + 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                edge = non_zero[lower]
            else:
                mix = position - lower
                edge = non_zero[lower] * (1.0 - mix) + non_zero[upper] * mix
            edges.append(edge)
        self.edges = sorted(set(edges))

    @property
    def total_buckets(self) -> int:
        if not self.edges:
            return 1
        return len(self.edges) + 1

    def bucket_index(self, value: float) -> int:
        if value == 0.0:
            return 0
        if not self.edges:
            return 1
        return bisect.bisect_right(self.edges, float(value)) + 1

    def to_dict(self) -> dict[str, object]:
        return {"bucket_count": self.bucket_count, "edges": self.edges or []}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NumericBucketizer":
        return cls(
            bucket_count=int(data["bucket_count"]),
            edges=[float(item) for item in data.get("edges", [])],  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class TokenizedEvent:
    key_ids: list[int]
    value_ids: list[int]
    value_positions: list[int]
    history_time: float
    calendar_features: list[float]


@dataclass(slots=True)
class TokenizedRecord:
    user_id: str
    profile_key_ids: list[int]
    profile_value_ids: list[int]
    profile_value_positions: list[int]
    profile_times: list[float]
    events: list[TokenizedEvent]
    label: Any = None


@dataclass(slots=True)
class PragmaBatch:
    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_value_positions: torch.Tensor
    profile_times: torch.Tensor
    profile_token_mask: torch.Tensor
    event_key_ids: torch.Tensor
    event_value_ids: torch.Tensor
    masked_event_value_ids: torch.Tensor
    event_value_positions: torch.Tensor
    event_history_times: torch.Tensor
    event_calendar_features: torch.Tensor
    event_token_mask: torch.Tensor
    event_mask: torch.Tensor
    mlm_labels: torch.Tensor
    downstream_labels: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "PragmaBatch":
        kwargs: dict[str, Any] = {}
        for dataclass_field in fields(self):
            field_name = dataclass_field.name
            value = getattr(self, field_name)
            if torch.is_tensor(value):
                kwargs[field_name] = value.to(device)
            else:
                kwargs[field_name] = value
        return PragmaBatch(**kwargs)


class PragmaTokenizer:
    def __init__(
        self,
        config: TokenizerConfig | None = None,
        masking_config: MaskingConfig | None = None,
    ) -> None:
        self.config = config or TokenizerConfig()
        self.masking_config = masking_config or MaskingConfig()
        self.vocab = Vocabulary(
            [
                self.masking_config.pad_token,
                self.masking_config.mask_token,
                self.masking_config.unk_token,
            ]
        )
        self.pad_token_id = self.vocab[self.masking_config.pad_token]
        self.mask_token_id = self.vocab[self.masking_config.mask_token]
        self.unk_token_id = self.vocab[self.masking_config.unk_token]

        self.field_types: dict[str, str] = {}
        self.key_token_ids: dict[str, int] = {}
        self.numeric_bucketizers: dict[str, NumericBucketizer] = {}
        self.categorical_value_ids: dict[str, dict[str, int]] = {}
        self.text_subword_ids: dict[str, int] = {}
        self.bpe = BPETokenizer(
            vocab_size=self.config.text_vocab_size,
            min_frequency=self.config.bpe_min_frequency,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, records: list[UserRecord | dict[str, Any]]) -> None:
        materialized = [self._ensure_record(record) for record in records]

        numeric_values: dict[str, list[float]] = defaultdict(list)
        string_values: dict[str, list[str]] = defaultdict(list)
        seen_numeric: set[str] = set()
        seen_string: set[str] = set()

        for record in materialized:
            for field, value in record.profile.items():
                self._accumulate_field_stats(field, value, numeric_values, string_values, seen_numeric, seen_string)
            for lifelong in record.lifelong:
                self._accumulate_field_stats(
                    lifelong.key,
                    lifelong.value,
                    numeric_values,
                    string_values,
                    seen_numeric,
                    seen_string,
                )
            for event in record.events:
                for field, value in event.fields.items():
                    self._accumulate_field_stats(
                        field,
                        value,
                        numeric_values,
                        string_values,
                        seen_numeric,
                        seen_string,
                    )

        all_fields = sorted(seen_numeric | seen_string)
        forced_categorical = set(self.config.force_categorical_fields)
        for field in all_fields:
            if field in seen_numeric and field not in seen_string:
                self.field_types[field] = "numeric"
            elif field in forced_categorical:
                self.field_types[field] = "categorical"
            else:
                distinct = len(set(string_values[field]))
                if distinct <= self.config.categorical_threshold:
                    self.field_types[field] = "categorical"
                else:
                    self.field_types[field] = "text"

        text_corpus: list[str] = []
        for field, field_type in self.field_types.items():
            if field_type == "numeric":
                bucketizer = NumericBucketizer(bucket_count=self.config.numeric_bucket_count)
                bucketizer.fit(numeric_values[field])
                self.numeric_bucketizers[field] = bucketizer
            elif field_type == "categorical":
                unique_values = sorted({str(value) for value in string_values[field]})
                self.categorical_value_ids[field] = {}
                for value in unique_values:
                    token = f"cat:{field}:{value}"
                    self.categorical_value_ids[field][value] = self.vocab.add(token)
            else:
                text_corpus.extend(string_values[field])

            self.key_token_ids[field] = self.vocab.add(f"key:{field}")

        self.bpe.fit(text_corpus)
        for subword in self.bpe.tokens:
            token = f"txt:{subword}"
            self.text_subword_ids[subword] = self.vocab.add(token)

        for field, bucketizer in self.numeric_bucketizers.items():
            for bucket_index in range(bucketizer.total_buckets + 1):
                self.vocab.add(f"num:{field}:{bucket_index}")

    def tokenize_record(self, record: UserRecord | dict[str, Any]) -> TokenizedRecord:
        user_record = self._ensure_record(record)
        evaluation_ts = parse_timestamp(user_record.evaluation_ts)

        profile_key_ids: list[int] = []
        profile_value_ids: list[int] = []
        profile_positions: list[int] = []
        profile_times: list[float] = []

        for field, value in user_record.profile.items():
            self._append_field_encoding(
                field=field,
                value=value,
                time_value=0.0,
                key_ids=profile_key_ids,
                value_ids=profile_value_ids,
                value_positions=profile_positions,
                time_values=profile_times,
            )

        for lifelong in user_record.lifelong:
            elapsed_seconds = max(
                0.0,
                (evaluation_ts - parse_timestamp(lifelong.timestamp)).total_seconds(),
            )
            self._append_field_encoding(
                field=lifelong.key,
                value=lifelong.value,
                time_value=self._soft_log_seconds(elapsed_seconds),
                key_ids=profile_key_ids,
                value_ids=profile_value_ids,
                value_positions=profile_positions,
                time_values=profile_times,
            )

        profile_limit = self.config.max_profile_tokens
        profile_key_ids = profile_key_ids[:profile_limit]
        profile_value_ids = profile_value_ids[:profile_limit]
        profile_positions = profile_positions[:profile_limit]
        profile_times = profile_times[:profile_limit]

        sorted_events = sorted(user_record.events, key=lambda item: parse_timestamp(item.timestamp))
        if self.config.max_events > 0 and len(sorted_events) > self.config.max_events:
            sorted_events = sorted_events[-self.config.max_events :]

        last_event_ts = parse_timestamp(sorted_events[-1].timestamp) if sorted_events else evaluation_ts
        tokenized_events: list[TokenizedEvent] = []
        for event in sorted_events:
            event_ts = parse_timestamp(event.timestamp)
            key_ids: list[int] = []
            value_ids: list[int] = []
            value_positions: list[int] = []
            time_gap = max(0.0, (last_event_ts - event_ts).total_seconds())
            for field, value in event.fields.items():
                self._append_field_encoding(
                    field=field,
                    value=value,
                    time_value=0.0,
                    key_ids=key_ids,
                    value_ids=value_ids,
                    value_positions=value_positions,
                    time_values=None,
                )

            token_limit = self.config.max_event_tokens
            if token_limit > 0:
                key_ids = key_ids[:token_limit]
                value_ids = value_ids[:token_limit]
                value_positions = value_positions[:token_limit]
            if not value_ids:
                continue
            tokenized_events.append(
                TokenizedEvent(
                    key_ids=key_ids,
                    value_ids=value_ids,
                    value_positions=value_positions,
                    history_time=self._soft_log_seconds(time_gap),
                    calendar_features=self._calendar_features(event_ts),
                )
            )

        return TokenizedRecord(
            user_id=user_record.user_id,
            profile_key_ids=profile_key_ids,
            profile_value_ids=profile_value_ids,
            profile_value_positions=profile_positions,
            profile_times=profile_times,
            events=tokenized_events,
            label=user_record.label,
        )

    def collate(
        self,
        records: list[TokenizedRecord | UserRecord | dict[str, Any]],
        *,
        apply_mlm: bool = True,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> PragmaBatch:
        tokenized_records = [
            record if isinstance(record, TokenizedRecord) else self.tokenize_record(record)
            for record in records
        ]
        batch_size = len(tokenized_records)
        max_profile_tokens = max((len(record.profile_key_ids) for record in tokenized_records), default=0)
        max_events = max((len(record.events) for record in tokenized_records), default=0)
        max_event_tokens = max(
            (len(event.value_ids) for record in tokenized_records for event in record.events),
            default=0,
        )

        profile_key_ids = torch.full(
            (batch_size, max_profile_tokens),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        profile_value_ids = torch.full_like(profile_key_ids, fill_value=self.pad_token_id)
        profile_value_positions = torch.zeros((batch_size, max_profile_tokens), dtype=torch.long)
        profile_times = torch.zeros((batch_size, max_profile_tokens), dtype=torch.float32)
        profile_token_mask = torch.zeros((batch_size, max_profile_tokens), dtype=torch.bool)

        event_shape = (batch_size, max_events, max_event_tokens)
        event_key_ids = torch.full(event_shape, fill_value=self.pad_token_id, dtype=torch.long)
        event_value_ids = torch.full_like(event_key_ids, fill_value=self.pad_token_id)
        event_value_positions = torch.zeros(event_shape, dtype=torch.long)
        event_token_mask = torch.zeros(event_shape, dtype=torch.bool)
        event_mask = torch.zeros((batch_size, max_events), dtype=torch.bool)
        event_history_times = torch.zeros((batch_size, max_events), dtype=torch.float32)
        event_calendar_features = torch.zeros((batch_size, max_events, 6), dtype=torch.float32)

        downstream_labels: torch.Tensor | None = None
        raw_labels = [record.label for record in tokenized_records]

        for batch_index, record in enumerate(tokenized_records):
            profile_length = len(record.profile_key_ids)
            if profile_length:
                profile_key_ids[batch_index, :profile_length] = torch.tensor(
                    record.profile_key_ids, dtype=torch.long
                )
                profile_value_ids[batch_index, :profile_length] = torch.tensor(
                    record.profile_value_ids, dtype=torch.long
                )
                profile_value_positions[batch_index, :profile_length] = torch.tensor(
                    record.profile_value_positions,
                    dtype=torch.long,
                )
                profile_times[batch_index, :profile_length] = torch.tensor(
                    record.profile_times, dtype=torch.float32
                )
                profile_token_mask[batch_index, :profile_length] = True

            for event_index, event in enumerate(record.events):
                token_length = len(event.value_ids)
                if token_length == 0:
                    continue
                event_key_ids[batch_index, event_index, :token_length] = torch.tensor(
                    event.key_ids, dtype=torch.long
                )
                event_value_ids[batch_index, event_index, :token_length] = torch.tensor(
                    event.value_ids, dtype=torch.long
                )
                event_value_positions[batch_index, event_index, :token_length] = torch.tensor(
                    event.value_positions,
                    dtype=torch.long,
                )
                event_token_mask[batch_index, event_index, :token_length] = True
                event_mask[batch_index, event_index] = True
                event_history_times[batch_index, event_index] = float(event.history_time)
                event_calendar_features[batch_index, event_index] = torch.tensor(
                    event.calendar_features, dtype=torch.float32
                )

        if raw_labels and all(label is not None for label in raw_labels):
            first = raw_labels[0]
            if isinstance(first, (list, tuple)):
                downstream_labels = torch.tensor(raw_labels, dtype=torch.float32)
            else:
                downstream_labels = torch.tensor(raw_labels, dtype=torch.float32)

        if apply_mlm and max_events > 0 and max_event_tokens > 0:
            masked_event_value_ids, mlm_labels = build_mlm_inputs(
                event_value_ids=event_value_ids,
                event_key_ids=event_key_ids,
                event_token_mask=event_token_mask,
                event_mask=event_mask,
                mask_token_id=self.mask_token_id,
                unk_token_id=self.unk_token_id,
                config=self.masking_config,
                generator=generator,
            )
        else:
            masked_event_value_ids = event_value_ids.clone()
            mlm_labels = torch.full_like(
                event_value_ids, fill_value=self.masking_config.ignore_index
            )

        batch = PragmaBatch(
            profile_key_ids=profile_key_ids,
            profile_value_ids=profile_value_ids,
            profile_value_positions=profile_value_positions,
            profile_times=profile_times,
            profile_token_mask=profile_token_mask,
            event_key_ids=event_key_ids,
            event_value_ids=event_value_ids,
            masked_event_value_ids=masked_event_value_ids,
            event_value_positions=event_value_positions,
            event_history_times=event_history_times,
            event_calendar_features=event_calendar_features,
            event_token_mask=event_token_mask,
            event_mask=event_mask,
            mlm_labels=mlm_labels,
            downstream_labels=downstream_labels,
        )
        if device is not None:
            batch = batch.to(device)
        return batch

    def save(self, path: str | Path) -> None:
        payload = {
            "config": asdict(self.config),
            "masking_config": asdict(self.masking_config),
            "vocab": self.vocab.to_list(),
            "field_types": self.field_types,
            "key_token_ids": self.key_token_ids,
            "numeric_bucketizers": {
                field: bucketizer.to_dict() for field, bucketizer in self.numeric_bucketizers.items()
            },
            "categorical_value_ids": self.categorical_value_ids,
            "text_subword_ids": self.text_subword_ids,
            "bpe": self.bpe.to_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PragmaTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            config=TokenizerConfig(**payload["config"]),
            masking_config=MaskingConfig(**payload["masking_config"]),
        )
        tokenizer.vocab = Vocabulary.from_list(payload["vocab"])
        tokenizer.pad_token_id = tokenizer.vocab[tokenizer.masking_config.pad_token]
        tokenizer.mask_token_id = tokenizer.vocab[tokenizer.masking_config.mask_token]
        tokenizer.unk_token_id = tokenizer.vocab[tokenizer.masking_config.unk_token]
        tokenizer.field_types = {str(k): str(v) for k, v in payload["field_types"].items()}
        tokenizer.key_token_ids = {
            str(k): int(v) for k, v in payload["key_token_ids"].items()
        }
        tokenizer.numeric_bucketizers = {
            str(field): NumericBucketizer.from_dict(data)
            for field, data in payload["numeric_bucketizers"].items()
        }
        tokenizer.categorical_value_ids = {
            str(field): {str(k): int(v) for k, v in mapping.items()}
            for field, mapping in payload["categorical_value_ids"].items()
        }
        tokenizer.text_subword_ids = {
            str(k): int(v) for k, v in payload["text_subword_ids"].items()
        }
        tokenizer.bpe = BPETokenizer.from_dict(payload["bpe"])
        return tokenizer

    def _ensure_record(self, record: UserRecord | dict[str, Any]) -> UserRecord:
        if isinstance(record, UserRecord):
            return record
        return UserRecord.from_dict(record)

    def _accumulate_field_stats(
        self,
        field: str,
        value: Any,
        numeric_values: dict[str, list[float]],
        string_values: dict[str, list[str]],
        seen_numeric: set[str],
        seen_string: set[str],
    ) -> None:
        for item in ensure_list(value):
            if _is_numeric_scalar(item):
                numeric_values[field].append(float(item))
                seen_numeric.add(field)
            else:
                string_values[field].append(str(item))
                seen_string.add(field)

    def _append_field_encoding(
        self,
        *,
        field: str,
        value: Any,
        time_value: float,
        key_ids: list[int],
        value_ids: list[int],
        value_positions: list[int],
        time_values: list[float] | None,
    ) -> None:
        if field not in self.field_types:
            return
        encoded_values = self._encode_value_ids(field, value)
        if not encoded_values:
            return
        key_id = self.key_token_ids[field]
        start = len(key_ids)
        for offset, value_id in enumerate(encoded_values):
            key_ids.append(key_id)
            value_ids.append(value_id)
            value_positions.append(offset)
            if time_values is not None:
                time_values.append(time_value)
        if start == len(key_ids):
            return

    def _encode_value_ids(self, field: str, value: Any) -> list[int]:
        field_type = self.field_types[field]
        encoded: list[int] = []
        for item in ensure_list(value):
            if field_type == "numeric":
                bucketizer = self.numeric_bucketizers[field]
                bucket_index = bucketizer.bucket_index(float(item))
                token = f"num:{field}:{bucket_index}"
                encoded.append(self.vocab[token])
            elif field_type == "categorical":
                mapping = self.categorical_value_ids[field]
                encoded.append(mapping.get(str(item), self.unk_token_id))
            else:
                subwords = self.bpe.encode(str(item))
                if not subwords:
                    encoded.append(self.unk_token_id)
                else:
                    for subword in subwords:
                        encoded.append(self.text_subword_ids.get(subword, self.unk_token_id))
        return encoded

    def _calendar_features(self, timestamp) -> list[float]:
        dt = parse_timestamp(timestamp)
        hour_angle = 2.0 * math.pi * (dt.hour / 24.0)
        dow_angle = 2.0 * math.pi * (dt.weekday() / 7.0)
        dom_angle = 2.0 * math.pi * ((dt.day - 1) / 31.0)
        return [
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.sin(dow_angle),
            math.cos(dow_angle),
            math.sin(dom_angle),
            math.cos(dom_angle),
        ]

    def _soft_log_seconds(self, seconds: float) -> float:
        return 8.0 * math.log1p(max(0.0, seconds) / 8.0)
