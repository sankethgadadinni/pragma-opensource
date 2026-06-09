from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .tokenizer import TokenizedEvent, TokenizedRecord


def _require_parquet_deps():
    try:
        import lmdb  # noqa: F401
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The sharded pretraining path requires both 'lmdb' and 'pyarrow'. "
            "Install them in the conda environment before building the shard store."
        ) from exc


@dataclass(slots=True)
class ShardSampleMetadata:
    sample_id: str
    shard_name: str
    row_index: int
    event_count: int
    profile_tokens: int
    event_tokens: int
    total_tokens: int


@dataclass(slots=True)
class ShardManifest:
    version: int
    shard_dir: str
    index_path: str
    sample_count: int
    shard_counts: dict[str, int]
    compression: str


def _materialize_tokenized_records(records: Iterable[Any], tokenizer) -> list[TokenizedRecord]:
    materialized: list[TokenizedRecord] = []
    for record in records:
        if isinstance(record, TokenizedRecord):
            tokenized = record
        else:
            tokenized = tokenizer.tokenize_record(record)
        if tokenized.events:
            materialized.append(tokenized)
    return materialized


def _record_to_row(record: TokenizedRecord) -> dict[str, Any]:
    return {
        "user_id": record.user_id,
        "profile_key_ids": record.profile_key_ids,
        "profile_value_ids": record.profile_value_ids,
        "profile_value_positions": record.profile_value_positions,
        "profile_text_values": record.profile_text_values,
        "profile_times": record.profile_times,
        "event_key_ids": [event.key_ids for event in record.events],
        "event_value_ids": [event.value_ids for event in record.events],
        "event_value_positions": [event.value_positions for event in record.events],
        "event_text_values": [event.text_values for event in record.events],
        "event_history_times": [event.history_time for event in record.events],
        "event_calendar_features": [event.calendar_features for event in record.events],
        "label_json": json.dumps(record.label),
    }


def _row_to_record(row: dict[str, Any]) -> TokenizedRecord:
    return TokenizedRecord(
        user_id=str(row["user_id"]),
        profile_key_ids=[int(item) for item in row["profile_key_ids"]],
        profile_value_ids=[int(item) for item in row["profile_value_ids"]],
        profile_value_positions=[int(item) for item in row["profile_value_positions"]],
        profile_text_values=[item if item is None else str(item) for item in row["profile_text_values"]],
        profile_times=[float(item) for item in row["profile_times"]],
        events=[
            TokenizedEvent(
                key_ids=[int(item) for item in key_ids],
                value_ids=[int(item) for item in value_ids],
                value_positions=[int(item) for item in positions],
                text_values=[item if item is None else str(item) for item in text_values],
                history_time=float(history_time),
                calendar_features=[float(item) for item in calendar_features],
            )
            for key_ids, value_ids, positions, text_values, history_time, calendar_features in zip(
                row["event_key_ids"],
                row["event_value_ids"],
                row["event_value_positions"],
                row["event_text_values"],
                row["event_history_times"],
                row["event_calendar_features"],
            )
        ],
        label=json.loads(row["label_json"]),
    )


def build_sharded_store(
    records: Iterable[Any],
    tokenizer,
    output_dir: str | Path,
    *,
    compression: str = "zstd",
) -> Path:
    _require_parquet_deps()
    import lmdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(output_dir)
    shard_dir = root / "event_shards"
    index_path = root / "user_index.lmdb"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tokenized_records = _materialize_tokenized_records(records, tokenizer)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_meta: dict[str, list[ShardSampleMetadata]] = defaultdict(list)

    sample_counter = 0
    for record in tokenized_records:
        event_count = len(record.events)
        shard_name = f"events_{event_count:05d}.parquet"
        row_index = len(grouped_rows[shard_name])
        grouped_rows[shard_name].append(_record_to_row(record))
        event_tokens = sum(len(event.value_ids) for event in record.events)
        profile_tokens = len(record.profile_value_ids)
        grouped_meta[shard_name].append(
            ShardSampleMetadata(
                sample_id=f"sample-{sample_counter:08d}",
                shard_name=shard_name,
                row_index=row_index,
                event_count=event_count,
                profile_tokens=profile_tokens,
                event_tokens=event_tokens,
                total_tokens=profile_tokens + event_tokens,
            )
        )
        sample_counter += 1

    for shard_name, rows in grouped_rows.items():
        columns: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            for key, value in row.items():
                columns[key].append(value)
        table = pa.Table.from_pydict(columns)
        pq.write_table(table, shard_dir / shard_name, compression=compression)

    env = lmdb.open(str(index_path), map_size=1 << 30)
    with env.begin(write=True) as txn:
        for shard_name, entries in grouped_meta.items():
            for entry in entries:
                txn.put(entry.sample_id.encode("utf-8"), json.dumps(asdict(entry)).encode("utf-8"))
    env.sync()
    env.close()

    manifest = ShardManifest(
        version=1,
        shard_dir=str(shard_dir),
        index_path=str(index_path),
        sample_count=sample_counter,
        shard_counts={name: len(rows) for name, rows in grouped_rows.items()},
        compression=compression,
    )
    (root / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return root


class ShardedRecordStore:
    def __init__(self, root: str | Path) -> None:
        _require_parquet_deps()
        import lmdb

        self.root = Path(root)
        manifest_payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.manifest = ShardManifest(**manifest_payload)
        self._env = lmdb.open(self.manifest.index_path, readonly=True, lock=False)
        self._metadata: list[ShardSampleMetadata] = self._load_metadata()
        self._rows_cache: dict[str, list[dict[str, Any]]] = {}

    def _load_metadata(self) -> list[ShardSampleMetadata]:
        entries: list[ShardSampleMetadata] = []
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                payload = json.loads(value.decode("utf-8"))
                entries.append(ShardSampleMetadata(**payload))
        return entries

    def _load_shard_rows(self, shard_name: str) -> list[dict[str, Any]]:
        if shard_name in self._rows_cache:
            return self._rows_cache[shard_name]
        import pyarrow.parquet as pq

        table = pq.read_table(Path(self.manifest.shard_dir) / shard_name)
        rows = table.to_pylist()
        self._rows_cache[shard_name] = rows
        return rows

    def iter_dynamic_batches(
        self,
        *,
        token_budget: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterable[list[TokenizedRecord]]:
        rng = random.Random(seed)
        by_shard: dict[str, list[ShardSampleMetadata]] = defaultdict(list)
        for entry in self._metadata:
            by_shard[entry.shard_name].append(entry)

        shard_names = list(by_shard)
        if shuffle:
            rng.shuffle(shard_names)

        for shard_name in shard_names:
            entries = list(by_shard[shard_name])
            if shuffle:
                rng.shuffle(entries)
            rows = self._load_shard_rows(shard_name)
            batch_entries: list[ShardSampleMetadata] = []
            batch_tokens = 0
            for entry in entries:
                sample_tokens = max(1, entry.total_tokens)
                if batch_entries and batch_tokens + sample_tokens > token_budget:
                    yield [_row_to_record(rows[item.row_index]) for item in batch_entries]
                    batch_entries = []
                    batch_tokens = 0
                batch_entries.append(entry)
                batch_tokens += sample_tokens
            if batch_entries:
                yield [_row_to_record(rows[item.row_index]) for item in batch_entries]

    def load_all_records(self) -> list[TokenizedRecord]:
        grouped: dict[str, list[ShardSampleMetadata]] = defaultdict(list)
        for entry in self._metadata:
            grouped[entry.shard_name].append(entry)
        records: list[TokenizedRecord] = []
        for shard_name, entries in grouped.items():
            rows = self._load_shard_rows(shard_name)
            records.extend(_row_to_record(rows[item.row_index]) for item in entries)
        return records
