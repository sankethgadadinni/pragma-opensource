from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pyarrow.csv as pa_csv
import pyarrow.dataset as pa_dataset

from .records import EventRecord, LifelongEvent, UserRecord, parse_timestamp


PathResolver = Callable[[str | Path], Path]

_ID_COLUMNS = ("client_id", "customer_id", "user_id")
_REPORT_COLUMNS = ("report_dt", "reporting_dt", "report_date", "reporting_date", "snapshot_date")
_TIME_COLUMNS = ("event_time", "timestamp", "ts", "datetime")
_FOLD_COLUMNS = ("fold", "split", "partition")
_TARGET_PATH_CANDIDATES = (
    "targets",
    "targets.parquet",
    "targets.csv",
    "target",
)
_TRANSACTION_PATH_CANDIDATES = (
    "detail/trx",
    "detail/trx.parquet",
    "trx",
    "trx.parquet",
    "transactions",
    "transactions.parquet",
)
_GEO_PATH_CANDIDATES = (
    "detail/geo",
    "detail/geo.parquet",
    "geo",
    "geo.parquet",
)
_DIALOG_PATH_CANDIDATES = (
    "detail/dialogs",
    "detail/dialogs.parquet",
    "dialogs",
    "dialogs.parquet",
)
_TRANSACTION_CATEGORICAL_FIELDS = {
    "currency",
    "event_type",
    "event_subtype",
}


@dataclass(slots=True)
class MbdTarget:
    client_id: str
    evaluation_ts: datetime
    fold: str | None
    label: Any


@dataclass(slots=True)
class MbdEvent:
    timestamp: datetime
    fields: dict[str, Any]
    modality: str


def load_mbd_records(
    config: Mapping[str, Any] | None = None,
    *,
    resolve_path: PathResolver | None = None,
) -> list[UserRecord]:
    mbd = dict(config or {})
    resolver = resolve_path or (lambda path_like: Path(path_like))

    root_dir = _resolve_optional_path(mbd.get("root_dir"), resolver)
    targets_path = _resolve_dataset_path(
        root_dir,
        mbd.get("targets_path"),
        _TARGET_PATH_CANDIDATES,
        resolver,
        required=True,
    )
    transactions_path = _resolve_dataset_path(
        root_dir,
        mbd.get("transactions_path"),
        _TRANSACTION_PATH_CANDIDATES,
        resolver,
        required=False,
    )
    geo_path = _resolve_dataset_path(
        root_dir,
        mbd.get("geo_path"),
        _GEO_PATH_CANDIDATES,
        resolver,
        required=False,
    )
    dialogs_path = _resolve_dataset_path(
        root_dir,
        mbd.get("dialogs_path"),
        _DIALOG_PATH_CANDIDATES,
        resolver,
        required=False,
    )

    allowed_folds = {str(item) for item in mbd.get("allowed_folds", [])}
    max_clients = int(mbd.get("max_clients", 0) or 0)
    max_records = int(mbd.get("max_records", 0) or 0)
    history_window_days = int(mbd.get("history_window_days", 365))
    attach_targets = bool(mbd.get("attach_targets", True))
    drop_empty_records = bool(mbd.get("drop_empty_records", False))
    shuffle_targets = bool(mbd.get("shuffle_targets", False))
    sample_seed = int(mbd.get("sample_seed", 0))

    targets_rows, target_columns = _read_rows(targets_path)
    target_client_column = _pick_column(target_columns, _ID_COLUMNS, context="targets")
    target_report_column = _pick_column(target_columns, _REPORT_COLUMNS, context="targets")
    target_fold_column = _pick_optional_column(target_columns, _FOLD_COLUMNS)
    label_fields = _resolve_label_fields(target_columns, mbd.get("label_fields"), target_client_column, target_report_column, target_fold_column)

    targets = _build_targets(
        targets_rows,
        client_column=target_client_column,
        report_column=target_report_column,
        fold_column=target_fold_column,
        label_fields=label_fields,
        label_mode=str(mbd.get("label_mode", "auto")),
        attach_targets=attach_targets,
        allowed_folds=allowed_folds,
    )
    if shuffle_targets and targets:
        random.Random(sample_seed).shuffle(targets)
    if max_clients > 0:
        allowed_clients: set[str] = set()
        truncated: list[MbdTarget] = []
        for target in targets:
            if target.client_id in allowed_clients or len(allowed_clients) < max_clients:
                allowed_clients.add(target.client_id)
                truncated.append(target)
        targets = truncated
    if max_records > 0:
        targets = targets[:max_records]
    if not targets:
        return []

    selected_clients = sorted({target.client_id for target in targets})
    window = timedelta(days=history_window_days) if history_window_days > 0 else None
    min_event_ts = min((target.evaluation_ts - window) for target in targets) if window is not None else None
    max_event_ts = max(target.evaluation_ts for target in targets)

    client_events: dict[str, list[MbdEvent]] = defaultdict(list)
    if transactions_path is not None:
        _extend_transaction_events(
            client_events,
            transactions_path,
            selected_clients=selected_clients,
            min_event_ts=min_event_ts,
            max_event_ts=max_event_ts,
        )
    if geo_path is not None:
        _extend_geo_events(
            client_events,
            geo_path,
            selected_clients=selected_clients,
            min_event_ts=min_event_ts,
            max_event_ts=max_event_ts,
        )
    if dialogs_path is not None:
        _extend_dialog_events(
            client_events,
            dialogs_path,
            selected_clients=selected_clients,
            min_event_ts=min_event_ts,
            max_event_ts=max_event_ts,
            strategy=str(mbd.get("dialog_embedding_strategy", "project")),
            projection_dim=int(mbd.get("dialog_projection_dim", 8)),
            projection_seed=int(mbd.get("dialog_seed", 0)),
        )

    for events in client_events.values():
        events.sort(key=lambda item: item.timestamp)

    records: list[UserRecord] = []
    for target in targets:
        events = client_events.get(target.client_id, [])
        history = _select_history(events, target.evaluation_ts, history_window_days)
        if drop_empty_records and not history:
            continue
        records.append(
            UserRecord(
                user_id=target.client_id,
                evaluation_ts=target.evaluation_ts,
                profile=_derive_profile(history, target.evaluation_ts),
                lifelong=_derive_lifelong(history),
                events=[EventRecord(timestamp=event.timestamp, fields=event.fields) for event in history],
                label=target.label,
            )
        )

    records.sort(key=lambda record: (record.user_id, _normalize_timestamp(record.evaluation_ts)))
    return records


def _resolve_optional_path(path_like: str | Path | None, resolve_path: PathResolver) -> Path | None:
    if path_like in {None, ""}:
        return None
    path = resolve_path(path_like)
    return path.expanduser()


def _resolve_dataset_path(
    root_dir: Path | None,
    explicit_path: str | Path | None,
    candidates: Iterable[str],
    resolve_path: PathResolver,
    *,
    required: bool,
) -> Path | None:
    if explicit_path not in {None, ""}:
        path = resolve_path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if root_dir is not None:
        for candidate in candidates:
            candidate_path = (root_dir / candidate).expanduser()
            if candidate_path.exists():
                return candidate_path
    if required:
        searched = ", ".join(candidates)
        root_hint = str(root_dir) if root_dir is not None else "<no root_dir>"
        raise FileNotFoundError(f"Could not find MBD data path under {root_hint}: {searched}")
    return None


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = path.expanduser()
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        parse_options = pa_csv.ParseOptions(delimiter="\t" if suffix == ".tsv" else ",")
        table = pa_csv.read_csv(path, parse_options=parse_options)
    else:
        dataset = pa_dataset.dataset(path)
        table = dataset.to_table()
    return table.to_pylist(), list(table.column_names)


def _read_columns(path: Path) -> list[str]:
    path = path.expanduser()
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        _, columns = _read_rows(path)
        return columns
    dataset = pa_dataset.dataset(path)
    return list(dataset.schema.names)


def _read_filtered_rows(
    path: Path,
    *,
    client_column: str,
    selected_clients: list[str],
    time_column: str,
    min_event_ts: datetime | None,
    max_event_ts: datetime | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    path = path.expanduser()
    selected_client_set = set(selected_clients)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        rows, columns = _read_rows(path)
        filtered: list[dict[str, Any]] = []
        for row in rows:
            client_id = str(row.get(client_column))
            if client_id not in selected_client_set:
                continue
            event_ts = _normalize_timestamp(row[time_column])
            if min_event_ts is not None and event_ts < min_event_ts:
                continue
            if max_event_ts is not None and event_ts >= max_event_ts:
                continue
            filtered.append(row)
        return filtered, columns

    dataset = pa_dataset.dataset(path)
    expression = pa_dataset.field(client_column).isin(selected_clients)
    table = dataset.to_table(filter=expression)
    rows = table.to_pylist()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        client_id = str(row.get(client_column))
        if client_id not in selected_client_set:
            continue
        event_ts = _normalize_timestamp(row[time_column])
        if min_event_ts is not None and event_ts < min_event_ts:
            continue
        if max_event_ts is not None and event_ts >= max_event_ts:
            continue
        filtered.append(row)
    return filtered, list(table.column_names)


def _pick_column(columns: Iterable[str], candidates: Iterable[str], *, context: str) -> str:
    found = _pick_optional_column(columns, candidates)
    if found is None:
        joined = ", ".join(columns)
        expected = ", ".join(candidates)
        raise ValueError(f"Could not find a {context} column. Looked for {expected}. Found: {joined}")
    return found


def _pick_optional_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _resolve_label_fields(
    columns: list[str],
    configured: Any,
    client_column: str,
    report_column: str,
    fold_column: str | None,
) -> list[str]:
    if configured is None:
        excluded = {client_column, report_column}
        if fold_column is not None:
            excluded.add(fold_column)
        return [column for column in columns if column not in excluded]
    if isinstance(configured, str):
        return [configured]
    return [str(item) for item in configured]


def _build_targets(
    rows: list[dict[str, Any]],
    *,
    client_column: str,
    report_column: str,
    fold_column: str | None,
    label_fields: list[str],
    label_mode: str,
    attach_targets: bool,
    allowed_folds: set[str],
) -> list[MbdTarget]:
    targets: list[MbdTarget] = []
    for row in rows:
        fold_value = None if fold_column is None else str(row.get(fold_column))
        if allowed_folds and fold_value not in allowed_folds:
            continue
        label = None
        if attach_targets:
            label = _format_label(row, label_fields, label_mode)
        targets.append(
            MbdTarget(
                client_id=str(row[client_column]),
                evaluation_ts=_normalize_timestamp(row[report_column]),
                fold=fold_value,
                label=label,
            )
        )
    return targets


def _format_label(row: Mapping[str, Any], label_fields: list[str], label_mode: str) -> Any:
    if not label_fields:
        return None
    values = [_coerce_scalar(row.get(field)) for field in label_fields]
    mode = label_mode.lower()
    if mode == "auto":
        mode = "scalar" if len(label_fields) == 1 else "vector"
    if mode == "scalar":
        if len(values) != 1:
            raise ValueError("label_mode=scalar requires exactly one label field.")
        return values[0]
    if mode == "vector":
        return values
    if mode == "dict":
        return {field: value for field, value in zip(label_fields, values, strict=True)}
    raise ValueError(f"Unsupported MBD label_mode={label_mode!r}.")


def _extend_transaction_events(
    client_events: dict[str, list[MbdEvent]],
    path: Path,
    *,
    selected_clients: list[str],
    min_event_ts: datetime | None,
    max_event_ts: datetime | None,
) -> None:
    columns = _read_columns(path)
    client_column = _pick_column(columns, _ID_COLUMNS, context="transaction client_id")
    time_column = _pick_column(columns, _TIME_COLUMNS, context="transaction event_time")
    rows, _ = _read_filtered_rows(
        path,
        client_column=client_column,
        selected_clients=selected_clients,
        time_column=time_column,
        min_event_ts=min_event_ts,
        max_event_ts=max_event_ts,
    )
    fold_column = _pick_optional_column(columns, _FOLD_COLUMNS)
    ignored = {client_column, time_column}
    if fold_column is not None:
        ignored.add(fold_column)
    for row in rows:
        fields: dict[str, Any] = {"modality": "trx"}
        for key, value in row.items():
            if key in ignored or value is None:
                continue
            coerced = _coerce_event_value(key, value, modality="trx")
            if coerced is not None:
                fields[key] = coerced
        if len(fields) == 1:
            continue
        client_events[str(row[client_column])].append(
            MbdEvent(
                timestamp=_normalize_timestamp(row[time_column]),
                fields=fields,
                modality="trx",
            )
        )


def _extend_geo_events(
    client_events: dict[str, list[MbdEvent]],
    path: Path,
    *,
    selected_clients: list[str],
    min_event_ts: datetime | None,
    max_event_ts: datetime | None,
) -> None:
    columns = _read_columns(path)
    client_column = _pick_column(columns, _ID_COLUMNS, context="geo client_id")
    time_column = _pick_column(columns, _TIME_COLUMNS, context="geo event_time")
    rows, _ = _read_filtered_rows(
        path,
        client_column=client_column,
        selected_clients=selected_clients,
        time_column=time_column,
        min_event_ts=min_event_ts,
        max_event_ts=max_event_ts,
    )
    fold_column = _pick_optional_column(columns, _FOLD_COLUMNS)
    ignored = {client_column, time_column}
    if fold_column is not None:
        ignored.add(fold_column)
    for row in rows:
        fields: dict[str, Any] = {"modality": "geo"}
        for key, value in row.items():
            if key in ignored or value is None:
                continue
            coerced = _coerce_event_value(key, value, modality="geo")
            if coerced is not None:
                fields[key] = coerced
        if len(fields) == 1:
            continue
        client_events[str(row[client_column])].append(
            MbdEvent(
                timestamp=_normalize_timestamp(row[time_column]),
                fields=fields,
                modality="geo",
            )
        )


def _extend_dialog_events(
    client_events: dict[str, list[MbdEvent]],
    path: Path,
    *,
    selected_clients: list[str],
    min_event_ts: datetime | None,
    max_event_ts: datetime | None,
    strategy: str,
    projection_dim: int,
    projection_seed: int,
) -> None:
    if strategy.lower() == "skip":
        return
    columns = _read_columns(path)
    client_column = _pick_column(columns, _ID_COLUMNS, context="dialog client_id")
    time_column = _pick_column(columns, _TIME_COLUMNS, context="dialog event_time")
    rows, _ = _read_filtered_rows(
        path,
        client_column=client_column,
        selected_clients=selected_clients,
        time_column=time_column,
        min_event_ts=min_event_ts,
        max_event_ts=max_event_ts,
    )
    fold_column = _pick_optional_column(columns, _FOLD_COLUMNS)
    ignored = {client_column, time_column}
    if fold_column is not None:
        ignored.add(fold_column)
    embedding_column = None
    for candidate in ("embedding", "dialog_embedding", "embeddings"):
        if candidate in columns:
            embedding_column = candidate
            ignored.add(candidate)
            break

    for row in rows:
        fields: dict[str, Any] = {"modality": "dialog"}
        if embedding_column is not None and row.get(embedding_column) is not None:
            fields.update(
                _encode_dialog_embedding(
                    row[embedding_column],
                    strategy=strategy,
                    projection_dim=projection_dim,
                    projection_seed=projection_seed,
                )
            )
        for key, value in row.items():
            if key in ignored or value is None:
                continue
            coerced = _coerce_event_value(key, value, modality="dialog")
            if coerced is not None:
                fields[key] = coerced
        if len(fields) == 1:
            continue
        client_events[str(row[client_column])].append(
            MbdEvent(
                timestamp=_normalize_timestamp(row[time_column]),
                fields=fields,
                modality="dialog",
            )
        )


def _encode_dialog_embedding(
    value: Any,
    *,
    strategy: str,
    projection_dim: int,
    projection_seed: int,
) -> dict[str, float]:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return {}
    mode = strategy.lower()
    fields: dict[str, float] = {
        "dialog_emb_norm": float(np.linalg.norm(vector)),
    }
    if mode == "summary":
        fields["dialog_emb_mean"] = float(vector.mean())
        fields["dialog_emb_std"] = float(vector.std())
        fields["dialog_emb_min"] = float(vector.min())
        fields["dialog_emb_max"] = float(vector.max())
        return fields
    if mode == "project":
        projected = _project_vector(vector, projection_dim=projection_dim, seed=projection_seed)
        for index, item in enumerate(projected):
            fields[f"dialog_proj_{index}"] = float(item)
        return fields
    raise ValueError(f"Unsupported dialog_embedding_strategy={strategy!r}.")


def _project_vector(vector: np.ndarray, *, projection_dim: int, seed: int) -> np.ndarray:
    if projection_dim <= 0:
        raise ValueError("dialog_projection_dim must be positive when using project strategy.")
    rng = np.random.default_rng(seed + int(vector.shape[0]))
    matrix = rng.standard_normal((vector.shape[0], projection_dim), dtype=np.float32)
    matrix = matrix / math.sqrt(float(projection_dim))
    return vector @ matrix


def _coerce_event_value(field: str, value: Any, *, modality: str) -> Any:
    coerced = _coerce_scalar(value)
    if coerced is None:
        return None
    field_key = field.lower()
    if modality == "trx":
        if field_key in _TRANSACTION_CATEGORICAL_FIELDS or field_key.startswith(("src_type", "dst_type")):
            return str(coerced)
        if field_key == "amount":
            return float(coerced)
    if modality == "geo" and field_key.startswith("geohash_"):
        return str(coerced)
    if isinstance(coerced, bool):
        return coerced
    if isinstance(coerced, (int, float)):
        return float(coerced) if isinstance(coerced, float) else int(coerced)
    return str(coerced)


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, datetime):
        return value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_scalar(item) for item in value]
    return str(value)


def _normalize_timestamp(value: datetime | str) -> datetime:
    dt = parse_timestamp(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _select_history(events: list[MbdEvent], evaluation_ts: datetime, history_window_days: int) -> list[MbdEvent]:
    if not events:
        return []
    timestamps = [event.timestamp for event in events]
    end = bisect_left(timestamps, evaluation_ts)
    if history_window_days <= 0:
        return events[:end]
    lower_bound = evaluation_ts - timedelta(days=history_window_days)
    start = bisect_left(timestamps, lower_bound)
    return events[start:end]


def _derive_profile(events: list[MbdEvent], evaluation_ts: datetime) -> dict[str, Any]:
    if not events:
        return {"history_days": 0.0}

    first_ts = events[0].timestamp
    profile: dict[str, Any] = {
        "history_days": round((evaluation_ts - first_ts).total_seconds() / 86400.0, 4),
        "event_count_30d": _count_events_within(events, evaluation_ts, 30),
        "event_count_90d": _count_events_within(events, evaluation_ts, 90),
    }

    tx_events = [event for event in events if event.modality == "trx"]
    geo_events = [event for event in events if event.modality == "geo"]
    dialog_events = [event for event in events if event.modality == "dialog"]

    if tx_events:
        profile["tx_count_30d"] = _count_events_within(tx_events, evaluation_ts, 30)
        profile["tx_count_90d"] = _count_events_within(tx_events, evaluation_ts, 90)
        amounts_30d = _collect_numeric(tx_events, "amount", evaluation_ts, days=30)
        amounts_90d = _collect_numeric(tx_events, "amount", evaluation_ts, days=90)
        profile["tx_amount_sum_30d"] = round(sum(amounts_30d), 4)
        profile["tx_amount_sum_90d"] = round(sum(amounts_90d), 4)
        if amounts_90d:
            profile["tx_amount_mean_90d"] = round(sum(amounts_90d) / len(amounts_90d), 4)
        currencies = [str(event.fields.get("currency")) for event in tx_events if event.fields.get("currency") is not None]
        if currencies:
            profile["dominant_currency"] = Counter(currencies).most_common(1)[0][0]
        profile["last_tx_days"] = round((evaluation_ts - tx_events[-1].timestamp).total_seconds() / 86400.0, 4)

    if geo_events:
        profile["geo_count_30d"] = _count_events_within(geo_events, evaluation_ts, 30)
        recent_geo = [event for event in geo_events if event.timestamp >= evaluation_ts - timedelta(days=30)]
        if recent_geo:
            active_days = {event.timestamp.date().isoformat() for event in recent_geo}
            profile["geo_active_days_30d"] = len(active_days)
        geohash4 = [str(event.fields.get("geohash_4")) for event in geo_events if event.fields.get("geohash_4") is not None]
        if geohash4:
            profile["home_geohash_4"] = Counter(geohash4).most_common(1)[0][0]

    if dialog_events:
        profile["dialog_count_90d"] = _count_events_within(dialog_events, evaluation_ts, 90)
        profile["last_dialog_days"] = round((evaluation_ts - dialog_events[-1].timestamp).total_seconds() / 86400.0, 4)

    return profile


def _derive_lifelong(events: list[MbdEvent]) -> list[LifelongEvent]:
    first_seen: dict[str, datetime] = {}
    for event in events:
        first_seen.setdefault(event.modality, event.timestamp)

    lifelong: list[LifelongEvent] = []
    if "trx" in first_seen:
        lifelong.append(LifelongEvent(key="first_transaction", value=True, timestamp=first_seen["trx"]))
    if "geo" in first_seen:
        lifelong.append(LifelongEvent(key="first_geo_event", value=True, timestamp=first_seen["geo"]))
    if "dialog" in first_seen:
        lifelong.append(LifelongEvent(key="first_dialog", value=True, timestamp=first_seen["dialog"]))
    return lifelong


def _count_events_within(events: list[MbdEvent], evaluation_ts: datetime, days: int) -> int:
    cutoff = evaluation_ts - timedelta(days=days)
    return sum(1 for event in events if event.timestamp >= cutoff)


def _collect_numeric(events: list[MbdEvent], field: str, evaluation_ts: datetime, *, days: int) -> list[float]:
    cutoff = evaluation_ts - timedelta(days=days)
    values: list[float] = []
    for event in events:
        if event.timestamp < cutoff:
            continue
        value = event.fields.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values
