from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TokenizerConfig
from data import PragmaTokenizer, load_mbd_records


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        write_parquet(
            root / "targets.parquet",
            [
                {
                    "client_id": "client_a",
                    "report_dt": "2024-03-31T00:00:00",
                    "fold": 0,
                    "product_1": 1,
                    "product_2": 0,
                },
                {
                    "client_id": "client_b",
                    "report_dt": "2024-03-31T00:00:00",
                    "fold": 1,
                    "product_1": 0,
                    "product_2": 1,
                },
            ],
        )
        write_parquet(
            root / "detail" / "trx.parquet",
            [
                {
                    "client_id": "client_a",
                    "event_time": "2024-01-05T08:00:00",
                    "amount": 100.0,
                    "event_type": 1,
                    "event_subtype": 2,
                    "currency": 840,
                    "src_type11": 7,
                    "dst_type11": 2,
                },
                {
                    "client_id": "client_a",
                    "event_time": "2024-03-20T13:00:00",
                    "amount": 25.0,
                    "event_type": 1,
                    "event_subtype": 3,
                    "currency": 840,
                    "src_type11": 7,
                    "dst_type11": 5,
                },
                {
                    "client_id": "client_a",
                    "event_time": "2024-04-05T09:00:00",
                    "amount": 999.0,
                    "event_type": 9,
                    "event_subtype": 9,
                    "currency": 840,
                    "src_type11": 1,
                    "dst_type11": 1,
                },
                {
                    "client_id": "client_b",
                    "event_time": "2024-03-10T10:00:00",
                    "amount": 80.0,
                    "event_type": 4,
                    "event_subtype": 1,
                    "currency": 978,
                    "src_type11": 3,
                    "dst_type11": 6,
                },
            ],
        )
        write_parquet(
            root / "detail" / "geo.parquet",
            [
                {
                    "client_id": "client_a",
                    "event_time": "2024-03-10T17:30:00",
                    "geohash_4": 1234,
                    "geohash_5": 12345,
                    "geohash_6": 123456,
                },
                {
                    "client_id": "client_b",
                    "event_time": "2024-03-05T07:15:00",
                    "geohash_4": 4321,
                    "geohash_5": 43210,
                    "geohash_6": 432109,
                },
            ],
        )
        write_parquet(
            root / "detail" / "dialogs.parquet",
            [
                {
                    "client_id": "client_a",
                    "event_time": "2024-03-25T09:30:00",
                    "embedding": [0.2, -0.1, 0.4, 0.3],
                },
                {
                    "client_id": "client_a",
                    "event_time": "2024-04-02T09:30:00",
                    "embedding": [0.9, 0.8, 0.7, 0.6],
                },
            ],
        )

        records = load_mbd_records(
            {
                "root_dir": root,
                "allowed_folds": [0],
                "label_fields": ["product_1", "product_2"],
                "label_mode": "vector",
                "history_window_days": 365,
                "dialog_embedding_strategy": "project",
                "dialog_projection_dim": 4,
                "dialog_seed": 11,
            }
        )

        if len(records) != 1:
            raise RuntimeError(f"Expected one filtered record, found {len(records)}.")

        record = records[0]
        if record.user_id != "client_a":
            raise RuntimeError(f"Unexpected client id: {record.user_id!r}")
        if record.label != [1, 0]:
            raise RuntimeError(f"Unexpected label payload: {record.label!r}")
        if len(record.events) != 4:
            raise RuntimeError(f"Expected four pre-cutoff events, found {len(record.events)}.")

        profile = record.profile
        expected_profile = {
            "tx_count_30d": 1,
            "tx_count_90d": 2,
            "tx_amount_sum_30d": 25.0,
            "tx_amount_sum_90d": 125.0,
            "dominant_currency": "840",
            "geo_count_30d": 1,
            "dialog_count_90d": 1,
            "home_geohash_4": "1234",
        }
        for key, expected_value in expected_profile.items():
            if profile.get(key) != expected_value:
                raise RuntimeError(f"Profile field {key!r} expected {expected_value!r}, found {profile.get(key)!r}.")

        lifelong_keys = {item.key for item in record.lifelong}
        if lifelong_keys != {"first_transaction", "first_geo_event", "first_dialog"}:
            raise RuntimeError(f"Unexpected lifelong keys: {lifelong_keys!r}")

        dialog_events = [event for event in record.events if event.fields.get("modality") == "dialog"]
        if len(dialog_events) != 1 or "dialog_proj_0" not in dialog_events[0].fields:
            raise RuntimeError("Dialog projection fields were not generated correctly.")

        tokenizer = PragmaTokenizer(TokenizerConfig(max_event_tokens=32, max_profile_tokens=64, max_events=64))
        tokenizer.fit(records)
        batch = tokenizer.collate(records, apply_mlm=True)
        if batch.downstream_labels is None or tuple(batch.downstream_labels.shape) != (1, 2):
            raise RuntimeError("Expected the MBD vector target to collate into a 1x2 downstream label tensor.")
        if int(batch.event_mask.sum().item()) != 4:
            raise RuntimeError("Event mask did not preserve all pre-cutoff events.")

        print(
            "mbd_loader_test_ok",
            {
                "records": len(records),
                "events": len(record.events),
                "lifelong": sorted(lifelong_keys),
                "profile_keys": sorted(expected_profile),
                "vocab_size": tokenizer.vocab_size,
            },
        )


if __name__ == "__main__":
    main()
