from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    load_yaml_config,
    masking_config_from_dict,
    text_encoder_config_from_dict,
    tokenizer_config_from_dict,
)
from data import (  # noqa: E402
    PragmaTokenizer,
    build_sharded_store,
    generate_synthetic_records,
    load_user_records,
    split_records,
)


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a sharded PRAGMA pretraining store.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def maybe_copy_config(config_path: Path, output_dir: Path) -> None:
    shutil.copyfile(config_path, output_dir / "config.yaml")


def load_records(config: dict) -> list:
    data_config = config.get("data", {})
    source = data_config.get("source", "synthetic")
    if source == "json":
        records_json = data_config.get("records_json")
        if not records_json:
            raise ValueError("data.records_json must be set when data.source=json.")
        return load_user_records(resolve_path(records_json))
    if source != "synthetic":
        raise ValueError("build_store.py supports synthetic or json sources.")
    return generate_synthetic_records(
        int(data_config.get("num_records", 256)),
        seed=int(data_config.get("synthetic_seed", 0)),
        min_events=int(data_config.get("min_events", 16)),
        max_events=int(data_config.get("max_events", 72)),
    )


def main() -> None:
    args = make_argument_parser().parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml_config(config_path)
    pretrain = config.get("pretrain", {})

    records = load_records(config)
    train_records, _ = split_records(records, train_fraction=float(config.get("data", {}).get("train_fraction", 0.8)))
    tokenizer = PragmaTokenizer(
        tokenizer_config_from_dict(config.get("tokenizer", {})),
        masking_config_from_dict(config.get("masking", {})),
        text_encoder_config_from_dict(config.get("text_encoder", {})),
    )
    tokenizer.fit(train_records)

    output_dir = resolve_path(pretrain.get("sharded_store_dir", "artifacts/pretrain_store"))
    output_dir.mkdir(parents=True, exist_ok=True)
    build_sharded_store(
        train_records,
        tokenizer,
        output_dir,
        compression=str(pretrain.get("parquet_compression", "zstd")),
    )
    tokenizer.save(output_dir / "tokenizer.json")
    maybe_copy_config(config_path, output_dir)
    print(f"built_store={output_dir}")


if __name__ == "__main__":
    main()
