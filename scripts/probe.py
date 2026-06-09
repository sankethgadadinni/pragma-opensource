from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_yaml_config, make_model_config  # noqa: E402
from data import PragmaTokenizer, ShardedRecordStore, generate_synthetic_records, load_user_records, split_records  # noqa: E402
from modeling import PragmaBackbone  # noqa: E402
from tasks import (  # noqa: E402
    StandardScaler,
    compute_embeddings,
    evaluate_probe,
    evaluate_t_learner,
    fit_lbfgs_probe,
    fit_t_learner_probe,
    infer_num_outputs,
    tensorize_targets,
    tensorize_uplift_targets,
)


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRAGMA embedding probe entrypoint.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_records(config: dict) -> list:
    probe = config.get("probe", {})
    data_source = str(probe.get("source", config.get("data", {}).get("source", "synthetic")))
    if data_source == "json":
        input_json = probe.get("input_json", config.get("data", {}).get("records_json"))
        if not input_json:
            raise ValueError("probe.input_json or data.records_json must be set for source=json.")
        return load_user_records(resolve_path(input_json))
    if data_source == "sharded":
        sharded_store_dir = probe.get("sharded_store_dir", config.get("data", {}).get("sharded_store_dir"))
        if not sharded_store_dir:
            raise ValueError("A sharded store directory is required for source=sharded.")
        return ShardedRecordStore(resolve_path(sharded_store_dir)).load_all_records()
    return generate_synthetic_records(
        int(probe.get("num_records", config.get("data", {}).get("num_records", 256))),
        seed=int(probe.get("synthetic_seed", config.get("data", {}).get("synthetic_seed", 0))),
        min_events=int(config.get("data", {}).get("min_events", 16)),
        max_events=int(config.get("data", {}).get("max_events", 72)),
    )


def main() -> None:
    args = make_argument_parser().parse_args()
    config = load_yaml_config(resolve_path(args.config))
    probe_config = config.get("probe", {})
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))

    tokenizer = PragmaTokenizer.load(resolve_path(probe_config.get("tokenizer_path", "artifacts/pretrain/tokenizer.json")))
    checkpoint = torch.load(
        resolve_path(probe_config.get("backbone_path", "artifacts/pretrain/backbone.pt")),
        map_location=device,
        weights_only=False,
    )
    backbone = PragmaBackbone(
        make_model_config(
            str(checkpoint.get("variant", config.get("model", {}).get("variant", "S"))),
            tokenizer.vocab_size,
            dropout=float(checkpoint.get("dropout", config.get("model", {}).get("dropout", 0.1))),
            label_smoothing=float(checkpoint.get("label_smoothing", tokenizer.masking_config.label_smoothing)),
            max_event_tokens=int(checkpoint.get("max_event_tokens", tokenizer.config.max_event_tokens)),
            text_encoder_dim=int(checkpoint.get("text_encoder_dim", tokenizer.text_embedding_dim)),
            text_loss_weight=float(checkpoint.get("text_loss_weight", config.get("text_encoder", {}).get("text_loss_weight", 1.0))),
        )
    ).to(device)
    backbone.load_state_dict(checkpoint["state_dict"], strict=False)
    backbone.eval()

    records = load_records(config)
    train_records, val_records = split_records(records, train_fraction=float(config.get("data", {}).get("train_fraction", 0.8)))
    pooling = str(probe_config.get("pooling", "usr_last"))
    batch_size = int(probe_config.get("batch_size", 32))
    task_type = str(probe_config.get("task_type", "binary")).lower()

    train_embeddings = compute_embeddings(backbone, tokenizer, train_records, device=device, pooling=pooling, batch_size=batch_size)
    val_embeddings = compute_embeddings(backbone, tokenizer, val_records, device=device, pooling=pooling, batch_size=batch_size)

    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_embeddings)
    val_inputs = scaler.transform(val_embeddings)
    result_payload: dict[str, object] = {
        "task_type": task_type,
        "pooling": pooling,
    }

    if task_type == "uplift":
        treatment_train, outcome_train, _ = tensorize_uplift_targets(train_records, train_inputs.device)
        treatment_val, outcome_val, propensity_val = tensorize_uplift_targets(val_records, val_inputs.device)
        probe = fit_t_learner_probe(
            train_inputs,
            treatment_train,
            outcome_train,
            max_iter=int(probe_config.get("max_iter", 128)),
        )
        result_payload["val_metrics"] = evaluate_t_learner(
            probe,
            val_inputs,
            treatment_val,
            outcome_val,
            propensity_val,
        )
    else:
        targets_train = tensorize_targets(train_records, task_type, train_inputs.device)
        targets_val = tensorize_targets(val_records, task_type, val_inputs.device)
        num_outputs = infer_num_outputs(
            train_records,
            task_type,
            None if probe_config.get("num_outputs") in {None, 0} else int(probe_config.get("num_outputs")),
        )
        probe = fit_lbfgs_probe(
            train_inputs,
            targets_train,
            task_type=task_type,
            num_outputs=num_outputs,
            max_iter=int(probe_config.get("max_iter", 128)),
        )
        result_payload["val_metrics"] = evaluate_probe(
            probe,
            val_inputs,
            targets_val,
            task_type=task_type,
        )

    output_path = resolve_path(probe_config.get("output_path", "artifacts/probe/results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
