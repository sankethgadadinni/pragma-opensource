from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_yaml_config, make_model_config  # noqa: E402
from data import (  # noqa: E402
    PragmaTokenizer,
    ShardedRecordStore,
    generate_synthetic_records,
    load_user_records,
    save_json,
)
from modeling import PragmaBackbone, PragmaClassifier  # noqa: E402
from modeling.lora import LoRAConfig, inject_lora  # noqa: E402


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRAGMA inference entrypoint.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def load_records(config: dict):
    inference = config.get("inference", {})
    data_source = str(inference.get("source", config.get("data", {}).get("source", "synthetic")))
    if data_source == "json":
        input_json = inference.get("input_json")
        if not input_json:
            raise ValueError("inference.input_json must be set when inference.source=json.")
        return load_user_records(resolve_path(input_json))
    if data_source == "sharded":
        sharded_store_dir = inference.get("sharded_store_dir", config.get("data", {}).get("sharded_store_dir"))
        if not sharded_store_dir:
            raise ValueError("A sharded store directory is required for inference.source=sharded.")
        return ShardedRecordStore(resolve_path(sharded_store_dir)).load_all_records()
    return generate_synthetic_records(
        int(inference.get("num_records", 16)),
        seed=int(inference.get("synthetic_seed", 7)),
    )


def main() -> None:
    args = make_argument_parser().parse_args()
    config = load_yaml_config(resolve_path(args.config))
    inference = config.get("inference", {})
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))

    tokenizer = PragmaTokenizer.load(resolve_path(inference.get("tokenizer_path", "artifacts/finetune/tokenizer.json")))
    checkpoint = torch.load(
        resolve_path(inference.get("checkpoint_path", "artifacts/finetune/classifier.pt")),
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
    inject_lora(
        backbone,
        LoRAConfig(
            rank=int(checkpoint.get("lora_rank", config.get("lora", {}).get("rank", 8))),
            alpha=int(checkpoint.get("lora_alpha", config.get("lora", {}).get("alpha", 8))),
            dropout=float(checkpoint.get("lora_dropout", config.get("lora", {}).get("dropout", 0.0))),
        ),
    )
    classifier = PragmaClassifier(
        backbone,
        num_outputs=int(checkpoint.get("num_outputs", 1)),
        pooling=str(checkpoint.get("pooling", "usr_last")),
        dropout=float(checkpoint.get("dropout", config.get("model", {}).get("dropout", 0.1))),
    ).to(device)
    classifier.load_state_dict(checkpoint["state_dict"], strict=False)
    classifier.eval()

    records = load_records(config)
    num_examples = int(inference.get("num_examples", 5))
    threshold = float(inference.get("threshold", 0.5))
    task_type = str(inference.get("task_type", checkpoint.get("task_type", "binary"))).lower()
    batch = tokenizer.collate(records[:num_examples], apply_mlm=False, device=device)

    with torch.no_grad():
        logits = classifier(batch)
        if task_type in {"binary", "ranking"}:
            scores = torch.sigmoid(logits).detach().cpu()
        else:
            scores = logits.detach().cpu()

    predictions = []
    if task_type == "binary":
        for record, score in zip(records[:num_examples], scores.tolist()):
            predictions.append(
                {
                    "user_id": record.user_id,
                    "score": float(score),
                    "prediction": int(score >= threshold),
                    "label": record.label,
                }
            )
    elif task_type == "regression":
        for record, value in zip(records[:num_examples], scores.tolist()):
            predictions.append(
                {
                    "user_id": record.user_id,
                    "prediction": float(value),
                    "label": record.label,
                }
            )
    elif task_type == "multiclass":
        for record, values in zip(records[:num_examples], scores.tolist()):
            predicted_class = int(max(range(len(values)), key=lambda idx: values[idx]))
            predictions.append(
                {
                    "user_id": record.user_id,
                    "logits": values,
                    "prediction": predicted_class,
                    "label": record.label,
                }
            )
    else:
        for record, values in zip(records[:num_examples], scores.tolist()):
            predictions.append(
                {
                    "user_id": record.user_id,
                    "scores": values,
                    "label": record.label,
                }
            )

    output_path = resolve_path(inference.get("prediction_output", "artifacts/inference/predictions.json"))
    save_json(output_path, predictions)
    print(f"device={device} predictions={len(predictions)} saved={output_path}")
    for item in predictions:
        print(item)


if __name__ == "__main__":
    main()
