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
from data import PragmaTokenizer, generate_synthetic_records, load_user_records, save_json  # noqa: E402
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
    input_json = inference.get("input_json")
    if input_json:
        return load_user_records(resolve_path(input_json))
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
    checkpoint = torch.load(resolve_path(inference.get("checkpoint_path", "artifacts/finetune/classifier.pt")), map_location=device)

    backbone = PragmaBackbone(
        make_model_config(
            str(checkpoint.get("variant", config.get("model", {}).get("variant", "S"))),
            tokenizer.vocab_size,
            dropout=float(checkpoint.get("dropout", config.get("model", {}).get("dropout", 0.1))),
            label_smoothing=float(checkpoint.get("label_smoothing", tokenizer.masking_config.label_smoothing)),
            max_event_tokens=int(checkpoint.get("max_event_tokens", tokenizer.config.max_event_tokens)),
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
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()

    records = load_records(config)
    num_examples = int(inference.get("num_examples", 5))
    threshold = float(inference.get("threshold", 0.5))
    batch = tokenizer.collate(records[:num_examples], apply_mlm=False, device=device)

    with torch.no_grad():
        logits = classifier(batch)
        if logits.ndim == 1:
            scores = torch.sigmoid(logits).tolist()
        else:
            scores = torch.sigmoid(logits).tolist()

    predictions = []
    if scores and isinstance(scores[0], list):
        for record, score in zip(records[:num_examples], scores):
            predictions.append(
                {
                    "user_id": record.user_id,
                    "scores": score,
                    "label": record.label,
                }
            )
    else:
        for record, score in zip(records[:num_examples], scores):
            predictions.append(
                {
                    "user_id": record.user_id,
                    "score": float(score),
                    "prediction": int(score >= threshold),
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
