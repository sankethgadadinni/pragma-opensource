from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    load_yaml_config,
    make_model_config,
    masking_config_from_dict,
    tokenizer_config_from_dict,
)
from data import (  # noqa: E402
    PragmaTokenizer,
    generate_synthetic_records,
    load_user_records,
    split_records,
)
from modeling import PragmaBackbone, PragmaClassifier  # noqa: E402
from modeling.lora import (  # noqa: E402
    LoRAConfig,
    freeze_non_lora_parameters,
    inject_lora,
    lora_parameter_count,
)


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRAGMA training entrypoint.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--task", choices=["pretrain", "finetune"], required=True)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def ensure_output_dir(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        raise ValueError(f"Unsupported data.source={source!r}.")
    return generate_synthetic_records(
        int(data_config.get("num_records", 256)),
        seed=int(data_config.get("synthetic_seed", 0)),
        min_events=int(data_config.get("min_events", 16)),
        max_events=int(data_config.get("max_events", 72)),
    )


def split_train_val(records, config: dict):
    train_fraction = float(config.get("data", {}).get("train_fraction", 0.8))
    return split_records(records, train_fraction=train_fraction)


def build_tokenizer(config: dict, train_records) -> PragmaTokenizer:
    tokenizer = PragmaTokenizer(
        tokenizer_config_from_dict(config.get("tokenizer", {})),
        masking_config_from_dict(config.get("masking", {})),
    )
    tokenizer.fit(train_records)
    return tokenizer


def run_pretrain(config: dict, config_path: Path) -> None:
    runtime = config.get("runtime", {})
    seed = int(runtime.get("seed", 0))
    torch.manual_seed(seed)
    sampler = random.Random(seed)
    device = resolve_device(str(runtime.get("device", "auto")))

    records = load_records(config)
    train_records, _ = split_train_val(records, config)
    tokenizer = build_tokenizer(config, train_records)

    model = PragmaBackbone(
        make_model_config(
            str(config.get("model", {}).get("variant", "S")),
            tokenizer.vocab_size,
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
            label_smoothing=float(tokenizer.masking_config.label_smoothing),
            max_event_tokens=int(tokenizer.config.max_event_tokens),
        )
    ).to(device)

    pretrain = config.get("pretrain", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(pretrain.get("learning_rate", 3e-4)),
    )
    steps = int(pretrain.get("steps", 20))
    batch_size = int(pretrain.get("batch_size", 8))

    print(f"task=pretrain device={device} vocab_size={tokenizer.vocab_size}")
    for step in range(1, steps + 1):
        batch_records = sampler.sample(train_records, k=min(batch_size, len(train_records)))
        batch = tokenizer.collate(batch_records, apply_mlm=True, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_pretrain(batch)
        output.loss.backward()
        optimizer.step()
        print(
            f"step={step:03d} loss={output.loss.item():.4f} "
            f"masked_tokens={int(output.masked_targets.numel())}"
        )

    output_dir = ensure_output_dir(pretrain.get("output_dir", "artifacts/pretrain"))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": str(config.get("model", {}).get("variant", "S")),
            "dropout": float(config.get("model", {}).get("dropout", 0.1)),
            "label_smoothing": float(tokenizer.masking_config.label_smoothing),
            "max_event_tokens": int(tokenizer.config.max_event_tokens),
        },
        output_dir / "backbone.pt",
    )
    tokenizer.save(output_dir / "tokenizer.json")
    maybe_copy_config(config_path, output_dir)
    print(f"saved={output_dir}")


def evaluate_classifier(classifier: PragmaClassifier, tokenizer: PragmaTokenizer, records, device) -> tuple[float, float]:
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for start in range(0, len(records), 16):
            batch_records = records[start : start + 16]
            batch = tokenizer.collate(batch_records, apply_mlm=False, device=device)
            if batch.downstream_labels is None:
                continue
            logits = classifier(batch)
            labels = batch.downstream_labels
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            predictions = (torch.sigmoid(logits) > 0.5).long()
            total_loss += loss.item() * len(batch_records)
            total_correct += int((predictions == labels.long()).sum().item())
            total_count += len(batch_records)
    classifier.train()
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def run_finetune(config: dict, config_path: Path) -> None:
    runtime = config.get("runtime", {})
    seed = int(runtime.get("seed", 0))
    torch.manual_seed(seed)
    sampler = random.Random(seed)
    device = resolve_device(str(runtime.get("device", "auto")))

    records = load_records(config)
    train_records, val_records = split_train_val(records, config)

    finetune = config.get("finetune", {})
    tokenizer_path = finetune.get("tokenizer_path")
    tokenizer_file = resolve_path(tokenizer_path) if tokenizer_path else None
    if tokenizer_file is not None and tokenizer_file.exists():
        tokenizer = PragmaTokenizer.load(tokenizer_file)
    else:
        tokenizer = build_tokenizer(config, train_records)

    backbone = PragmaBackbone(
        make_model_config(
            str(config.get("model", {}).get("variant", "S")),
            tokenizer.vocab_size,
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
            label_smoothing=float(tokenizer.masking_config.label_smoothing),
            max_event_tokens=int(tokenizer.config.max_event_tokens),
        )
    ).to(device)

    pretrained_path = finetune.get("pretrained_backbone_path")
    if pretrained_path:
        pretrained_file = resolve_path(pretrained_path)
        if not pretrained_file.exists():
            raise FileNotFoundError(f"Pretrained backbone not found: {pretrained_file}")
        checkpoint = torch.load(pretrained_file, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        backbone.load_state_dict(state_dict)

    lora_config = config.get("lora", {})
    inject_lora(
        backbone,
        LoRAConfig(
            rank=int(lora_config.get("rank", 8)),
            alpha=int(lora_config.get("alpha", 8)),
            dropout=float(lora_config.get("dropout", 0.0)),
        ),
    )
    freeze_non_lora_parameters(backbone)

    classifier = PragmaClassifier(
        backbone,
        num_outputs=int(finetune.get("num_outputs", 1)),
        pooling=str(finetune.get("pooling", "usr_last")),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    for parameter in classifier.head.parameters():
        parameter.requires_grad = True

    trainable, total = lora_parameter_count(classifier)
    print(f"task=finetune device={device} trainable={trainable} total={total}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=float(finetune.get("learning_rate", 2e-4)),
    )
    steps = int(finetune.get("steps", 20))
    batch_size = int(finetune.get("batch_size", 8))

    for step in range(1, steps + 1):
        batch_records = sampler.sample(train_records, k=min(batch_size, len(train_records)))
        batch = tokenizer.collate(batch_records, apply_mlm=False, device=device)
        if batch.downstream_labels is None:
            raise RuntimeError("Training batch is missing downstream labels.")
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch.downstream_labels)
        loss.backward()
        optimizer.step()
        print(f"step={step:03d} train_loss={loss.item():.4f}")

    val_loss, val_accuracy = evaluate_classifier(classifier, tokenizer, val_records, device)
    print(f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")

    output_dir = ensure_output_dir(finetune.get("output_dir", "artifacts/finetune"))
    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "variant": str(config.get("model", {}).get("variant", "S")),
            "dropout": float(config.get("model", {}).get("dropout", 0.1)),
            "pooling": str(finetune.get("pooling", "usr_last")),
            "num_outputs": int(finetune.get("num_outputs", 1)),
            "max_event_tokens": int(tokenizer.config.max_event_tokens),
            "label_smoothing": float(tokenizer.masking_config.label_smoothing),
            "lora_rank": int(lora_config.get("rank", 8)),
            "lora_alpha": int(lora_config.get("alpha", 8)),
            "lora_dropout": float(lora_config.get("dropout", 0.0)),
        },
        output_dir / "classifier.pt",
    )
    tokenizer.save(output_dir / "tokenizer.json")
    maybe_copy_config(config_path, output_dir)
    print(f"saved={output_dir}")


def main() -> None:
    args = make_argument_parser().parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml_config(config_path)
    if args.task == "pretrain":
        run_pretrain(config, config_path)
    else:
        run_finetune(config, config_path)


if __name__ == "__main__":
    main()
