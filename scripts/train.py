from __future__ import annotations

import argparse
import random
import shutil
import sys
from contextlib import nullcontext
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
    text_encoder_config_from_dict,
    tokenizer_config_from_dict,
)
from data import (  # noqa: E402
    PragmaTokenizer,
    ShardedRecordStore,
    build_sharded_store,
    generate_synthetic_records,
    load_user_records,
    split_records,
)
from modeling import PragmaBackbone, PragmaClassifier, build_muon_adamw_optimizer  # noqa: E402
from modeling.lora import (  # noqa: E402
    LoRAConfig,
    freeze_non_lora_parameters,
    inject_lora,
    lora_parameter_count,
)
from tasks import (  # noqa: E402
    binary_classification_metrics,
    infer_num_outputs,
    ranking_metrics,
    regression_metrics,
    tensorize_targets,
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
    if source == "sharded":
        sharded_store_dir = data_config.get("sharded_store_dir")
        if not sharded_store_dir:
            raise ValueError("data.sharded_store_dir must be set when data.source=sharded.")
        return ShardedRecordStore(resolve_path(sharded_store_dir)).load_all_records()
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
        text_encoder_config_from_dict(config.get("text_encoder", {})),
    )
    tokenizer.fit(train_records)
    return tokenizer


def autocast_context(device: torch.device, precision: str):
    precision_key = precision.lower()
    if precision_key == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def task_loss(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> torch.Tensor:
    task_key = task_type.lower()
    if task_key == "binary":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if task_key == "regression":
        return F.mse_loss(logits, targets)
    if task_key == "ranking":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if task_key == "multiclass":
        return F.cross_entropy(logits, targets)
    raise ValueError(f"Unsupported downstream task type: {task_type!r}")


def task_metrics(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> dict[str, float]:
    task_key = task_type.lower()
    if task_key == "binary":
        return binary_classification_metrics(logits, targets)
    if task_key == "regression":
        return regression_metrics(logits, targets)
    if task_key == "ranking":
        return ranking_metrics(logits, targets)
    if task_key == "multiclass":
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).to(torch.float32).mean().item()
        return {"accuracy": float(accuracy)}
    raise ValueError(f"Unsupported downstream task type: {task_type!r}")


def pretrain_batch_iterator(records, tokenizer, config: dict, seed: int):
    pretrain = config.get("pretrain", {})
    use_sharded_store = bool(pretrain.get("use_sharded_store", False))
    if not use_sharded_store:
        batch_size = int(pretrain.get("batch_size", 8))
        sampler = random.Random(seed)
        while True:
            yield sampler.sample(records, k=min(batch_size, len(records)))
        return

    store_dir = resolve_path(pretrain.get("sharded_store_dir", "artifacts/pretrain_store"))
    if config.get("data", {}).get("source") == "sharded":
        store = ShardedRecordStore(resolve_path(config["data"]["sharded_store_dir"]))
    else:
        if bool(pretrain.get("rebuild_sharded_store", False)) or not (store_dir / "manifest.json").exists():
            build_sharded_store(
                records,
                tokenizer,
                store_dir,
                compression=str(pretrain.get("parquet_compression", "zstd")),
            )
        store = ShardedRecordStore(store_dir)

    token_budget = int(pretrain.get("token_budget", 8192))
    local_seed = seed
    while True:
        yielded = False
        for batch_records in store.iter_dynamic_batches(
            token_budget=token_budget,
            shuffle=True,
            seed=local_seed,
        ):
            yielded = True
            yield batch_records
        if not yielded:
            raise RuntimeError("The sharded pretraining store did not yield any batches.")
        local_seed += 1


def evaluate_classifier(
    classifier: PragmaClassifier,
    tokenizer: PragmaTokenizer,
    records,
    device: torch.device,
    *,
    task_type: str,
    precision: str,
) -> tuple[float, dict[str, float]]:
    classifier.eval()
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            batch_records = records[start : start + 16]
            batch = tokenizer.collate(batch_records, apply_mlm=False, device=device)
            targets = tensorize_targets(batch_records, task_type, device)
            with autocast_context(device, precision):
                logits = classifier(batch)
                loss = task_loss(logits, targets, task_type)
            logits_parts.append(logits.detach().cpu().to(torch.float32))
            target_parts.append(targets.detach().cpu())
            losses.append(float(loss.item()))
    classifier.train()
    if not logits_parts:
        return 0.0, {}
    logits = torch.cat(logits_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    metrics = task_metrics(logits, targets, task_type)
    return float(sum(losses) / len(losses)), metrics


def run_pretrain(config: dict, config_path: Path) -> None:
    runtime = config.get("runtime", {})
    pretrain = config.get("pretrain", {})
    seed = int(runtime.get("seed", 0))
    torch.manual_seed(seed)
    device = resolve_device(str(runtime.get("device", "auto")))
    precision = str(pretrain.get("precision", "bf16"))

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
            text_encoder_dim=tokenizer.text_embedding_dim,
            text_loss_weight=float(config.get("text_encoder", {}).get("text_loss_weight", 1.0)),
        )
    ).to(device)

    optimizer_name = str(pretrain.get("optimizer", "muon_adamw")).lower()
    learning_rate = float(pretrain.get("learning_rate", 3e-4))
    if optimizer_name == "muon_adamw":
        optimizer = build_muon_adamw_optimizer(
            model,
            lr=learning_rate,
            adamw_lr=float(pretrain.get("adamw_learning_rate", learning_rate)),
            momentum=float(pretrain.get("muon_momentum", 0.95)),
            weight_decay=float(pretrain.get("weight_decay", 0.01)),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(pretrain.get("weight_decay", 0.01)),
        )
    steps = int(pretrain.get("steps", 20))
    batch_iter = pretrain_batch_iterator(train_records, tokenizer, config, seed)
    use_sequence_packing = bool(pretrain.get("use_sequence_packing", True))

    print(
        f"task=pretrain device={device} vocab_size={tokenizer.vocab_size} "
        f"precision={precision} optimizer={optimizer_name}"
    )
    for step in range(1, steps + 1):
        batch_records = next(batch_iter)
        batch = tokenizer.collate(
            batch_records,
            apply_mlm=True,
            device=device,
            pack_events=use_sequence_packing,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            output = model.forward_pretrain(batch)
        output.loss.backward()
        optimizer.step()
        print(
            f"step={step:03d} loss={output.loss.item():.4f} "
            f"token_loss={float(output.token_loss.item()):.4f} "
            f"text_loss={float(output.text_loss.item()):.4f} "
            f"batch_size={len(batch_records)}"
        )

    output_dir = ensure_output_dir(pretrain.get("output_dir", "artifacts/pretrain"))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": str(config.get("model", {}).get("variant", "S")),
            "dropout": float(config.get("model", {}).get("dropout", 0.1)),
            "label_smoothing": float(tokenizer.masking_config.label_smoothing),
            "max_event_tokens": int(tokenizer.config.max_event_tokens),
            "text_encoder_dim": int(tokenizer.text_embedding_dim),
            "text_loss_weight": float(config.get("text_encoder", {}).get("text_loss_weight", 1.0)),
        },
        output_dir / "backbone.pt",
    )
    tokenizer.save(output_dir / "tokenizer.json")
    maybe_copy_config(config_path, output_dir)
    print(f"saved={output_dir}")


def run_finetune(config: dict, config_path: Path) -> None:
    runtime = config.get("runtime", {})
    finetune = config.get("finetune", {})
    seed = int(runtime.get("seed", 0))
    torch.manual_seed(seed)
    sampler = random.Random(seed)
    device = resolve_device(str(runtime.get("device", "auto")))
    precision = str(finetune.get("precision", "fp32"))
    task_type = str(finetune.get("task_type", "binary")).lower()
    if task_type == "uplift":
        raise ValueError("Use scripts/probe.py for uplift evaluation on frozen embeddings.")

    records = load_records(config)
    train_records, val_records = split_train_val(records, config)

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
            text_encoder_dim=tokenizer.text_embedding_dim,
            text_loss_weight=float(config.get("text_encoder", {}).get("text_loss_weight", 1.0)),
        )
    ).to(device)

    pretrained_path = finetune.get("pretrained_backbone_path")
    if pretrained_path:
        pretrained_file = resolve_path(pretrained_path)
        if not pretrained_file.exists():
            raise FileNotFoundError(f"Pretrained backbone not found: {pretrained_file}")
        checkpoint = torch.load(pretrained_file, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        backbone.load_state_dict(state_dict, strict=False)

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

    num_outputs_config = finetune.get("num_outputs")
    num_outputs = infer_num_outputs(
        train_records,
        task_type,
        None if num_outputs_config in {None, 0} else int(num_outputs_config),
    )
    classifier = PragmaClassifier(
        backbone,
        num_outputs=num_outputs,
        pooling=str(finetune.get("pooling", "usr_last")),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    for parameter in classifier.head.parameters():
        parameter.requires_grad = True

    trainable, total = lora_parameter_count(classifier)
    print(
        f"task=finetune device={device} precision={precision} "
        f"task_type={task_type} trainable={trainable} total={total}"
    )

    optimizer = torch.optim.Adam(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=float(finetune.get("learning_rate", 2e-4)),
    )
    steps = int(finetune.get("steps", 20))
    batch_size = int(finetune.get("batch_size", 8))

    for step in range(1, steps + 1):
        batch_records = sampler.sample(train_records, k=min(batch_size, len(train_records)))
        batch = tokenizer.collate(batch_records, apply_mlm=False, device=device)
        targets = tensorize_targets(batch_records, task_type, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            logits = classifier(batch)
            loss = task_loss(logits, targets, task_type)
        loss.backward()
        optimizer.step()
        print(f"step={step:03d} train_loss={loss.item():.4f}")

    val_loss, metrics = evaluate_classifier(
        classifier,
        tokenizer,
        val_records,
        device,
        task_type=task_type,
        precision=precision,
    )
    metrics_display = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
    print(f"val_loss={val_loss:.4f} {metrics_display}".strip())

    output_dir = ensure_output_dir(finetune.get("output_dir", "artifacts/finetune"))
    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "variant": str(config.get("model", {}).get("variant", "S")),
            "dropout": float(config.get("model", {}).get("dropout", 0.1)),
            "pooling": str(finetune.get("pooling", "usr_last")),
            "num_outputs": int(num_outputs),
            "task_type": task_type,
            "max_event_tokens": int(tokenizer.config.max_event_tokens),
            "label_smoothing": float(tokenizer.masking_config.label_smoothing),
            "lora_rank": int(lora_config.get("rank", 8)),
            "lora_alpha": int(lora_config.get("alpha", 8)),
            "lora_dropout": float(lora_config.get("dropout", 0.0)),
            "text_encoder_dim": int(tokenizer.text_embedding_dim),
            "text_loss_weight": float(config.get("text_encoder", {}).get("text_loss_weight", 1.0)),
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
