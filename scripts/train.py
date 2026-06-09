from __future__ import annotations

import argparse
import random
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    TokenizedRecord,
    build_sharded_store,
    generate_synthetic_records,
    load_user_records,
    split_records,
    validate_frozen_text_encoder,
)
from modeling import PragmaBackbone, PragmaClassifier, build_muon_adamw_optimizer  # noqa: E402
from modeling.lora import (  # noqa: E402
    LoRAConfig,
    freeze_non_lora_parameters,
    inject_lora,
    lora_parameter_count,
)
from runtime import (  # noqa: E402
    DistributedContext,
    barrier,
    cpu_state_dict,
    finalize_runtime,
    initialize_runtime,
    is_main_process,
    load_training_checkpoint,
    make_cpu_generator,
    resolve_resume_checkpoint,
    save_training_checkpoint,
    seed_everything,
    seed_step_generators,
    unwrap_module,
    wrap_ddp,
)
from tasks import (  # noqa: E402
    binary_classification_metrics,
    infer_num_outputs,
    ranking_metrics,
    regression_metrics,
    tensorize_targets,
)


@dataclass(slots=True)
class IteratorState:
    epoch: int = 0
    cursor: int = 0


class PretrainBatchStream:
    def __init__(
        self,
        records,
        tokenizer: PragmaTokenizer,
        config: dict[str, Any],
        *,
        seed: int,
        context: DistributedContext,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.context = context
        self.seed = int(seed)
        self.pretrain = config.get("pretrain", {})
        self.data = config.get("data", {})
        self.state = IteratorState()
        self.use_sharded_store = bool(self.pretrain.get("use_sharded_store", False))
        self.batch_size = int(self.pretrain.get("batch_size", 8))
        self.token_budget = int(self.pretrain.get("token_budget", 8192))
        self.store: ShardedRecordStore | None = None
        self._batches: list[list[Any]] = []
        if self.use_sharded_store:
            self.store = self._prepare_store()
        self._refresh_batches()

    def _prepare_store(self) -> ShardedRecordStore:
        if self.data.get("source") == "sharded":
            store_dir = resolve_path(self.data["sharded_store_dir"])
            barrier(self.context)
            return ShardedRecordStore(store_dir)

        store_dir = resolve_path(self.pretrain.get("sharded_store_dir", "artifacts/pretrain_store"))
        should_build = bool(self.pretrain.get("rebuild_sharded_store", False)) or not (store_dir / "manifest.json").exists()
        if should_build and is_main_process(self.context):
            build_sharded_store(
                self.records,
                self.tokenizer,
                store_dir,
                compression=str(self.pretrain.get("parquet_compression", "zstd")),
            )
        barrier(self.context)
        return ShardedRecordStore(store_dir)

    def _refresh_batches(self) -> None:
        if self.use_sharded_store:
            if self.store is None:
                raise RuntimeError("Sharded pretraining store is not initialized.")
            batches = list(
                self.store.iter_dynamic_batches(
                    token_budget=self.token_budget,
                    shuffle=True,
                    seed=self.seed + self.state.epoch,
                )
            )
        else:
            indices = list(range(len(self.records)))
            random.Random(self.seed + self.state.epoch).shuffle(indices)
            batches = [
                [self.records[index] for index in indices[start : start + self.batch_size]]
                for start in range(0, len(indices), self.batch_size)
            ]

        if self.context.world_size > 1:
            batches = batches[self.context.rank :: self.context.world_size]
        self._batches = [batch for batch in batches if batch]
        if not self._batches:
            raise RuntimeError(
                "No training batches were assigned to this rank. Reduce WORLD_SIZE or increase the dataset size."
            )

    def next_batch(self):
        if self.state.cursor >= len(self._batches):
            self.state.epoch += 1
            self.state.cursor = 0
            self._refresh_batches()
        batch = self._batches[self.state.cursor]
        self.state.cursor += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": int(self.state.epoch), "cursor": int(self.state.cursor)}

    def load_state_dict(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        self.state = IteratorState(
            epoch=int(payload.get("epoch", 0)),
            cursor=int(payload.get("cursor", 0)),
        )
        self._refresh_batches()
        while self.state.cursor >= len(self._batches):
            self.state.cursor -= len(self._batches)
            self.state.epoch += 1
            self._refresh_batches()


class FinetuneBatchStream:
    def __init__(
        self,
        records,
        *,
        batch_size: int,
        seed: int,
        context: DistributedContext,
    ) -> None:
        self.records = records
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.context = context
        self.state = IteratorState()
        self._batches: list[list[Any]] = []
        self._refresh_batches()

    def _refresh_batches(self) -> None:
        indices = list(range(len(self.records)))
        random.Random(self.seed + self.state.epoch).shuffle(indices)
        if self.context.world_size > 1:
            indices = indices[self.context.rank :: self.context.world_size]
        self._batches = [
            [self.records[index] for index in indices[start : start + self.batch_size]]
            for start in range(0, len(indices), self.batch_size)
        ]
        self._batches = [batch for batch in self._batches if batch]
        if not self._batches:
            raise RuntimeError(
                "No fine-tuning batches were assigned to this rank. Reduce WORLD_SIZE or increase the dataset size."
            )

    def next_batch(self):
        if self.state.cursor >= len(self._batches):
            self.state.epoch += 1
            self.state.cursor = 0
            self._refresh_batches()
        batch = self._batches[self.state.cursor]
        self.state.cursor += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": int(self.state.epoch), "cursor": int(self.state.cursor)}

    def load_state_dict(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        self.state = IteratorState(
            epoch=int(payload.get("epoch", 0)),
            cursor=int(payload.get("cursor", 0)),
        )
        self._refresh_batches()
        while self.state.cursor >= len(self._batches):
            self.state.cursor -= len(self._batches)
            self.state.epoch += 1
            self._refresh_batches()


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRAGMA training entrypoint.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--task", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def resolve_optional_path(path_like: str | Path | None) -> Path | None:
    if path_like in {None, ""}:
        return None
    return resolve_path(path_like)


def ensure_output_dir(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_copy_config(config_path: Path, output_dir: Path) -> None:
    shutil.copyfile(config_path, output_dir / "config.yaml")


def maybe_compile(module: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and hasattr(torch, "compile"):
        return torch.compile(module)
    return module


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
    if train_records and isinstance(train_records[0], TokenizedRecord):
        raise ValueError(
            "Tokenized records require an existing tokenizer.json. Point config at a sharded store or checkpoint tokenizer."
        )
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


def build_backbone(config: dict, tokenizer: PragmaTokenizer, checkpoint: dict[str, Any] | None = None) -> PragmaBackbone:
    metadata = checkpoint or {}
    text_encoder_config = config.get("text_encoder", {})
    model_config = config.get("model", {})
    return PragmaBackbone(
        make_model_config(
            str(metadata.get("variant", model_config.get("variant", "S"))),
            tokenizer.vocab_size,
            dropout=float(metadata.get("dropout", model_config.get("dropout", 0.1))),
            label_smoothing=float(metadata.get("label_smoothing", tokenizer.masking_config.label_smoothing)),
            max_event_tokens=int(metadata.get("max_event_tokens", tokenizer.config.max_event_tokens)),
            text_encoder_dim=int(metadata.get("text_encoder_dim", tokenizer.text_embedding_dim)),
            text_loss_weight=float(metadata.get("text_loss_weight", text_encoder_config.get("text_loss_weight", 1.0))),
            attention_backend=str(metadata.get("attention_backend", model_config.get("attention_backend", "auto"))),
        )
    )


def backbone_metadata(config: dict, tokenizer: PragmaTokenizer, *, step: int) -> dict[str, Any]:
    return {
        "variant": str(config.get("model", {}).get("variant", "S")),
        "dropout": float(config.get("model", {}).get("dropout", 0.1)),
        "label_smoothing": float(tokenizer.masking_config.label_smoothing),
        "max_event_tokens": int(tokenizer.config.max_event_tokens),
        "text_encoder_dim": int(tokenizer.text_embedding_dim),
        "text_loss_weight": float(config.get("text_encoder", {}).get("text_loss_weight", 1.0)),
        "attention_backend": str(config.get("model", {}).get("attention_backend", "auto")),
        "step": int(step),
    }


def classifier_metadata(
    config: dict,
    tokenizer: PragmaTokenizer,
    *,
    num_outputs: int,
    task_type: str,
    step: int,
) -> dict[str, Any]:
    lora_config = config.get("lora", {})
    finetune = config.get("finetune", {})
    metadata = backbone_metadata(config, tokenizer, step=step)
    metadata.update(
        {
            "pooling": str(finetune.get("pooling", "usr_last")),
            "num_outputs": int(num_outputs),
            "task_type": task_type,
            "lora_rank": int(lora_config.get("rank", 8)),
            "lora_alpha": int(lora_config.get("alpha", 8)),
            "lora_dropout": float(lora_config.get("dropout", 0.0)),
        }
    )
    return metadata


def prepare_tokenizer(
    task: str,
    config: dict,
    train_records,
    *,
    output_dir: Path,
    context: DistributedContext,
    resume_path: Path | None,
) -> PragmaTokenizer:
    target_path = output_dir / "tokenizer.json"
    task_config = config.get(task, {})
    data_config = config.get("data", {})

    candidate_path: Path | None = target_path if (resume_path is not None and target_path.exists()) else None
    if candidate_path is None:
        explicit_tokenizer_path = resolve_optional_path(task_config.get("tokenizer_path"))
        if explicit_tokenizer_path is not None and explicit_tokenizer_path.exists():
            candidate_path = explicit_tokenizer_path
    if candidate_path is None and data_config.get("source") == "sharded":
        store_tokenizer_path = resolve_path(data_config.get("sharded_store_dir", "artifacts/pretrain_store")) / "tokenizer.json"
        if store_tokenizer_path.exists():
            candidate_path = store_tokenizer_path
    if candidate_path is None and target_path.exists():
        candidate_path = target_path

    if is_main_process(context):
        if candidate_path is not None:
            tokenizer = PragmaTokenizer.load(candidate_path)
        else:
            tokenizer = build_tokenizer(config, train_records)
        tokenizer.save(target_path)
    barrier(context)
    if not target_path.exists():
        raise FileNotFoundError(f"Tokenizer was not materialized at {target_path}")
    tokenizer = PragmaTokenizer.load(target_path)
    if is_main_process(context) and tokenizer.text_encoder is not None:
        validation = validate_frozen_text_encoder(tokenizer.text_encoder)
        print(
            "text_encoder",
            " ".join(f"{key}={value}" for key, value in sorted(validation.items())),
        )
    return tokenizer


def build_pretrain_optimizer(model: torch.nn.Module, config: dict):
    pretrain = config.get("pretrain", {})
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
    return optimizer_name, optimizer


def maybe_save_checkpoint(
    context: DistributedContext,
    *,
    output_dir: Path,
    prefix: str,
    step: int,
    model: torch.nn.Module,
    optimizer: Any | None,
    iterator_state: dict[str, Any],
    tokenizer_path: Path,
    metadata: dict[str, Any],
    keep_last: int,
) -> None:
    barrier(context)
    if is_main_process(context):
        save_training_checkpoint(
            output_dir,
            prefix=prefix,
            step=step,
            model=model,
            optimizer=optimizer,
            iterator_state=iterator_state,
            tokenizer_path=str(tokenizer_path),
            metadata=metadata,
            keep_last=keep_last,
        )
    barrier(context)


def run_pretrain(
    config: dict,
    config_path: Path,
    runtime_config,
    context: DistributedContext,
    *,
    resume_override: Path | None,
) -> None:
    pretrain = config.get("pretrain", {})
    output_dir = ensure_output_dir(pretrain.get("output_dir", "artifacts/pretrain"))
    configured_resume_path = resolve_optional_path(runtime_config.checkpointing.resume_path)
    override_resume_path = resolve_optional_path(resume_override)
    resume_path = resolve_resume_checkpoint(
        output_dir,
        prefix="pretrain",
        configured_path=str(configured_resume_path) if configured_resume_path is not None else None,
        override_path=str(override_resume_path) if override_resume_path is not None else None,
    )

    records = load_records(config)
    train_records, _ = split_train_val(records, config)
    tokenizer = prepare_tokenizer(
        "pretrain",
        config,
        train_records,
        output_dir=output_dir,
        context=context,
        resume_path=resume_path,
    )

    model = build_backbone(config, tokenizer).to(context.device)
    model = maybe_compile(model, runtime_config.compile)
    optimizer_name, optimizer = build_pretrain_optimizer(unwrap_module(model), config)
    batch_stream = PretrainBatchStream(
        train_records,
        tokenizer,
        config,
        seed=int(runtime_config.seed),
        context=context,
    )

    start_step = 1
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            map_location=context.device,
        )
        batch_stream.load_state_dict(checkpoint.get("iterator_state"))
        start_step = int(checkpoint.get("step", 0)) + 1
        if is_main_process(context):
            print(f"resumed={resume_path} from_step={start_step - 1}")

    model = wrap_ddp(
        model,
        context,
        find_unused_parameters=runtime_config.distributed.find_unused_parameters,
        broadcast_buffers=runtime_config.distributed.broadcast_buffers,
    )

    steps = int(pretrain.get("steps", 20))
    precision = str(pretrain.get("precision", "bf16"))
    use_sequence_packing = bool(pretrain.get("use_sequence_packing", True))
    save_every = int(runtime_config.checkpointing.save_every)
    keep_last = int(runtime_config.checkpointing.keep_last)
    if is_main_process(context):
        print(
            f"task=pretrain device={context.device} vocab_size={tokenizer.vocab_size} "
            f"precision={precision} optimizer={optimizer_name} start_step={start_step}"
        )

    last_step = start_step - 1
    for step in range(start_step, steps + 1):
        step_seed = seed_step_generators(int(runtime_config.seed), step=step, rank=context.rank)
        batch_records = batch_stream.next_batch()
        batch = tokenizer.collate(
            batch_records,
            apply_mlm=True,
            device=context.device,
            generator=make_cpu_generator(step_seed),
            pack_events=use_sequence_packing,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(context.device, precision):
            output = model.forward_pretrain(batch)
        output.loss.backward()
        optimizer.step()
        last_step = step

        if is_main_process(context) and step % max(1, int(runtime_config.log_every)) == 0:
            print(
                f"step={step:03d} loss={output.loss.item():.4f} "
                f"token_loss={float(output.token_loss.item()):.4f} "
                f"text_loss={float(output.text_loss.item()):.4f} "
                f"batch_size={len(batch_records)}"
            )

        if save_every > 0 and step % save_every == 0:
            maybe_save_checkpoint(
                context,
                output_dir=output_dir,
                prefix="pretrain",
                step=step,
                model=model,
                optimizer=optimizer,
                iterator_state=batch_stream.state_dict(),
                tokenizer_path=output_dir / "tokenizer.json",
                metadata=backbone_metadata(config, tokenizer, step=step),
                keep_last=keep_last,
            )

    final_step = last_step if last_step > 0 else start_step - 1
    maybe_save_checkpoint(
        context,
        output_dir=output_dir,
        prefix="pretrain",
        step=final_step,
        model=model,
        optimizer=optimizer,
        iterator_state=batch_stream.state_dict(),
        tokenizer_path=output_dir / "tokenizer.json",
        metadata=backbone_metadata(config, tokenizer, step=final_step),
        keep_last=keep_last,
    )

    barrier(context)
    if is_main_process(context):
        metadata = backbone_metadata(config, tokenizer, step=final_step)
        torch.save(
            {
                "state_dict": cpu_state_dict(model),
                **metadata,
            },
            output_dir / "backbone.pt",
        )
        maybe_copy_config(config_path, output_dir)
        print(f"saved={output_dir}")


def run_finetune(
    config: dict,
    config_path: Path,
    runtime_config,
    context: DistributedContext,
    *,
    resume_override: Path | None,
) -> None:
    finetune = config.get("finetune", {})
    output_dir = ensure_output_dir(finetune.get("output_dir", "artifacts/finetune"))
    configured_resume_path = resolve_optional_path(runtime_config.checkpointing.resume_path)
    override_resume_path = resolve_optional_path(resume_override)
    resume_path = resolve_resume_checkpoint(
        output_dir,
        prefix="finetune",
        configured_path=str(configured_resume_path) if configured_resume_path is not None else None,
        override_path=str(override_resume_path) if override_resume_path is not None else None,
    )

    task_type = str(finetune.get("task_type", "binary")).lower()
    if task_type == "uplift":
        raise ValueError("Use scripts/probe.py for uplift evaluation on frozen embeddings.")

    records = load_records(config)
    train_records, val_records = split_train_val(records, config)
    tokenizer = prepare_tokenizer(
        "finetune",
        config,
        train_records,
        output_dir=output_dir,
        context=context,
        resume_path=resume_path,
    )

    backbone = build_backbone(config, tokenizer).to(context.device)
    pretrained_path = resolve_optional_path(finetune.get("pretrained_backbone_path"))
    if pretrained_path is not None and pretrained_path.exists():
        checkpoint = torch.load(pretrained_path, map_location=context.device, weights_only=False)
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

    num_outputs = infer_num_outputs(
        train_records,
        task_type,
        None if finetune.get("num_outputs") in {None, 0} else int(finetune.get("num_outputs")),
    )
    classifier = PragmaClassifier(
        backbone,
        num_outputs=num_outputs,
        pooling=str(finetune.get("pooling", "usr_last")),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(context.device)
    for parameter in classifier.head.parameters():
        parameter.requires_grad = True

    classifier = maybe_compile(classifier, runtime_config.compile)
    optimizer = torch.optim.Adam(
        [parameter for parameter in unwrap_module(classifier).parameters() if parameter.requires_grad],
        lr=float(finetune.get("learning_rate", 2e-4)),
    )
    batch_stream = FinetuneBatchStream(
        train_records,
        batch_size=int(finetune.get("batch_size", 8)),
        seed=int(runtime_config.seed),
        context=context,
    )

    start_step = 1
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=classifier,
            optimizer=optimizer,
            map_location=context.device,
        )
        batch_stream.load_state_dict(checkpoint.get("iterator_state"))
        start_step = int(checkpoint.get("step", 0)) + 1
        if is_main_process(context):
            print(f"resumed={resume_path} from_step={start_step - 1}")

    classifier = wrap_ddp(
        classifier,
        context,
        find_unused_parameters=runtime_config.distributed.find_unused_parameters,
        broadcast_buffers=runtime_config.distributed.broadcast_buffers,
    )

    trainable, total = lora_parameter_count(unwrap_module(classifier))
    precision = str(finetune.get("precision", "fp32"))
    steps = int(finetune.get("steps", 20))
    save_every = int(runtime_config.checkpointing.save_every)
    keep_last = int(runtime_config.checkpointing.keep_last)
    if is_main_process(context):
        print(
            f"task=finetune device={context.device} precision={precision} task_type={task_type} "
            f"trainable={trainable} total={total} start_step={start_step}"
        )

    last_step = start_step - 1
    for step in range(start_step, steps + 1):
        seed_step_generators(int(runtime_config.seed), step=step, rank=context.rank)
        batch_records = batch_stream.next_batch()
        batch = tokenizer.collate(batch_records, apply_mlm=False, device=context.device)
        targets = tensorize_targets(batch_records, task_type, context.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(context.device, precision):
            logits = classifier(batch)
            loss = task_loss(logits, targets, task_type)
        loss.backward()
        optimizer.step()
        last_step = step

        if is_main_process(context) and step % max(1, int(runtime_config.log_every)) == 0:
            print(f"step={step:03d} train_loss={loss.item():.4f}")

        if save_every > 0 and step % save_every == 0:
            maybe_save_checkpoint(
                context,
                output_dir=output_dir,
                prefix="finetune",
                step=step,
                model=classifier,
                optimizer=optimizer,
                iterator_state=batch_stream.state_dict(),
                tokenizer_path=output_dir / "tokenizer.json",
                metadata=classifier_metadata(config, tokenizer, num_outputs=num_outputs, task_type=task_type, step=step),
                keep_last=keep_last,
            )

    final_step = last_step if last_step > 0 else start_step - 1
    maybe_save_checkpoint(
        context,
        output_dir=output_dir,
        prefix="finetune",
        step=final_step,
        model=classifier,
        optimizer=optimizer,
        iterator_state=batch_stream.state_dict(),
        tokenizer_path=output_dir / "tokenizer.json",
        metadata=classifier_metadata(config, tokenizer, num_outputs=num_outputs, task_type=task_type, step=final_step),
        keep_last=keep_last,
    )

    barrier(context)
    if is_main_process(context):
        classifier_module = unwrap_module(classifier)
        val_loss, metrics = evaluate_classifier(
            classifier_module,
            tokenizer,
            val_records,
            context.device,
            task_type=task_type,
            precision=precision,
        )
        metrics_display = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        print(f"val_loss={val_loss:.4f} {metrics_display}".strip())
        metadata = classifier_metadata(config, tokenizer, num_outputs=num_outputs, task_type=task_type, step=final_step)
        torch.save(
            {
                "state_dict": cpu_state_dict(classifier_module),
                **metadata,
            },
            output_dir / "classifier.pt",
        )
        maybe_copy_config(config_path, output_dir)
        print(f"saved={output_dir}")


def main() -> None:
    args = make_argument_parser().parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml_config(config_path)
    runtime_config, context = initialize_runtime(config.get("runtime", {}))
    seed_everything(int(runtime_config.seed), rank=context.rank)
    try:
        if args.task == "pretrain":
            run_pretrain(
                config,
                config_path,
                runtime_config,
                context,
                resume_override=args.resume,
            )
        else:
            run_finetune(
                config,
                config_path,
                runtime_config,
                context,
                resume_override=args.resume,
            )
    finally:
        finalize_runtime(context)


if __name__ == "__main__":
    main()
