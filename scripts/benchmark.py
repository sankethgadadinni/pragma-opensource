from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    benchmark_config_from_dict,
    load_yaml_config,
    make_model_config,
    masking_config_from_dict,
    text_encoder_config_from_dict,
    tokenizer_config_from_dict,
)
from data import PragmaTokenizer, generate_synthetic_records  # noqa: E402
from modeling import PragmaBackbone, build_muon_adamw_optimizer, resolve_attention_backend  # noqa: E402
from runtime import finalize_runtime, initialize_runtime, make_cpu_generator, seed_everything, seed_step_generators  # noqa: E402


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PRAGMA attention and packing backends.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def autocast_context(device: torch.device, precision: str):
    if precision.lower() == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_tokens(batch) -> int:
    total = int(batch.profile_token_mask.sum().item())
    if batch.event_token_mask is not None:
        total += int(batch.event_token_mask.sum().item())
    return total


def build_tokenizer(config: dict, records) -> PragmaTokenizer:
    tokenizer = PragmaTokenizer(
        tokenizer_config_from_dict(config.get("tokenizer", {})),
        masking_config_from_dict(config.get("masking", {})),
        text_encoder_config_from_dict(config.get("text_encoder", {})),
    )
    tokenizer.fit(records)
    return tokenizer


def main() -> None:
    args = make_argument_parser().parse_args()
    config = load_yaml_config(resolve_path(args.config))
    runtime_config, context = initialize_runtime(config.get("runtime", {}))
    seed_everything(int(runtime_config.seed), rank=context.rank)
    try:
        benchmark_config = benchmark_config_from_dict(config.get("benchmark", {}))
        records = generate_synthetic_records(
            int(benchmark_config.num_records),
            seed=int(config.get("data", {}).get("synthetic_seed", 0)),
            min_events=int(config.get("data", {}).get("min_events", 16)),
            max_events=int(config.get("data", {}).get("max_events", 72)),
        )
        tokenizer = build_tokenizer(config, records)
        batch_records = records[: int(benchmark_config.batch_size)]
        precision = str(config.get("pretrain", {}).get("precision", "bf16"))
        results: list[dict[str, object]] = []

        for attention_backend in benchmark_config.backends:
            for pack_events in benchmark_config.pack_events:
                trial_config = copy.deepcopy(config)
                trial_config.setdefault("model", {})
                trial_config["model"]["attention_backend"] = attention_backend
                model = PragmaBackbone(
                    make_model_config(
                        str(trial_config.get("model", {}).get("variant", "S")),
                        tokenizer.vocab_size,
                        dropout=float(trial_config.get("model", {}).get("dropout", 0.1)),
                        label_smoothing=float(tokenizer.masking_config.label_smoothing),
                        max_event_tokens=int(tokenizer.config.max_event_tokens),
                        text_encoder_dim=int(tokenizer.text_embedding_dim),
                        text_loss_weight=float(trial_config.get("text_encoder", {}).get("text_loss_weight", 1.0)),
                        attention_backend=attention_backend,
                    )
                ).to(context.device)
                optimizer = build_muon_adamw_optimizer(model, lr=float(config.get("pretrain", {}).get("learning_rate", 3e-4)))
                total_tokens = 0
                synchronize(context.device)

                for warmup_step in range(1, int(benchmark_config.warmup_steps) + 1):
                    step_seed = seed_step_generators(int(runtime_config.seed), step=warmup_step, rank=context.rank)
                    batch = tokenizer.collate(
                        batch_records,
                        apply_mlm=True,
                        device=context.device,
                        generator=make_cpu_generator(step_seed),
                        pack_events=pack_events,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(context.device, precision):
                        output = model.forward_pretrain(batch)
                    output.loss.backward()
                    optimizer.step()

                synchronize(context.device)
                start = time.perf_counter()
                measured_steps = int(benchmark_config.steps)
                for step_index in range(1, measured_steps + 1):
                    step_seed = seed_step_generators(
                        int(runtime_config.seed),
                        step=int(benchmark_config.warmup_steps) + step_index,
                        rank=context.rank,
                    )
                    batch = tokenizer.collate(
                        batch_records,
                        apply_mlm=True,
                        device=context.device,
                        generator=make_cpu_generator(step_seed),
                        pack_events=pack_events,
                    )
                    total_tokens += benchmark_tokens(batch)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(context.device, precision):
                        output = model.forward_pretrain(batch)
                    output.loss.backward()
                    optimizer.step()
                synchronize(context.device)
                elapsed = max(time.perf_counter() - start, 1e-9)
                backend_info = resolve_attention_backend(
                    attention_backend,
                    device=context.device,
                    dtype=torch.bfloat16 if precision.lower() == "bf16" else torch.float32,
                    valid_mask=None,
                )
                results.append(
                    {
                        "attention_backend": attention_backend,
                        "resolved_backend": backend_info.resolved,
                        "pack_events": bool(pack_events),
                        "steps": measured_steps,
                        "elapsed_seconds": elapsed,
                        "steps_per_second": measured_steps / elapsed,
                        "tokens_per_second": total_tokens / elapsed,
                        "avg_tokens_per_step": total_tokens / max(measured_steps, 1),
                    }
                )

        output_path = resolve_path(benchmark_config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
    finally:
        finalize_runtime(context)


if __name__ == "__main__":
    main()
