from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def run_command(*args: str) -> None:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.strip())


def build_config(root: Path) -> dict:
    pretrain_dir = root / "pretrain"
    finetune_dir = root / "finetune"
    store_dir = root / "store"
    return {
        "runtime": {
            "seed": 3,
            "device": "auto",
            "deterministic": False,
            "compile": False,
            "matmul_precision": "high",
            "log_every": 1,
            "distributed": {
                "enabled": False,
                "backend": "nccl",
                "find_unused_parameters": False,
                "broadcast_buffers": False,
            },
            "checkpointing": {
                "save_every": 1,
                "keep_last": 2,
                "resume_path": None,
            },
        },
        "data": {
            "source": "synthetic",
            "num_records": 48,
            "train_fraction": 0.8,
            "synthetic_seed": 5,
            "min_events": 8,
            "max_events": 16,
            "sharded_store_dir": str(store_dir),
        },
        "tokenizer": {
            "categorical_threshold": 128,
            "numeric_bucket_count": 32,
            "text_vocab_size": 1024,
            "bpe_min_frequency": 1,
            "max_event_tokens": 16,
            "max_profile_tokens": 64,
            "max_events": 512,
        },
        "masking": {
            "token_mask_probability": 0.15,
            "event_mask_probability": 0.10,
            "key_mask_probability": 0.10,
            "unk_probability": 0.10,
            "label_smoothing": 0.1,
        },
        "text_encoder": {
            "enabled": True,
            "provider": "hash",
            "target_fields": ["description"],
            "output_dim": 64,
            "text_loss_weight": 1.0,
        },
        "model": {
            "variant": "S",
            "dropout": 0.1,
            "attention_backend": "sdpa",
        },
        "pretrain": {
            "batch_size": 4,
            "steps": 2,
            "learning_rate": 3e-4,
            "adamw_learning_rate": 3e-4,
            "optimizer": "muon_adamw",
            "precision": "fp32",
            "token_budget": 2048,
            "use_sharded_store": True,
            "sharded_store_dir": str(store_dir),
            "rebuild_sharded_store": True,
            "parquet_compression": "zstd",
            "use_sequence_packing": True,
            "weight_decay": 0.01,
            "muon_momentum": 0.95,
            "output_dir": str(pretrain_dir),
        },
        "finetune": {
            "batch_size": 4,
            "steps": 2,
            "learning_rate": 2e-4,
            "precision": "fp32",
            "output_dir": str(finetune_dir),
            "pooling": "usr_last",
            "num_outputs": 1,
            "task_type": "binary",
            "pretrained_backbone_path": str(pretrain_dir / "backbone.pt"),
            "tokenizer_path": str(pretrain_dir / "tokenizer.json"),
        },
        "lora": {
            "rank": 4,
            "alpha": 4,
            "dropout": 0.0,
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = build_config(root)
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        run_command("scripts/train.py", "--config", str(config_path), "--task", "pretrain")
        pretrain_latest = root / "pretrain" / "checkpoints" / "pretrain_latest.pt"
        if not pretrain_latest.exists():
            raise RuntimeError("Pretraining checkpoint was not written.")

        config["pretrain"]["steps"] = 3
        config["runtime"]["checkpointing"]["resume_path"] = str(pretrain_latest)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_command("scripts/train.py", "--config", str(config_path), "--task", "pretrain")
        pretrain_checkpoint = torch.load(pretrain_latest, map_location="cpu", weights_only=False)
        if int(pretrain_checkpoint.get("step", 0)) != 3:
            raise RuntimeError("Pretraining resume did not advance to the requested step.")

        config["runtime"]["checkpointing"]["resume_path"] = None
        config["finetune"]["steps"] = 2
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_command("scripts/train.py", "--config", str(config_path), "--task", "finetune")
        finetune_latest = root / "finetune" / "checkpoints" / "finetune_latest.pt"
        if not finetune_latest.exists():
            raise RuntimeError("Fine-tuning checkpoint was not written.")

        config["finetune"]["steps"] = 3
        config["runtime"]["checkpointing"]["resume_path"] = str(finetune_latest)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_command("scripts/train.py", "--config", str(config_path), "--task", "finetune")
        finetune_checkpoint = torch.load(finetune_latest, map_location="cpu", weights_only=False)
        if int(finetune_checkpoint.get("step", 0)) != 3:
            raise RuntimeError("Fine-tuning resume did not advance to the requested step.")

        print(
            "integration_test_ok",
            {
                "pretrain_step": int(pretrain_checkpoint["step"]),
                "finetune_step": int(finetune_checkpoint["step"]),
                "pretrain_dir": str(root / "pretrain"),
                "finetune_dir": str(root / "finetune"),
            },
        )


if __name__ == "__main__":
    main()
