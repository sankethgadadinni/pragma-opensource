from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TextEncoderConfig, TokenizerConfig, make_model_config
from data import (
    PragmaTokenizer,
    build_sharded_store,
    generate_synthetic_records,
    validate_frozen_text_encoder,
)
from modeling import PragmaBackbone, build_muon_adamw_optimizer, resolve_attention_backend
from runtime import load_training_checkpoint, make_cpu_generator, save_training_checkpoint, seed_step_generators
from tasks import (
    StandardScaler,
    binary_classification_metrics,
    compute_embeddings,
    fit_lbfgs_probe,
    tensorize_targets,
)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = generate_synthetic_records(48, seed=0)
    tokenizer = PragmaTokenizer(
        TokenizerConfig(),
        text_encoder_config=TextEncoderConfig(
            enabled=True,
            provider="hash",
            target_fields=("description",),
            output_dim=64,
        ),
    )
    tokenizer.fit(records)
    text_validation = validate_frozen_text_encoder(tokenizer.text_encoder)
    batch = tokenizer.collate(
        records[:8],
        apply_mlm=True,
        device=device,
        generator=make_cpu_generator(seed_step_generators(0, step=1, rank=0)),
        pack_events=True,
    )
    if batch.packed_event_lengths is None or batch.packed_event_lengths.numel() == 0:
        raise RuntimeError("Packed event path did not produce packed event lengths.")

    backbone = PragmaBackbone(
        make_model_config(
            "S",
            tokenizer.vocab_size,
            max_event_tokens=tokenizer.config.max_event_tokens,
            text_encoder_dim=tokenizer.text_embedding_dim,
            text_loss_weight=1.0,
            attention_backend="sdpa",
        )
    ).to(device)
    optimizer = build_muon_adamw_optimizer(backbone, lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    pretrain_output = backbone.forward_pretrain(batch)
    pretrain_output.loss.backward()
    optimizer.step()

    resolved_backend = resolve_attention_backend("sdpa", device=device, dtype=torch.float32, valid_mask=None)
    checkpoint_step = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        store_dir = tmpdir_path / "store"
        build_sharded_store(records, tokenizer, store_dir)
        if not (store_dir / "manifest.json").exists():
            raise RuntimeError("Sharded store manifest was not written.")

        tokenizer_path = tmpdir_path / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        checkpoint_dir = tmpdir_path / "pretrain"
        save_training_checkpoint(
            checkpoint_dir,
            prefix="smoke",
            step=1,
            model=backbone,
            optimizer=optimizer,
            iterator_state={"epoch": 0, "cursor": 1},
            tokenizer_path=str(tokenizer_path),
            metadata={"variant": "S"},
            keep_last=1,
        )

        resumed_backbone = PragmaBackbone(
            make_model_config(
                "S",
                tokenizer.vocab_size,
                max_event_tokens=tokenizer.config.max_event_tokens,
                text_encoder_dim=tokenizer.text_embedding_dim,
                text_loss_weight=1.0,
                attention_backend="sdpa",
            )
        ).to(device)
        resumed_optimizer = build_muon_adamw_optimizer(resumed_backbone, lr=1e-3)
        checkpoint = load_training_checkpoint(
            checkpoint_dir / "checkpoints" / "smoke_latest.pt",
            model=resumed_backbone,
            optimizer=resumed_optimizer,
            map_location=device,
        )
        checkpoint_step = int(checkpoint.get("step", 0))

    train_records = records[:32]
    val_records = records[32:40]
    train_embeddings = compute_embeddings(backbone, tokenizer, train_records, device=device, pooling="usr_last")
    val_embeddings = compute_embeddings(backbone, tokenizer, val_records, device=device, pooling="usr_last")
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(train_embeddings)
    val_inputs = scaler.transform(val_embeddings)
    train_targets = tensorize_targets(train_records, "binary", train_inputs.device)
    val_targets = tensorize_targets(val_records, "binary", val_inputs.device)
    probe = fit_lbfgs_probe(train_inputs, train_targets, task_type="binary", num_outputs=1, max_iter=32)
    with torch.no_grad():
        val_logits = probe(val_inputs)
    metrics = binary_classification_metrics(val_logits, val_targets)

    hf_validation: dict[str, object] = {"enabled": False, "provider": "hf", "skipped": True}
    hf_model_name = os.environ.get("PRAGMA_TEST_HF_MODEL")
    if hf_model_name:
        hf_tokenizer = PragmaTokenizer(
            TokenizerConfig(),
            text_encoder_config=TextEncoderConfig(
                enabled=True,
                provider="hf",
                model_name=hf_model_name,
                target_fields=("description",),
                local_files_only=True,
            ),
        )
        hf_tokenizer.fit(records[:8])
        hf_validation = validate_frozen_text_encoder(hf_tokenizer.text_encoder)
        hf_validation["skipped"] = False

    print(
        "smoke_test_ok",
        {
            "device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "text_validation": text_validation,
            "hf_validation": hf_validation,
            "resolved_backend": resolved_backend.resolved,
            "packed_events": int(batch.packed_event_lengths.shape[0]),
            "masked_targets": int(pretrain_output.masked_targets.numel()),
            "token_loss": round(float(pretrain_output.token_loss.item()), 4),
            "text_loss": round(float(pretrain_output.text_loss.item()), 4),
            "checkpoint_step": checkpoint_step,
            "probe_pr_auc": round(float(metrics["pr_auc"]), 4),
        },
    )


if __name__ == "__main__":
    main()
