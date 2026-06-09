from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TextEncoderConfig, TokenizerConfig, make_model_config
from data import PragmaTokenizer, build_sharded_store, generate_synthetic_records
from modeling import PragmaBackbone, build_muon_adamw_optimizer
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
    batch = tokenizer.collate(records[:8], apply_mlm=True, device=device, pack_events=True)
    if batch.packed_event_lengths is None or batch.packed_event_lengths.numel() == 0:
        raise RuntimeError("Packed event path did not produce packed event lengths.")

    backbone = PragmaBackbone(
        make_model_config(
            "S",
            tokenizer.vocab_size,
            max_event_tokens=tokenizer.config.max_event_tokens,
            text_encoder_dim=tokenizer.text_embedding_dim,
            text_loss_weight=1.0,
        )
    ).to(device)
    optimizer = build_muon_adamw_optimizer(backbone, lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    pretrain_output = backbone.forward_pretrain(batch)
    pretrain_output.loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        store_dir = Path(tmpdir) / "store"
        build_sharded_store(records, tokenizer, store_dir)
        if not (store_dir / "manifest.json").exists():
            raise RuntimeError("Sharded store manifest was not written.")

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

    print(
        "smoke_test_ok",
        {
            "device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "text_embedding_dim": tokenizer.text_embedding_dim,
            "packed_events": int(batch.packed_event_lengths.shape[0]),
            "masked_targets": int(pretrain_output.masked_targets.numel()),
            "token_loss": round(float(pretrain_output.token_loss.item()), 4),
            "text_loss": round(float(pretrain_output.text_loss.item()), 4),
            "probe_pr_auc": round(float(metrics["pr_auc"]), 4),
        },
    )


if __name__ == "__main__":
    main()
