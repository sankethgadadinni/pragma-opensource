from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import TokenizerConfig, make_model_config
from data import PragmaTokenizer, generate_synthetic_records
from modeling import PragmaBackbone, PragmaClassifier
from modeling.lora import LoRAConfig, freeze_non_lora_parameters, inject_lora


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = generate_synthetic_records(48, seed=0)
    tokenizer = PragmaTokenizer(TokenizerConfig())
    tokenizer.fit(records)
    batch = tokenizer.collate(records[:8], apply_mlm=True, device=device)

    backbone = PragmaBackbone(
        make_model_config("S", tokenizer.vocab_size, max_event_tokens=tokenizer.config.max_event_tokens)
    ).to(device)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    pretrain_output = backbone.forward_pretrain(batch)
    pretrain_output.loss.backward()
    optimizer.step()

    inject_lora(backbone, LoRAConfig(rank=4, alpha=4))
    freeze_non_lora_parameters(backbone)
    classifier = PragmaClassifier(backbone).to(device)
    for parameter in classifier.head.parameters():
        parameter.requires_grad = True

    classifier_batch = tokenizer.collate(records[8:16], apply_mlm=False, device=device)
    if classifier_batch.downstream_labels is None:
        raise RuntimeError("Synthetic dataset labels are missing.")
    logits = classifier(classifier_batch)
    loss = F.binary_cross_entropy_with_logits(logits, classifier_batch.downstream_labels)
    loss.backward()

    print(
        "smoke_test_ok",
        {
            "device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "masked_targets": int(pretrain_output.masked_targets.numel()),
            "classification_loss": round(float(loss.item()), 4),
        },
    )


if __name__ == "__main__":
    main()
