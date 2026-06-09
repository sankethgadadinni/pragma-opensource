from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pragma_repro import PragmaBackbone, PragmaClassifier, PragmaTokenizer, TokenizerConfig, make_model_config
from pragma_repro.lora import LoRAConfig, freeze_non_lora_parameters, inject_lora, lora_parameter_count
from pragma_repro.synthetic import generate_synthetic_records, split_records


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny PRAGMA-style LoRA finetuning run.")
    parser.add_argument("--variant", default="S", choices=["S", "M", "L"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--pretrained", type=Path, default=None)
    return parser


def evaluate(classifier: PragmaClassifier, tokenizer: PragmaTokenizer, records, device) -> tuple[float, float]:
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for start in range(0, len(records), 16):
            batch_records = records[start : start + 16]
            batch = tokenizer.collate(batch_records, apply_mlm=False, device=device)
            logits = classifier(batch)
            labels = batch.downstream_labels
            if labels is None:
                continue
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            predictions = (torch.sigmoid(logits) > 0.5).long()
            total_loss += loss.item() * len(batch_records)
            total_correct += int((predictions == labels.long()).sum().item())
            total_count += len(batch_records)
    classifier.train()
    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def main() -> None:
    args = make_argument_parser().parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    records = generate_synthetic_records(args.num_records, seed=args.seed)
    train_records, val_records = split_records(records)

    tokenizer = PragmaTokenizer(TokenizerConfig())
    tokenizer.fit(train_records)
    config = make_model_config(
        args.variant,
        tokenizer.vocab_size,
        max_event_tokens=tokenizer.config.max_event_tokens,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = PragmaBackbone(config).to(device)
    if args.pretrained is not None and args.pretrained.exists():
        backbone.load_state_dict(torch.load(args.pretrained, map_location=device))

    inject_lora(backbone, LoRAConfig(rank=args.rank, alpha=args.alpha))
    freeze_non_lora_parameters(backbone)
    classifier = PragmaClassifier(backbone, pooling="usr_last").to(device)
    for parameter in classifier.head.parameters():
        parameter.requires_grad = True

    trainable, total = lora_parameter_count(classifier)
    print(f"device={device} trainable={trainable} total={total}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    for step in range(1, args.steps + 1):
        batch_records = rng.sample(train_records, k=min(args.batch_size, len(train_records)))
        batch = tokenizer.collate(batch_records, apply_mlm=False, device=device)
        if batch.downstream_labels is None:
            raise RuntimeError("Synthetic dataset should always provide labels.")
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch.downstream_labels)
        loss.backward()
        optimizer.step()
        print(f"step={step:03d} train_loss={loss.item():.4f}")

    val_loss, val_accuracy = evaluate(classifier, tokenizer, val_records, device)
    print(f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")


if __name__ == "__main__":
    main()
