from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pragma_repro import PragmaBackbone, PragmaTokenizer, TokenizerConfig, make_model_config
from pragma_repro.synthetic import generate_synthetic_records, split_records


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny PRAGMA-style MLM pretraining run.")
    parser.add_argument("--variant", default="S", choices=["S", "M", "L"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pretrain"))
    return parser


def main() -> None:
    args = make_argument_parser().parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    records = generate_synthetic_records(args.num_records, seed=args.seed)
    train_records, _ = split_records(records)

    tokenizer = PragmaTokenizer(TokenizerConfig())
    tokenizer.fit(train_records)

    model_config = make_model_config(
        args.variant,
        tokenizer.vocab_size,
        max_event_tokens=tokenizer.config.max_event_tokens,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PragmaBackbone(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device} vocab_size={tokenizer.vocab_size} variant={args.variant}")
    for step in range(1, args.steps + 1):
        batch_records = rng.sample(train_records, k=min(args.batch_size, len(train_records)))
        batch = tokenizer.collate(batch_records, apply_mlm=True, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_pretrain(batch)
        output.loss.backward()
        optimizer.step()
        masked_count = int(output.masked_targets.numel())
        print(f"step={step:03d} loss={output.loss.item():.4f} masked_tokens={masked_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "backbone.pt")
    tokenizer.save(args.output_dir / "tokenizer.json")
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
