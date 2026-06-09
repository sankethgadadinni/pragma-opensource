# PRAGMA Reproduction

This workspace contains a clean-room PyTorch implementation of the core PRAGMA recipe from the Revolut paper:

- key-value-time tokenization for profile state and event histories
- shared key/value embeddings with within-field positional encodings
- profile-state, event, and history encoders
- mixed masking for MLM pretraining
- LoRA adapters over QKV and MLP projections for downstream tuning

The implementation is intentionally dependency-light so it can run in a fresh workspace. That means a few production details from the paper are approximated rather than copied verbatim:

- it uses padded attention instead of FlashAttention varlen kernels
- it uses a small in-repo BPE-style tokenizer instead of an external tokenizer package
- the exact categorical threshold, numeric bucket count, and `[UNK]` replacement rate are configurable because the paper does not publish those values
- the data layer uses in-memory records and JSON-friendly schemas instead of the paper's LMDB plus Parquet sharding stack

## Layout

- `pragma_repro/config.py`: model and tokenizer presets, including PRAGMA-S/M/L
- `pragma_repro/records.py`: dataset schema
- `pragma_repro/tokenizer.py`: schema inference, vocab building, tokenization, collation
- `pragma_repro/model.py`: PRAGMA-style backbone and MLM head
- `pragma_repro/lora.py`: LoRA injection utilities
- `pragma_repro/synthetic.py`: synthetic transactional dataset generator
- `scripts/pretrain_synthetic.py`: tiny MLM pretraining run
- `scripts/finetune_synthetic.py`: tiny downstream LoRA run
- `scripts/smoke_test.py`: one forward pass and one optimizer step

## Record Schema

Each training example is a `UserRecord` with:

- `evaluation_ts`: the record cutoff timestamp
- `profile`: profile-state key/value pairs
- `lifelong`: milestone events with their own timestamps
- `events`: time-ordered events, each with a timestamp and key/value fields
- `label`: optional downstream label

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch numpy
python scripts/smoke_test.py
python scripts/pretrain_synthetic.py --steps 20
python scripts/finetune_synthetic.py --steps 20
```

## Paper-Faithful Defaults

- PRAGMA-S: `d_model=192`, `d_ffn=768`, profile/event/history layers = `1/5/2`, heads = `3`
- PRAGMA-M: `d_model=512`, `d_ffn=2048`, profile/event/history layers = `3/16/6`, heads = `8`
- PRAGMA-L: `d_model=1024`, `d_ffn=4096`, profile/event/history layers = `9/45/18`, heads = `16`
- event truncation: `24` tokens
- profile truncation: `200` tokens
- max history length: `6500` events
- masking mix: token `15%`, event `10%`, key-level `10%`
- LoRA defaults: rank `8`, alpha `8`
