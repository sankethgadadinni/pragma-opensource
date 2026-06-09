# PRAGMA Reproduction

This repo contains a clean-room PyTorch implementation of the core PRAGMA recipe from the Revolut paper.

## What Is Here

- key-value-time tokenization for profile state and event histories
- shared key/value embeddings with within-field positional encodings
- profile-state, event, and history encoders
- mixed masking for MLM pretraining
- LoRA adapters over QKV and MLP projections for downstream tuning

The implementation is intentionally dependency-light so it can run in a fresh workspace. A few production details from the paper are approximated rather than copied verbatim:

- padded attention is used instead of FlashAttention varlen kernels
- the tokenizer is implemented in-repo instead of depending on an external tokenizer stack
- categorical thresholds, numeric bucket counts, and `[UNK]` replacement rates are configurable because the paper does not publish exact values
- the data layer uses in-memory records instead of the paper's LMDB plus Parquet sharding layout

## Folder Structure

- `environment.yml`: Conda environment definition
- `config.yaml`: shared runtime, training, and inference settings
- `src/config.py`: model presets plus YAML config loading helpers
- `src/data/`: records, tokenization, masking, JSON I/O, and synthetic data generation
- `src/modeling/`: backbone and LoRA modules
- `src/tests/`: smoke-level execution checks
- `scripts/`: training and inference entrypoints
- `research/`: local paper notes and ignored research assets

## Record Schema

Each training example is a `UserRecord` with:

- `evaluation_ts`: the record cutoff timestamp
- `profile`: profile-state key/value pairs
- `lifelong`: milestone events with their own timestamps
- `events`: time-ordered events, each with a timestamp and key/value fields
- `label`: optional downstream label

## Quick Start

```bash
conda env create -f environment.yml
conda activate pragma-opensource
python src/tests/smoke_test.py
python scripts/train.py --config config.yaml --task pretrain
python scripts/train.py --config config.yaml --task finetune
python scripts/infer.py --config config.yaml
```

## Config-Driven Workflow

`config.yaml` is the single control surface for:

- runtime device and random seed
- dataset source and synthetic dataset size
- tokenizer settings
- model variant and dropout
- pretraining hyperparameters and output paths
- finetuning and LoRA settings
- inference checkpoint paths and prediction output

That means you can change batch size, model size, output locations, or checkpoint paths without editing Python files.

## Paper-Faithful Defaults

- PRAGMA-S: `d_model=192`, `d_ffn=768`, profile/event/history layers = `1/5/2`, heads = `3`
- PRAGMA-M: `d_model=512`, `d_ffn=2048`, profile/event/history layers = `3/16/6`, heads = `8`
- PRAGMA-L: `d_model=1024`, `d_ffn=4096`, profile/event/history layers = `9/45/18`, heads = `16`
- event truncation: `24` tokens
- profile truncation: `200` tokens
- max history length: `6500` events
- masking mix: token `15%`, event `10%`, key-level `10%`
- LoRA defaults: rank `8`, alpha `8`
