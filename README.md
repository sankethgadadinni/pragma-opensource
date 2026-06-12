# PRAGMA Reproduction

This repo contains a clean-room PyTorch implementation of the PRAGMA recipe from the Revolut paper, including the paper’s core architecture, the scalable shard-based pretraining path, embedding probes, and the optional frozen text-encoder variant.

## What Is Here

- key-value-time tokenization for profile state and event histories
- shared key/value embeddings with within-field positional encodings
- profile-state, event, and history encoders
- mixed masking for MLM pretraining
- LMDB-backed user index plus Parquet event shards for pretraining
- dynamic token-budget batching with packed event processing
- Muon + AdamW pretraining split with configurable bf16 autocast
- resumable checkpoints with deterministic step seeding
- optional single-node DDP via `torchrun`
- selectable attention backends (`auto`, `sdpa`, `flash`, `manual`)
- LoRA adapters over QKV and MLP projections for downstream tuning
- LBFGS embedding probes with standard-scaling
- binary, regression, ranking, and uplift-style downstream evaluation
- optional frozen text encoder with continuous text reconstruction targets

The implementation is still pragmatic in a few places:

- the packed event path is implemented in pure PyTorch and groups equal-length sequences, rather than requiring a specific external FlashAttention build
- the tokenizer is implemented in-repo instead of depending on an external tokenizer stack
- categorical thresholds, numeric bucket counts, and `[UNK]` replacement rates are configurable because the paper does not publish exact values
- the optional text encoder defaults to a deterministic frozen hash encoder for local smoke tests, and can switch to a local Hugging Face encoder when you provide a model name

## Folder Structure

- `environment.yml`: Conda environment definition
- `config.yaml`: shared runtime, training, and inference settings
- `src/config.py`: model presets plus YAML config loading helpers
- `src/runtime.py`: runtime, checkpointing, and DDP helpers
- `src/data/`: records, tokenization, masking, JSON I/O, shard storage, and synthetic data generation
- `src/data/mbd.py`: MBD and MBD-mini loader that maps public multimodal banking tables into `UserRecord`s
- `src/modeling/`: backbone, LoRA, and optimizer modules
- `src/tasks/`: probe fitting, label handling, and metrics
- `src/tests/`: smoke and end-to-end integration checks
- `scripts/`: store building, training, benchmarking, probing, and inference entrypoints
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
python src/tests/integration_test.py
python scripts/build_store.py --config config.yaml
python scripts/train.py --config config.yaml --task pretrain
python scripts/train.py --config config.yaml --task finetune
python scripts/benchmark.py --config config.yaml
python scripts/probe.py --config config.yaml
python scripts/infer.py --config config.yaml
```

## Config-Driven Workflow

`config.yaml` is the single control surface for:

- runtime device and random seed
- runtime checkpointing, logging cadence, and distributed settings
- dataset source, shard storage, and synthetic dataset size
- MBD/MBD-mini root paths, fold filters, target attachment, and dialog embedding compression
- tokenizer settings
- optional frozen text encoder settings
- model variant, dropout, and attention backend
- pretraining hyperparameters, shard batching, and output paths
- benchmark combinations for packed vs. unpacked attention
- finetuning, probe, and LoRA settings
- inference checkpoint paths and prediction output

That means you can switch between in-memory records and sharded pretraining, change downstream task types, resume from checkpoints, benchmark backend choices, or enable the text-encoder ablation without editing Python files.

## MBD Loader

The repo now has a first-class `data.source: mbd` path for `MBD` and `MBD-mini`.

- one `UserRecord` is built per `(client_id, report_date)` cutoff
- transactions, geo rows, and dialog rows are merged into one time-ordered event history
- profile state is derived from the history window, including recent transaction counts, amount aggregates, dominant currency, geo activity, and dialog recency
- milestone-style `lifelong` events are derived from first transaction, geo, and dialog timestamps
- multi-product targets can stay attached as a label vector for downstream ranking, or be disabled with `attach_targets: false` for pure pretraining
- dialog embeddings are compacted with either `summary`, `project`, or `skip` strategies so they fit the tokenizer cleanly

Example config snippet:

```yaml
data:
  source: mbd
  train_fraction: 0.8
  mbd:
    root_dir: /path/to/mbd-mini
    allowed_folds: [0]
    label_fields: [product_1, product_2, product_3, product_4]
    label_mode: vector
    history_window_days: 365
    dialog_embedding_strategy: project
    dialog_projection_dim: 8
    attach_targets: true
```

Then run the normal flow:

```bash
python scripts/build_store.py --config config.yaml
python scripts/train.py --config config.yaml --task pretrain
python scripts/probe.py --config config.yaml
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
- probe protocol: standard-scaled frozen embeddings plus L-BFGS linear probe
- uplift protocol: T-learner over frozen embeddings

## Training Paths

- `scripts/build_store.py` builds the LMDB + Parquet pretraining store.
- `scripts/train.py --task pretrain` runs the masked-model pretraining path with Muon + AdamW or plain AdamW, periodic checkpoints, and resume support.
- `scripts/train.py --task finetune` runs LoRA fine-tuning for `binary`, `regression`, `ranking`, or `multiclass` tasks with the same checkpoint/resume path.
- `scripts/benchmark.py` measures packed-event throughput across attention backends.
- `scripts/probe.py` runs frozen embedding probes, including uplift evaluation.
- `scripts/infer.py` loads a fine-tuned checkpoint and writes predictions to JSON.

## Resume And DDP

Resume from the latest local checkpoint:

```bash
python scripts/train.py --config config.yaml --task pretrain --resume artifacts/pretrain/checkpoints/pretrain_latest.pt
```

Run single-node distributed pretraining:

```bash
torchrun --nproc_per_node=4 scripts/train.py --config config.yaml --task pretrain
```
