# PRAGMA Overview

This note explains what PRAGMA is trying to solve, what tasks it supports, what data the paper uses, and why it is different from earlier tabular, recommendation, and finance foundation models.

## 1. What Problem PRAGMA Is Solving

PRAGMA is built for a very specific problem:

- banks and fintechs have long, irregular, multi-source user histories
- those histories contain transactions, app activity, communications, product usage, and profile state
- most production ML systems still solve each downstream problem with a separate feature-engineering pipeline
- generic text LLM tokenization is a poor fit for structured financial events

In plain English, the paper is trying to replace this pattern:

- one fraud model with its own SQL features
- one credit model with another feature set
- one recommendation model with another pipeline
- one engagement model with another training stack

with this pattern:

- one shared encoder backbone trained on raw banking histories
- lightweight task-specific adaptation on top of that backbone

So PRAGMA is not just "a model for transactions."
It is a reusable representation layer for banking event histories.

## 2. Why The Problem Is Hard

The paper says multi-source banking histories are hard for three main reasons:

1. Each event is a variable-length record with mixed field types.
   Example: categorical fields, numerical amounts, free text, and timestamps all appear in one record.

2. Histories are long and irregular in time.
   Some users have only a few events; others have thousands.
   Events also have strong hourly, daily, and weekly patterns.

3. Real financial modeling has privacy, regulatory, and operational constraints.
   You cannot casually treat all raw data as plain text and hope a generic model learns the right inductive bias.

The paper argues that naive text serialization creates two major failures:

- it inflates sequence length because field names and delimiters become extra tokens
- it destroys useful numeric structure because amounts get broken into digit fragments

That is why PRAGMA uses a banking-native structured tokenization scheme instead of "just put rows into an LLM."

## 3. What PRAGMA Actually Is

PRAGMA is an encoder-only Transformer family for banking user histories.

The core design is:

- a **Profile State Encoder** for contextual attributes
- an **Event Encoder** for each event record
- a **History Encoder** that fuses profile state with the event sequence

The model family scales across three sizes:

- `PRAGMA-S`: 10M params
- `PRAGMA-M`: 100M params
- `PRAGMA-L`: 1B params

The paper chooses an **encoder-only** design because the goal is not open-ended generation.
The goal is strong transferable embeddings for discriminative finance tasks.

## 4. How PRAGMA Represents Data

PRAGMA treats each field as:

- a **key**
- a **value**
- a **time coordinate**

This is the heart of the method.

Example raw event:

```text
Type: card_payment
Currency: USD
Amount: 42.75
Merchant: netflix
Timestamp: 2024-04-07 19:20:18
```

PRAGMA would conceptually break this into key/value/time pieces such as:

- `key=Type`, `value=card_payment`, `time=2024-04-07 19:20:18`
- `key=Currency`, `value=USD`, `time=2024-04-07 19:20:18`
- `key=Amount`, `value=42.75`, `time=2024-04-07 19:20:18`
- `key=Merchant`, `value=netflix`, `time=2024-04-07 19:20:18`

So the model does not see one flat sentence.
It sees a structured event made of typed fields with values and temporal context.

### Keys

Each field name is tokenized as a semantic type token.
Examples:

- `Type`
- `Channel`
- `Currency`
- `Plan`

### Values

Values are tokenized by type:

- **numerical** values are bucketized by percentile
- **categorical** values become single tokens
- **text** values use BPE-style subword tokenization

### Time

Time is encoded in two ways:

- time since the most recent event using a soft log transform
- calendar cycles such as hour-of-day, day-of-week, and day-of-month

The paper also adds **life-long events** for milestone facts that may be very old but still important.
Examples:

- first top-up
- account age
- first key milestone on the platform

This matters because long histories are truncated, and early milestones would otherwise disappear.

## 5. What Data The Paper Uses

### Pretraining Data

PRAGMA is pretrained on **internal Revolut data**, not a public Kaggle or Hugging Face dataset.

The paper describes the pretraining corpus as:

- **26M user records**
- **111 countries**
- **24B events**
- **207B tokens**

The pretraining dataset is built at the **record level**:

- one observation is a pseudonymized user history
- that history is collected up to an **evaluation point**
- the record also includes contextual/profile attributes at that evaluation point

The event sources are grouped broadly into:

- transactions
- app
- trading
- communication

The profile state includes contextual attributes such as:

- balance quantile
- plan
- insurance state
- service region

The selected pretraining time range is:

- **25 months from 2023 to 2025**

The paper is explicit that:

- data is fully anonymized
- there is no personally identifiable information
- figures/examples shown in the paper are synthetic
- absolute downstream metrics are not disclosed for commercial sensitivity

### Downstream Data

The downstream datasets are also internal task-specific benchmarks.

For each downstream task, they build a record using:

- an identifier
- an evaluation point
- the history before that point
- the profile attributes available at that point

So the downstream datasets mirror the same basic record structure as pretraining.

## 6. What Tasks PRAGMA Solves

The paper evaluates one pretrained backbone across six main downstream tasks.

### 6.1 Credit Scoring

**Question:** Is this retail credit applicant likely to default?

The paper defines it as predicting the **probability of default within the first 12 months of use**.

Why it matters:

- lending decisions
- approval/risk ranking
- expected losses

Task type:

- binary classification

Metrics:

- ROC-AUC
- PR-AUC

### 6.2 Communication Engagement

**Question:** If we send a re-engagement message to a user who abandoned a credit application, will they open it?

This is not a pure credit-risk task.
It is more like CRM / funnel recovery prediction.

Why it matters:

- helps recover unfinished credit applications
- measures whether the history embedding captures behavior before the drop-off

Task type:

- binary classification

Metrics:

- ROC-AUC
- PR-AUC

### 6.3 External Fraud

**Question:** Does this case look fraudulent?

The paper presents this as a representative fraud detection problem.

Why it matters:

- operational fraud alerts
- blocking/review workflows

Task type:

- binary classification

Metrics:

- precision
- recall

### 6.4 Product Recommendation

**Question:** Which products is this user likely to adopt soon, conditioned on a communication?

This is not just "recommend the next item."
It is closer to conversion propensity across multiple products.

Why it matters:

- cross-sell
- targeting
- campaign prioritization

Task type:

- multilabel classification

Metric:

- mAP

### 6.5 Recurrent Transactions

**Question:** Is this transaction part of a recurring subscription pattern that will repeat next month?

Example intuition:

- rent
- subscription
- recurring utility bill

Why it matters:

- subscription detection
- budget analysis
- cash-flow understanding

Task type:

- binary classification

Metric:

- macro F1

### 6.6 Lifetime Value (LTV)

**Question:** Is this user likely to generate positive gross profit over a future horizon?

The paper notes this task is hard because users often have short observed histories but a much longer prediction horizon.

Why it matters:

- customer prioritization
- retention
- growth strategy

Task type:

- binary classification

Metrics:

- ROC-AUC
- PR-AUC

## 7. What Value PRAGMA Adds Beyond "One Model For Many Tasks"

That is part of the value, but it is not the whole value.

The paper is really making several claims at once.

### 7.1 Shared Representation Learning

Instead of training six independent models from scratch, PRAGMA learns one reusable behavioral representation and adapts it.

### 7.2 Less Hand-Crafted Feature Engineering

The model is intended to learn directly from raw event histories and profile state instead of relying on bespoke SQL features for each downstream team.

### 7.3 Better Inductive Bias For Banking Data

PRAGMA is designed for:

- mixed-type event records
- long irregular timelines
- explicit profile state
- milestone events
- temporal cycles

This is different from forcing banking data through generic text or flat-tabular assumptions.

### 7.4 Cheap Adaptation

The paper uses two adaptation paths:

- frozen embedding probe with a linear head
- LoRA fine-tuning with only about 2-4% trainable parameter overhead

That means the pretrained backbone can be reused cheaply across tasks.

### 7.5 Better Performance On High-Value Tasks

The headline claim is not only convenience.
The paper reports strong relative gains over internal baselines, especially on hard, sparse tasks like credit scoring and communication engagement.

## 8. How PRAGMA Differs From Other Dataset Shapes

The PRAGMA data shape is very different from a normal public benchmark CSV.

### PRAGMA-style dataset shape

A PRAGMA-style dataset is:

- user-level
- multi-source
- event-based
- irregular in time
- profile-aware
- built around evaluation-point snapshots

One record is not just a row.
It is:

- profile state at a cutoff
- optional life-long milestones
- all events before the cutoff
- a downstream label defined after or at that cutoff

### What it is not

It is not:

- a single fixed-schema tabular row
- a transaction table without user-level state
- a text corpus
- a pure recommendation interaction list
- a single-task benchmark with one narrow label type

This is why most public datasets only partially match PRAGMA.

## 9. How PRAGMA Differs From Other Models

### Compared with generic text LLMs

PRAGMA is different because:

- it does not serialize everything as text by default
- it preserves numeric and field structure
- it is encoder-only, not chat/generation-first
- it is optimized for discriminative finance tasks

### Compared with tabular Transformers

Models like TabTransformer or FT-Transformer operate on fixed-schema tables.
PRAGMA instead models:

- variable-length histories
- multiple event sources
- explicit time structure
- profile state plus event history together

### Compared with recommender models

Sequential recommendation models often reduce history to item interactions.
PRAGMA models richer events with:

- typed fields
- amounts
- text
- time gaps
- profile context

and is evaluated on more than recommendation alone.

### Compared with finance text models

FinBERT, BloombergGPT, and FinGPT are mainly about financial language.
PRAGMA is mainly about structured banking event histories.

### Compared with newer transaction-ledger models

The paper positions PRAGMA as broader than models such as:

- **nuFormer**, which is highlighted mainly around product recommendation and risk-style transaction modeling
- **TransactionGPT**, which focuses on anomaly detection and trajectory generation

PRAGMA's distinguishing claims are:

- multi-source banking data
- explicit profile state
- one backbone across six downstream tasks
- encoder-style embeddings with lightweight adaptation

## 10. Important Limitations In The Paper

PRAGMA is strong, but the paper is not claiming it solves every finance problem equally well.

### AML limitation

The paper reports that PRAGMA performs poorly on Anti-Money Laundering relative to a network-aware production baseline.

Why:

- AML is highly relational
- cross-record and graph structure matter
- PRAGMA processes histories mostly in isolation

This is important because it shows PRAGMA is not a universal replacement for every financial model.

### Text encoder is optional, not default

The paper experiments with a frozen pretrained text encoder (Nemotron) for text-heavy settings.
It helps some tasks, especially credit scoring, but hurts product recommendation and increases training latency.
So it is an optional extension, not the core default architecture.

## 11. What This Means For Our Repo

If we want this repo to be truly PRAGMA-like, the target is not just "train on transactions."

We need data that looks like:

- one user record per evaluation point
- multiple event modalities
- contextual profile state
- optional life-long milestone events
- long pre-cutoff histories
- downstream labels attached to the same record format

That is why public datasets like MBD/MBD-mini are useful for us: not because they are identical to Revolut data, but because they are much closer to the required record shape than a flat fraud CSV or a credit-score-only table.

## 12. Source Notes

This note is based primarily on:

- the bundled PRAGMA paper source in `research/paper_assets/paper_src/`
- the PRAGMA arXiv paper
- the NVIDIA Revolut transaction foundation model case study
