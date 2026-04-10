# Hopper Architecture

## 1. Purpose

This document is the single source of truth for the `signals -> infer -> agent -> backtest` pipeline.

The goal is not just to train a model. The goal is to:

1. train a deployable `signals` model,
2. run inference on held-out / future dates with a fixed model,
3. let the Agent consume only online-available context,
4. evaluate the Agent with backtests that do not leak training labels.

This document therefore separates four concerns that were previously mixed together:

- training artifacts,
- inference artifacts,
- Agent decision context,
- backtest evaluation inputs.

If a future change touches `signals`, `agent`, or `backtest`, this document should be updated together with code.

---

## 2. Scope

This document covers:

- the `signals` training pipeline,
- ONNX export and runtime inference,
- Agent feature consumption,
- weekly backtest input semantics,
- the recommended data split for a 4-year history,
- the standard operational commands.

This document does not try to fully specify:

- prompt wording details,
- LLM provider choice,
- every possible research workflow variant,
- day-level guardrail policy details.

Those are downstream concerns. The primary contract here is the data and model flow.

---

## 3. System Overview

The production path is:

```text
monthly sub-inference parquet files
  -> raw.parquet
  -> signals dataset cache
  -> signals train
      -> checkpoint
      -> full-history agent_features.parquet
      -> OOF agent_features.oof.parquet
  -> signals export-onnx
      -> ONNX bundle
  -> signals infer
      -> held-out / future agent_features.oof.parquet
  -> agent
      -> decision_context
      -> level1_plan + level2_plan
  -> backtest
      -> weekly PnL / holdings / meta-sector attribution
```

The core design rule is:

> Agent and backtest should consume inference outputs, not training-time labels or in-memory training objects.

---

## 4. Current Architecture

### 4.1 Signals

The maintained `signals` path is:

- canonical small-category space from [`data/label_stats.json`](/home/nn/workspace/New2ETF-Agent/data/label_stats.json),
- fan-in TCN over canonical sub-categories,
- one LightGBM per meta sector,
- IsolationForest for abnormal news heat,
- manual ONNX deployment bundle export.

Current training outputs include:

- `tcn_fanin.pt`
- `lgbm/*.txt`
- `iforest_model.pkl`
- `signals_checkpoint.json`
- `latest.txt`
- `data/agent_features.parquet`
- `data/agent_features.oof.parquet`

`signals train` saves a self-contained checkpoint. `signals export-onnx` converts that checkpoint into a deployment bundle containing:

- `tcn.onnx`
- best-effort `lgbm/*.onnx`
- best-effort `iforest.onnx`
- `manifest.json`

### 4.2 Agent

The Agent is meta-sector first:

- Level 1: `level1_plan[]` for 8 meta sectors,
- Level 2: `level2_plan[]` for ETF selection under approved sectors,
- `build_decision_context()` builds the shared runtime context,
- the trader prompt consumes that shared context,
- the Agent prefers OOF / infer feature parquet over fallback computation.

### 4.3 Backtest

The backtest path is no longer industry-first.

Current backtest semantics:

- holdings are `meta_sector -> weight`,
- selected ETFs are stored explicitly,
- weekly returns are computed from the selected ETF when available,
- results persist:
  - `meta_sector_returns`
  - `meta_sector_contributions`
  - `selected_etfs`

---

## 5. Data Contracts

### 5.1 Canonical Sub-Category Space

Canonical sub-category order comes from [`data/label_stats.json`](/home/nn/workspace/New2ETF-Agent/data/label_stats.json).

Current contract:

- training input dimensionality is fixed by canonical sub-categories,
- missing categories in a given year remain zero columns,
- dimensions must not shrink just because a subset of years only contains fewer active categories.

This is required so that:

- training shape stays stable,
- ONNX export shape stays stable,
- Agent inference schema stays stable across years.

### 5.2 Meta-Sector Mapping

Meta-sector grouping comes from [`data/meta_sector_mapping.json`](/home/nn/workspace/New2ETF-Agent/data/meta_sector_mapping.json).

This file defines:

- the 8 meta sectors,
- sub-category membership,
- weighting hints,
- global leader links.

No hidden second mapping should exist in code.

### 5.3 Signals Raw Input

The training input path in [`trainer/config.toml`](/home/nn/workspace/New2ETF-Agent/trainer/config.toml) is:

- `signals.dataset.raw_data_path = ./trainer/data/labeled/signals/raw.parquet`

If `raw.parquet` does not exist, the dataset layer now rebuilds it automatically from:

- [`trainer/data/labeled/signals/raw/.raw_sub_monthly_checkpoints`](/home/nn/workspace/New2ETF-Agent/trainer/data/labeled/signals/raw/.raw_sub_monthly_checkpoints)

but only after validating:

- every monthly parquet is readable,
- month sequence is contiguous,
- no missing month exists inside the detected range.

This is intentional. Training should fail early on a broken raw source instead of silently training on partial data.

Operationally, this means:

- monthly checkpoint validation is a built-in dataset precondition,
- `signals train` is allowed to trigger the rebuild automatically,
- manual checkpoint validation is a debugging tool, not part of the standard production runbook.

---

## 6. Model Contracts

### 6.1 TCN Input / Output

The maintained TCN contract is:

```text
input shape:
  (batch, seq_len=10, n_sub=46, channels=6)

output shape:
  (batch, 8)
```

The 6 input channels are:

- `sentiment_ema`
- `sentiment_acceleration`
- `sentiment_std`
- `log_news_count`
- `event_type_score`
- `sentiment_vs_price_residual`

The output is an 8-meta-sector forward score aligned to the configured target mode.

### 6.2 TCN Target

The maintained target is:

- `meta_excess_return` by default

Configured in:

- [`trainer/config.toml`](/home/nn/workspace/New2ETF-Agent/trainer/config.toml)
- [`trainer/src/config/signals.py`](/home/nn/workspace/New2ETF-Agent/trainer/src/config/signals.py)

The project should not revert to “future sentiment change” as the primary target.

### 6.3 LightGBM

LightGBM remains one model per meta sector.

Critical rule:

> Every LightGBM feature must be available both at training time and at runtime inference time.

That means:

- no target-derived features,
- no future-return leakage,
- no train-only convenience features that do not exist in infer mode.

### 6.4 IForest

IForest is treated as a news heat / anomaly detector.

The exact tensor shape is not the public contract.
The contract is:

- the Agent and LightGBM can consume a stable abnormal-news signal per date,
- the inference bundle can reproduce it consistently enough for runtime use.

### 6.5 SHAP

SHAP is for offline inspection only.

It is allowed to answer:

- which features dominate a LightGBM,
- whether TCN-derived features are overwhelming the model,
- whether ablation is needed.

It is not allowed to directly drive live decisions.

---

## 7. Inference Artifacts

There are two feature artifacts with different purposes:

### 7.1 `data/agent_features.parquet`

This is the full-history export produced after training.

Use cases:

- offline debugging,
- schema inspection,
- feature QA,
- quick smoke checks.

This file should not be the default input for held-out evaluation if a cleaner OOF / infer artifact exists.

### 7.2 `data/agent_features.oof.parquet`

This is the preferred runtime feature artifact.

Use cases:

- held-out validation,
- future-period inference,
- Agent runtime feature cache,
- backtest input.

Current code already prefers this file in [`src/agent/features.py`](/home/nn/workspace/New2ETF-Agent/src/agent/features.py).

---

## 8. Agent Context Contract

The Agent decision context is assembled by [`build_decision_context()`](/home/nn/workspace/New2ETF-Agent/src/agent/tools.py).

Current context categories are:

### 8.1 Model Context

Derived from `agent_features.oof.parquet` or ONNX inference:

- `tcn_sequence`
- `ml_signal_snapshot`
- `sent_p_divergence`
- residual and stability features

### 8.2 News Context

Derived from raw news and ONNX classifiers:

- `news_summary`
- sector/top-news views
- inferred major/sub/sentiment labels

### 8.3 Market Context

Derived from ETF price aggregates:

- market return
- market volatility
- `volume_ratio`
- market state label

### 8.4 Position / Performance Context

Derived from backtest state history:

- holdings
- weekly returns
- `agent_perf_1w`
- `agent_perf_4w`
- top holdings

### 8.5 Memory Context

Derived from:

- `PromptManager`
- `Memos`

This includes:

- good patterns
- bad patterns
- reasoning summary
- retrieved similar cases

### 8.6 Context Output Format

`build_decision_context()` now returns:

- structured JSON,
- `schema_version`,
- `human_summary`

The trader prompt uses the same schema and prepends `human_summary` for readability.

---

## 9. Backtest Contract

Backtest now validates:

- decisions made from online-available context,
- not direct access to training labels,
- not hidden dependence on training-time Python objects.

The backtest should be interpreted as:

```text
Agent consumes infer/OOF features
  + news / memory / market / holdings
  -> produces meta-sector allocation and ETF selection
  -> weekly backtest evaluates resulting decisions
```

This is the correct evaluation path for the system.

The backtest is not intended to answer:

- “how low was train loss?”
- “how good is TCN alone?”

It is intended to answer:

- “how does the decision system behave when only inference-time inputs are available?”

---

## 10. Recommended 4-Year Data Split

Assume the raw monthly history covers:

- `2021-01-01` to `2024-12-31`

The recommended split is:

### 10.1 Development Split

Use this to debug and compare model quality:

- `2021-01-01` to `2022-12-31`: training
- `2023-01-01` to `2023-12-31`: validation / OOF
- `2024-01-01` to `2024-12-31`: untouched holdout

This is the `2 + 1 + 1` split.

### 10.2 Final Deployment Split

After model choices are fixed:

- `2021-01-01` to `2023-12-31`: final training
- `2024-01-01` to `2024-12-31`: pure inference + Agent + backtest

This is the `3 + 1` split.

This is the split that should feed the final Agent / backtest run.

### 10.3 Why Not Train on All 4 Years?

Because then the Agent would be evaluated on dates the model has already seen during training, which contaminates the interpretation of backtest performance.

---

## 11. Standard Operating Commands

The standard commands are defined in [`justfile`](/home/nn/workspace/New2ETF-Agent/justfile).

### 11.1 Environment

GPU environment:

```bash
just gpu-sync
```

CPU environment:

```bash
just cpu-sync
```

### 11.2 Raw Dataset Behavior

`signals train` expects:

- `trainer/data/labeled/signals/raw.parquet`, or
- a valid monthly checkpoint directory under `trainer/data/labeled/signals/raw/.raw_sub_monthly_checkpoints`

If `raw.parquet` is missing, the dataset layer will:

1. validate monthly parquet continuity,
2. validate readability of every monthly parquet,
3. rebuild `raw.parquet`,
4. continue training.

This step is automatic and is not treated as a separate standard command.

### 11.3 Development-Phase Signals Training

Run the `2 + 1 + 1` development split:

```bash
just signals-train-dev-2y1y
```

### 11.4 Final Signals Training

Run the final `3 + 1` split:

```bash
just signals-train-final-3y
```

### 11.5 Manual ONNX Export

Export a deployable bundle from the saved checkpoint:

```bash
just signals-export-onnx-final-3y
```

The development split can also be exported manually:

```bash
just signals-export-onnx-dev-2y1y
```

### 11.6 Pure 2024 Inference

Generate held-out runtime features:

```bash
just signals-infer-2024
```

### 11.7 Agent Backtest on Held-Out Data

Run weekly backtest on 2024 only:

```bash
just backtest-2024
```

### 11.8 One-Pass Final Pipeline

For the final 2024 evaluation path:

```bash
just signals-agent-pipeline-2024
```

This expands to:

1. train the final 3-year model,
2. infer 2024 features,
3. run the 2024 backtest.

---

## 12. Validation Checklist

Before trusting a run, verify all of the following.

### 12.1 Data

- monthly parquet files are contiguous,
- every monthly parquet is readable,
- `raw.parquet` exists or can be rebuilt,
- canonical sub-category count matches `label_stats.json`,
- no unexpected schema drift exists in raw news features.

### 12.2 Training

- `signals train` finishes without shape mismatch,
- walk-forward metrics are logged,
- no fake metric inflation from constant targets or label leakage,
- checkpoint metadata is exported,
- OOF file is exported.

### 12.3 Inference

- `signals infer` writes only the intended date range,
- `agent_features.oof.parquet` is populated,
- Agent prefers OOF / infer features,
- ONNX and reference feature values remain within acceptable tolerance.

Use:

```bash
python scripts/check_signals_onnx_consistency.py --help
```

### 12.4 Agent / Backtest

- `decision_context` contains model, news, market, position, and memory sections,
- `human_summary` is present,
- holdings are meta-sector based,
- selected ETFs are recorded,
- backtest output contains `meta_sector_returns` and `meta_sector_contributions`.

---

## 13. Current Implementation Status

As of the current codebase:

- `signals train` saves a self-contained checkpoint,
- `signals export-onnx` exports a deployable ONNX bundle,
- `signals infer` exists as an explicit CLI command,
- `AgentFeatureBuilder` prefers `agent_features.oof.parquet`,
- Agent features can be incrementally refreshed up to a target date,
- `build_decision_context()` is standardized,
- backtest is meta-sector based,
- monthly raw checkpoint validation and rebuild are available.

The system is now past the “wiring is missing” phase.

The main remaining improvement areas are:

1. stronger walk-forward / OOF rigor,
2. better ETF-level attribution and reporting,
3. continued cleanup of downstream guardrail/logging placeholders,
4. model quality improvement rather than pipeline plumbing.

---

## 14. Non-Negotiable Rules

1. Do not evaluate Agent/backtest on dates that were used for final training.
2. Do not let Agent consume training labels or future-derived features.
3. Do not shrink canonical input dimensions just because a subset of years has fewer active sub-categories.
4. Do not maintain parallel hidden mappings outside `label_stats.json` and `meta_sector_mapping.json`.
5. Do not treat full-history training export as equivalent to held-out inference export.

If a change breaks one of these rules, the change is architecturally incorrect even if it improves a local metric.
