# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

News2ETF-Agent is a financial ML system that uses LLM-driven analysis of news articles to generate weekly ETF trading signals. The system combines transformer models (FinBERT, SetFit) for sentiment classification with temporal CNN + LightGBM stacking for signal generation.

## Common Commands

```bash
# Environment setup (choose one)
just gpu-sync   # GPU training environment
just cpu-sync    # CPU-only environment

# Agent inference
python main.py decide --week 2023-06-12   # Single week decision (debug)
python main.py backtest --start-date 2021-01-01 --end-date 2023-12-31  # Walk-forward backtest

# Model training (via trainer CLI)
python -m trainer.main finbert train        # Train FinBERT sentiment model
python -m trainer.main setfit train         # Train SetFit sub-category classifiers
python -m trainer.main signals train         # Train TCN + LightGBM stacking pipeline

# ONNX export
python -m trainer.main finbert export-onnx --model-path ... --onnx-path ...
python -m trainer.main setfit export-onnx --model-path ... --onnx-path ...

# Inference pipeline
python -m trainer.main predict all          # Full FinBERT → SetFit pipeline
python -m trainer.main predict finbert      # Phase 1 only
python -m trainer.main predict setfit        # Phase 2 only

# Linting
ruff check .
ruff format .
```

## Architecture

```
src/
├── agent/              # LangGraph ReAct agent for trading decisions
│   ├── single_agent.py # Node functions (agent_node, decide_node, risk_check_node, tools_node)
│   ├── tools.py        # Tool implementations (read_market_news, compute_ml_signals, etc.)
│   ├── workflow.py     # LangGraph topology definition
│   └── state.py        # AgentState TypedDict
├── signals/            # ONNX inference pipeline
│   ├── onnx_inference.py   # FinBERT + SetFit ONNX batch inference
│   └── memos_retrieval.py # Historical memory retrieval from Memos API
├── backtest/           # Walk-forward backtesting engine
│   └── engine.py       # WalkForwardEngine (weekly rebalancing)
└── utils/
    └── industry_map.py # ETF/industry mapping with beta metadata

trainer/                # Model training code
├── finbert/            # FinBERT sentiment model (8 L1 classes + 3 sentiment)
├── setfit_module/      # SetFit sub-category classifier (47 sub-categories)
└── signals/            # TCN + LightGBM stacking pipeline
    ├── dataset.py      # WeeklySignalDataset with 6-channel TCN input
    └── train.py        # TCN pretrain → finetune → LightGBM stack
```

## Key Data Flow

```
Raw News → FinBERT ONNX (L1 category + sentiment) → SetFit ONNX (sub-category)
                                                            ↓
                                        TCN (5-day sequence, 6 channels) → LightGBM → Signals
                                                            ↓
                                        LangGraph Agent (ReAct) → Trade Decisions
                                                            ↓
                                        Walk-Forward Backtest Engine
```

## Configuration

- **Agent config**: `config.toml` (root) — data paths, model directories, LLM settings
- **Trainer config**: `trainer/config.toml` — training hyperparameters for FinBERT, SetFit, Signals

Model I/O specs are documented in `docs/model-io.md` (TCN 6-channel input, LightGBM 13 features, etc.).

## Three-Phase Data Split

- Phase 1 (60%): 2023-01 ~ 2024-09 — TCN + LightGBM training
- Phase 2 (20%): 2024-10 ~ 2025-06 — Agent dry-run / validation
- Phase 3 (20%): 2025-07 ~ 2026-03 — Final backtest (unseen data)

## Important Notes

- `src/config.py` loads `config.toml` and resolves relative paths to project root
- `trainer/config.py` is a separate config system for the trainer CLI (not used by main.py)
- ONNX models are used for inference (not PyTorch) — trained models must be exported first
- The agent uses an LLM for genuine reasoning; rules only handle guardrails and risk checks
