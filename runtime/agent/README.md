# Runtime Agent

`runtime/agent` is the standalone runtime app for the ETF agent.
The repository no longer keeps a root-level `src/` or `main.py` runtime copy.

## What Lives Here

- `main.py`: runtime CLI entrypoint
- `config.toml`: runtime config
- `config/prompts/`: runtime prompt templates
- `src/`: runtime source tree
- `tests/`: runtime-specific migration and layout tests
- `checkpoints/`: per-run checkpoints and logs
- `data/`: runtime outputs and caches
- `wandb/`: runtime W&B directory

## Shared Inputs

The runtime app still reads shared inputs from the repository root by default:

- `data/converted/`
- `data/meta_sector_mapping.json`
- `data/industry_dict.json`
- `trainer/models/`

You can override roots with environment variables:

- `NEWS2ETF_RUNTIME_ROOT`
- `NEWS2ETF_SHARED_DATA_ROOT`
- `NEWS2ETF_TRAINER_ROOT`
- `NEWS2ETF_CONFIG_PATH`
- `NEWS2ETF_REPO_ROOT`

## Default Commands

Run the runtime app directly:

```bash
./.venv/bin/python runtime/agent/main.py --help
./.venv/bin/python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
./.venv/bin/python runtime/agent/main.py diagnose-backtest --run-id bt_example
```

## Output Layout

By default the runtime app writes to:

- `runtime/agent/checkpoints/{run_id}/`
- `runtime/agent/data/backtest_results.parquet`
- `runtime/agent/data/backtest_metrics.parquet`
- `runtime/agent/data/onnx_cache/`
- `runtime/agent/wandb/`

## Test Commands

Runtime-only tests:

```bash
./.venv/bin/pytest -q runtime/agent/tests
```

Core migration regression:

```bash
./.venv/bin/pytest -q \
  tests/test_agent_state.py \
  tests/test_prompt_manager.py \
  tests/test_single_agent.py \
  tests/test_tools.py \
  tests/test_logger.py \
  tests/test_config.py \
  tests/test_workflow.py \
  tests/test_backtest_engine.py \
  tests/test_backtest_diagnostics.py \
  tests/test_features.py \
  tests/test_news_loader.py \
  tests/test_meta_sector_map.py \
  tests/test_etf_universe.py \
  runtime/agent/tests
```

## Repository Layout

The repository is now organized around two real subprojects:

- `trainer/`
- `runtime/agent/`

Shared raw data and trained model artifacts still intentionally live at the repository root.
