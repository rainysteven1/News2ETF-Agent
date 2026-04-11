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
- `runtime/agent/models/`
- `runtime/agent/data/inputs/`

You can override roots with environment variables:

- `NEWS2ETF_RUNTIME_ROOT`
- `NEWS2ETF_SHARED_DATA_ROOT`
- `NEWS2ETF_CONFIG_PATH`
- `NEWS2ETF_REPO_ROOT`

## Default Commands

Run the runtime app directly:

```bash
./.venv/bin/python runtime/agent/main.py --help
./.venv/bin/python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
./.venv/bin/python runtime/agent/main.py diagnose-backtest --run-id bt_example
./.venv/bin/python runtime/agent/main.py visualize-backtest --run-id bt_example
```

## Artifact Migration

The runtime app now expects its deployable artifacts to live under `runtime/agent/` instead of `trainer/`.

Default runtime-local locations:

- `runtime/agent/data/inputs/sentiment_weekly.parquet`
- `runtime/agent/models/major/`
- `runtime/agent/models/sub/0407-1415/`
- `runtime/agent/models/signals/final-3y/`

Helpful commands:

```bash
# Check whether the runtime-local artifacts already exist
just runtime-check-artifacts

# Copy the legacy trainer outputs into runtime/agent once
just runtime-migrate-artifacts
```

For the current runtime-only backtest flow, the hard requirements are:

- `runtime/agent/data/inputs/sentiment_weekly.parquet`
- `runtime/agent/models/signals/final-3y/`

The legacy FinBERT / SetFit ONNX bundles under `runtime/agent/models/major/` and
`runtime/agent/models/sub/0407-1415/` are optional. They are only needed if you
want runtime to fall back to raw-news ONNX labeling instead of using precomputed
labels / sentiment inputs.

## Docker

Build the runtime-oriented image from the repository root:

```bash
docker build -f runtime/agent/Dockerfile -t news2etf-agent .
```

The Docker image is built in two stages:

- builder stage: uses `uv export` to turn the locked `runtime` dependency group from `pyproject.toml` / `uv.lock` into an explicit `requirements.txt`, then installs that into `/opt/venv`
- final stage: copies the prepared runtime virtualenv into a plain `python:3.11-slim` image

That flow is a little more verbose than `uv sync --only-group runtime`, but it is easier to reason about inside Docker:

- `pyproject.toml` stays the single dependency source of truth
- `uv.lock` still pins the resolved versions
- the image build installs from one exported requirements file, so the final dependency input is visible and reproducible

For day-to-day usage, the `just` recipes now call `runtime/agent/scripts/docker_backtest.sh`, so the long shared `docker run` arguments only live in one place.

Run a 2024 backtest in Docker:

```bash
docker run -d --rm --network host \
  --env-file .env \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/news2etf-home \
  -e XDG_CACHE_HOME=/tmp/news2etf-cache \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e NEWS2ETF_REPO_ROOT=/app \
  -e NEWS2ETF_RUNTIME_ROOT=/app/runtime/agent \
  -e NEWS2ETF_SHARED_DATA_ROOT=/app/data \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/runtime/agent/models:/app/runtime/agent/models" \
  -v "$(pwd)/runtime/agent/data/inputs:/app/runtime/agent/data/inputs" \
  -v "$(pwd)/runtime/agent/checkpoints:/app/runtime/agent/checkpoints" \
  -v "$(pwd)/runtime/agent/data:/app/runtime/agent/data" \
  -v "$(pwd)/runtime/agent/wandb:/app/runtime/agent/wandb" \
  news2etf-agent \
  python main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
```

Resume the latest checkpoint for an existing run:

```bash
docker run -d --rm --network host \
  --env-file .env \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/news2etf-home \
  -e XDG_CACHE_HOME=/tmp/news2etf-cache \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e NEWS2ETF_REPO_ROOT=/app \
  -e NEWS2ETF_RUNTIME_ROOT=/app/runtime/agent \
  -e NEWS2ETF_SHARED_DATA_ROOT=/app/data \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/runtime/agent/models:/app/runtime/agent/models" \
  -v "$(pwd)/runtime/agent/data/inputs:/app/runtime/agent/data/inputs" \
  -v "$(pwd)/runtime/agent/checkpoints:/app/runtime/agent/checkpoints" \
  -v "$(pwd)/runtime/agent/data:/app/runtime/agent/data" \
  -v "$(pwd)/runtime/agent/wandb:/app/runtime/agent/wandb" \
  news2etf-agent \
  python main.py backtest --start-date 2024-01-01 --end-date 2024-12-31 --run-id bt_xxxxxxxx --resume-latest
```

Resume only part of a date range:

```bash
docker run -d --rm --network host \
  --env-file .env \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/news2etf-home \
  -e XDG_CACHE_HOME=/tmp/news2etf-cache \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e NEWS2ETF_REPO_ROOT=/app \
  -e NEWS2ETF_RUNTIME_ROOT=/app/runtime/agent \
  -e NEWS2ETF_SHARED_DATA_ROOT=/app/data \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/runtime/agent/models:/app/runtime/agent/models" \
  -v "$(pwd)/runtime/agent/data/inputs:/app/runtime/agent/data/inputs" \
  -v "$(pwd)/runtime/agent/checkpoints:/app/runtime/agent/checkpoints" \
  -v "$(pwd)/runtime/agent/data:/app/runtime/agent/data" \
  -v "$(pwd)/runtime/agent/wandb:/app/runtime/agent/wandb" \
  news2etf-agent \
  python main.py backtest \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --run-id bt_xxxxxxxx \
    --resume-from-week 2024-06-03 \
    --resume-to-week 2024-07-01
```

Common `just` shortcuts:

```bash
just docker-build-runtime
just docker-backtest 2024-01-01 2024-12-31
just docker-backtest-run bt_demo 2024-01-01 2024-12-31
just docker-backtest-2024
```

Note:

- `just docker-build-runtime` is the only helper that builds the image
- all other `just docker-*` commands now run against the existing `news2etf-agent` image without rebuilding
- rebuild only when you actually changed `runtime/agent/Dockerfile` or dependency inputs
- helper-backed runs now include `--network host` by default
- helper-backed runs bind the container process to your current host uid/gid instead of running as root
- helper-backed runs start the container in detached mode and print the container id
- helper-backed runs prefer `runtime/agent/.env` / `runtime/agent/.env.*`; if those do not exist, they fall back to the repo-root `.env`
- inspect progress with `docker logs -f <container_id>`

Notes:

- The image is runtime-oriented, but it still expects shared data and trained model artifacts to be mounted in.
- move your exported ONNX bundles into `runtime/agent/models/` manually before running Docker or local runtime-only flows.
- move your runtime sentiment parquet into `runtime/agent/data/inputs/sentiment_weekly.parquet` manually if you use the default config.
- Checkpoints and W&B state are mounted out so resume works across container restarts.
- The helper-backed `just` recipes mount the whole `runtime/agent/` directory so edited configs, prompts, checkpoints, logs, and W&B state are all visible inside the container.

You can also invoke the helper directly:

```bash
bash runtime/agent/scripts/docker_backtest.sh diagnose --run-id bt_xxxxxxxx
bash runtime/agent/scripts/docker_backtest.sh diagnose --run-id bt_xxxxxxxx --start-week 2024-06-03 --end-week 2024-07-01
```

## Minimal Container Workflow

If you just want the smallest useful loop, this is the common sequence:

```bash
# 1) Build the runtime image
just docker-build-runtime

# 2) Run a custom date-range backtest
just docker-backtest 2024-01-01 2024-12-31

# 3) Run the full 2024 backtest
just docker-backtest-2024

# 4) Run a custom run_id if you want repeatable checkpoint naming
just docker-backtest-run bt_demo 2024-01-01 2024-12-31
```

## Output Layout

By default the runtime app writes to:

- `runtime/agent/checkpoints/{run_id}/`
- `runtime/agent/data/backtest_results.parquet`
- `runtime/agent/data/backtest_metrics.parquet`
- `runtime/agent/data/onnx_cache/`
- `runtime/agent/wandb/`

Backtest runs also write an interactive Plotly dashboard by default:

- `runtime/agent/checkpoints/{run_id}/visualizations/report.html`
- `runtime/agent/checkpoints/{run_id}/visualizations/summary.json`
- `runtime/agent/checkpoints/{run_id}/visualizations/*.html`
- `runtime/agent/checkpoints/{run_id}/visualizations/*.png`

Charts include NAV / total value, weekly returns, drawdown, cash vs invested
weight, allocation drift, sector contribution, and sector return heatmap when
the corresponding result columns exist.

Disable automatic visualization for a run with:

```bash
./.venv/bin/python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31 --no-visualize
```

Log standalone visualization PNGs to the existing W&B run stored in
`runtime/agent/checkpoints/{run_id}/run_meta.json` as `wandb.Image` media with:

```bash
./.venv/bin/python runtime/agent/main.py visualize-backtest --run-id bt_example --upload-wandb
```

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
