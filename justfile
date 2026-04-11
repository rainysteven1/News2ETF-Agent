set dotenv-load := true
set dotenv-filename := ".env"

root := justfile_directory()
docker_image := "news2etf-agent"
docker_backtest_script := "runtime/agent/scripts/docker_backtest.sh"
docker_run := "bash " + docker_backtest_script + " --image " + docker_image + " --skip-build"

# ── Dev environment ──────────────────────────────────────────────────────────────

cpu-sync:
    uv sync --group dev --group trainer --group torch_cpu --no-group torch_gpu 

gpu-sync:
    uv sync --group dev --group torch_gpu --no-group torch_cpu

# ── Inference / Debug ───────────────────────────────────────────────────────────

decide week:
    python runtime/agent/main.py decide --week {{ week }}

backtest start end:
    python runtime/agent/main.py backtest --start-date {{ start }} --end-date {{ end }}

upload-backtest-viz run_id:
    ./.venv/bin/python runtime/agent/main.py visualize-backtest --run-id {{ run_id }} --upload-wandb

# ── Trainer CLI (trainer/main.py) ──────────────────────────────────────────────

major-train:
    ./.venv/bin/python -m trainer.main major train

setfit-train:
    ./.venv/bin/python -m trainer.main sub setfit train

signals-train:
    ./.venv/bin/python -m trainer.main signals train

# ── Signals / Agent / Backtest (4-year split) ────────────────────────────────

# Recommended development split:
#   - 2021-2022: train
#   - 2023: validation / OOF
#   - 2024: holdout for agent + backtest
signals-train-dev-2y1y:
    tmp_cfg="$(mktemp trainer/.tmp-signals-dev-2y1y-XXXX.toml)"; python -c "import re; from pathlib import Path; src = Path('trainer/config.toml').read_text(encoding='utf-8'); src = re.sub(r'train_end_week = \".*\"', 'train_end_week = \"2022-12-31\"', src, count=1); src = re.sub(r'output_checkpoint = \".*\"', 'output_checkpoint = \"./trainer/checkpoints/signals/dev-2y1y\"', src, count=1); Path('$tmp_cfg').write_text(src, encoding='utf-8')"; TRAINER_CONFIG_PATH="$tmp_cfg" ./.venv/bin/python -m trainer.main signals train --force

# Final model for agent / backtest:
#   - 2021-2023: train
#   - 2024: pure inference / agent / backtest
signals-train-final-3y:
    tmp_cfg="$(mktemp trainer/.tmp-signals-final-3y-XXXX.toml)"; python -c "import re; from pathlib import Path; src = Path('trainer/config.toml').read_text(encoding='utf-8'); src = re.sub(r'train_end_week = \".*\"', 'train_end_week = \"2023-12-31\"', src, count=1); src = re.sub(r'output_checkpoint = \".*\"', 'output_checkpoint = \"./trainer/checkpoints/signals/final-3y\"', src, count=1); Path('$tmp_cfg').write_text(src, encoding='utf-8')"; TRAINER_CONFIG_PATH="$tmp_cfg" ./.venv/bin/python -m trainer.main signals train --force

# Export the development split checkpoint to a deployable ONNX bundle
signals-export-onnx-dev-2y1y:
    ./.venv/bin/python -m trainer.main signals export-onnx \
        --checkpoint-dir ./trainer/checkpoints/signals/dev-2y1y \
        --bundle-dir ./trainer/models/signals/dev-2y1y

# Export the final 3-year checkpoint to a deployable ONNX bundle
signals-export-onnx-final-3y:
    ./.venv/bin/python -m trainer.main signals export-onnx \
        --checkpoint-dir ./trainer/checkpoints/signals/final-3y \
        --bundle-dir ./trainer/models/signals/final-3y

# Run pure 2024 inference with the final 3-year model and export OOF/infer features for agent/backtest
signals-infer-2024:
    ./.venv/bin/python -m trainer.main signals infer \
        --bundle-dir ./trainer/models/signals/final-3y \
        --output-path ./data/agent_features.oof.parquet \
        --start-date 2024-01-01 \
        --end-date 2024-12-31

# Run weekly backtest with the agent consuming only 2024 inference outputs
backtest-2024:
    ./.venv/bin/python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31

# Inspect whether runtime-local model and sentiment artifacts are in place
runtime-check-artifacts:
    @echo "runtime artifact check"
    @{ [ -e runtime/agent/data/inputs/sentiment_weekly.parquet ] && echo "  REQ  runtime/agent/data/inputs/sentiment_weekly.parquet" || echo "  MISS runtime/agent/data/inputs/sentiment_weekly.parquet"; \
       [ -e runtime/agent/models/signals/final-3y/manifest.json ] && echo "  REQ  runtime/agent/models/signals/final-3y/manifest.json" || echo "  MISS runtime/agent/models/signals/final-3y/manifest.json"; \
       [ -e runtime/agent/models/major/best.onnx ] && echo "  OPT  runtime/agent/models/major/best.onnx" || echo "  OPT  runtime/agent/models/major/best.onnx (missing, only needed for raw news ONNX labeling)"; \
       [ -e runtime/agent/models/sub/0407-1415 ] && echo "  OPT  runtime/agent/models/sub/0407-1415" || echo "  OPT  runtime/agent/models/sub/0407-1415 (missing, only needed for raw news ONNX labeling)"; \
       [ -e runtime/agent/data/inputs/sentiment_weekly.parquet ] && [ -e runtime/agent/models/signals/final-3y/manifest.json ]; }

# One-shot helper to copy legacy trainer artifacts into runtime/agent
runtime-migrate-artifacts:
    @echo "migrating runtime artifacts"
    @mkdir -p runtime/agent/data/inputs runtime/agent/models/major runtime/agent/models/sub/0407-1415 runtime/agent/models/signals/final-3y
    @if [ -f trainer/data/labeled/signals/sentiment_weekly.parquet ]; then echo "  REQ  trainer/data/labeled/signals/sentiment_weekly.parquet -> runtime/agent/data/inputs/sentiment_weekly.parquet"; cp trainer/data/labeled/signals/sentiment_weekly.parquet runtime/agent/data/inputs/sentiment_weekly.parquet; else echo "  MISS trainer/data/labeled/signals/sentiment_weekly.parquet -> runtime/agent/data/inputs/sentiment_weekly.parquet"; fi
    @if [ -d trainer/models/signals/final-3y ]; then echo "  REQ  trainer/models/signals/final-3y -> runtime/agent/models/signals/final-3y"; cp -R trainer/models/signals/final-3y/. runtime/agent/models/signals/final-3y/; else echo "  MISS trainer/models/signals/final-3y -> runtime/agent/models/signals/final-3y"; fi
    @if [ -d trainer/models/major ]; then echo "  OPT  trainer/models/major -> runtime/agent/models/major"; cp -R trainer/models/major/. runtime/agent/models/major/; else echo "  OPT  trainer/models/major -> runtime/agent/models/major (missing, only needed for raw news ONNX labeling)"; fi
    @if [ -d trainer/models/sub/0407-1415 ]; then echo "  OPT  trainer/models/sub/0407-1415 -> runtime/agent/models/sub/0407-1415"; cp -R trainer/models/sub/0407-1415/. runtime/agent/models/sub/0407-1415/; else echo "  OPT  trainer/models/sub/0407-1415 -> runtime/agent/models/sub/0407-1415 (missing, only needed for raw news ONNX labeling)"; fi

# Build the runtime-oriented Docker image
docker-build-runtime: runtime-check-artifacts
    docker build --network host -f runtime/agent/Dockerfile -t {{ docker_image }} .

# Run a Docker backtest for any date range
docker-backtest start_date end_date:
    {{ docker_run }} --start-date {{ start_date }} --end-date {{ end_date }}

# Run a Docker backtest for any date range with an explicit run_id
docker-backtest-run run_id start_date end_date:
    {{ docker_run }} --run-id {{ run_id }} --start-date {{ start_date }} --end-date {{ end_date }}

# Run the 2024 backtest inside Docker using an already-built image
docker-backtest-2024:
    {{ docker_run }} --start-date 2024-01-01 --end-date 2024-12-31

# Recommended one-pass order for a 4-year dataset:
#   1) Evaluate the development split: just signals-train-dev-2y1y
#   2) Export the dev ONNX bundle if needed: just signals-export-onnx-dev-2y1y
#   3) Retrain the final 3-year model: just signals-train-final-3y
#   4) Export the final ONNX bundle: just signals-export-onnx-final-3y
#   5) Export pure 2024 inference features: just signals-infer-2024
#   6) Run the agent backtest on 2024: just backtest-2024
#   7) Build the runtime image: just docker-build-runtime
#   8) Run a custom Docker backtest: just docker-backtest 2024-01-01 2024-12-31
#   9) Run a Docker backtest with an explicit run_id: just docker-backtest-run bt_demo 2024-01-01 2024-12-31
#  10) Run the full 2024 Docker backtest: just docker-backtest-2024

# Run the local 2024 signal-to-backtest pipeline end-to-end
signals-agent-pipeline-2024: signals-train-final-3y signals-export-onnx-final-3y signals-infer-2024 backtest-2024

# Major defaults are taken from trainer/config.toml:
#   - major_shard_workers = 2
#   - major_workers = 1

# - batch_size = 256
predict-major:
    ./.venv/bin/python -m trainer.main predict major

predict-major-overwrite:
    ./.venv/bin/python -m trainer.main predict major --overwrite

# 64-core recommendation for sub:
#   - 4 shard processes

# - 8 per-major workers inside each shard
predict-sub:
    ./.venv/bin/python -m trainer.main predict sub --sub-shard-workers 4 --sub-major-workers 8

predict-sub-overwrite:
    ./.venv/bin/python -m trainer.main predict sub --sub-shard-workers 4 --sub-major-workers 8 --overwrite

# Full pipeline with the same recommendation:
predict-all:
    ./.venv/bin/python -m trainer.main predict all

predict-all-overwrite:
    ./.venv/bin/python -m trainer.main predict all --overwrite
