set dotenv-load := true
set dotenv-filename := ".env"

root := justfile_directory()

# ── Dev environment ──────────────────────────────────────────────────────────────

cpu-sync:
    uv sync --group dev --group trainer --group torch_cpu --no-group torch_gpu 

gpu-sync:
    uv sync --group dev --group torch_gpu --no-group torch_cpu

# ── Inference / Debug ───────────────────────────────────────────────────────────

decide week:
    python main.py decide --week {{ week }}

backtest start end:
    python main.py backtest --start-date {{ start }} --end-date {{ end }}

# ── Trainer CLI (trainer/main.py) ──────────────────────────────────────────────

major-train:
    ./.venv/bin/python -m trainer.main major train

setfit-train:
    ./.venv/bin/python -m trainer.main sub setfit train

signals-train:
    ./.venv/bin/python -m trainer.main signals train

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
