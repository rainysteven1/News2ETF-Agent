set dotenv-load := true
set dotenv-filename := ".env"

root := justfile_directory()

# ── Dev environment ──────────────────────────────────────────────────────────────

cpu-sync:
    uv sync --group dev --group torch_cpu --no-group torch_gpu

gpu-sync:
    uv sync --group dev --group torch_gpu --no-group torch_cpu

# ── Inference / Debug ───────────────────────────────────────────────────────────

decide week:
    python main.py decide --week {{week}}

backtest start end:
    python main.py backtest --start-date {{start}} --end-date {{end}}

# ── Trainer CLI (trainer/main.py) ──────────────────────────────────────────────

signals-train:
    python -m trainer.main signals-train

finbert-train data:
    python -m trainer.main finbert-train --data {{data}}

setfit-train data:
    python -m trainer.main setfit-train --data {{data}}
