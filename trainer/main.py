"""Trainer CLI — all training commands in one place.

All configuration is read from config.toml; the --config flag only needed
if your config file is not at trainer/config.toml.
"""

import typer
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file at startup

app = typer.Typer(add_completion=False)


# ── Signals ──────────────────────────────────────────────────────────────────


@app.command("signals-train")
def signals_train(
    config: str | None = None,
) -> None:
    """Run full LSTM+Attention pipeline: pretrain → finetune → LightGBM stacking."""
    from trainer.signals.train import run_training

    run_training(config_path=config)


# ── FinBERT ─────────────────────────────────────────────────────────────────


@app.command("finbert-train")
def finbert_train() -> None:
    """Train FinBERT (8 L1 classes + 3 sentiment) on labeled news data."""
    from trainer.finbert.train import train_finbert

    train_finbert()


# ── SetFit ─────────────────────────────────────────────────────────────────


@app.command("setfit-train")
def setfit_train() -> None:
    """Train one SetFit model per major category (each major = separate wandb run)."""
    from trainer.setfit.train import train_per_major

    train_per_major()


if __name__ == "__main__":
    app()
