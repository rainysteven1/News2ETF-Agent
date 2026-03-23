"""Trainer CLI — all training commands in one place."""

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


# ── Signals ──────────────────────────────────────────────────────────────────


@app.command("signals-train")
def signals_train(
    config: str | None = None,
    project: str = "news2etf",
    name: str | None = None,
) -> None:
    """Run full LSTM+Attention pipeline: pretrain → finetune → LightGBM stacking."""
    from trainer.signals.train import run_training
    run_training(config_path=config, wandb_project=project, wandb_name=name)


# ── FinBERT ─────────────────────────────────────────────────────────────────


@app.command("finbert-train")
def finbert_train(
    data: str = typer.Option(..., help="Path to labeled news parquet file"),
    config: str | None = None,
    output: str | None = None,
    project: str = "news2etf",
    name: str | None = None,
) -> None:
    """Train FinBERT (8 L1 classes + 3 sentiment) on labeled news data."""
    from trainer.finbert.train import train_finbert
    train_finbert(
        data_path=Path(data),
        output_dir=Path(output) if output else None,
        config_path=config,
        wandb_project=project,
        wandb_name=name,
    )


# ── SetFit ──────────────────────────────────────────────────────────────────


@app.command("setfit-train")
def setfit_train(
    data: str = typer.Option(..., help="Path to labeled raw parquet file"),
    output: str | None = None,
    project: str = "news2etf",
) -> None:
    """Train one SetFit model per major category (each major = separate wandb run)."""
    from trainer.setfit.train import train_per_major
    train_per_major(
        data_path=Path(data),
        output_dir=Path(output) if output else None,
        wandb_project=project,
    )


if __name__ == "__main__":
    app()
