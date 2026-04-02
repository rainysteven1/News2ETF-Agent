"""Trainer CLI — all training commands in one place.

Usage:
    python -m trainer.main finbert train
    python -m trainer.main finbert export-onnx --model-path ... --onnx-path ...
    python -m trainer.main setfit train
    python -m trainer.main setfit export-onnx --model-path ... --onnx-path ...
    python -m trainer.main signals train
    python -m trainer.main signals train-tcn
    python -m trainer.main signals train-lgbm
"""

import functools
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'from trainer.xxx' absolute imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from collections.abc import Callable

import numpy as np
import torch
import typer
from dotenv import load_dotenv
from rich.console import Console

from trainer.config import get_config, init_config
from trainer.logger import init_logger
from trainer.wandb_handler import WandbRegistry

load_dotenv()  # Load environment variables from .env file at startup

app = typer.Typer(add_completion=False)
console = Console()

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_trainer(context: typer.Context) -> None:
    """Initialize config, logger, and wandb from the parent app name."""

    app_name: str = context.parent.params.get("name", "finbert")  # type: ignore[union-attr]
    console.print(f"[bold blue]Initializing trainer for app: {app_name}[/bold blue]")

    init_config(app_name)  # type: ignore
    init_logger()

    if app_name == "setfit":
        pass
    else:
        WandbRegistry.init(app_name, tags=[app_name])

    _init_seed(get_config().seed)


def with_trainer_init(func: Callable) -> Callable:
    """Decorator: initializes config/logger/wandb before running a train command."""

    @functools.wraps(func)
    def wrapper(ctx: typer.Context, *args, **kwargs):
        _init_trainer(ctx)
        func(ctx, *args, **kwargs)
        WandbRegistry.finish_all()

    return wrapper


# ── Signals subapp ────────────────────────────────────────────────────────────


signals_app = typer.Typer(add_completion=False)


@signals_app.command("train")
@with_trainer_init
def signals_train(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f", help="Force re-process raw data even if cached parquet exists."),
) -> None:
    """Run full TCN pipeline: pretrain → finetune → LightGBM stacking."""
    from trainer.signals.train import run_training

    run_training(force=force)


# ── FinBERT subapp ──────────────────────────────────────────────────────────


finbert_app = typer.Typer(add_completion=False)


@finbert_app.command("train")
@with_trainer_init
def finbert_train(ctx: typer.Context) -> None:
    """Train FinBERT (8 L1 classes + 3 sentiment) on labeled news data."""
    from trainer.finbert.train import train_finbert

    train_finbert(device=device)


@finbert_app.command("export-onnx")
def finbert_export_onnx(
    model_path: str = typer.Option(..., "--model-path", "-i", help="Path to trained FinBERT model directory"),
    onnx_path: str = typer.Option(..., "--onnx-path", "-o", help="Output path for the ONNX file"),
    max_seq_length: int = typer.Option(128, "--max-seq-length", help="Maximum sequence length for ONNX export"),
    opset_version: int = typer.Option(14, "--opset-version", help="ONNX opset version"),
) -> None:
    """Export a trained FinBERT model to ONNX format."""
    from trainer.finbert.model import export_finbert_to_onnx

    export_finbert_to_onnx(
        Path(model_path),
        Path(onnx_path),
        max_seq_length,
        opset_version,
    )

    console.print(f"[bold green]ONNX model saved to: {onnx_path}[/bold green]")


# ── SetFit subapp ────────────────────────────────────────────────────────────


setfit_app = typer.Typer(add_completion=False)


@setfit_app.command("train")
def setfit_train() -> None:
    """Train one SetFit model per major category (each major = separate wandb run)."""
    from trainer.setfit_module.train import train_per_major

    train_per_major()


@setfit_app.command("export-onnx")
def setfit_export_onnx(
    model_path: str = typer.Option(..., "--model-path", "-i", help="Path to trained SetFit model directory"),
    onnx_path: str = typer.Option(..., "--onnx-path", "-o", help="Output path for the ONNX file"),
    max_seq_length: int = typer.Option(256, "--max-seq-length", help="Maximum sequence length for ONNX export"),
    opset_version: int = typer.Option(14, "--opset-version", help="ONNX opset version"),
) -> None:
    """Export a trained SetFit model to ONNX format."""
    from trainer.setfit_module.model import export_setfit_to_onnx

    export_setfit_to_onnx(
        Path(model_path),
        Path(onnx_path),
        max_seq_length,
        opset_version,
    )

    console.print(f"[bold green]ONNX model saved to: {onnx_path}[/bold green]")


# ── Predict subapp ────────────────────────────────────────────────────────────


predict_app = typer.Typer(add_completion=False)


@predict_app.command("all")
def predict_cmd(
    rows: int | None = typer.Option(
        None,
        "--rows",
        "-n",
        help="Limit input rows (for quick testing). Default: all rows.",
    ),
) -> None:
    """Run full pipeline: FinBERT → SetFit sub-category."""
    from trainer.predict import run as run_predict

    run_predict(limit_rows=rows)


@predict_app.command("finbert")
def finbert_cmd(
    rows: int | None = typer.Option(
        None,
        "--rows",
        "-n",
        help="Limit input rows (for quick testing). Default: all rows.",
    ),
) -> None:
    """Phase 1 only: FinBERT inference → intermediate parquet."""
    from trainer.predict import run_finbert

    path = run_finbert(limit_rows=rows)
    console.print(f"[bold green]FinBERT intermediate saved to: {path}[/bold green]")


@predict_app.command("setfit")
def setfit_cmd(
    rows: int | None = typer.Option(
        None,
        "--rows",
        "-n",
        help="Limit input rows (for quick testing). Default: all rows.",
    ),
) -> None:
    """Phase 2 only: SetFit sub-category classification on FinBERT intermediate."""
    from trainer.predict import run_setfit

    run_setfit(limit_rows=rows)


# ── Register subapps ──────────────────────────────────────────────────────────


app.add_typer(signals_app, name="signals")
app.add_typer(finbert_app, name="finbert")
app.add_typer(setfit_app, name="setfit")
app.add_typer(predict_app, name="predict")


if __name__ == "__main__":
    app()
