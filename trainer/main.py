"""Trainer CLI — all training commands in one place.

Usage:
    python -m trainer.main major train
    python -m trainer.main major export-onnx --model-path ... --onnx-path ...

    python -m trainer.main sub setfit prepare [majors...]
    python -m trainer.main sub setfit train [majors...]
    python -m trainer.main sub setfit export-onnx --model-path ... --onnx-path ...

    python -m trainer.main sub supervised train [majors...]
    python -m trainer.main sub supervised export-onnx --model-path ... --onnx-path ...

    python -m trainer.main predict all
    python -m trainer.main predict major
    python -m trainer.main predict sub
"""

import functools
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import typer
from dotenv import load_dotenv
from rich.console import Console

from trainer.src.config import get_config, init_config
from trainer.src.utils import WandbRegistry, init_logger

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_app_name(context: typer.Context) -> str:
    current: typer.Context | None = context.parent
    while current is not None:
        info_name = current.info_name
        if info_name in {"major", "sub", "signals", "predict"}:
            return info_name
        current = current.parent
    raise ValueError("Unable to determine app name from context.")


def _init_trainer(context: typer.Context, init_wandb: bool = True) -> None:
    app_name = _resolve_app_name(context)

    console.print(f"[bold blue]Initializing trainer for app: {app_name}[/bold blue]")
    init_config(app_name)
    init_logger()

    if init_wandb and app_name in {"major", "signals"}:
        WandbRegistry.init(app_name, tags=[app_name])

    _init_seed(get_config().seed)


def with_trainer_init(func: Callable | None = None, *, init_wandb: bool = True) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(ctx: typer.Context, *args, **kwargs):
            _init_trainer(ctx, init_wandb=init_wandb)
            fn(ctx, *args, **kwargs)
            WandbRegistry.finish_all()

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


# ── Major subapp ──────────────────────────────────────────────────────────────


major_app = typer.Typer(add_completion=False)


@major_app.command("train")
@with_trainer_init
def major_train(ctx: typer.Context) -> None:
    """Train major category + sentiment classifier."""
    from trainer.src.pipelines.train_major import train_major

    train_major(device=device)


@major_app.command("export-onnx")
@with_trainer_init(init_wandb=False)
def major_export_onnx(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model-path", "-i"),
    onnx_path: str = typer.Option(..., "--onnx-path", "-o"),
    max_seq_length: int = typer.Option(128, "--max-seq-length"),
    opset_version: int = typer.Option(14, "--opset-version"),
) -> None:
    """Export a trained major model to ONNX."""
    from trainer.src.models.major import export_major_to_onnx

    export_major_to_onnx(Path(model_path), Path(onnx_path), max_seq_length, opset_version)
    console.print(f"[bold green]ONNX saved to: {onnx_path}[/bold green]")


# ── Sub parent app (setfit + supervised) ──────────────────────────────────────


sub_app = typer.Typer(add_completion=False)


# ── Sub: setfit ────────────────────────────────────────────────────────────────


setfit_app = typer.Typer(add_completion=False)


@setfit_app.command("prepare")
@with_trainer_init(init_wandb=False)
def setfit_prepare(
    ctx: typer.Context,
    majors: list[str] = typer.Argument(None, help="Specific major categories (default: all)"),
) -> None:
    """Prepare datasets for SetFit training (cached)."""
    from trainer.src.datasets.sub import SetFitDatasetPreparer

    SetFitDatasetPreparer().prepare_all(majors=majors)


@setfit_app.command("train")
@with_trainer_init
def setfit_train(
    ctx: typer.Context,
    majors: list[str] = typer.Argument(None, help="Specific major categories (default: all)"),
) -> None:
    """Train SetFit models per major (contrastive learning)."""
    from trainer.src.pipelines.train_sub_setfit import SetFitMultiMajorTrainer

    SetFitMultiMajorTrainer(device=device).train(majors=majors)


@setfit_app.command("export-onnx")
@with_trainer_init(init_wandb=False)
def setfit_export(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model-path", "-i"),
    onnx_path: str = typer.Option(..., "--onnx-path", "-o"),
    max_seq_length: int = typer.Option(256, "--max-seq-length"),
    opset_version: int = typer.Option(14, "--opset-version"),
) -> None:
    """Export a trained SetFit model to ONNX."""
    from trainer.src.models.sub import export_sub_to_onnx

    export_sub_to_onnx(Path(model_path), Path(onnx_path), max_seq_length, opset_version)
    console.print(f"[bold green]ONNX saved to: {onnx_path}[/bold green]")


# ── Sub: supervised ────────────────────────────────────────────────────────────


supervised_app = typer.Typer(add_completion=False)


@supervised_app.command("train")
@with_trainer_init
def supervised_train(
    ctx: typer.Context,
    majors: list[str] = typer.Argument(None, help="Specific major categories (default: all)"),
) -> None:
    """Train sub-category classifiers (supervised fine-tune) per major."""
    from trainer.src.pipelines.train_sub_supervised import SubMultiMajorTrainer

    SubMultiMajorTrainer(get_config().sub, device).train(majors=majors)


@supervised_app.command("export-onnx")
@with_trainer_init(init_wandb=False)
def supervised_export(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model-path", "-i"),
    onnx_path: str = typer.Option(..., "--onnx-path", "-o"),
    max_seq_length: int = typer.Option(128, "--max-seq-length"),
    opset_version: int = typer.Option(14, "--opset-version"),
) -> None:
    """Export a trained Sub model to ONNX."""
    from trainer.src.models.sub import export_sub_to_onnx

    export_sub_to_onnx(Path(model_path), Path(onnx_path), max_seq_length, opset_version)
    console.print(f"[bold green]ONNX saved to: {onnx_path}[/bold green]")


# ── Predict subapp ────────────────────────────────────────────────────────────


predict_app = typer.Typer(add_completion=False)


@predict_app.command("all")
@with_trainer_init(init_wandb=False)
def predict_all(
    ctx: typer.Context,
    rows: int | None = typer.Option(None, "--rows", "-n"),
) -> None:
    """Run full pipeline: Major → SetFit."""
    from trainer.src.pipelines.predict import run as predict_run

    predict_run(limit_rows=rows)


@predict_app.command("major")
@with_trainer_init(init_wandb=False)
def predict_major(
    ctx: typer.Context,
    rows: int | None = typer.Option(None, "--rows", "-n"),
) -> None:
    """Phase 1: Major inference → intermediate parquet."""
    from trainer.src.pipelines.predict import run_finbert

    path = run_finbert(limit_rows=rows)
    console.print(f"[bold green]Major intermediate saved to: {path}[/bold green]")


@predict_app.command("sub")
@with_trainer_init(init_wandb=False)
def predict_sub(
    ctx: typer.Context,
    rows: int | None = typer.Option(None, "--rows", "-n"),
) -> None:
    """Phase 2: sub-category classification on Major intermediate."""
    from trainer.src.pipelines.predict import run_setfit

    run_setfit(limit_rows=rows)


# ── Signals subapp ────────────────────────────────────────────────────────────


signals_app = typer.Typer(add_completion=False)


@signals_app.command("train")
@with_trainer_init
def signals_train(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Run full TCN pipeline: pretrain → finetune → LightGBM."""
    from trainer.src.pipelines.train_signals import run_training

    run_training(force=force)


# ── Register subapps ──────────────────────────────────────────────────────────


app.add_typer(major_app, name="major")
app.add_typer(sub_app, name="sub")
app.add_typer(predict_app, name="predict")
app.add_typer(signals_app, name="signals")

# sub setfit and sub supervised are nested under sub_app
sub_app.add_typer(setfit_app, name="setfit")
sub_app.add_typer(supervised_app, name="supervised")


if __name__ == "__main__":
    app()
