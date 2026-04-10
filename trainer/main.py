"""Trainer CLI — all training commands in one place.

Usage:
    python -m trainer.main major train
    python -m trainer.main major export-onnx --model-path ... --onnx-path ...

    python -m trainer.main sub setfit prepare [majors...]
    python -m trainer.main sub setfit train [majors...]
    python -m trainer.main sub setfit export-onnx --model-path ... --onnx-path ...

    python -m trainer.main sub supervised train [majors...]
    python -m trainer.main sub supervised export-onnx --model-path ... --onnx-path ...

    python -m trainer.main signals train
    python -m trainer.main signals export-onnx --checkpoint-dir ... --bundle-dir ...
    python -m trainer.main signals infer --bundle-dir ...

    python -m trainer.main predict all
    python -m trainer.main predict major
    python -m trainer.main predict sub --sub-shard-workers 4 --sub-major-workers 8

Recommended for multi-file inference:
    - Put raw news parquet shards into `predict.major_input_dir`
    - Put major-labeled parquet shards into `predict.sub_input_dir`
    - Run `predict major` first, then `predict sub`, or point `predict sub` at a cached input directory
    - Use shard workers for process-level parallelism and sub major workers for per-major parallelism
    - Re-run without `--overwrite` to resume from finished shards
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
    from trainer.src.pipelines.train_sub_supervised import SupervisedMultiMajorTrainer

    SupervisedMultiMajorTrainer(device).train(majors=majors)


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
    overwrite: bool = typer.Option(False, "--overwrite"),
    major_shard_workers: int | None = typer.Option(None, "--major-shard-workers"),
    sub_shard_workers: int | None = typer.Option(None, "--sub-shard-workers"),
    sub_major_workers: int | None = typer.Option(None, "--sub-major-workers"),
    start_month: str | None = typer.Option(None, "--start-month", help="Inclusive YYYY-MM filter for sub inference."),
    end_month: str | None = typer.Option(None, "--end-month", help="Inclusive YYYY-MM filter for sub inference."),
) -> None:
    """Run full pipeline using `predict.major_input_*` then `predict.sub_input_*`."""
    from trainer.src.pipelines.predict import run as predict_run

    predict_run(
        limit_rows=rows,
        overwrite=overwrite,
        major_shard_workers=major_shard_workers,
        sub_shard_workers=sub_shard_workers,
        sub_major_workers=sub_major_workers,
        start_month=start_month,
        end_month=end_month,
    )


@predict_app.command("major")
@with_trainer_init(init_wandb=False)
def predict_major(
    ctx: typer.Context,
    rows: int | None = typer.Option(None, "--rows", "-n"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    major_shard_workers: int | None = typer.Option(None, "--major-shard-workers"),
) -> None:
    """Phase 1: Major inference → intermediate parquet.

    Reads:
        `predict.major_input_dir` or `predict.major_input_path(s)`

    Writes:
        `predict.major_output_dir` or `predict.major_output_path`

    Example:
        python -m trainer.main predict major
    """
    from trainer.src.pipelines.predict import run_major

    paths = run_major(limit_rows=rows, overwrite=overwrite, shard_workers=major_shard_workers)
    console.print(f"[bold green]Major intermediate saved to: {paths}[/bold green]")


@predict_app.command("sub")
@with_trainer_init(init_wandb=False)
def predict_sub(
    ctx: typer.Context,
    rows: int | None = typer.Option(None, "--rows", "-n"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    sub_shard_workers: int | None = typer.Option(None, "--sub-shard-workers"),
    sub_major_workers: int | None = typer.Option(None, "--sub-major-workers"),
    start_month: str | None = typer.Option(None, "--start-month", help="Inclusive YYYY-MM filter for sub inference."),
    end_month: str | None = typer.Option(None, "--end-month", help="Inclusive YYYY-MM filter for sub inference."),
) -> None:
    """Phase 2: sub-category classification on Major intermediate.

    Reads:
        `predict.sub_input_dir` or `predict.sub_input_path(s)`
        Falls back to `predict.major_output_dir` only when sub input is not explicitly configured.

    Writes:
        `predict.output_dir` or `predict.output_path`

    Example:
        python -m trainer.main predict sub --sub-shard-workers 4 --sub-major-workers 8
    """
    from trainer.src.pipelines.predict import run_sub

    run_sub(
        limit_rows=rows,
        overwrite=overwrite,
        shard_workers=sub_shard_workers,
        sub_major_workers=sub_major_workers,
        start_month=start_month,
        end_month=end_month,
    )


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


@signals_app.command("infer")
@with_trainer_init(init_wandb=False)
def signals_infer(
    ctx: typer.Context,
    bundle_dir: str | None = typer.Option(None, "--bundle-dir", help="Signals ONNX bundle directory."),
    output_path: str | None = typer.Option(None, "--output-path", "-o", help="Output parquet path."),
    start_date: str | None = typer.Option(None, "--start-date", help="Inclusive YYYY-MM-DD."),
    end_date: str | None = typer.Option(None, "--end-date", help="Inclusive YYYY-MM-DD."),
    force_dataset: bool = typer.Option(False, "--force-dataset", help="Rebuild cached sentiment dataset first."),
) -> None:
    """Run explicit signals inference from the deployed ONNX bundle."""
    from trainer.src.pipelines.infer_signals import run_inference

    run_inference(
        bundle_dir=Path(bundle_dir) if bundle_dir else None,
        output_path=Path(output_path) if output_path else None,
        start_date=start_date,
        end_date=end_date,
        force_dataset=force_dataset,
    )


@signals_app.command("export-onnx")
@with_trainer_init(init_wandb=False)
def signals_export_onnx(
    ctx: typer.Context,
    checkpoint_dir: str = typer.Option(..., "--checkpoint-dir", "-i", help="Signals checkpoint directory."),
    bundle_dir: str | None = typer.Option(None, "--bundle-dir", "-o", help="Target ONNX bundle directory."),
) -> None:
    """Export a trained signals checkpoint to an ONNX deployment bundle."""
    from trainer.src.pipelines.train_signals import export_signals_onnx_bundle_from_checkpoint

    export_signals_onnx_bundle_from_checkpoint(
        checkpoint_dir=Path(checkpoint_dir),
        bundle_dir=Path(bundle_dir) if bundle_dir else None,
    )


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
