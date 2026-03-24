"""Trainer CLI — all training commands in one place.

Usage:
    python -m trainer.main finbert train
    python -m trainer.main finbert export-onnx --model-path ... --onnx-path ...
    python -m trainer.main setfit train
    python -m trainer.main setfit export-onnx --model-path ... --onnx-path ...
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'from trainer.xxx' absolute imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()  # Load environment variables from .env file at startup

app = typer.Typer(add_completion=False)
console = Console()


# ── Signals ──────────────────────────────────────────────────────────────────


@app.command("signals-train")
def signals_train(
    config: str | None = None,
) -> None:
    """Run full LSTM+Attention pipeline: pretrain → finetune → LightGBM stacking."""
    from trainer.signals.train import run_training

    run_training(config_path=config)


# ── FinBERT subapp ──────────────────────────────────────────────────────────


finbert_app = typer.Typer(add_completion=False)


@finbert_app.command("train")
def finbert_train() -> None:
    """Train FinBERT (8 L1 classes + 3 sentiment) on labeled news data."""
    from trainer.finbert.train import train_finbert

    train_finbert()


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


app.add_typer(finbert_app, name="finbert")
app.add_typer(setfit_app, name="setfit")
app.add_typer(predict_app, name="predict")


if __name__ == "__main__":
    app()
