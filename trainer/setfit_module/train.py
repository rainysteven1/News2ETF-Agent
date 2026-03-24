"""Train one SetFit model per major category for sub-category classification.

Two-phase training (same as SetFit official approach):
  Phase 1: Contrastive learning — generate positive/negative text pairs
  Phase 2: Train classification head on the embeddings
"""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch
from datasets import Dataset
from loguru import logger
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

from trainer.config import SetFitModelConfig, SetFitTrainingConfig
from trainer.config import load_config as load_trainer_config
from trainer.setfit_module.model import LabelStats, _safe_name, export_setfit_to_onnx
from trainer.utils.seed import set_seed
from trainer.wandb_handler import WandbHandler

# ─── Helpers ──────────────────────────────────────────────────────────────────


def prepare_hf_dataset(df: pl.DataFrame, major: str) -> tuple[Dataset, list[str]]:
    """Build a HuggingFace Dataset from a filtered DataFrame for one major category.

    Returns (dataset, unique_labels) to ensure label order is consistent
    between training and inference.
    """
    sub_df = df.filter(pl.col("major_category") == major)
    text_col = "content" if "content" in df.columns else "title"
    sub_df = sub_df.drop_nulls(subset=["sub_category", text_col])

    unique_labels = sorted(sub_df["sub_category"].unique().to_list())
    dataset = Dataset.from_dict(
        {
            "text": sub_df[text_col].to_list(),
            "label": [unique_labels.index(label) for label in sub_df["sub_category"].to_list()],
            "label_text": sub_df["sub_category"].to_list(),
        }
    )
    dataset = dataset.class_encode_column("label")
    return dataset, unique_labels


def _adaptive_num_iterations(n_samples: int, base_iterations: int) -> int:
    """Reduce num_iterations for large classes to save GPU memory / time."""
    if n_samples > 1500:
        return max(5, base_iterations // 3)
    if n_samples > 800:
        return max(5, base_iterations // 2)
    return base_iterations


def train_setfit_for_major(
    df: pl.DataFrame,
    major: str,
    output_dir: Path,
    mcfg: SetFitModelConfig,
    tcfg: SetFitTrainingConfig,
    device: torch.device,
    wb: WandbHandler,
) -> dict[str, Any]:
    """Train and save a SetFit model for one major category. Returns metrics."""
    logger.info(f"[SetFit] Training for major: {major}")

    dataset, unique_labels = prepare_hf_dataset(df, major)

    if len(unique_labels) < 2:
        logger.warning(f"[SetFit] {major}: only {len(unique_labels)} sub-categories, skipping")
        return {"major": major, "status": "skipped", "reason": "fewer than 2 sub-categories"}

    n_samples = len(dataset)
    num_iters = _adaptive_num_iterations(n_samples, tcfg.num_iterations)

    try:
        train_ds = dataset.train_test_split(test_size=tcfg.test_size, seed=tcfg.seed, stratify_by_column="label")
    except Exception:
        train_ds = dataset.train_test_split(test_size=tcfg.test_size, seed=tcfg.seed)

    # ── Phase 1: Contrastive fine-tuning ──────────────────────────────────────
    id2label = {i: label for i, label in enumerate(unique_labels)}
    model = SetFitModel.from_pretrained(
        mcfg.pretrained_model,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    model.to(device)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds["train"],
        eval_dataset=train_ds["test"],
        loss_class=CosineSimilarityLoss,
        batch_size=tcfg.batch_size,
        num_iterations=num_iters,
        num_epochs=tcfg.num_epochs,
        learning_rate=tcfg.learning_rate,
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    logger.info(f"[SetFit] {major} — samples={n_samples}, iters={num_iters}, metrics={metrics}")

    accuracy = float(metrics.get("accuracy", 0))
    f1 = float(metrics.get("f1", 0))

    wb.log_epoch(
        "setfit",
        1,
        accuracy,
        {
            "f1": f1,
            "samples": n_samples,
            "num_iterations": num_iters,
        },
    )

    wb.log_summary({"best_accuracy": accuracy, "best_f1": f1})

    # ── Save best model ───────────────────────────────────────────────────────
    best_dir = output_dir / "best"

    # Always save the trained model as "best" (SetFitTrainer trains once per run,
    # so this IS the best we've got for this major category)
    model.save_pretrained(str(best_dir))

    # Save label list alongside (id2label order)
    with open(best_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(unique_labels, f, ensure_ascii=False, indent=2)

    logger.success(f"[SetFit] {major} best model saved to {best_dir} (accuracy={accuracy:.4f}, f1={f1:.4f})")

    # ── Export best model to ONNX ──────────────────────────────────────────────
    onnx_path = output_dir / "best.onnx"
    try:
        export_setfit_to_onnx(model_dir=best_dir, onnx_path=onnx_path, max_seq_length=mcfg.max_seq_length or 256)

        # ── Upload ONNX as W&B artifact ──────────────────────────────────────
        wb.upload_artifact(
            artifact_path=onnx_path,
            name=f"setfit-onnx-{_safe_name(major)}",
            artifact_type="model",
            aliases=["best"],
        )
        onnx_status = "exported"
    except Exception as e:
        logger.warning(
            f"[SetFit] ONNX export failed for {major}: {e}, pth checkpoint saved at {best_dir} for manual conversion"
        )
        onnx_path = None
        onnx_status = "failed"

    return {
        "major": major,
        "status": "ok",
        "metrics": metrics,
        "accuracy": accuracy,
        "f1": f1,
        "output_dir": str(best_dir),
        "onnx_path": str(onnx_path) if onnx_path else None,
        "onnx_status": onnx_status,
    }


# ─── Main training ───────────────────────────────────────────────────────────


def train_per_major() -> dict[str, dict[str, Any]]:
    """Train one SetFit model per major category.

    Returns dict mapping major category -> metrics/results.
    Each major category gets its own wandb run (tags: setfit + <major>).
    GPU memory is released after each major to avoid OOM.
    """
    cfg = load_trainer_config()

    mcfg = cfg.setfit
    tcfg = cfg.setfit_training
    wcfg = cfg.wandb

    set_seed(tcfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[SetFit] Device: {device}")

    data_path = tcfg.raw_data_path if tcfg.raw_data_path else (Path("data/labeled/raw.parquet"))
    df = pl.read_parquet(data_path)
    logger.info(f"[SetFit] Loaded {len(df)} rows from {data_path}")

    run_prefix = f"setfit-{datetime.now():%m%d-%H%M}"
    run_prefix_dir = (
        Path(tcfg.output_dir) / run_prefix if tcfg.output_dir else Path("trainer/checkpoints/setfit") / run_prefix
    )
    run_prefix_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for major in LabelStats().get_major_categories():
        major_count = len(df.filter(pl.col("major_category") == major))
        logger.info(f"[SetFit] Major '{major}': {major_count} rows")

        run_name = f"{run_prefix}-{_safe_name(major)}"
        run_output_dir = run_prefix_dir / _safe_name(major)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        wb = WandbHandler(
            project=wcfg.project,
            entity=wcfg.entity,
            name=run_name,
            config_dict={
                "pretrained_model": mcfg.pretrained_model,
                "batch_size": tcfg.batch_size,
                "num_iterations": tcfg.num_iterations,
                "num_epochs": tcfg.num_epochs,
                "learning_rate": tcfg.learning_rate,
            },
            tags=["setfit", major],
            mode=cfg.wandb.mode,
        )

        result = train_setfit_for_major(df, major, run_output_dir, mcfg, tcfg, device, wb)
        results[major] = result
        wb.finish()

        # ── GPU memory cleanup ─────────────────────────────────────────────────
        del wb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    summary_path = run_prefix_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.success(f"[SetFit] All models trained. Summary: {summary_path}")
    return results
