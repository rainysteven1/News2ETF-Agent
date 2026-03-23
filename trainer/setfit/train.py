"""Train one SetFit model per major category for sub-category classification.

Two-phase training (same as SetFit official approach):
  Phase 1: Contrastive learning — generate positive/negative text pairs
  Phase 2: Train classification head on the embeddings
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
from datasets import Dataset
from loguru import logger
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

from trainer.config import load_config as load_trainer_config
from trainer.setfit.model import _safe_name, get_major_categories

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
    mcfg: Any,
    tcfg: Any,
    device: torch.device,
    wb: Any,
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
        train_ds = dataset.train_test_split(test_size=tcfg.test_size, seed=tcfg.seed, stratify="label")
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

    wb.log_epoch(
        "setfit",
        1,
        float(metrics.get("accuracy", 0)),
        {
            "f1": float(metrics.get("f1", 0)),
            "samples": n_samples,
            "num_iterations": num_iters,
        },
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    major_dir = Path(output_dir) / _safe_name(major)
    model.save(str(major_dir))

    # Save label list alongside (id2label order)
    with open(major_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(unique_labels, f, ensure_ascii=False, indent=2)

    logger.success(f"[SetFit] {major} saved to {major_dir}")
    return {"major": major, "status": "ok", "metrics": metrics, "output_dir": str(major_dir)}


# ─── Main training ───────────────────────────────────────────────────────────


def train_per_major(
    data_path: Path,
    output_dir: Path | None = None,
    model_config: Any | None = None,
    training_config: Any | None = None,
    wandb_project: str = "news2etf",
) -> dict[str, dict[str, Any]]:
    """Train one SetFit model per major category.

    Returns dict mapping major category -> metrics/results.
    Each major category gets its own wandb run (tags: setfit + <major>).
    GPU memory is released after each major to avoid OOM.
    """
    cfg = load_trainer_config()
    mcfg = model_config or cfg.setfit
    tcfg = training_config or cfg.setfit_training

    output_dir = output_dir or Path("checkpoints/setfit")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pl.read_parquet(data_path)
    logger.info(f"[SetFit] Loaded {len(df)} rows from {data_path}")

    results = {}
    for major in get_major_categories():
        major_count = len(df.filter(pl.col("major_category") == major))
        logger.info(f"[SetFit] Major '{major}': {major_count} rows")

        from trainer.wandb_handler import WandbHandler

        wb = WandbHandler(
            project=wandb_project,
            name=major,
            config_dict={
                "pretrained_model": mcfg.pretrained_model,
                "batch_size": tcfg.batch_size,
                "num_iterations": tcfg.num_iterations,
                "num_epochs": tcfg.num_epochs,
                "learning_rate": tcfg.learning_rate,
            },
            tags=["setfit", major],
        )

        result = train_setfit_for_major(df, major, output_dir, mcfg, tcfg, device, wb)
        results[major] = result
        wb.finish()

        # ── GPU memory cleanup ─────────────────────────────────────────────────
        del wb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.success(f"[SetFit] All models trained. Summary: {summary_path}")
    return results
