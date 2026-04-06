"""Train one SetFit model per major category for sub-category classification.

Contrastive learning approach — uses SetFit's contrastive pair generation.
Run `setfit prepare` first to prepare datasets.
"""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import confusion_matrix

from trainer.src.config import LabelStats, get_config, safe_name
from trainer.src.utils import WandbRegistry, get_logger


class SetFitMultiMajorTrainer:
    """Train one SetFit model per major category."""

    def __init__(self, device: torch.device):
        self.cfg = get_config()
        self.dcfg = self.cfg.sub.setfit.data
        self.mcfg = self.cfg.sub.setfit.model
        self.tcfg = self.cfg.sub.setfit.training

        self.logger = get_logger()
        self.device = device

    def train(self, majors: list[str] | None = None) -> None:
        """Train all (or a subset of) majors (reads from prepared cache only)."""
        run_prefix = f"setfit-{datetime.now():%m%d-%H%M}"

        assert self.dcfg.raw_data_dir is not None, "setfit.data.raw_data_dir must be set in config"

        all_majors = LabelStats.load().get_major_categories()
        target_majors = majors if majors is not None else all_majors

        results = {}
        for major in target_majors:
            cache_dir = self.dcfg.raw_data_dir / safe_name(major)
            parquet_path = cache_dir / "prepared.parquet"
            meta_path = cache_dir / "meta.json"

            if not cache_dir.exists() or not parquet_path.exists():
                raise FileNotFoundError(
                    f"[SetFit] Prepared dataset not found for '{major}'. Run `setfit prepare --majors {major}` first."
                )

            self.logger.info(f"[SetFit] Loading prepared dataset for '{major}' from cache: {cache_dir}")
            df_cache = pl.read_parquet(parquet_path)
            unique_labels = sorted(df_cache["label_text"].unique().to_list())
            dataset = Dataset.from_dict(
                {
                    "text": df_cache["text"].to_list(),
                    "label": [unique_labels.index(label) for label in df_cache["label_text"].to_list()],
                    "label_text": df_cache["label_text"].to_list(),
                }
            )
            dataset = dataset.class_encode_column("label")

            with open(meta_path) as f:
                meta_content = json.load(f)

            self.logger.info(f"[SetFit] Training '{major}': {len(dataset)} samples")

            run_output_dir = Path(self.tcfg.output_dir or "trainer/checkpoints/setfit") / safe_name(major) / run_prefix
            run_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = self._train_one_major(major, run_output_dir, dataset, meta_content, unique_labels)
                results[major] = result
            except Exception as e:
                self.logger.error(f"[Sub] [Setfit] Failed to train {major}: {e}")
                continue

        summary_path = Path(self.tcfg.output_dir or "trainer/checkpoints/setfit") / f"{run_prefix}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[Sub] [Setfit] All models trained. Summary: {summary_path}")

    def _train_one_major(
        self,
        major: str,
        output_dir: Path,
        dataset: Dataset,
        dataset_meta: dict[str, Any],
        unique_labels: list[str],
    ) -> dict[str, Any]:
        self.logger.info(f"[SetFit] Training for major: {major}")

        n_samples = len(dataset)
        num_iters = self.tcfg.num_iterations

        try:
            train_ds = dataset.train_test_split(
                test_size=self.dcfg.val_ratio,
                seed=self.cfg.seed,
                stratify_by_column="label",
            )
        except Exception:
            train_ds = dataset.train_test_split(
                test_size=self.dcfg.val_ratio,
                seed=self.cfg.seed,
            )

        self.dcfg.extra_args = dataset_meta

        wb_key = f"sub-{safe_name(major)}"
        WandbRegistry.init(
            wb_key,
            run_name=wb_key,
            cfg_dict=self.cfg.sub.to_wandb(),
            tags=["setfit", major],
        )
        wb = WandbRegistry.get(wb_key)

        id2label = {i: label for i, label in enumerate(unique_labels)}
        model = SetFitModel.from_pretrained(
            self.mcfg.pretrained_model,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )
        model.to(self.device)

        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_ds["train"],
            eval_dataset=train_ds["test"],
            loss_class=CosineSimilarityLoss,
            batch_size=self.dcfg.batch_size,
            seed=self.cfg.seed,
            num_iterations=num_iters,
            num_epochs=self.tcfg.num_epochs,
            learning_rate=self.tcfg.learning_rate,
            column_mapping={"text": "text", "label": "label"},
        )
        trainer.train()

        metrics = trainer.evaluate()
        accuracy = float(metrics.get("accuracy", 0))

        eval_texts = train_ds["test"]["text"]
        eval_labels = train_ds["test"]["label"]
        preds = model.predict(eval_texts)

        cm = confusion_matrix(eval_labels, preds, labels=list(range(len(unique_labels))))
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        wb.log_confusion_matrix(cm_normalized.tolist(), unique_labels, title=f"{major} Confusion Matrix")
        wb.log_summary(
            {
                "samples": n_samples,
                "num_iterations": num_iters,
                "best_accuracy": accuracy,
                **metrics,
            }
        )

        self.logger.info(f"[SetFit] {major} — samples={n_samples}, iters={num_iters}, metrics={metrics}")
        self.logger.info("[SetFit] Confusion matrix (rows=truth, cols=pred):")
        for i, row in enumerate(cm_normalized):
            self.logger.info(f"  {unique_labels[i]}: " + " ".join(f"{v:.2f}" for v in row))

        results: dict[str, Any] = {
            "major": major,
            "status": "ok",
            "metrics": metrics,
            "accuracy": accuracy,
        }

        if self.tcfg.save_checkpoint:
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            with open(best_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump(unique_labels, f, ensure_ascii=False, indent=2)
            results["model_dir"] = str(best_dir)
            self.logger.info(f"[SetFit] {major} best model saved to {best_dir} (accuracy={accuracy:.4f})")

        wb.finish()
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results


def run_training(majors: list[str] | None = None) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SetFitMultiMajorTrainer(device).train(majors)
