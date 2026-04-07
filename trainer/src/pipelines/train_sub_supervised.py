"""Training loop for sub (L2) supervised fine-tuning — two-phase BERT.

Phase 1: Freeze BERT backbone, train classification head.
Phase 2: Unfreeze BERT, fine-tune all params with lower BERT LR.

Uses focal loss for class imbalance handling.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from trainer.src.config import LabelStats, get_config, safe_name
from trainer.src.datasets.sub import SubCatDataset, preprocess_split
from trainer.src.models.sub import SubClassifier, load_sub_classifier, save_sub_classifier
from trainer.src.utils import WandbRegistry, get_logger


class EvalMetrics:
    def __init__(
        self,
        loss: float,
        accuracy: float,
        y_true: list[int] | None = None,
        y_pred: list[int] | None = None,
    ):
        self.loss = loss
        self.accuracy = accuracy
        self._y_true = y_true or []
        self._y_pred = y_pred or []

    def wandb_dict(self) -> dict:
        return {"loss": self.loss, "accuracy": self.accuracy}


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                label=batch["label"],
            )
        total_loss += outputs["loss"].item() * batch["input_ids"].size(0)
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["input_ids"].size(0)
        all_true.extend(batch["label"].cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    return EvalMetrics(
        loss=total_loss / total,
        accuracy=correct / total,
        y_true=all_true,
        y_pred=all_pred,
    )


def freeze_bert(model: SubClassifier) -> None:
    for param in model.bert.parameters():
        param.requires_grad = False
    get_logger().info("[Sub] BERT backbone frozen (Phase 1)")


def unfreeze_bert(model: SubClassifier) -> None:
    for param in model.bert.parameters():
        param.requires_grad = True
    get_logger().info("[Sub] BERT backbone unfrozen (Phase 2)")


def build_confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[float]]:
    cm = [[0.0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    for i in range(num_classes):
        row_sum = sum(cm[i])
        if row_sum > 0:
            cm[i] = [v / row_sum for v in cm[i]]
    return cm


class SupervisedMultiMajorTrainer:
    def __init__(self, device: torch.device):
        self.cfg = get_config().sub.supervised
        self.device = device
        self.logger = get_logger()

        self.dcfg = self.cfg.data
        self.mcfg = self.cfg.model
        self.tcfg = self.cfg.training

    def train(self, majors: list[str] | None = None) -> None:
        """Train all (or a subset of) majors (reads from prepared cache only)."""
        run_prefix = f"supervised-{datetime.now():%m%d-%H%M}"
        tokenizer = AutoTokenizer.from_pretrained(self.mcfg.pretrained_model)

        assert self.dcfg.raw_data_dir is not None, "supervised.data.raw_data_dir must be set in config"
        assert self.tcfg.output_dir is not None, "supervised.training.output_dir must be set in config"

        all_majors = LabelStats.load().get_major_categories()
        target_majors = majors if majors is not None else all_majors

        results: dict[str, dict[str, Any]] = {}
        for major in target_majors:
            run_output_dir = Path(self.tcfg.output_dir) / safe_name(major) / run_prefix
            run_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = self._train_one_major(major, tokenizer, run_output_dir)
                results[major] = result
            except Exception as e:
                self.logger.error(f"[Sub] [Supervised] Failed to train {major}: {e}")
                self.logger.error(traceback.format_exc())
                continue

        summary_path = Path(self.tcfg.output_dir) / f"{run_prefix}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[Sub] [Supervised] All models trained. Summary: {summary_path}")

    def _train_one_major(self, major: str, tokenizer: AutoTokenizer, output_dir: Path) -> dict[str, Any]:
        self.logger.info(f"[Supervised] Training for major: {major}")

        assert self.dcfg.raw_data_dir is not None, "supervised.data.raw_data_dir must be set in config"
        raw_path = Path(self.dcfg.raw_data_dir) / "raw.parquet"
        train_path, val_path = preprocess_split(raw_path, major, val_ratio=self.dcfg.val_ratio, seed=get_config().seed)

        df = pl.read_parquet(raw_path).filter(pl.col("major_category") == major)
        sub_labels = sorted(df["sub_category"].unique().to_list())
        label_to_idx = {name: i for i, name in enumerate(sub_labels)}
        idx_to_label = {i: name for i, name in enumerate(sub_labels)}

        label_map_path = output_dir / "label_map.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {"label_to_idx": label_to_idx, "idx_to_label": {str(k): v for k, v in idx_to_label.items()}},
                f,
                ensure_ascii=False,
            )

        num_classes = len(sub_labels)
        self.logger.info(f"[{major}] Classes ({num_classes}): {sub_labels}")

        train_ds = SubCatDataset(train_path, tokenizer, label_to_idx, max_length=self.dcfg.max_seq_length)
        val_ds = SubCatDataset(val_path, tokenizer, label_to_idx, max_length=self.dcfg.max_seq_length)
        self.logger.info(f"[{major}] Train: {len(train_ds)}, Val: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds, batch_size=self.dcfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(val_ds, batch_size=self.dcfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        wb_key = f"sub-{safe_name(major)}"
        WandbRegistry.init(
            wb_key,
            run_name=wb_key,
            cfg_dict=self.cfg.to_wandb(),
            tags=["setfit", major],
        )
        wb = WandbRegistry.get(wb_key)

        model = load_sub_classifier(
            self.mcfg.pretrained_model,
            num_classes=num_classes,
            dropout=self.mcfg.dropout,
            focal_gamma=self.tcfg.focal_loss_gamma,
        )
        model.set_class_weights(train_ds.get_class_weights())
        model.to(self.device)

        scaler = GradScaler(enabled=self.tcfg.fp16 and self.device.type == "cuda", device=self.device.type)
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

        best_val_acc = 0.0
        epochs_without_improvement = 0
        global_step = 0

        # Phase 1: Freeze BERT, train head
        self.logger.info(f"[{major}] === Phase 1: {self.tcfg.epochs_phase1} epochs with frozen BERT ===")
        freeze_bert(model)

        head_params = (
            list(model.fc1.parameters())
            + list(model.activation.parameters())
            + list(model.fc1_dropout.parameters())
            + list(model.fc2.parameters())
        )
        optimizer = optim.AdamW(head_params, lr=self.tcfg.heads_lr, weight_decay=self.tcfg.weight_decay)

        phase1_steps = len(train_loader) * self.tcfg.epochs_phase1 // self.tcfg.grad_accum_steps
        phase1_warmup = int(phase1_steps * self.tcfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, phase1_warmup, phase1_steps)

        for epoch in range(self.tcfg.epochs_phase1):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(enabled=self.tcfg.fp16 and self.device.type == "cuda", device_type=self.device.type):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        label=batch["label"],
                    )
                    loss = outputs["loss"] / self.tcfg.grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.tcfg.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), self.tcfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                epoch_loss += outputs["loss"].item()

            elapsed = time.time() - t0
            val_metrics = evaluate(model, val_loader, self.device)
            self.logger.info(
                f"[{major}] P1 Epoch {epoch + 1}/{self.tcfg.epochs_phase1} {elapsed:.1f}s — "
                f"loss={epoch_loss / len(train_loader):.4f}, val_acc={val_metrics.accuracy:.4f}"
            )
            wb.log_metrics(
                {
                    "P1/train_loss": epoch_loss / len(train_loader),
                    "P1/val_acc": val_metrics.accuracy,
                    "P1/epoch": epoch + 1,
                }
            )

            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                if self.tcfg.save_checkpoint:
                    save_sub_classifier(model, output_dir / "best")
                    tokenizer.save_pretrained(output_dir / "best")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.tcfg.early_stopping_patience:
                    self.logger.info(f"[{major}] Early stopping in Phase 1")
                    break

        # Phase 2: Unfreeze BERT, fine-tune all
        self.logger.info(f"[{major}] === Phase 2: Unfreezing BERT, {self.tcfg.epochs_phase2} epochs ===")
        epochs_without_improvement = 0
        unfreeze_bert(model)

        named_bert_params = [(n, p) for n, p in model.bert.named_parameters()]
        head_params = (
            list(model.fc1.parameters())
            + list(model.activation.parameters())
            + list(model.fc1_dropout.parameters())
            + list(model.fc2.parameters())
            + list(model.dropout.parameters())
        )

        optimizer = optim.AdamW(
            [
                {
                    "params": [p for n, p in named_bert_params if not any(nd in n for nd in no_decay)],
                    "lr": self.tcfg.bert_lr,
                    "weight_decay": self.tcfg.weight_decay,
                },
                {
                    "params": [p for n, p in named_bert_params if any(nd in n for nd in no_decay)],
                    "lr": self.tcfg.bert_lr,
                    "weight_decay": 0.0,
                },
                {"params": head_params, "lr": self.tcfg.heads_lr, "weight_decay": self.tcfg.weight_decay},
            ]
        )

        phase2_steps = len(train_loader) * self.tcfg.epochs_phase2 // self.tcfg.grad_accum_steps
        phase2_warmup = int(phase2_steps * self.tcfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, phase2_warmup, phase2_steps)

        cm: list[list[float]] = []
        for epoch in range(self.tcfg.epochs_phase2):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(enabled=self.tcfg.fp16 and self.device.type == "cuda", device_type=self.device.type):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        label=batch["label"],
                    )
                    loss = outputs["loss"] / self.tcfg.grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % self.tcfg.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), self.tcfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                epoch_loss += outputs["loss"].item()

            elapsed = time.time() - t0
            val_metrics = evaluate(model, val_loader, self.device)
            self.logger.info(
                f"[{major}] P2 Epoch {epoch + 1}/{self.tcfg.epochs_phase2} {elapsed:.1f}s — "
                f"loss={epoch_loss / len(train_loader):.4f}, val_acc={val_metrics.accuracy:.4f}"
            )
            wb.log_metrics(
                {
                    "P2/train_loss": epoch_loss / len(train_loader),
                    "P2/val_acc": val_metrics.accuracy,
                    "P2/epoch": epoch + self.tcfg.epochs_phase1 + 1,
                }
            )

            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                if val_metrics._y_true and val_metrics._y_pred:
                    cm = build_confusion_matrix(val_metrics._y_true, val_metrics._y_pred, num_classes)
                if self.tcfg.save_checkpoint:
                    save_sub_classifier(model, output_dir / "best")
                    tokenizer.save_pretrained(output_dir / "best")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.tcfg.early_stopping_patience:
                    self.logger.info(f"[{major}] Early stopping in Phase 2")
                    break

        self.logger.info(f"[{major}] Best val accuracy: {best_val_acc:.4f}")
        if cm:
            cm_normalized = [[v / sum(row) if sum(row) > 0 else 0 for v in row] for row in cm]
            wb.log_confusion_matrix(
                cm, [idx_to_label[i] for i in range(num_classes)], title=f"{major} Confusion Matrix"
            )
            self.logger.info("[Supervised] Confusion matrix (rows=truth, cols=pred):")
            for i, row in enumerate(cm_normalized):
                self.logger.info(f"  {idx_to_label[i]}: " + " ".join(f"{v:.2f}" for v in row))
        wb.log_summary({"best_val_acc": best_val_acc})

        results: dict[str, Any] = {
            "major": major,
            "status": "ok",
            "accuracy": best_val_acc,
        }

        if self.tcfg.save_checkpoint:
            best_dir = output_dir / "best"
            save_sub_classifier(model, best_dir)
            results["model_dir"] = str(best_dir)
            self.logger.info(f"[Supervised] {major} best model saved to {best_dir} (accuracy={best_val_acc:.4f})")

        wb.finish()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
