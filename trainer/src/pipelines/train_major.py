"""Training loop for major (L1) classifier — two-phase BERT fine-tune.

Phase 1: Freeze BERT backbone, train L1 + Sentiment heads.
Phase 2: Unfreeze BERT, fine-tune all params with lower BERT LR.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from trainer.src.config import get_config
from trainer.src.datasets.major import (
    IDX_TO_L1,
    L1_TO_IDX,
    SENTIMENT_LABELS,
    NewsClassificationDataset,
    preprocess_split,
)
from trainer.src.models.major import MajorClassifier, load_major_classifier
from trainer.src.utils import WandbRegistry, get_logger


class EvalMetrics:
    def __init__(
        self,
        loss: float,
        l1_accuracy: float,
        sentiment_accuracy: float,
        l1_true: list[int] | None = None,
        l1_pred: list[int] | None = None,
        sent_true: list[int] | None = None,
        sent_pred: list[int] | None = None,
    ):
        self.loss = loss
        self.l1_accuracy = l1_accuracy
        self.sentiment_accuracy = sentiment_accuracy
        self._l1_true = l1_true or []
        self._l1_pred = l1_pred or []
        self._sent_true = sent_true or []
        self._sent_pred = sent_pred or []

    def wandb_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "l1_accuracy": self.l1_accuracy,
            "sentiment_accuracy": self.sentiment_accuracy,
        }


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    l1_correct = 0
    sent_correct = 0
    total = 0
    all_l1_true: list[int] = []
    all_l1_pred: list[int] = []
    all_sent_true: list[int] = []
    all_sent_pred: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                l1_label=batch["l1_label"],
                sentiment_label=batch["sentiment_label"],
            )
            total_loss += outputs["loss"].item() * batch["input_ids"].size(0)
            l1_preds = outputs["l1_logits"].argmax(dim=-1)
            sent_preds = outputs["sentiment_logits"].argmax(dim=-1)
            l1_correct += (l1_preds == batch["l1_label"]).sum().item()
            sent_correct += (sent_preds == batch["sentiment_label"]).sum().item()
            total += batch["input_ids"].size(0)
            all_l1_true.extend(batch["l1_label"].cpu().tolist())
            all_l1_pred.extend(l1_preds.cpu().tolist())
            all_sent_true.extend(batch["sentiment_label"].cpu().tolist())
            all_sent_pred.extend(sent_preds.cpu().tolist())

    return EvalMetrics(
        loss=total_loss / total,
        l1_accuracy=l1_correct / total,
        sentiment_accuracy=sent_correct / total,
        l1_true=all_l1_true,
        l1_pred=all_l1_pred,
        sent_true=all_sent_true,
        sent_pred=all_sent_pred,
    )


def freeze_bert(model: MajorClassifier) -> None:
    for param in model.bert.parameters():
        param.requires_grad = False
    get_logger().info("[Major] BERT backbone frozen (Phase 1)")


def unfreeze_bert(model: MajorClassifier) -> None:
    for param in model.bert.parameters():
        param.requires_grad = True
    get_logger().info("[Major] BERT backbone unfrozen (Phase 2)")


def train_major(device: torch.device) -> None:
    cfg = get_config()
    logger = get_logger()

    logger.info(f"[Major] Device: {device}")

    major_cfg = cfg.major
    dcfg = major_cfg.data
    mcfg = major_cfg.model
    tcfg = major_cfg.training

    run_name = f"major-{datetime.now():%m%d-%H%M}"
    run_output_dir = (
        Path(tcfg.output_dir) / run_name if tcfg.output_dir else Path("trainer/checkpoints/major") / run_name
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    wb = WandbRegistry.get("major")

    tokenizer = AutoTokenizer.from_pretrained(mcfg.pretrained_model)

    assert dcfg.raw_data_dir is not None, "raw_data_dir must be set in config"

    preprocess_split(dcfg.raw_data_dir / "raw.parquet", val_ratio=dcfg.val_ratio, seed=cfg.seed)
    train_ds = NewsClassificationDataset(
        dcfg.raw_data_dir / "train.parquet",
        tokenizer,
        max_length=mcfg.max_seq_length,
        use_content=dcfg.use_content,
    )
    val_ds = NewsClassificationDataset(
        dcfg.raw_data_dir / "val.parquet",
        tokenizer,
        max_length=mcfg.max_seq_length,
        use_content=dcfg.use_content,
    )
    logger.info(f"[Major] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=dcfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=dcfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = load_major_classifier(
        pretrained_model=mcfg.pretrained_model,
        num_level1=mcfg.num_level1,
        num_sentiment=mcfg.num_sentiment,
        dropout=mcfg.dropout,
        alpha=0.1,
        gamma=0.1,
    )
    model.to(device)

    scaler = GradScaler(enabled=tcfg.fp16 and device == "cuda", device=device.type)
    patience = tcfg.early_stopping_patience
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    best_val_l1_acc = 0.0
    best_val_sent_acc = 0.0
    epochs_without_improvement = 0
    global_step = 0

    # Phase 1: Freeze BERT, train heads
    logger.info(f"=== Phase 1: {tcfg.epochs_phase1} epochs with frozen BERT ===")
    freeze_bert(model)

    heads_params_decay = []
    heads_params_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            heads_params_no_decay.append(param)
        else:
            heads_params_decay.append(param)

    optimizer = optim.AdamW(
        [
            {"params": heads_params_decay, "lr": tcfg.heads_lr, "weight_decay": tcfg.weight_decay},
            {"params": heads_params_no_decay, "lr": tcfg.heads_lr, "weight_decay": 0.0},
        ]
    )

    phase1_steps = len(train_loader) * tcfg.epochs_phase1 // tcfg.grad_accum_steps
    phase1_warmup = int(phase1_steps * tcfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, phase1_warmup, phase1_steps)

    for epoch in range(tcfg.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=tcfg.fp16 and device.type == "cuda", device_type=device.type):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    l1_label=batch["l1_label"],
                    sentiment_label=batch["sentiment_label"],
                )
                loss = outputs["loss"] / tcfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % tcfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += outputs["loss"].item()

            if global_step > 0 and global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"  [P1 {epoch + 1}] batch= {step + 1}/{len(train_loader)} "
                    f"loss={outputs['loss'].item():.4f} "
                    f"l1={outputs.get('l1_loss', torch.tensor(0)).item():.4f} "
                    f"sent={outputs.get('sentiment_loss', torch.tensor(0)).item():.4f} "
                    f"lr={lr:.2e}"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"P1 Epoch {epoch + 1}/{tcfg.epochs_phase1} {elapsed:.1f}s — "
            f"loss={avg_loss:.4f}, val_l1_acc={val_metrics.l1_accuracy:.4f}, "
            f"val_sent_acc={val_metrics.sentiment_accuracy:.4f}"
        )
        wb.log_metrics(
            {
                "P1/train_l1_loss": outputs.get("l1_loss", torch.tensor(0)).item(),
                "P1/train_sent_loss": outputs.get("sentiment_loss", torch.tensor(0)).item(),
                "P1/val_l1_acc": val_metrics.l1_accuracy,
                "P1/val_sent_acc": val_metrics.sentiment_accuracy,
                "P1/epoch": epoch + 1,
            },
        )

        improved = val_metrics.l1_accuracy > best_val_l1_acc
        if improved:
            best_val_l1_acc = val_metrics.l1_accuracy
            best_val_sent_acc = val_metrics.sentiment_accuracy
            if tcfg.save_checkpoint:
                best_dir = run_output_dir / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"  Best saved (l1_acc={best_val_l1_acc:.4f})")
            else:
                logger.info(f"  Best (l1_acc={best_val_l1_acc:.4f}, checkpoint skipped)")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("[Major] Early stopping triggered in Phase 1.")
                break

    # Phase 2: Unfreeze BERT, fine-tune all
    logger.info("=== Phase 2: Unfreezing BERT, training all params ===")
    epochs_without_improvement = 0
    unfreeze_bert(model)

    bert_params_decay = []
    bert_params_no_decay = []
    heads_params_decay = []
    heads_params_no_decay = []

    for name, param in model.named_parameters():
        if not any(nd in name for nd in no_decay):
            if "bert." in name:
                bert_params_decay.append(param)
            else:
                heads_params_decay.append(param)
        else:
            if "bert." in name:
                bert_params_no_decay.append(param)
            else:
                heads_params_no_decay.append(param)

    optimizer = optim.AdamW(
        [
            {"params": bert_params_decay, "lr": tcfg.bert_lr, "weight_decay": tcfg.weight_decay},
            {"params": bert_params_no_decay, "lr": tcfg.bert_lr, "weight_decay": 0.0},
            {"params": heads_params_decay, "lr": tcfg.heads_lr, "weight_decay": tcfg.weight_decay},
            {"params": heads_params_no_decay, "lr": tcfg.heads_lr, "weight_decay": 0.0},
        ]
    )

    phase2_steps = len(train_loader) * tcfg.epochs_phase2 // tcfg.grad_accum_steps
    phase2_warmup = int(phase2_steps * tcfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, phase2_warmup, phase2_steps)

    for epoch in range(tcfg.epochs_phase2):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=tcfg.fp16 and device.type == "cuda", device_type=device.type):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    l1_label=batch["l1_label"],
                    sentiment_label=batch["sentiment_label"],
                )
                loss = outputs["loss"] / tcfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % tcfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += outputs["loss"].item()

            if global_step > 0 and global_step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"  [P2 {epoch + 1}] batch= {step + 1}/{len(train_loader)} "
                    f"loss={outputs['loss'].item():.4f} "
                    f"l1={outputs.get('l1_loss', torch.tensor(0)).item():.4f} "
                    f"sent={outputs.get('sentiment_loss', torch.tensor(0)).item():.4f} "
                    f"lr={lr:.2e}"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"P2 Epoch {epoch + 1}/{tcfg.epochs_phase2} {elapsed:.1f}s — "
            f"loss={avg_loss:.4f}, val_l1_acc={val_metrics.l1_accuracy:.4f}, "
            f"val_sent_acc={val_metrics.sentiment_accuracy:.4f}"
        )
        wb.log_metrics(
            {
                "P2/train_l1_loss": outputs.get("l1_loss", torch.tensor(0)).item(),
                "P2/train_sent_loss": outputs.get("sentiment_loss", torch.tensor(0)).item(),
                "P2/val_l1_acc": val_metrics.l1_accuracy,
                "P2/val_sent_acc": val_metrics.sentiment_accuracy,
                "P2/epoch": epoch + tcfg.epochs_phase1 + 1,
            },
        )

        improved = val_metrics.l1_accuracy > best_val_l1_acc
        if improved:
            best_val_l1_acc = val_metrics.l1_accuracy
            best_val_sent_acc = val_metrics.sentiment_accuracy
            if tcfg.save_checkpoint:
                best_dir = run_output_dir / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"Best saved (l1_acc={best_val_l1_acc:.4f})")
            else:
                logger.info(f"Best (l1_acc={best_val_l1_acc:.4f}, checkpoint skipped)")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("[Major] Early stopping triggered in Phase 2.")
                break

    label_info = {
        "l1_to_idx": L1_TO_IDX,
        "idx_to_l1": {str(k): v for k, v in IDX_TO_L1.items()},
        "sentiment_labels": SENTIMENT_LABELS,
    }
    with open(run_output_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)

    wb.log_summary(
        {
            "best_val_l1_acc": best_val_l1_acc,
            "best_val_sent_acc": best_val_sent_acc,
        }
    )

    logger.info(f"\n[Major] Training complete. Best val L1 accuracy: {best_val_l1_acc:.4f}")
