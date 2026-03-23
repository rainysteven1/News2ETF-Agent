"""Training loop for FinBERT (8 L1 classes + 3 sentiment).

Two-phase training:
  Phase 1: Freeze BERT backbone, train L1 + Sentiment heads.
  Phase 2: Unfreeze BERT, fine-tune all params with lower BERT LR.

Usage:
    python -m trainer.finbert.train
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from trainer.config import FinBERTModelConfig, FinBERTTrainingConfig, TrainerConfig
from trainer.finbert.dataset import (
    IDX_TO_L1,
    L1_TO_IDX,
    SENTIMENT_LABELS,
    NewsClassificationDataset,
    preprocess_split,
)
from trainer.finbert.model import FinBERTClassifier, load_finbert_classifier
from trainer.wandb_handler import WandbHandler

# ─── Helpers ───────────────────────────────────────────────────────────────────


class EvalMetrics:
    """Evaluation metrics returned by the evaluate() function."""

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> EvalMetrics:
    """Run evaluation, return loss / accuracy metrics."""
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


# ─── Phase control ───────────────────────────────────────────────────────────────


def freeze_bert(model: FinBERTClassifier) -> None:
    for param in model.bert.parameters():
        param.requires_grad = False
    logger.info("[FinBERT] BERT backbone frozen (Phase 1)")


def unfreeze_bert(model: FinBERTClassifier) -> None:
    for param in model.bert.parameters():
        param.requires_grad = True
    logger.info("[FinBERT] BERT backbone unfrozen (Phase 2)")


# ─── Main training function ───────────────────────────────────────────────────


def train_finbert(
    data_path: Path,
    output_dir: Path | None = None,
    model_config: FinBERTModelConfig | None = None,
    training_config: FinBERTTrainingConfig | None = None,
    wandb_project: str = "news2etf",
    wandb_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Train FinBERT on labeled news data.

    Returns dict of saved checkpoint paths.
    """
    from trainer.config import load_config as load_trainer_config

    cfg = load_trainer_config(config_path) if config_path else TrainerConfig()

    mcfg = model_config or cfg.finbert
    tcfg = training_config or cfg.finbert_training

    set_seed(tcfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[FinBERT] Device: {device}")

    wb = WandbHandler(
        project=wandb_project,
        name=wandb_name or f"finbert-{datetime.now():%m%d-%H%M}",
        config_dict={
            "pretrained_model": mcfg.pretrained_model,
            "num_level1": mcfg.num_level1,
            "num_sentiment": mcfg.num_sentiment,
            "max_seq_length": mcfg.max_seq_length,
            "dropout": mcfg.dropout,
            "batch_size": tcfg.batch_size,
            "bert_lr": tcfg.bert_lr,
            "heads_lr": tcfg.heads_lr,
            "epochs_phase1": tcfg.epochs_phase1,
            "epochs_phase2": tcfg.epochs_phase2,
        },
        tags=["finbert"],
    )

    run_output_dir = output_dir or (Path("checkpoints") / "finbert")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(mcfg.pretrained_model)

    preprocess_split(data_path, run_output_dir, val_ratio=0.15, seed=tcfg.seed)
    train_ds = NewsClassificationDataset(
        run_output_dir / "train.parquet",
        tokenizer,
        max_length=mcfg.max_seq_length,
        l1_to_idx=L1_TO_IDX,
        use_content=tcfg.use_content,
    )
    val_ds = NewsClassificationDataset(
        run_output_dir / "val.parquet",
        tokenizer,
        max_length=mcfg.max_seq_length,
        l1_to_idx=L1_TO_IDX,
        use_content=tcfg.use_content,
    )
    logger.info(f"[FinBERT] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = load_finbert_classifier(
        pretrained_model=mcfg.pretrained_model,
        num_level1=mcfg.num_level1,
        num_sentiment=mcfg.num_sentiment,
        dropout=mcfg.dropout,
        alpha=0.1,
        gamma=0.1,
    )
    model.to(device)

    scaler = GradScaler(enabled=tcfg.fp16 and device.type == "cuda")

    # ═══ Phase 1: Freeze BERT, train heads ══════════════════════════════════════
    freeze_bert(model)

    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    heads_params_decay = []
    heads_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            heads_params_no_decay.append(param)
        else:
            heads_params_decay.append(param)

    optimizer = AdamW(
        [
            {"params": heads_params_decay, "lr": tcfg.heads_lr, "weight_decay": tcfg.weight_decay},
            {"params": heads_params_no_decay, "lr": tcfg.heads_lr, "weight_decay": 0.0},
        ]
    )

    phase1_steps = len(train_loader) * tcfg.epochs_phase1 // tcfg.grad_accum_steps
    phase1_warmup = int(phase1_steps * tcfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, phase1_warmup, phase1_steps)

    best_val_l1_acc = 0.0
    global_step = 0

    logger.info(f"=== Phase 1: {tcfg.epochs_phase1} epochs with frozen BERT ===")
    for epoch in range(tcfg.epochs_phase1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=tcfg.fp16 and device.type == "cuda"):
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
                    f"  [P1 {epoch + 1}] step {global_step} "
                    f"loss={outputs['loss'].item():.4f} "
                    f"l1={outputs['l1_loss'].item():.4f} "
                    f"sent={outputs.get('sentiment_loss', torch.tensor(0)).item():.4f} "
                    f"lr={lr:.2e}"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"P1 Epoch {epoch + 1}/{tcfg.epochs_phase1} {elapsed:.1f}s — "
            f"loss={avg_loss:.4f}, val_l1={val_metrics.l1_accuracy:.4f}, "
            f"val_sent={val_metrics.sentiment_accuracy:.4f}"
        )
        wb.log_epoch(
            "P1",
            epoch + 1,
            avg_loss,
            {
                "train/l1_loss": outputs.get("l1_loss", torch.tensor(0)).item(),
                "train/sentiment_loss": outputs.get("sentiment_loss", torch.tensor(0)).item(),
                "epoch/val_l1_accuracy": val_metrics.l1_accuracy,
                "epoch/val_sentiment_accuracy": val_metrics.sentiment_accuracy,
            },
        )

        if val_metrics.l1_accuracy > best_val_l1_acc:
            best_val_l1_acc = val_metrics.l1_accuracy
            best_dir = run_output_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.success(f"  ✓ Best saved (l1_acc={best_val_l1_acc:.4f})")

    # ═══ Phase 2: Unfreeze BERT, fine-tune all ═══════════════════════════════════
    logger.info("=== Phase 2: Unfreezing BERT, training all params ===")
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

    optimizer = AdamW(
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

            with autocast(enabled=tcfg.fp16 and device.type == "cuda"):
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
                    f"  [P2 {epoch + 1}] step {global_step} "
                    f"loss={outputs['loss'].item():.4f} "
                    f"l1={outputs['l1_loss'].item():.4f} "
                    f"sent={outputs.get('sentiment_loss', torch.tensor(0)).item():.4f} "
                    f"lr={lr:.2e}"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"P2 Epoch {epoch + 1}/{tcfg.epochs_phase2} {elapsed:.1f}s — "
            f"loss={avg_loss:.4f}, val_l1={val_metrics.l1_accuracy:.4f}, "
            f"val_sent={val_metrics.sentiment_accuracy:.4f}"
        )
        wb.log_epoch(
            "P2",
            epoch + tcfg.epochs_phase1 + 1,
            avg_loss,
            {
                "train/l1_loss": outputs.get("l1_loss", torch.tensor(0)).item(),
                "train/sentiment_loss": outputs.get("sentiment_loss", torch.tensor(0)).item(),
                "epoch/val_l1_accuracy": val_metrics.l1_accuracy,
                "epoch/val_sentiment_accuracy": val_metrics.sentiment_accuracy,
            },
        )

        if val_metrics.l1_accuracy > best_val_l1_acc:
            best_val_l1_acc = val_metrics.l1_accuracy
            best_dir = run_output_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.success(f"  ✓ Best saved (l1_acc={best_val_l1_acc:.4f})")

    # ── Save final model ─────────────────────────────────────────────
    final_dir = run_output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    label_info = {
        "l1_to_idx": L1_TO_IDX,
        "idx_to_l1": {str(k): v for k, v in IDX_TO_L1.items()},
        "sentiment_labels": SENTIMENT_LABELS,
    }
    with open(run_output_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)

    wb.log_summary(
        {
            "best_val_l1_accuracy": best_val_l1_acc,
            "final_val_l1_accuracy": val_metrics.l1_accuracy,
            "final_val_sentiment_accuracy": val_metrics.sentiment_accuracy,
        }
    )
    wb.finish()

    logger.info(f"\n[FinBERT] Training complete. Best val L1 accuracy: {best_val_l1_acc:.4f}")
    logger.info(f"[FinBERT] Checkpoints: {run_output_dir}")

    return {"best": run_output_dir / "best", "final": run_output_dir / "final"}


# ─── CLI ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def train(
        data: str = typer.Option(..., help="Path to labeled news parquet file"),
        config: str | None = None,
        output: str | None = None,
        project: str = "news2etf",
        name: str | None = None,
    ) -> None:
        """Train FinBERT on labeled news data."""
        train_finbert(
            data_path=Path(data),
            output_dir=Path(output) if output else None,
            config_path=config,
            wandb_project=project,
            wandb_name=name,
        )

    app()
