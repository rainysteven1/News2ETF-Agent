"""Major (L1) model and training configuration — Level1 major category + sentiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class MajorDataConfig(BaseModel):
    raw_data_dir: Path | None = None
    batch_size: int = 32
    use_content: bool = False
    val_ratio: float = 0.15

    def to_wandb(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "use_content": self.use_content,
            "val_ratio": self.val_ratio,
        }


class MajorModelConfig(BaseModel):
    pretrained_model: str = "bert-base-chinese"
    num_level1: int = 8
    num_sentiment: int = 3
    max_seq_length: int = 128
    dropout: float = 0.1

    def to_wandb(self) -> dict[str, Any]:
        return {
            "pretrained_model": self.pretrained_model,
            "num_level1": self.num_level1,
            "num_sentiment": self.num_sentiment,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout,
        }


class MajorTrainingConfig(BaseModel):
    output_dir: Path | None = None
    early_stopping_patience: int = 1
    epochs_phase1: int = 3
    epochs_phase2: int = 5
    bert_lr: float = 2e-5
    heads_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    save_checkpoint: bool = True

    def to_wandb(self) -> dict[str, Any]:
        return {
            "early_stopping_patience": self.early_stopping_patience,
            "epochs_phase1": self.epochs_phase1,
            "epochs_phase2": self.epochs_phase2,
            "bert_lr": self.bert_lr,
            "heads_lr": self.heads_lr,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "grad_accum_steps": self.grad_accum_steps,
            "max_grad_norm": self.max_grad_norm,
            "fp16": self.fp16,
            "save_checkpoint": self.save_checkpoint,
        }


class MajorConfig(BaseModel):
    data: MajorDataConfig = MajorDataConfig()
    model: MajorModelConfig = MajorModelConfig()
    training: MajorTrainingConfig = MajorTrainingConfig()

    def to_wandb(self) -> dict[str, Any]:
        return {
            "data": self.data.to_wandb(),
            "model": self.model.to_wandb(),
            "training": self.training.to_wandb(),
        }
