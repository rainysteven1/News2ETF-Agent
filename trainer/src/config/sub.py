"""Sub (L2) model and training configuration — split into setfit and supervised."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

# ── SetFit config ──────────────────────────────────────────────────────────────


class SetFitDataConfig(BaseModel):
    raw_data_dir: Path | None = None
    batch_size: int = 32
    val_ratio: float = 0.2
    cluster_sampling_majors: list[str] = ["科技信息", "基础设施/公共", "主题策略"]
    prepare_max_workers: int = 4
    confidence_floor: float = 0.75
    extra_args: dict[str, Any] = {}

    class RandomConfig(BaseModel):
        samples_per_class: int = 50
        min_samples_per_class: int = 10

    class ClusterConfig(BaseModel):
        n_cap: int = 1000
        n_clusters: int = 300
        samples_per_cluster: int = 3
        min_samples_per_class: int = 30
        hard_negative_boost: int = 3
        confused_pairs: list[list[str]] = [
            ["区域经济", "宽基/策略"],
            ["宽基/策略", "民企/综合策略"],
            ["民企/综合策略", "区域经济"],
        ]

    random: RandomConfig = RandomConfig()
    cluster: ClusterConfig = ClusterConfig()

    def to_wandb(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "val_ratio": self.val_ratio,
            "confidence_floor": self.confidence_floor,
        }


class SetFitModelConfig(BaseModel):
    pretrained_model: Path | str = "./trainer/data/pretrained_models/mengzi-bert-base-fin"
    label_stats: Path | str = "./trainer/data/labeled/setfit/label_stats.json"
    max_seq_length: int = 128

    def to_wandb(self) -> dict[str, Any]:
        return {
            "pretrained_model": self.pretrained_model,
            "max_seq_length": self.max_seq_length,
        }


class SetFitTrainingConfig(BaseModel):
    output_dir: Path | None = None
    num_iterations: int = 10
    num_epochs: int = 1
    learning_rate: float = 7.5e-6
    save_checkpoint: bool = False

    def to_wandb(self) -> dict[str, Any]:
        return {
            "num_iterations": self.num_iterations,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
        }


class SetFitConfig(BaseModel):
    data: SetFitDataConfig = SetFitDataConfig()
    model: SetFitModelConfig = SetFitModelConfig()
    training: SetFitTrainingConfig = SetFitTrainingConfig()

    def to_wandb(self) -> dict[str, Any]:
        return {
            "data": self.data.to_wandb(),
            "model": self.model.to_wandb(),
            "training": self.training.to_wandb(),
        }


# ── Supervised config ──────────────────────────────────────────────────────────


class SupervisedDataConfig(BaseModel):
    raw_data_dir: Path | None = None
    batch_size: int = 32
    val_ratio: float = 0.15
    max_seq_length: int = 128
    use_content: bool = True

    def to_wandb(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "val_ratio": self.val_ratio,
            "max_seq_length": self.max_seq_length,
            "use_content": self.use_content,
        }


class SupervisedModelConfig(BaseModel):
    pretrained_model: Path | str = "bert-base-chinese"
    dropout: float = 0.1

    def to_wandb(self) -> dict[str, Any]:
        return {
            "pretrained_model": self.pretrained_model,
            "dropout": self.dropout,
        }


class SupervisedTrainingConfig(BaseModel):
    output_dir: Path | None = None
    epochs_phase1: int = 3
    epochs_phase2: int = 5
    bert_lr: float = 2e-5
    heads_lr: float = 1e-4
    weight_decay: float = 0.02
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    early_stopping_patience: int = 2
    focal_loss_gamma: float = 2.0
    save_checkpoint: bool = True

    def to_wandb(self) -> dict[str, Any]:
        return {
            "epochs_phase1": self.epochs_phase1,
            "epochs_phase2": self.epochs_phase2,
            "bert_lr": self.bert_lr,
            "heads_lr": self.heads_lr,
            "focal_loss_gamma": self.focal_loss_gamma,
        }


class SupervisedConfig(BaseModel):
    data: SupervisedDataConfig = SupervisedDataConfig()
    model: SupervisedModelConfig = SupervisedModelConfig()
    training: SupervisedTrainingConfig = SupervisedTrainingConfig()

    def to_wandb(self) -> dict[str, Any]:
        return {
            "data": self.data.to_wandb(),
            "model": self.model.to_wandb(),
            "training": self.training.to_wandb(),
        }


# ── SubConfig (container) ──────────────────────────────────────────────────────


class SubConfig(BaseModel):
    setfit: SetFitConfig = SetFitConfig()
    supervised: SupervisedConfig = SupervisedConfig()

    def to_wandb(self) -> dict[str, Any]:
        return {
            "setfit": self.setfit.to_wandb(),
            "supervised": self.supervised.to_wandb(),
        }
