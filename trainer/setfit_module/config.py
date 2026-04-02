"""SetFit model and training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Data ───────────────────────────────────────────────────────────────────
class SetFitDataConfig(BaseModel):
    raw_data_path: Path | None = None
    batch_size: int = 16
    val_ratio: float = 0.2

    def to_wandb(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "val_ratio": self.val_ratio,
        }


# ─── Model ───────────────────────────────────────────────────────────────────


class SetFitModelConfig(BaseModel):
    pretrained_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    label_stats: Path = _ROOT / "trainer" / "label_stats.json"
    max_seq_length: int = 256

    def to_wandb(self) -> dict[str, Any]:
        return {
            "pretrained_model": self.pretrained_model,
            "max_seq_length": self.max_seq_length,
        }


# ─── Training ─────────────────────────────────────────────────────────────────


class SetFitTrainingConfig(BaseModel):
    output_dir: Path | None = None
    num_iterations: int = 20
    num_epochs: int = 1
    learning_rate: float = 2e-5
    min_samples_per_class: int = 2

    def to_wandb(self) -> dict[str, Any]:
        return {
            "num_iterations": self.num_iterations,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "min_samples_per_class": self.min_samples_per_class,
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


class LabelStats:
    """Load and access label_stats.json (singleton)."""

    _instance: LabelStats | None = None
    _initialized: bool = False

    def __new__(cls, stats_path: Path | None = None) -> LabelStats:
        if cls._instance is None:
            instance = super().__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self, stats_path: Path | None = None):
        if LabelStats._initialized:
            return
        if stats_path is None:
            cfg = load_setfit_config()
            stats_path = cfg.model.label_stats
        with open(stats_path, encoding="utf-8") as f:
            self._stats: dict[str, Any] = json.load(f)
        LabelStats._initialized = True

    def get_major_categories(self) -> list[str]:
        return sorted(self._stats["major_category"].keys())

    def get_sub_categories(self, major: str) -> list[str]:
        return sorted(self._stats["sub_category_by_major"][major].keys())
