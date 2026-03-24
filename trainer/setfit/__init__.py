"""Trainer setfit sub-package — per-major SetFit sub-category classifiers."""

from trainer.setfit.model import LabelStats
from trainer.setfit.train import train_per_major

__all__ = [
    "LabelStats",
    "train_per_major",
]
