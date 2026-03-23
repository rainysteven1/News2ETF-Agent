"""Trainer setfit sub-package — per-major SetFit sub-category classifiers."""

from trainer.setfit.model import get_major_categories, get_sub_categories, load_label_stats
from trainer.setfit.train import train_per_major

__all__ = [
    "get_major_categories",
    "get_sub_categories",
    "load_label_stats",
    "train_per_major",
]
