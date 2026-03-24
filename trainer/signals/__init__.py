"""Trainer signals sub-package — TCN + LightGBM stacking."""

from trainer.signals.models import TCN
from trainer.signals.train import (
    build_lgbm_features,
    build_sequences,
    finetune_per_industry,
    run_training,
    train_lgbm_stacking,
    train_tcn_pretrain,
)

__all__ = [
    "TCN",
    "build_sequences",
    "build_lgbm_features",
    "train_tcn_pretrain",
    "finetune_per_industry",
    "train_lgbm_stacking",
    "run_training",
]
