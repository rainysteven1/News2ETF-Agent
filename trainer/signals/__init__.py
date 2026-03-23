"""Trainer signals sub-package — LSTM+Attention + LightGBM stacking."""

from trainer.signals.models import LSTMWithAttention
from trainer.signals.train import (
    build_lgbm_features,
    build_sequences,
    finetune_per_industry,
    run_training,
    train_lgbm_stacking,
    train_lstm_attention_pretrain,
)

__all__ = [
    "LSTMWithAttention",
    "build_sequences",
    "build_lgbm_features",
    "train_lstm_attention_pretrain",
    "finetune_per_industry",
    "train_lgbm_stacking",
    "run_training",
]
