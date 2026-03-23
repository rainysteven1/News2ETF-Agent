"""Trainer package — LSTM+Attention + LightGBM stacking, FinBERT, and SetFit.

Fully standalone: no dependency on src/.
"""

from trainer.config import (
    DataConfig,
    FinBERTModelConfig,
    FinBERTTrainingConfig,
    SignalsTrainingConfig,
    TrainerConfig,
    load_config as load_trainer_config,
)
from trainer.finbert import (
    FinBERTClassifier,
    FinBERTClassifierConfig,
    IDX_TO_L1,
    L1_CATEGORIES,
    L1_TO_IDX,
    NewsClassificationDataset,
    NewsInferenceDataset,
    SENTIMENT_LABELS,
    load_finbert_classifier,
    preprocess_split,
    train_finbert,
)
from trainer.signals.models import LSTMWithAttention
from trainer.signals.train import (
    build_lgbm_features,
    build_sequences,
    finetune_per_industry,
    run_training,
    train_lgbm_stacking,
    train_lstm_attention_pretrain,
)
from trainer.wandb_handler import WandbHandler
from trainer.setfit import (
    get_major_categories,
    get_sub_categories,
    load_label_stats,
    train_per_major,
)

__all__ = [
    # Config
    "TrainerConfig",
    "SignalsTrainingConfig",
    "FinBERTModelConfig",
    "FinBERTTrainingConfig",
    "DataConfig",
    "load_trainer_config",
    # Shared
    "WandbHandler",
    # Signals (LSTM pipeline)
    "LSTMWithAttention",
    "build_sequences",
    "build_lgbm_features",
    "train_lstm_attention_pretrain",
    "finetune_per_industry",
    "train_lgbm_stacking",
    "run_training",
    # FinBERT
    "FinBERTClassifier",
    "FinBERTClassifierConfig",
    "IDX_TO_L1",
    "L1_CATEGORIES",
    "L1_TO_IDX",
    "NewsClassificationDataset",
    "NewsInferenceDataset",
    "SENTIMENT_LABELS",
    "load_finbert_classifier",
    "preprocess_split",
    "train_finbert",
    # SetFit
    "get_major_categories",
    "get_sub_categories",
    "load_label_stats",
    "train_per_major",
]
