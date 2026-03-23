"""FinBERT — 8 large categories + 3 sentiment classification."""

from trainer.finbert.dataset import (
    IDX_TO_L1,
    L1_CATEGORIES,
    L1_TO_IDX,
    NewsClassificationDataset,
    NewsInferenceDataset,
    SENTIMENT_LABELS,
    preprocess_split,
)
from trainer.finbert.model import FinBERTClassifier, FinBERTClassifierConfig, load_finbert_classifier
from trainer.finbert.predict import predict as finbert_predict
from trainer.finbert.train import train_finbert

__all__ = [
    "L1_CATEGORIES",
    "L1_TO_IDX",
    "IDX_TO_L1",
    "SENTIMENT_LABELS",
    "NewsClassificationDataset",
    "NewsInferenceDataset",
    "preprocess_split",
    "FinBERTClassifier",
    "FinBERTClassifierConfig",
    "load_finbert_classifier",
    "train_finbert",
    "finbert_predict",
]
