"""Datasets package — major (L1) and sub (L2) datasets."""

from trainer.src.datasets.major import (
    IDX_TO_L1,
    L1_CATEGORIES,
    L1_TO_IDX,
    SENTIMENT_LABELS,
    SENTIMENT_STR_TO_INT,
    NewsClassificationDataset,
    preprocess_split,
)
from trainer.src.datasets.signals import (
    WeeklySignalDataset,
    build_sub_category_sequences,
    compute_global_leader_sentiment,
    compute_market_beta,
    export_phase2_dataset,
)
from trainer.src.datasets.signals import (
    build_lgbm_features as build_signals_lgbm_features,
)
from trainer.src.datasets.signals import (
    build_sequences as build_signals_sequences,
)
from trainer.src.datasets.sub import (
    SetFitDatasetPreparer,
    SubCatDataset,
)
from trainer.src.datasets.sub import (
    preprocess_split as subcat_preprocess_split,
)

__all__ = [
    "NewsClassificationDataset",
    "preprocess_split",
    "L1_CATEGORIES",
    "L1_TO_IDX",
    "IDX_TO_L1",
    "SENTIMENT_LABELS",
    "SENTIMENT_STR_TO_INT",
    "WeeklySignalDataset",
    "build_signals_lgbm_features",
    "build_signals_sequences",
    "build_sub_category_sequences",
    "compute_global_leader_sentiment",
    "compute_market_beta",
    "export_phase2_dataset",
    "SubCatDataset",
    "subcat_preprocess_split",
    "SetFitDatasetPreparer",
]
