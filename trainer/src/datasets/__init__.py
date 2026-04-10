"""Datasets package with lazy exports.

This keeps runtime consumers of ``trainer.src.datasets.signals`` from
eagerly importing the major/sub training stacks and their heavy dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

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

_MODULE_BY_NAME = {
    "NewsClassificationDataset": "trainer.src.datasets.major",
    "preprocess_split": "trainer.src.datasets.major",
    "L1_CATEGORIES": "trainer.src.datasets.major",
    "L1_TO_IDX": "trainer.src.datasets.major",
    "IDX_TO_L1": "trainer.src.datasets.major",
    "SENTIMENT_LABELS": "trainer.src.datasets.major",
    "SENTIMENT_STR_TO_INT": "trainer.src.datasets.major",
    "WeeklySignalDataset": "trainer.src.datasets.signals",
    "build_signals_lgbm_features": "trainer.src.datasets.signals",
    "build_signals_sequences": "trainer.src.datasets.signals",
    "build_sub_category_sequences": "trainer.src.datasets.signals",
    "compute_global_leader_sentiment": "trainer.src.datasets.signals",
    "compute_market_beta": "trainer.src.datasets.signals",
    "export_phase2_dataset": "trainer.src.datasets.signals",
    "SubCatDataset": "trainer.src.datasets.sub",
    "subcat_preprocess_split": "trainer.src.datasets.sub",
    "SetFitDatasetPreparer": "trainer.src.datasets.sub",
}

_ALIASES = {
    "build_signals_lgbm_features": "build_lgbm_features",
    "build_signals_sequences": "build_sequences",
    "subcat_preprocess_split": "preprocess_split",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    target_name = _ALIASES.get(name, name)
    value = getattr(module, target_name)
    globals()[name] = value
    return value
