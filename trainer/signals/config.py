"""Signals pipeline configuration — TCN, LightGBM, IsolationForest, dataset, OHLCV."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── TCN ──────────────────────────────────────────────────────────────────────


class SignalsTCNConfig(BaseModel):
    sequence_length: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


# ─── Training ──────────────────────────────────────────────────────────────────


class SignalsTrainingConfig(BaseModel):
    epochs_pretrain: int = 15
    epochs_finetune: int = 10
    batch_size: int = 64
    lr: float = 0.001
    num_heads: int = 4
    anomaly_threshold: float = 0.03
    output_checkpoint: Path | None = None


# ─── IsolationForest ───────────────────────────────────────────────────────────


class SignalsIsolationForestConfig(BaseModel):
    contamination: float = 0.1
    n_estimators: int = 100


# ─── LightGBM ─────────────────────────────────────────────────────────────────


class SignalsLightGBMConfig(BaseModel):
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200


# ─── Dataset ──────────────────────────────────────────────────────────────────


class SignalsDatasetConfig(BaseModel):
    raw_data_path: Path | None = None
    output_sentiment: Path | None = None
    train_end_week: str = "2021-01-03"
    freq: str = "weekly"
    cross_industry: bool = True


# ─── OHLCV ───────────────────────────────────────────────────────────────────


class SignalsOhlcvConfig(BaseModel):
    ohlcv_path: Path | None = None
    industry_dict_path: Path | None = None
    etf_info_path: Path | None = None


# ─── Signals Root Config ──────────────────────────────────────────────────────


class SignalsConfig(BaseModel):
    tcn: SignalsTCNConfig = SignalsTCNConfig()
    training: SignalsTrainingConfig = SignalsTrainingConfig()
    isolation_forest: SignalsIsolationForestConfig = SignalsIsolationForestConfig()
    lightgbm: SignalsLightGBMConfig = SignalsLightGBMConfig()
    dataset: SignalsDatasetConfig = SignalsDatasetConfig()
    ohlcv: SignalsOhlcvConfig = SignalsOhlcvConfig()


def load_signals_config(path: str | Path | None = None) -> SignalsConfig:
    """Load signals config from trainer/config.toml."""
    if path is None:
        path = _ROOT / "trainer" / "config.toml"
    path = Path(path)

    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    toml_section = raw.get("signals", {})

    filtered: dict[str, Any] = {}
    toml_to_field = {
        "tcn": "tcn",
        "training": "training",
        "isolation_forest": "isolation_forest",
        "lightgbm": "lightgbm",
        "dataset": "dataset",
        "ohlcv": "ohlcv",
    }
    for toml_key, field_name in toml_to_field.items():
        if toml_key in toml_section:
            filtered[field_name] = toml_section[toml_key]

    cfg = SignalsConfig.model_validate(filtered)

    # Resolve relative paths
    dataset_section = toml_section.get("dataset", {})
    if "raw_data_path" in dataset_section:
        cfg.dataset.raw_data_path = _ROOT / dataset_section["raw_data_path"]
    if "output_sentiment" in dataset_section:
        cfg.dataset.output_sentiment = _ROOT / dataset_section["output_sentiment"]

    training_section = toml_section.get("training", {})
    if "output_checkpoint" in training_section:
        cfg.training.output_checkpoint = _ROOT / training_section["output_checkpoint"]

    ohlcv_section = toml_section.get("ohlcv", {})
    for key in ("ohlcv_path", "industry_dict_path", "etf_info_path"):
        if key in ohlcv_section:
            setattr(cfg.ohlcv, key, _ROOT / ohlcv_section[key])

    return cfg
