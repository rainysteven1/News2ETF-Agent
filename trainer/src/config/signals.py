"""Signals pipeline configuration — TCN, LightGBM, IsolationForest, dataset, OHLCV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

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
    """Compatibility loader that delegates to the shared root config."""
    from trainer.src.config.root import load_config

    return load_config(path).signals
