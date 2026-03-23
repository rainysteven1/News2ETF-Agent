"""Trainer configuration — Pydantic models loaded from config.toml (project root).

No dependency on src/ — trainer is fully standalone.

TOML structure:
  [signals.model]       → LSTMWithAttention 超参
  [signals.training]     → LSTM 训练超参
  [signals.isolation_forest]
  [signals.lightgbm]
  [finbert.model]        → FinBERT 模型超参
  [finbert.training]     → FinBERT 训练超参
  [data]                 → 数据输出路径
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent


# ─── Signals (LSTM pipeline) ────────────────────────────────────────────────────────


class SignalsModelConfig(BaseModel):
    """LSTM + Attention 超参."""

    sequence_length: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class SignalsTrainingConfig(BaseModel):
    """LSTM + Attention 训练超参."""

    epochs_pretrain: int = 15
    epochs_finetune: int = 10
    batch_size: int = 64
    lr: float = 0.001
    num_heads: int = 4
    anomaly_threshold: float = 0.03


class SignalsIsolationForestConfig(BaseModel):
    contamination: float = 0.1
    n_estimators: int = 100


class SignalsLightGBMConfig(BaseModel):
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200


# ─── FinBERT ──────────────────────────────────────────────────────────────────────


class FinBERTModelConfig(BaseModel):
    pretrained_model: str = "bert-base-chinese"
    num_level1: int = 8
    num_sentiment: int = 3
    max_seq_length: int = 128
    dropout: float = 0.1


class FinBERTTrainingConfig(BaseModel):
    raw_data_path: Path | None = None
    output_dir: Path | None = None
    batch_size: int = 32
    early_stopping_patience: int = 1
    epochs_phase1: int = 3
    epochs_phase2: int = 5
    bert_lr: float = 2e-5
    heads_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    use_content: bool = False


# ─── SetFit ─────────────────────────────────────────────────────────────────────────


class SetFitModelConfig(BaseModel):
    """SetFit 模型超参."""

    pretrained_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class SetFitTrainingConfig(BaseModel):
    """SetFit 训练超参."""

    raw_data_path: Path | None = None
    test_size: float = 0.2
    seed: int = 42
    batch_size: int = 16
    num_iterations: int = 20
    num_epochs: int = 1
    learning_rate: float = 2e-5
    min_samples_per_class: int = 2


# ─── WandB ──────────────────────────────────────────────────────────────────────


class WandbConfig(BaseModel):
    project: str = "news2etf"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"  # "online" | "offline" | "disabled"


# ─── Data paths ──────────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    output_sentiment: Path = _ROOT / "data" / "industry_sentiment.parquet"
    output_signals: Path = _ROOT / "data" / "ml_signals.parquet"
    output_backtest: Path = _ROOT / "data" / "backtest_results.parquet"
    output_weekly_returns: Path = _ROOT / "data" / "weekly_returns.parquet"


# ─── Root config ───────────────────────────────────────────────────────────────────


class TrainerConfig(BaseModel):
    wandb: WandbConfig = WandbConfig()
    signals: SignalsModelConfig = SignalsModelConfig()
    training: SignalsTrainingConfig = SignalsTrainingConfig()
    isolation_forest: SignalsIsolationForestConfig = SignalsIsolationForestConfig()
    lightgbm: SignalsLightGBMConfig = SignalsLightGBMConfig()
    finbert: FinBERTModelConfig = FinBERTModelConfig()
    finbert_training: FinBERTTrainingConfig = FinBERTTrainingConfig()
    setfit: SetFitModelConfig = SetFitModelConfig()
    setfit_training: SetFitTrainingConfig = SetFitTrainingConfig()
    data: DataConfig = DataConfig()


def load_config(path: str | Path | None = None) -> TrainerConfig:
    """Load trainer/config.toml and resolve relative paths against the project root."""
    if path is None:
        path = _ROOT / "trainer" / "config.toml"
    path = Path(path)
    print(path)

    with open(path, "rb") as f:
        raw: dict = tomllib.load(f)

    # Map TOML section names to TrainerConfig field names
    toml_to_field = {
        "wandb": "wandb",
        "signals.model": "signals",
        "signals.training": "training",
        "signals.isolation_forest": "isolation_forest",
        "signals.lightgbm": "lightgbm",
        "finbert.model": "finbert",
        "finbert.training": "finbert_training",
        "setfit.model": "setfit",
        "setfit.training": "setfit_training",
        "data": "data",
    }

    filtered: dict = {}
    for toml_key, field_name in toml_to_field.items():
        parts = toml_key.split(".")
        val = raw
        for p in parts:
            val = val.get(p, {})
        if val:
            filtered[field_name] = val

    cfg = TrainerConfig.model_validate(filtered)

    # Resolve relative data paths to absolute
    data_section = raw.get("data", {})
    for key in (
        "output_sentiment",
        "output_signals",
        "output_backtest",
        "output_weekly_returns",
    ):
        if key in data_section:
            resolved = _ROOT / data_section[key]
            setattr(cfg.data, key, resolved)

    # Resolve finbert training raw_data_path and output_dir
    finbert_section = raw.get("finbert", {}).get("training", {})
    if "raw_data_path" in finbert_section:
        cfg.finbert_training.raw_data_path = _ROOT / finbert_section["raw_data_path"]
    if "output_dir" in finbert_section:
        cfg.finbert_training.output_dir = _ROOT / finbert_section["output_dir"]

    # Resolve setfit training raw_data_path
    setfit_section = raw.get("setfit", {}).get("training", {})
    if "raw_data_path" in setfit_section:
        cfg.setfit_training.raw_data_path = _ROOT / setfit_section["raw_data_path"]

    return cfg
