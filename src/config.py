"""Agent configuration — Pydantic models loaded from config.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent


class LSTMConfig(BaseModel):
    sequence_length: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class IsolationForestConfig(BaseModel):
    contamination: float = 0.1
    n_estimators: int = 100


class LightGBMConfig(BaseModel):
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200


# ─── FinBERT Config ─────────────────────────────────────────────────────────────


class FinBERTModelConfig(BaseModel):
    pretrained_model: str = "bert-base-chinese"
    num_level1: int = 8
    num_sentiment: int = 3
    max_seq_length: int = 128
    dropout: float = 0.1


class FinBERTTrainingConfig(BaseModel):
    """Training hyperparameters for FinBERT fine-tuning."""

    batch_size: int = 32
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
    use_l2_class_weights: bool = False
    use_content: bool = False


class TrainingConfig(BaseModel):
    """Training hyperparameters for LSTM+Attention and LightGBM stacking."""

    epochs_pretrain: int = 15
    epochs_finetune: int = 10
    batch_size: int = 64
    lr: float = 0.001
    num_heads: int = 4
    anomaly_threshold: float = 0.03  # 3% weekly return = anomaly


class ModelConfig(BaseModel):
    lstm: LSTMConfig = LSTMConfig()
    isolation_forest: IsolationForestConfig = IsolationForestConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()
    training: TrainingConfig = TrainingConfig()
    finbert: FinBERTModelConfig = FinBERTModelConfig()


class AgentConfig(BaseModel):
    llm_model: str = "glm-4-flash"
    llm_temperature: float = 0.0
    max_weight_per_industry: float = 0.3
    max_total_weight: float = 1.0


class BacktestConfig(BaseModel):
    initial_capital: float = 1_000_000.0
    transaction_fee: float = 0.0003
    slippage: float = 0.0005
    risk_free_rate: float = 0.03


class DataConfig(BaseModel):
    # Raw inputs
    input_news_raw: Path = _ROOT / "data" / "converted" / "tushare_news_2021_today_merged.parquet"
    etf_info: Path = _ROOT / "data" / "converted" / "主题ETF信息表-快照1_主题ETF.parquet"
    etf_prices: Path = _ROOT / "data" / "converted" / "主题ETF历史量价.parquet"
    industry_dict: Path = _ROOT / "data" / "industry_dict.json"
    # Outputs
    output_sentiment: Path = _ROOT / "data" / "industry_sentiment.parquet"
    output_signals: Path = _ROOT / "data" / "ml_signals.parquet"
    output_trades: Path = _ROOT / "data" / "trade_signals.parquet"
    output_logs: Path = _ROOT / "data" / "decision_logs.jsonl"
    output_backtest: Path = _ROOT / "data" / "backtest_results.parquet"
    output_weekly_returns: Path = _ROOT / "data" / "weekly_returns.parquet"
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"


class AgentRootConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    agent: AgentConfig = AgentConfig()
    backtest: BacktestConfig = BacktestConfig()
    data: DataConfig = DataConfig()
    training_finbert: FinBERTTrainingConfig = FinBERTTrainingConfig()


def load_config(path: Path | str | None = None) -> AgentRootConfig:
    """Load config.toml and resolve relative paths against the project root."""
    if path is None:
        path = _ROOT / "config.toml"
    path = Path(path)

    with open(path, "rb") as f:
        raw: dict = tomllib.load(f)

    # Resolve relative paths to absolute, anchored at project root
    data_section = raw.get("data", {})
    for key in (
        "input_news_raw",
        "etf_info",
        "etf_prices",
        "industry_dict",
        "output_sentiment",
        "output_signals",
        "output_trades",
        "output_logs",
        "output_backtest",
        "output_weekly_returns",
    ):
        if key in data_section:
            raw["data"][key] = str(_ROOT / data_section[key])

    return AgentRootConfig.model_validate(raw)
