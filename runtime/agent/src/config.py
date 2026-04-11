"""Unified src configuration, loaded from config.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class AgentConfig(BaseModel):
    llm_model: str = "Minimax-M2.7-highspeed"
    llm_temperature: float = 0.0
    enable_history_retrieval: bool = False
    max_weight_per_industry: float = 0.3
    max_total_weight: float = 1.0


class BacktestConfig(BaseModel):
    initial_capital: float = 1_000_000.0
    transaction_fee: float = 0.0003
    slippage: float = 0.0005
    risk_free_rate: float = 0.03
    max_abs_weekly_return_guardrail: float = 0.30
    auto_visualize: bool = True


class DataConfig(BaseModel):
    input_news_raw: Path | None = None
    etf_info: Path | None = None
    etf_prices: Path | None = None
    industry_dict: Path | None = None
    meta_sector_mapping: Path | None = None
    output_sentiment: Path | None = None
    output_logs: Path | None = None
    output_backtest: Path | None = None
    output_backtest_metrics: Path | None = None
    output_agent_features: Path | None = None
    output_agent_features_oof: Path | None = None


class PredictConfig(BaseModel):
    finbert_onnx_dir: Path | None = None
    finbert_max_length: int = 128
    setfit_base_dir: Path | None = None
    setfit_max_length: int = 256
    signals_onnx_dir: Path | None = None
    onnx_cache_dir: Path | None = None
    max_cache_weeks: int = 4


class MemosConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://memos.memtensor.cn/api/openmem/v1"


class WandbConfig(BaseModel):
    mode: str = "disabled"
    project: str = "news2etf-agent"
    entity: str | None = None
    tags: list[str] = []


class AgentRootConfig(BaseModel):
    seed: int = 42
    agent: AgentConfig = AgentConfig()
    backtest: BacktestConfig = BacktestConfig()
    data: DataConfig = DataConfig()
    predict: PredictConfig = PredictConfig()
    memos: MemosConfig = MemosConfig()
    wandb: WandbConfig = WandbConfig()


_config_instance: AgentRootConfig | None = None


def repo_root() -> Path:
    env_path = os.getenv("NEWS2ETF_REPO_ROOT")
    if env_path:
        return Path(env_path).resolve()

    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "trainer").exists() and (candidate / "data").exists():
            return candidate
    return current.parent.parent


def project_root() -> Path:
    return repo_root()


def runtime_root() -> Path:
    env_path = os.getenv("NEWS2ETF_RUNTIME_ROOT")
    if env_path:
        return Path(env_path).resolve()
    return (repo_root() / "runtime" / "agent").resolve()


def shared_data_root() -> Path:
    env_path = os.getenv("NEWS2ETF_SHARED_DATA_ROOT")
    if env_path:
        return Path(env_path).resolve()
    return repo_root() / "data"


def trainer_root() -> Path:
    env_path = os.getenv("NEWS2ETF_TRAINER_ROOT")
    if env_path:
        return Path(env_path).resolve()
    return repo_root() / "trainer"


def runtime_inputs_root() -> Path:
    return runtime_root() / "data" / "inputs"


def runtime_models_root() -> Path:
    return runtime_root() / "models"


def best_etf_by_index_path(etf_info_path: str | Path | None = None) -> Path:
    if etf_info_path is not None:
        path = Path(etf_info_path)
        if path.parent.name == "converted":
            return path.parent.parent / "best_etf_by_index.parquet"
        return path.parent / "best_etf_by_index.parquet"
    return shared_data_root() / "best_etf_by_index.parquet"


def _default_config_path() -> Path:
    env_path = os.getenv("NEWS2ETF_CONFIG_PATH")
    if env_path:
        return Path(env_path).resolve()
    runtime_cfg = (runtime_root() / "config.toml").resolve()
    if runtime_cfg.exists():
        return runtime_cfg
    return (repo_root() / "config.toml").resolve()


def _looks_like_relative_path(value: str) -> bool:
    return value.startswith("./") or value.startswith("../")


def _resolve_nested_path_fields(section: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, val in section.items():
        if isinstance(val, dict):
            resolved[key] = _resolve_nested_path_fields(val, root)
        elif isinstance(val, list) and key.endswith("_paths"):
            resolved[key] = [root / item if isinstance(item, str) and _looks_like_relative_path(item) else item for item in val]
        elif isinstance(val, str) and (
            key.endswith("_path")
            or key.endswith("_dir")
            or key.endswith("_file")
            or key.endswith("_mapping")
            or key.endswith("_dict")
            or key.endswith("_raw")
            or key.endswith("_info")
            or key.endswith("_prices")
            or key.startswith("output_")
        ):
            resolved[key] = root / val if _looks_like_relative_path(val) else Path(val)
        else:
            resolved[key] = val
    return resolved


def _apply_defaults(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized = dict(raw)
    runtime = runtime_root()
    shared_data = shared_data_root()
    runtime_inputs = runtime_inputs_root()
    runtime_models = runtime_models_root()
    normalized.setdefault("seed", 42)
    normalized.setdefault("agent", {})
    normalized.setdefault("backtest", {})
    normalized.setdefault("data", {})
    normalized.setdefault("predict", {})
    normalized.setdefault("memos", {})
    normalized.setdefault("wandb", {})

    normalized["data"].setdefault("input_news_raw", str(shared_data / "converted" / "tushare_news_2021_today_merged.parquet"))
    normalized["data"].setdefault("etf_info", str(shared_data / "converted" / "主题ETF信息表-快照1_主题ETF.parquet"))
    normalized["data"].setdefault("etf_prices", str(shared_data / "converted" / "主题ETF历史量价.parquet"))
    normalized["data"].setdefault("industry_dict", str(shared_data / "industry_dict.json"))
    normalized["data"].setdefault("meta_sector_mapping", str(shared_data / "meta_sector_mapping.json"))
    normalized["data"].setdefault("output_sentiment", str(runtime_inputs / "sentiment_weekly.parquet"))
    normalized["data"].setdefault("output_logs", str(runtime / "data" / "decision_logs.jsonl"))
    normalized["data"].setdefault("output_backtest", str(runtime / "data" / "backtest_results.parquet"))
    normalized["data"].setdefault("output_backtest_metrics", str(runtime / "data" / "backtest_metrics.parquet"))
    normalized["data"].setdefault("output_agent_features", str(shared_data / "agent_features.parquet"))
    normalized["data"].setdefault("output_agent_features_oof", str(shared_data / "agent_features.oof.parquet"))

    normalized["predict"].setdefault("finbert_onnx_dir", str(runtime_models / "major"))
    normalized["predict"].setdefault("setfit_base_dir", str(runtime_models / "sub" / "0407-1415"))
    normalized["predict"].setdefault("signals_onnx_dir", str(runtime_models / "signals" / "final-3y"))
    normalized["predict"].setdefault("onnx_cache_dir", str(runtime / "data" / "onnx_cache"))

    normalized["data"] = _resolve_nested_path_fields(normalized["data"], root)
    normalized["predict"] = _resolve_nested_path_fields(normalized["predict"], root)
    return normalized


def load_config(path: str | Path | None = None) -> AgentRootConfig:
    """Load config.toml and resolve relative paths against the project root."""
    cfg_path = _default_config_path() if path is None else Path(path).resolve()
    with open(cfg_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    return AgentRootConfig.model_validate(_apply_defaults(raw, cfg_path.parent))


def init_config(path: str | Path | None = None) -> AgentRootConfig:
    """Initialize the singleton config from a TOML file."""
    global _config_instance
    _config_instance = load_config(path)
    return _config_instance


def get_config() -> AgentRootConfig:
    """Return the singleton config. Must call init_config() first."""
    assert _config_instance is not None, "Config not initialized. Call init_config() first."
    return _config_instance
