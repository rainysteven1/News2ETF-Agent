from __future__ import annotations

from pathlib import Path

from src.config import best_etf_by_index_path, get_config, init_config, load_config, repo_root, runtime_root


def test_load_config_resolves_data_and_predict_paths() -> None:
    cfg = load_config()

    assert cfg.data.input_news_raw is not None
    assert cfg.data.etf_prices is not None
    assert cfg.data.output_backtest is not None
    assert cfg.predict.signals_onnx_dir is not None
    assert cfg.predict.onnx_cache_dir is not None

    assert isinstance(cfg.data.input_news_raw, Path)
    assert isinstance(cfg.predict.signals_onnx_dir, Path)
    assert cfg.data.input_news_raw.is_absolute()
    assert cfg.data.output_backtest.is_absolute()
    assert cfg.predict.signals_onnx_dir.is_absolute()
    assert cfg.data.output_sentiment == runtime_root() / "data" / "inputs" / "sentiment_weekly.parquet"
    assert cfg.data.output_agent_features == repo_root() / "data" / "agent_features.parquet"
    assert cfg.data.output_agent_features_oof == repo_root() / "data" / "agent_features.oof.parquet"
    assert cfg.predict.finbert_onnx_dir == runtime_root() / "models" / "major"
    assert cfg.predict.setfit_base_dir == runtime_root() / "models" / "sub" / "0407-1415"
    assert cfg.predict.signals_onnx_dir == runtime_root() / "models" / "signals" / "final-3y"


def test_init_config_sets_singleton_instance() -> None:
    cfg = init_config()

    assert get_config() is cfg


def test_runtime_and_repo_roots_are_distinct() -> None:
    assert runtime_root() == repo_root() / "runtime" / "agent"
    assert runtime_root() != repo_root()


def test_best_etf_path_derives_from_etf_info_location() -> None:
    path = best_etf_by_index_path(Path("/tmp/shared/converted/etf_info.parquet"))

    assert path == Path("/tmp/shared/best_etf_by_index.parquet")
