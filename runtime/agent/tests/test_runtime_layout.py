from __future__ import annotations

import importlib.util
from pathlib import Path

from src.config import _default_config_path, load_config, runtime_root


def test_runtime_config_is_default() -> None:
    assert _default_config_path() == runtime_root() / "config.toml"


def test_runtime_outputs_live_under_runtime_agent() -> None:
    cfg = load_config()

    assert cfg.data.output_backtest == runtime_root() / "data" / "backtest_results.parquet"
    assert cfg.predict.onnx_cache_dir == runtime_root() / "data" / "onnx_cache"


def test_runtime_entrypoint_exposes_typer_app() -> None:
    entry_path = runtime_root() / "main.py"
    spec = importlib.util.spec_from_file_location("runtime_agent_main", entry_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "app")
    assert module.app is not None
    assert isinstance(entry_path, Path)
