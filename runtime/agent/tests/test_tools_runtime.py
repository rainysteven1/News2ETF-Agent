from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

from src.agent.tools import TOOL_REGISTRY, check_last_week_pnl, read_market_news


def test_runtime_tools_module_resolves_local_file() -> None:
    runtime_agent_root = Path(__file__).resolve().parents[1] / "src" / "agent"
    runtime_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import src.agent.tools as tools_module
print(json.dumps({"tools": tools_module.__file__}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(runtime_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert Path(payload["tools"]).resolve() == runtime_agent_root / "tools.py"


def test_runtime_tool_registry_still_complete() -> None:
    for name in (
        "read_market_news",
        "compute_ml_signals",
        "check_last_week_pnl",
        "retrieve_history",
        "get_industry_top_news",
        "get_etf_candidates",
        "store_decision",
        "build_decision_context",
    ):
        assert name in TOOL_REGISTRY


def test_runtime_read_market_news_falls_back_to_summary(tmp_path: Path) -> None:
    news_path = tmp_path / "news.parquet"
    pl.DataFrame(
        {
            "datetime": ["2024-01-03 09:00:00", "2024-01-04 10:00:00"],
            "content": ["人工智能芯片景气回升", "半导体设备国产化推进"],
            "title": ["人工智能芯片景气回升", "半导体设备国产化推进"],
            "source": ["s1", "s2"],
        }
    ).write_parquet(news_path)

    config = SimpleNamespace(data=SimpleNamespace(input_news_raw=news_path))

    with patch("src.agent.tools._current_config", return_value=config), patch(
        "src.signals.onnx_inference.get_onnx_predictions",
        side_effect=ModuleNotFoundError("missing onnxruntime"),
    ):
        out = read_market_news.invoke({"date": "2024-01-01"})

    assert "News Summary" in out
    assert "[Fallback]" in out


def test_runtime_check_last_week_pnl_prefers_runtime_checkpoint_results(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    run_id = "bt_runtime_tools"
    run_dir = checkpoint_dir / run_id
    run_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "week_start": ["2024-01-01"],
            "weekly_return": [0.02],
            "nav": [1_020_000.0],
            "holdings": ['{"科技成长": 0.2}'],
            "invested_weight": [0.2],
        }
    ).write_parquet(run_dir / "backtest_results.parquet")

    config = SimpleNamespace(data=SimpleNamespace(output_backtest=tmp_path / "global_backtest.parquet"))

    with patch("src.agent.tools._current_config", return_value=config), patch(
        "src.agent.tools.get_runtime",
        return_value=SimpleNamespace(run_id=run_id, checkpoint_dir=checkpoint_dir),
    ):
        out = check_last_week_pnl.invoke({})

    assert '"week_start": "2024-01-01"' in out
    assert '"weekly_return": 0.02' in out
