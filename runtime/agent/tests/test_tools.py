"""Tests for src/agent/tools.py — TOOL_REGISTRY completeness."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest


class TestTOOLREGISTRY:
    """TOOL_REGISTRY must contain all required tools."""

    def test_registry_has_build_decision_context(self) -> None:
        """build_decision_context must be registered."""
        from src.agent.tools import TOOL_REGISTRY
        assert "build_decision_context" in TOOL_REGISTRY

    def test_registry_has_all_base_tools(self) -> None:
        """Base tools (read_market_news, etc.) must still exist."""
        from src.agent.tools import TOOL_REGISTRY
        for name in (
            "read_market_news",
            "compute_ml_signals",
            "check_last_week_pnl",
            "retrieve_history",
            "get_industry_top_news",
            "get_etf_candidates",
            "store_decision",
        ):
            assert name in TOOL_REGISTRY, f"{name} missing from TOOL_REGISTRY"

    def test_registry_count(self) -> None:
        """Total tools should be 8 (7 original + build_decision_context)."""
        from src.agent.tools import TOOL_REGISTRY
        assert len(TOOL_REGISTRY) == 8

    def test_tools_are_langchain_tools(self) -> None:
        """Each tool in registry must be a LangChain StructuredTool."""
        from langchain_core.tools import StructuredTool
        from src.agent.tools import TOOL_REGISTRY

        for name, tool in TOOL_REGISTRY.items():
            assert isinstance(tool, StructuredTool), (
                f"{name} is {type(tool)}, expected StructuredTool"
            )

    def test_tool_names_match_keys(self) -> None:
        """Each tool's .name attribute must match its registry key."""
        from src.agent.tools import TOOL_REGISTRY

        for key, tool in TOOL_REGISTRY.items():
            assert tool.name == key, (
                f"TOOL_REGISTRY key='{key}' but tool.name='{tool.name}'"
            )

    def test_build_decision_context_schema(self) -> None:
        """build_decision_context must accept a 'date' parameter."""
        from src.agent.tools import TOOL_REGISTRY

        tool = TOOL_REGISTRY["build_decision_context"]
        schema = tool.args_schema
        assert hasattr(schema, "model_fields"), "args_schema should be a Pydantic model"
        assert "date" in schema.model_fields, "build_decision_context must accept 'date' parameter"


def test_get_industry_top_news_falls_back_to_raw_keyword_match(tmp_path: Path) -> None:
    from src.agent.tools import get_industry_top_news

    news_path = tmp_path / "news.parquet"
    pl.DataFrame(
        {
            "datetime": [
                "2024-01-03 09:00:00",
                "2024-01-04 10:00:00",
                "2024-01-05 11:00:00",
            ],
            "content": [
                "半导体设备国产化提速，芯片产线扩建",
                "医药板块出现反弹",
                "芯片设计企业发布新品",
            ],
            "title": [
                "半导体设备景气回升",
                "创新药反弹",
                "芯片新品发布",
            ],
            "source": ["s1", "s2", "s3"],
        }
    ).write_parquet(news_path)

    config = SimpleNamespace(data=SimpleNamespace(input_news_raw=news_path))

    with patch("src.agent.tools._current_config", return_value=config), patch(
        "src.signals.onnx_inference.get_onnx_predictions",
        side_effect=FileNotFoundError("missing onnx"),
    ):
        out = get_industry_top_news.invoke({"date": "2024-01-01", "industry": "半导体/芯片", "top_k": 2})

    assert "raw keyword fallback" in out
    assert "半导体设备景气回升" in out or "芯片新品发布" in out
    assert "[Fallback]" in out


def test_read_market_news_falls_back_to_compressed_summary(tmp_path: Path) -> None:
    from src.agent.tools import read_market_news

    news_path = tmp_path / "news.parquet"
    pl.DataFrame(
        {
            "datetime": [
                "2024-01-03 09:00:00",
                "2024-01-03 10:00:00",
                "2024-01-04 11:00:00",
                "2024-01-05 12:00:00",
            ],
            "content": [
                "人工智能芯片景气回升，半导体需求改善",
                "半导体设备国产化持续推进",
                "人工智能应用落地加速",
                "芯片设计企业发布新品",
            ],
            "title": [
                "人工智能芯片景气回升",
                "半导体设备国产化推进",
                "人工智能应用落地",
                "芯片新品发布",
            ],
            "source": ["s1", "s1", "s2", "s3"],
        }
    ).write_parquet(news_path)

    config = SimpleNamespace(data=SimpleNamespace(input_news_raw=news_path))

    with patch("src.agent.tools._current_config", return_value=config), patch(
        "src.signals.onnx_inference.get_onnx_predictions",
        side_effect=ModuleNotFoundError("missing onnxruntime"),
    ):
        out = read_market_news.invoke({"date": "2024-01-01"})

    assert "News Summary" in out
    assert "Top sources:" in out
    assert "Hot terms:" in out
    assert "Recent headlines:" in out
    assert "[Fallback]" in out


def test_read_market_news_skips_onnx_when_news_already_labeled(tmp_path: Path) -> None:
    from src.agent.tools import read_market_news

    news_path = tmp_path / "labeled_news.parquet"
    pl.DataFrame(
        {
            "datetime": ["2024-01-03 09:00:00"],
            "content": ["半导体需求改善"],
            "title": ["半导体景气回升"],
            "source": ["s1"],
            "major_category": ["科技信息"],
            "sub_category": ["半导体/芯片"],
            "sentiment": ["positive"],
        }
    ).write_parquet(news_path)

    config = SimpleNamespace(data=SimpleNamespace(input_news_raw=news_path))

    with patch("src.agent.tools._current_config", return_value=config), patch(
        "src.signals.onnx_inference.get_onnx_predictions",
        side_effect=AssertionError("ONNX should not be called for labeled news"),
    ):
        out = read_market_news.invoke({"date": "2024-01-01"})

    assert "科技信息/半导体/芯片" in out
    assert "positive" in out
    assert "[Fallback]" not in out


def test_get_industry_top_news_skips_onnx_when_news_already_labeled(tmp_path: Path) -> None:
    from src.agent.tools import get_industry_top_news

    news_path = tmp_path / "labeled_news.parquet"
    pl.DataFrame(
        {
            "datetime": ["2024-01-03 09:00:00", "2024-01-04 09:00:00"],
            "content": ["半导体需求改善", "医药需求改善"],
            "title": ["半导体景气回升", "创新药景气回升"],
            "source": ["s1", "s2"],
            "major_category": ["科技信息", "医药健康"],
            "sub_category": ["半导体/芯片", "创新药"],
            "sentiment": ["positive", "neutral"],
        }
    ).write_parquet(news_path)

    config = SimpleNamespace(data=SimpleNamespace(input_news_raw=news_path))

    with patch("src.agent.tools._current_config", return_value=config), patch(
        "src.signals.onnx_inference.get_onnx_predictions",
        side_effect=AssertionError("ONNX should not be called for labeled news"),
    ):
        out = get_industry_top_news.invoke({"date": "2024-01-01", "industry": "半导体/芯片", "top_k": 1})

    assert "半导体景气回升" in out
    assert "[positive]" in out


def test_check_last_week_pnl_prefers_current_run_checkpoint_results(tmp_path: Path) -> None:
    from src.agent.tools import check_last_week_pnl

    checkpoint_dir = tmp_path / "checkpoints"
    run_id = "bt_test"
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


def test_retrieve_history_disabled_by_default_config() -> None:
    from src.agent.tools import retrieve_history

    config = SimpleNamespace(
        agent=SimpleNamespace(enable_history_retrieval=False),
        memos=SimpleNamespace(api_key="dummy", base_url="https://example.com"),
    )

    with patch("src.agent.tools._current_config", return_value=config):
        out = retrieve_history.invoke({"date": "2024-01-01", "query": "test"})

    assert out == "History retrieval disabled by config."
