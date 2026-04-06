"""Tests for src/agent/tools.py — TOOL_REGISTRY completeness."""

from __future__ import annotations

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
