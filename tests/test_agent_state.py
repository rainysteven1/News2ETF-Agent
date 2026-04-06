"""Tests for src/agent/state.py"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.agent.state import (
    AgentState,
    ETFSelections,
    MetaSectorPlan,
    SectorStatus,
    TradeDecision,
)


class TestSectorStatus:
    """Tests for SectorStatus enum."""

    def test_normal_value(self) -> None:
        """Test SectorStatus.NORMAL has correct value."""
        assert SectorStatus.NORMAL.value == "normal"

    def test_forbidden_zone_value(self) -> None:
        """Test SectorStatus.FORBIDDEN_ZONE has correct value."""
        assert SectorStatus.FORBIDDEN_ZONE.value == "forbidden"

    def test_enum_values_are_strings(self) -> None:
        """Test that enum values are strings."""
        assert isinstance(SectorStatus.NORMAL.value, str)
        assert isinstance(SectorStatus.FORBIDDEN_ZONE.value, str)

    def test_enum_count(self) -> None:
        """Test that there are exactly 2 enum values."""
        assert len(SectorStatus) == 2


class TestMetaSectorPlan:
    """Tests for MetaSectorPlan model validation."""

    def test_valid_model_creation(self) -> None:
        """Test creating a valid MetaSectorPlan."""
        plan = MetaSectorPlan(
            meta_sector="科技成长",
            action="buy",
            weight=0.3,
            reason="上涨趋势确认",
        )
        assert plan.meta_sector == "科技成长"
        assert plan.action == "buy"
        assert plan.weight == 0.3
        assert plan.reason == "上涨趋势确认"

    def test_required_meta_sector(self) -> None:
        """Test that meta_sector is required."""
        with pytest.raises(ValidationError) as exc_info:
            MetaSectorPlan(action="buy", weight=0.3)
        assert "meta_sector" in str(exc_info.value)

    def test_required_action(self) -> None:
        """Test that action is required."""
        with pytest.raises(ValidationError) as exc_info:
            MetaSectorPlan(meta_sector="科技成长", weight=0.3)
        assert "action" in str(exc_info.value)

    def test_weight_constraints(self) -> None:
        """Test that weight must be between 0 and 1."""
        # Valid weight
        plan = MetaSectorPlan(meta_sector="科技成长", action="buy", weight=0.0)
        assert plan.weight == 0.0

        plan = MetaSectorPlan(meta_sector="科技成长", action="buy", weight=1.0)
        assert plan.weight == 1.0

        # Invalid: negative weight
        with pytest.raises(ValidationError):
            MetaSectorPlan(meta_sector="科技成长", action="buy", weight=-0.1)

        # Invalid: weight > 1
        with pytest.raises(ValidationError):
            MetaSectorPlan(meta_sector="科技成长", action="buy", weight=1.5)

    def test_default_reason(self) -> None:
        """Test that reason defaults to empty string."""
        plan = MetaSectorPlan(meta_sector="科技成长", action="hold", weight=0.0)
        assert plan.reason == ""


class TestETFSelections:
    """Tests for ETFSelections model validation."""

    def test_valid_model_creation(self) -> None:
        """Test creating a valid ETFSelections."""
        selection = ETFSelections(
            meta_sector="科技成长",
            selected_indices=["半导体/芯片指数", "TMT指数"],
            selected_etf="512000 华夏国证半导体ETF",
        )
        assert selection.meta_sector == "科技成长"
        assert selection.selected_indices == ["半导体/芯片指数", "TMT指数"]
        assert selection.selected_etf == "512000 华夏国证半导体ETF"

    def test_required_meta_sector(self) -> None:
        """Test that meta_sector is required."""
        with pytest.raises(ValidationError) as exc_info:
            ETFSelections(selected_indices=["index1"], selected_etf="ETF1")
        assert "meta_sector" in str(exc_info.value)

    def test_default_selected_indices(self) -> None:
        """Test that selected_indices defaults to empty list."""
        selection = ETFSelections(meta_sector="科技成长")
        assert selection.selected_indices == []

    def test_default_selected_etf(self) -> None:
        """Test that selected_etf defaults to empty string."""
        selection = ETFSelections(meta_sector="科技成长")
        assert selection.selected_etf == ""


class TestTradeDecision:
    """Tests for TradeDecision model."""

    def test_has_level1_plan_field(self) -> None:
        """Test that TradeDecision has level1_plan field."""
        decision = TradeDecision(
            industry="半导体/芯片",
            action="buy",
            weight=0.3,
        )
        assert hasattr(decision, "level1_plan")

    def test_has_level2_plan_field(self) -> None:
        """Test that TradeDecision has level2_plan field."""
        decision = TradeDecision(
            industry="半导体/芯片",
            action="buy",
            weight=0.3,
        )
        assert hasattr(decision, "level2_plan")

    def test_level1_plan_is_list_of_meta_sector_plan(self) -> None:
        """Test that level1_plan is a list of MetaSectorPlan."""
        plan1 = MetaSectorPlan(meta_sector="科技成长", action="buy", weight=0.3)
        plan2 = MetaSectorPlan(meta_sector="高端制造", action="hold", weight=0.0)

        decision = TradeDecision(
            industry="半导体/芯片",
            action="buy",
            weight=0.3,
            level1_plan=[plan1, plan2],
        )
        assert len(decision.level1_plan) == 2
        assert isinstance(decision.level1_plan[0], MetaSectorPlan)

    def test_level2_plan_is_list_of_etf_selections(self) -> None:
        """Test that level2_plan is a list of ETFSelections."""
        selection1 = ETFSelections(meta_sector="科技成长", selected_etf="512000")
        selection2 = ETFSelections(meta_sector="高端制造", selected_etf="512660")

        decision = TradeDecision(
            industry="半导体/芯片",
            action="buy",
            weight=0.3,
            level2_plan=[selection1, selection2],
        )
        assert len(decision.level2_plan) == 2
        assert isinstance(decision.level2_plan[0], ETFSelections)

    def test_has_sector_status_field(self) -> None:
        """Test that TradeDecision has sector_status field."""
        decision = TradeDecision(industry="半导体/芯片", action="buy", weight=0.3)
        assert hasattr(decision, "sector_status")

    def test_sector_status_default_is_dict(self) -> None:
        """Test that sector_status defaults to empty dict."""
        decision = TradeDecision(industry="半导体/芯片", action="buy", weight=0.3)
        assert decision.sector_status == {}


class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_is_typed_dict(self) -> None:
        """Test that AgentState is a TypedDict."""
        state: AgentState = {
            "date": "2024-10-01",
            "last_week_pnl": 0.05,
            "last_week_holdings": {"科技成长": 0.3, "高端制造": 0.2},
            "last_week_returns": {"科技成长": 0.02, "高端制造": -0.01},
            "observations": {},
            "messages": [],
            "decisions": [],
            "is_risk_passed": True,
            "retry_count": 0,
            "last_error": "",
            "loop_step": 0,
            "forbidden_sectors": {},
            "tcn_sequence": {},
            "decision_context": {},
            "last_guardrail_events": [],
        }
        assert state["date"] == "2024-10-01"

    def test_has_forbidden_sectors_field(self) -> None:
        """Test that AgentState has forbidden_sectors field."""
        state: AgentState = {
            "date": "2024-10-01",
            "last_week_pnl": 0.0,
            "last_week_holdings": {},
            "last_week_returns": {},
            "observations": {},
            "messages": [],
            "decisions": [],
            "is_risk_passed": False,
            "retry_count": 0,
            "last_error": "",
            "loop_step": 0,
            "forbidden_sectors": {"科技成长": "DAILY_LOSS_5PCT"},
            "tcn_sequence": {},
            "decision_context": {},
            "last_guardrail_events": [],
        }
        assert "forbidden_sectors" in state
        assert state["forbidden_sectors"]["科技成长"] == "DAILY_LOSS_5PCT"

    def test_has_tcn_sequence_field(self) -> None:
        """Test that AgentState has tcn_sequence field."""
        tcn_seq = {
            "科技成长": [0.1, 0.2, 0.15, 0.18, 0.22],
            "高端制造": [-0.05, -0.03, 0.01, 0.02, 0.05],
        }
        state: AgentState = {
            "date": "2024-10-01",
            "last_week_pnl": 0.0,
            "last_week_holdings": {},
            "last_week_returns": {},
            "observations": {},
            "messages": [],
            "decisions": [],
            "is_risk_passed": False,
            "retry_count": 0,
            "last_error": "",
            "loop_step": 0,
            "forbidden_sectors": {},
            "tcn_sequence": tcn_seq,
            "decision_context": {},
            "last_guardrail_events": [],
        }
        assert "tcn_sequence" in state
        assert state["tcn_sequence"]["科技成长"] == [0.1, 0.2, 0.15, 0.18, 0.22]

    def test_has_decision_context_field(self) -> None:
        """Test that AgentState has decision_context field."""
        context = {
            "market_state": "bullish",
            "vol_percentile": 0.6,
            "sector_signals": {"科技成长": 0.8, "高端制造": 0.3},
        }
        state: AgentState = {
            "date": "2024-10-01",
            "last_week_pnl": 0.0,
            "last_week_holdings": {},
            "last_week_returns": {},
            "observations": {},
            "messages": [],
            "decisions": [],
            "is_risk_passed": False,
            "retry_count": 0,
            "last_error": "",
            "loop_step": 0,
            "forbidden_sectors": {},
            "tcn_sequence": {},
            "decision_context": context,
            "last_guardrail_events": [],
        }
        assert "decision_context" in state
        assert state["decision_context"]["market_state"] == "bullish"

    def test_has_last_guardrail_events_field(self) -> None:
        """Test that AgentState has last_guardrail_events field."""
        events = [
            {
                "meta_sector": "科技成长",
                "trigger_type": "DAILY_LOSS_5PCT",
                "severity": 0.7,
                "reason": "Daily loss exceeds 5% threshold",
            }
        ]
        state: AgentState = {
            "date": "2024-10-01",
            "last_week_pnl": 0.0,
            "last_week_holdings": {},
            "last_week_returns": {},
            "observations": {},
            "messages": [],
            "decisions": [],
            "is_risk_passed": False,
            "retry_count": 0,
            "last_error": "",
            "loop_step": 0,
            "forbidden_sectors": {},
            "tcn_sequence": {},
            "decision_context": {},
            "last_guardrail_events": events,
        }
        assert "last_guardrail_events" in state
        assert len(state["last_guardrail_events"]) == 1
        assert state["last_guardrail_events"][0]["trigger_type"] == "DAILY_LOSS_5PCT"

    def test_all_required_fields_present(self) -> None:
        """Test that all required fields are present in AgentState."""
        required_fields = [
            "date",
            "last_week_pnl",
            "last_week_holdings",
            "last_week_returns",
            "observations",
            "messages",
            "decisions",
            "is_risk_passed",
            "retry_count",
            "last_error",
            "loop_step",
        ]
        state: AgentState = {
            "date": "2024-10-01",
            "last_week_pnl": 0.0,
            "last_week_holdings": {},
            "last_week_returns": {},
            "observations": {},
            "messages": [],
            "decisions": [],
            "is_risk_passed": False,
            "retry_count": 0,
            "last_error": "",
            "loop_step": 0,
            "forbidden_sectors": {},
            "tcn_sequence": {},
            "decision_context": {},
            "last_guardrail_events": [],
        }
        for field in required_fields:
            assert field in state, f"Required field '{field}' not found in AgentState"
