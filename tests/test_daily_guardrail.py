"""Tests for src/agent/daily_guardrail.py"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.agent.daily_guardrail import (
    COOLDOWN_RULES,
    DailyGuardrailMonitor,
    FORBIDDEN_ZONEEntry,
    FORBIDDEN_ZONEStateMachine,
    GuardrailSignal,
    SectorStatus,
)


class TestCOOLDOWNRules:
    """Tests for COOLDOWN_RULES constant."""

    def test_contains_expected_keys(self) -> None:
        """Test that COOLDOWN_RULES has expected trigger types."""
        expected_keys = [
            "DAILY_LOSS_5PCT",
            "DAILY_LOSS_3PCT_HIGH_BETA",
            "BREAKING_NEWS",
            "MARKET_VOL_SPIKE",
        ]
        for key in expected_keys:
            assert key in COOLDOWN_RULES, f"Key '{key}' not found in COOLDOWN_RULES"

    def test_daily_loss_5pct_cooldown(self) -> None:
        """Test DAILY_LOSS_5PCT has 3 day cooldown."""
        assert COOLDOWN_RULES["DAILY_LOSS_5PCT"] == 3

    def test_daily_loss_3pct_high_beta_cooldown(self) -> None:
        """Test DAILY_LOSS_3PCT_HIGH_BETA has 2 day cooldown."""
        assert COOLDOWN_RULES["DAILY_LOSS_3PCT_HIGH_BETA"] == 2

    def test_breaking_news_no_auto_release(self) -> None:
        """Test BREAKING_NEWS has None cooldown (no auto-release)."""
        assert COOLDOWN_RULES["BREAKING_NEWS"] is None

    def test_market_vol_spike_cooldown(self) -> None:
        """Test MARKET_VOL_SPIKE has 1 day cooldown."""
        assert COOLDOWN_RULES["MARKET_VOL_SPIKE"] == 1


class TestFORBIDDEN_ZONEEntry:
    """Tests for FORBIDDEN_ZONEEntry dataclass."""

    def test_creation(self) -> None:
        """Test creating a FORBIDDEN_ZONEEntry."""
        entry = FORBIDDEN_ZONEEntry(
            sector="科技成长",
            reason="Daily loss exceeds 5%",
            trigger_type="DAILY_LOSS_5PCT",
            start_date="2024-10-01",
        )
        assert entry.sector == "科技成长"
        assert entry.reason == "Daily loss exceeds 5%"
        assert entry.trigger_type == "DAILY_LOSS_5PCT"
        assert entry.start_date == "2024-10-01"

    def test_default_end_date(self) -> None:
        """Test default end_date is None."""
        entry = FORBIDDEN_ZONEEntry(
            sector="科技成长",
            reason="Test",
            trigger_type="BREAKING_NEWS",
            start_date="2024-10-01",
        )
        assert entry.end_date is None

    def test_default_created_at(self) -> None:
        """Test default created_at is empty string."""
        entry = FORBIDDEN_ZONEEntry(
            sector="科技成长",
            reason="Test",
            trigger_type="DAILY_LOSS_5PCT",
            start_date="2024-10-01",
        )
        assert entry.created_at == ""


class TestFORBIDDEN_ZONEStateMachine:
    """Tests for FORBIDDEN_ZONEStateMachine."""

    @pytest.fixture
    def machine(self) -> FORBIDDEN_ZONEStateMachine:
        """Create a fresh FORBIDDEN_ZONEStateMachine for each test."""
        return FORBIDDEN_ZONEStateMachine()

    class TestMarkForbidden:
        """Tests for mark_forbidden()."""

        def test_marks_sector_as_forbidden(self, machine: FORBIDDEN_ZONEStateMachine) -> None:
            """Test that mark_forbidden marks a sector as forbidden."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            assert "科技成长" in machine._forbidden

        def test_sets_correct_trigger_type(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that correct trigger_type is stored."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            entry = machine._forbidden["科技成长"]
            assert entry.trigger_type == "DAILY_LOSS_5PCT"

        def test_sets_auto_release_date_for_daily_loss(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that end_date is set for DAILY_LOSS_5PCT (3 day cooldown)."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            entry = machine._forbidden["科技成长"]
            assert entry.end_date is not None
            assert entry.end_date == "2024-10-04"  # 3 days later

        def test_no_auto_release_for_breaking_news(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that end_date is None for BREAKING_NEWS."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Black swan event",
                trigger_type="BREAKING_NEWS",
                current_date="2024-10-01",
            )

            entry = machine._forbidden["科技成长"]
            assert entry.end_date is None

    class TestIsForbidden:
        """Tests for is_forbidden()."""

        def test_returns_false_for_normal_sector(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that is_forbidden returns False for unmarked sectors."""
            result = machine.is_forbidden("科技成长", "2024-10-01")
            assert result is False

        def test_returns_true_for_forbidden_sector(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that is_forbidden returns True for forbidden sector."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            result = machine.is_forbidden("科技成长", "2024-10-01")
            assert result is True

        def test_returns_false_after_cooldown_expired(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that is_forbidden returns False after cooldown expires."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            # After 4 days (past the 3-day cooldown)
            result = machine.is_forbidden("科技成长", "2024-10-05")
            assert result is False

        def test_breaking_news_never_auto_releases(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that BREAKING_NEWS never auto-releases (end_date is None)."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Black swan",
                trigger_type="BREAKING_NEWS",
                current_date="2024-10-01",
            )

            # Even after a long time, should still be forbidden
            result = machine.is_forbidden("科技成长", "2024-12-01")
            assert result is True  # BREAKING_NEWS should never auto-release, still forbidden

    class TestCooldownExpired:
        """Tests for cooldown_expired()."""

        def test_returns_true_for_unmarked_sector(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that cooldown_expired returns True for unmarked sectors."""
            result = machine.cooldown_expired("科技成长", "2024-10-01")
            assert result is True

        def test_returns_false_before_cooldown(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that cooldown_expired returns False before cooldown expires."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            result = machine.cooldown_expired("科技成长", "2024-10-02")
            assert result is False

        def test_returns_true_after_cooldown(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that cooldown_expired returns True after cooldown expires."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            result = machine.cooldown_expired("科技成长", "2024-10-05")
            assert result is True

        def test_breaking_news_never_expires(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that BREAKING_NEWS cooldown never expires."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Black swan",
                trigger_type="BREAKING_NEWS",
                current_date="2024-10-01",
            )

            result = machine.cooldown_expired("科技成长", "2030-01-01")
            assert result is False

    class TestAutoRelease:
        """Tests for auto_release()."""

        def test_returns_false_if_not_expired(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that auto_release returns False if cooldown not expired."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            result = machine.auto_release("科技成长", "2024-10-02")
            assert result is False
            assert "科技成长" in machine._forbidden

        def test_returns_true_and_releases_if_expired(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that auto_release returns True and removes sector if cooldown expired."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss exceeds 5%",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            result = machine.auto_release("科技成长", "2024-10-05")
            assert result is True
            assert "科技成长" not in machine._forbidden

        def test_returns_false_for_unknown_sector(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that auto_release returns False for unknown sector."""
            result = machine.auto_release("不存在的板块", "2024-10-01")
            assert result is False

    class TestGetForbiddenSectors:
        """Tests for get_forbidden_sectors()."""

        def test_returns_empty_list_when_none_forbidden(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that get_forbidden_sectors returns empty list when none forbidden."""
            result = machine.get_forbidden_sectors("2024-10-01")
            assert result == []

        def test_returns_forbidden_sectors(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that get_forbidden_sectors returns list of forbidden sectors."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )
            machine.mark_forbidden(
                sector="高端制造",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            result = machine.get_forbidden_sectors("2024-10-01")
            assert "科技成长" in result
            assert "高端制造" in result

    class TestGetForbiddenInfo:
        """Tests for get_forbidden_info()."""

        def test_returns_dict_for_forbidden_sector(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that get_forbidden_info returns dict for forbidden sector."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            result = machine.get_forbidden_info("科技成长")
            assert result is not None
            assert isinstance(result, dict)
            assert result["sector"] == "科技成长"
            assert result["trigger_type"] == "DAILY_LOSS_5PCT"

        def test_returns_none_for_unknown_sector(
            self, machine: FORBIDDEN_ZONEStateMachine
        ) -> None:
            """Test that get_forbidden_info returns None for unknown sector."""
            result = machine.get_forbidden_info("不存在的板块")
            assert result is None

    class TestClearAll:
        """Tests for clear_all()."""

        def test_clears_all_entries(self, machine: FORBIDDEN_ZONEStateMachine) -> None:
            """Test that clear_all removes all forbidden entries."""
            machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )
            machine.mark_forbidden(
                sector="高端制造",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            machine.clear_all()

            assert len(machine._forbidden) == 0


class TestDailyGuardrailMonitor:
    """Tests for DailyGuardrailMonitor."""

    @pytest.fixture
    def monitor(self) -> DailyGuardrailMonitor:
        """Create a fresh DailyGuardrailMonitor for each test."""
        return DailyGuardrailMonitor()

    class TestCheckGuardrailTrigger:
        """Tests for check_guardrail_trigger()."""

        def test_returns_list(self, monitor: DailyGuardrailMonitor) -> None:
            """Test that check_guardrail_trigger returns a list."""
            result = monitor.check_guardrail_trigger(
                current_date="2024-10-01",
                positions={},
                etf_prices={},
            )
            assert isinstance(result, list)

        def test_empty_positions_returns_empty(
            self, monitor: DailyGuardrailMonitor
        ) -> None:
            """Test that empty positions returns empty list."""
            result = monitor.check_guardrail_trigger(
                current_date="2024-10-01",
                positions={},
                etf_prices={},
            )
            assert result == []

        def test_low_weight_positions_skipped(
            self, monitor: DailyGuardrailMonitor
        ) -> None:
            """Test that positions with weight < 0.01 are skipped."""
            result = monitor.check_guardrail_trigger(
                current_date="2024-10-01",
                positions={"科技成长": 0.005},  # Below threshold
                etf_prices={"512000": 1.0},
            )
            assert result == []

    class TestApplyForbiddenZone:
        """Tests for apply_forbidden_zone()."""

        def test_returns_tuple(self, monitor: DailyGuardrailMonitor) -> None:
            """Test that apply_forbidden_zone returns a tuple."""
            result = monitor.apply_forbidden_zone(
                agent_plan=[],
                current_date="2024-10-01",
            )
            assert isinstance(result, tuple)
            assert len(result) == 2

        def test_returns_adjusted_plan_and_overrides(
            self, monitor: DailyGuardrailMonitor
        ) -> None:
            """Test that returns (adjusted_plan, overrides)."""
            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.3},
                {"meta_sector": "高端制造", "action": "hold", "weight": 0.0},
            ]

            adjusted_plan, overrides = monitor.apply_forbidden_zone(
                agent_plan=plan,
                current_date="2024-10-01",
            )

            assert isinstance(adjusted_plan, list)
            assert isinstance(overrides, list)

        def test_buy_downgraded_to_hold(
            self, monitor: DailyGuardrailMonitor
        ) -> None:
            """Test that buy is downgraded to hold for forbidden sectors."""
            # Mark 科技成长 as forbidden
            monitor.state_machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.3, "reason": "Test"},
            ]

            adjusted_plan, overrides = monitor.apply_forbidden_zone(
                agent_plan=plan,
                current_date="2024-10-01",
            )

            assert adjusted_plan[0]["action"] == "hold"
            assert len(overrides) == 1

        def test_hold_unchanged(self, monitor: DailyGuardrailMonitor) -> None:
            """Test that hold actions are not changed for forbidden sectors."""
            monitor.state_machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            plan = [
                {"meta_sector": "科技成长", "action": "hold", "weight": 0.0, "reason": "Test"},
            ]

            adjusted_plan, overrides = monitor.apply_forbidden_zone(
                agent_plan=plan,
                current_date="2024-10-01",
            )

            assert adjusted_plan[0]["action"] == "hold"
            assert len(overrides) == 0

        def test_non_forbidden_sectors_unchanged(
            self, monitor: DailyGuardrailMonitor
        ) -> None:
            """Test that non-forbidden sectors are not changed."""
            # Mark 科技成长 as forbidden
            monitor.state_machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.3, "reason": "Test"},
                {"meta_sector": "高端制造", "action": "buy", "weight": 0.2, "reason": "Test"},
            ]

            adjusted_plan, overrides = monitor.apply_forbidden_zone(
                agent_plan=plan,
                current_date="2024-10-01",
            )

            # 科技成长 is changed
            assert adjusted_plan[0]["action"] == "hold"
            # 高端制造 is not changed
            assert adjusted_plan[1]["action"] == "buy"
            assert len(overrides) == 1

    class TestEmergencyExit:
        """Tests for emergency_exit()."""

        def test_returns_dict(self, monitor: DailyGuardrailMonitor) -> None:
            """Test that emergency_exit returns a dict."""
            signal = GuardrailSignal(
                meta_sector="科技成长",
                trigger_type="DAILY_LOSS_5PCT",
                severity=0.7,
                reason="Daily loss exceeds 5%",
                current_date="2024-10-01",
            )

            result = monitor.emergency_exit(signal, "2024-10-01")

            assert isinstance(result, dict)

        def test_marks_sector_forbidden(
            self, monitor: DailyGuardrailMonitor
        ) -> None:
            """Test that emergency_exit marks sector as forbidden."""
            signal = GuardrailSignal(
                meta_sector="科技成长",
                trigger_type="DAILY_LOSS_5PCT",
                severity=0.7,
                reason="Daily loss exceeds 5%",
                current_date="2024-10-01",
            )

            monitor.emergency_exit(signal, "2024-10-01")

            assert monitor.state_machine.is_forbidden("科技成长", "2024-10-01")

        def test_returns_exit_action(
            self, monitor: DailyGuardrailMonitor
        ) -> None:
            """Test that emergency_exit returns correct action."""
            signal = GuardrailSignal(
                meta_sector="科技成长",
                trigger_type="DAILY_LOSS_5PCT",
                severity=0.7,
                reason="Daily loss exceeds 5%",
                current_date="2024-10-01",
            )

            result = monitor.emergency_exit(signal, "2024-10-01")

            assert result["action"] == "emergency_exit"
            assert result["sector"] == "科技成长"

    class TestReset:
        """Tests for reset()."""

        def test_clears_state(self, monitor: DailyGuardrailMonitor) -> None:
            """Test that reset clears all state."""
            monitor.state_machine.mark_forbidden(
                sector="科技成长",
                reason="Daily loss",
                trigger_type="DAILY_LOSS_5PCT",
                current_date="2024-10-01",
            )

            monitor.reset()

            assert len(monitor.state_machine._forbidden) == 0
            assert len(monitor._event_history) == 0

    class TestGetLastGuardrailEvents:
        """Tests for get_last_guardrail_events()."""

        def test_returns_list(self, monitor: DailyGuardrailMonitor) -> None:
            """Test that get_last_guardrail_events returns a list."""
            result = monitor.get_last_guardrail_events()
            assert isinstance(result, list)
