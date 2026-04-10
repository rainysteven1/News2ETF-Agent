from __future__ import annotations

import pytest

from src.agent.daily_guardrail import (
    DailyGuardrailMonitor,
    FORBIDDEN_ZONEEntry,
    FORBIDDEN_ZONEStateMachine,
    GuardrailSignal,
    MIN_FORBIDDEN_DAYS,
    RELEASE_CLEAN_DAYS,
)


def test_forbidden_entry_defaults() -> None:
    entry = FORBIDDEN_ZONEEntry(
        sector="科技成长",
        reason="risk",
        trigger_type="ABNORMAL_VOLATILITY",
        start_date="2024-10-01",
    )

    assert entry.end_date is None
    assert entry.last_trigger_date == ""
    assert entry.clean_days == 0
    assert entry.created_at == ""


def test_mark_forbidden_sets_minimum_release_window() -> None:
    machine = FORBIDDEN_ZONEStateMachine()

    machine.mark_forbidden(
        sector="科技成长",
        reason="vol spike",
        trigger_type="ABNORMAL_VOLATILITY",
        current_date="2024-10-01",
    )

    entry = machine._forbidden["科技成长"]
    assert entry.end_date == "2024-10-04"
    assert entry.last_trigger_date == "2024-10-01"
    assert entry.created_at == "2024-10-01"


def test_retrigger_resets_clean_days_and_keeps_latest_end_date() -> None:
    machine = FORBIDDEN_ZONEStateMachine()
    machine.mark_forbidden(
        sector="科技成长",
        reason="first trigger",
        trigger_type="ABNORMAL_VOLATILITY",
        current_date="2024-10-01",
    )
    machine.record_clean_day("科技成长", "2024-10-04")
    assert machine._forbidden["科技成长"].clean_days == 1

    machine.mark_forbidden(
        sector="科技成长",
        reason="second trigger",
        trigger_type="HARD_RISK",
        current_date="2024-10-02",
    )

    entry = machine._forbidden["科技成长"]
    assert entry.reason == "second trigger"
    assert entry.trigger_type == "HARD_RISK"
    assert entry.clean_days == 0
    assert entry.end_date == "2024-10-05"


def test_cooldown_requires_end_date_and_clean_days() -> None:
    machine = FORBIDDEN_ZONEStateMachine()
    machine.mark_forbidden(
        sector="科技成长",
        reason="risk",
        trigger_type="ABNORMAL_VOLATILITY",
        current_date="2024-10-01",
    )

    assert machine.cooldown_expired("科技成长", "2024-10-03") is False
    assert machine.cooldown_expired("科技成长", "2024-10-04") is False

    for current_date in ("2024-10-04", "2024-10-05", "2024-10-06"):
        machine.record_clean_day("科技成长", current_date)

    assert machine.cooldown_expired("科技成长", "2024-10-06") is True
    assert machine.is_forbidden("科技成长", "2024-10-06") is False


def test_auto_release_removes_sector_once_cooldown_expires() -> None:
    machine = FORBIDDEN_ZONEStateMachine()
    machine.mark_forbidden(
        sector="科技成长",
        reason="risk",
        trigger_type="ABNORMAL_VOLATILITY",
        current_date="2024-10-01",
    )
    for current_date in ("2024-10-04", "2024-10-05", "2024-10-06"):
        machine.record_clean_day("科技成长", current_date)

    assert machine.auto_release("科技成长", "2024-10-06") is True
    assert machine.get_forbidden_info("科技成长") is None


def test_check_guardrail_trigger_emits_metric_based_signals() -> None:
    monitor = DailyGuardrailMonitor()

    signals = monitor.check_guardrail_trigger(
        current_date="2024-10-01",
        positions={},
        etf_prices={},
        sector_metrics={
            "科技成长": {
                "intraday_vol": 0.1,
                "news_heat": 0.98,
                "data_missing": True,
                "hard_risk": True,
            }
        },
    )

    assert [signal.trigger_type for signal in signals] == [
        "ABNORMAL_VOLATILITY",
        "ABNORMAL_HEAT",
        "DATA_MISSING",
        "HARD_RISK",
    ]
    assert [signal.action for signal in signals] == [
        "cap_weight",
        "forbid_open",
        "forbid_open",
        "force_flat",
    ]


def test_check_guardrail_trigger_accumulates_recent_history() -> None:
    monitor = DailyGuardrailMonitor()

    monitor.check_guardrail_trigger(
        current_date="2024-10-01",
        positions={},
        etf_prices={},
        sector_metrics={"科技成长": {"hard_risk": True}},
    )

    events = monitor.get_last_guardrail_events()
    assert events == [
        {
            "meta_sector": "科技成长",
            "trigger_type": "HARD_RISK",
            "severity": 1.0,
            "reason": "Hard risk event detected, force flatten required",
            "date": "2024-10-01",
        }
    ]


def test_clean_days_progress_and_auto_release_through_monitor() -> None:
    monitor = DailyGuardrailMonitor()
    monitor.state_machine.mark_forbidden(
        sector="科技成长",
        reason="risk",
        trigger_type="ABNORMAL_VOLATILITY",
        current_date="2024-10-01",
    )

    for current_date in ("2024-10-04", "2024-10-05", "2024-10-06"):
        monitor.check_guardrail_trigger(
            current_date=current_date,
            positions={},
            etf_prices={},
            sector_metrics={},
        )

    assert monitor.state_machine.get_forbidden_info("科技成长") is None


def test_apply_forbidden_zone_downgrades_buy_and_tags_guardrail_action() -> None:
    monitor = DailyGuardrailMonitor()
    monitor.state_machine.mark_forbidden(
        sector="科技成长",
        reason="vol spike",
        trigger_type="ABNORMAL_VOLATILITY",
        current_date="2024-10-01",
    )

    adjusted_plan, overrides = monitor.apply_forbidden_zone(
        agent_plan=[{"meta_sector": "科技成长", "action": "buy", "weight": 0.3, "reason": "alpha"}],
        current_date="2024-10-01",
    )

    assert adjusted_plan == [
        {
            "meta_sector": "科技成长",
            "action": "hold",
            "weight": 0.0,
            "reason": "[FORBIDDEN_ZONE] alpha",
            "original_action": "buy",
            "guardrail_action": "forbid_add",
        }
    ]
    assert overrides[0]["guardrail_action"] == "forbid_add"


def test_emergency_exit_marks_forbidden_and_returns_payload() -> None:
    monitor = DailyGuardrailMonitor()
    signal = GuardrailSignal(
        meta_sector="科技成长",
        trigger_type="HARD_RISK",
        severity=0.8,
        reason="hard risk",
        current_date="2024-10-01",
        action="force_flat",
    )

    result = monitor.emergency_exit(signal, "2024-10-01")

    assert result["action"] == "emergency_exit"
    assert result["guardrail_action"] == "force_flat"
    assert result["pnl_impact"] == pytest.approx(-0.04)
    assert monitor.state_machine.is_forbidden("科技成长", "2024-10-01") is True


def test_guardrail_constants_are_positive() -> None:
    assert MIN_FORBIDDEN_DAYS > 0
    assert RELEASE_CLEAN_DAYS > 0
