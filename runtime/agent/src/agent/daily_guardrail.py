"""Daily Guardrail Monitor - enforces forbidden zone rules and emergency exits.

This module provides:
  - COOLDOWN_RULES: Mapping of trigger types to cooldown periods
  - FORBIDDEN_ZONEStateMachine: Manages sector forbidden zone states
  - DailyGuardrailMonitor: Checks guardrail triggers and applies forbidden zones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

MIN_FORBIDDEN_DAYS = 3
RELEASE_CLEAN_DAYS = 3
ALLOWED_GUARDRAIL_ACTIONS = {"cap_weight", "forbid_open", "forbid_add", "force_flat"}


class SectorStatus(Enum):
    """Sector trading status."""

    NORMAL = "normal"
    FORBIDDEN_ZONE = "forbidden"


@dataclass
class GuardrailSignal:
    """Represents a guardrail trigger signal."""

    meta_sector: str
    trigger_type: str  # COOLDOWN_RULES key
    severity: float  # 0-1 severity score
    reason: str
    current_date: str
    action: str = "forbid_open"


@dataclass
class FORBIDDEN_ZONEEntry:
    """Entry in the forbidden zone state machine."""

    sector: str
    reason: str
    trigger_type: str
    start_date: str
    end_date: str | None = None
    last_trigger_date: str = ""
    clean_days: int = 0
    created_at: str = ""


class FORBIDDEN_ZONEStateMachine:
    """Manages sector forbidden zone states with cooldown logic."""

    def __init__(self):
        """Initialize the state machine."""
        self._forbidden: dict[str, FORBIDDEN_ZONEEntry] = {}

    def mark_forbidden(
        self,
        sector: str,
        reason: str,
        trigger_type: str,
        current_date: str,
    ) -> None:
        """Mark a sector as forbidden.

        Args:
            sector: Meta sector name
            reason: Human-readable reason
            trigger_type: Key from COOLDOWN_RULES
            current_date: Current date string (YYYY-MM-DD)
        """
        start_dt = datetime.strptime(current_date, "%Y-%m-%d")
        end_date = (start_dt + timedelta(days=MIN_FORBIDDEN_DAYS)).strftime("%Y-%m-%d")
        existing = self._forbidden.get(sector)
        if existing is not None:
            existing.reason = reason
            existing.trigger_type = trigger_type
            existing.last_trigger_date = current_date
            existing.clean_days = 0
            existing.end_date = max(existing.end_date or end_date, end_date)
            return

        self._forbidden[sector] = FORBIDDEN_ZONEEntry(
            sector=sector,
            reason=reason,
            trigger_type=trigger_type,
            start_date=current_date,
            end_date=end_date,
            last_trigger_date=current_date,
            clean_days=0,
            created_at=current_date,
        )

    def is_forbidden(self, sector: str, current_date: str) -> bool:
        """Check if a sector is currently in forbidden zone.

        Args:
            sector: Meta sector name
            current_date: Current date string (YYYY-MM-DD)

        Returns:
            True if sector is forbidden
        """
        if sector not in self._forbidden:
            return False

        return not self.cooldown_expired(sector, current_date)

    def get_forbidden_sectors(self, current_date: str) -> list[str]:
        """Get all sectors currently in forbidden zone.

        Args:
            current_date: Current date string

        Returns:
            List of forbidden sector names
        """
        forbidden = []
        for sector in list(self._forbidden.keys()):
            if self.is_forbidden(sector, current_date):
                forbidden.append(sector)
        return forbidden

    def cooldown_expired(self, sector: str, current_date: str) -> bool:
        """Check if cooldown period has expired.

        Args:
            sector: Meta sector name
            current_date: Current date string

        Returns:
            True if cooldown has expired (sector can be released)
        """
        if sector not in self._forbidden:
            return True

        entry = self._forbidden[sector]

        if entry.end_date is None or current_date < entry.end_date:
            return False
        return entry.clean_days >= RELEASE_CLEAN_DAYS

    def auto_release(self, sector: str, current_date: str) -> bool:
        """Attempt to auto-release a sector from forbidden zone.

        Args:
            sector: Meta sector name
            current_date: Current date string

        Returns:
            True if sector was released
        """
        if not self.cooldown_expired(sector, current_date):
            return False

        if sector in self._forbidden:
            del self._forbidden[sector]
            return True

        return False

    def record_clean_day(self, sector: str, current_date: str) -> None:
        entry = self._forbidden.get(sector)
        if entry is None:
            return
        if current_date >= (entry.end_date or current_date):
            entry.clean_days += 1

    def get_forbidden_info(self, sector: str) -> dict[str, Any] | None:
        """Get information about a forbidden sector.

        Args:
            sector: Meta sector name

        Returns:
            Dict with forbidden info or None
        """
        if sector not in self._forbidden:
            return None

        entry = self._forbidden[sector]
        return {
            "sector": entry.sector,
            "reason": entry.reason,
            "trigger_type": entry.trigger_type,
            "start_date": entry.start_date,
            "end_date": entry.end_date,
        }

    def clear_all(self) -> None:
        """Clear all forbidden zone entries (for testing)."""
        self._forbidden.clear()


class DailyGuardrailMonitor:
    """Monitors guardrail triggers and applies forbidden zones."""

    def __init__(self):
        """Initialize the monitor."""
        self.state_machine = FORBIDDEN_ZONEStateMachine()
        self._event_history: list[GuardrailSignal] = []

    def check_guardrail_trigger(
        self,
        current_date: str,
        positions: dict[str, float],
        etf_prices: dict[str, float],
        news_df: Any | None = None,
        sector_metrics: dict[str, dict[str, Any]] | None = None,
    ) -> list[GuardrailSignal]:
        """Check if any guardrail triggers are activated.

        Args:
            current_date: Current date string
            positions: Current holdings {meta_sector: weight}
            etf_prices: Current ETF prices {etf_code: price}
            news_df: Optional news DataFrame for breaking news detection

        Returns:
            List of triggered GuardrailSignal
        """
        signals = []
        sector_metrics = sector_metrics or {}

        for sector, metrics in sector_metrics.items():
            intraday_vol = float(metrics.get("intraday_vol", 0.0) or 0.0)
            news_heat = float(metrics.get("news_heat", 0.0) or 0.0)
            data_missing = bool(metrics.get("data_missing", False))
            hard_risk = bool(metrics.get("hard_risk", False))

            if intraday_vol >= 0.08:
                signals.append(
                    GuardrailSignal(
                        meta_sector=sector,
                        trigger_type="ABNORMAL_VOLATILITY",
                        severity=min(intraday_vol / 0.12, 1.0),
                        reason=f"Intraday volatility {intraday_vol:.2%} exceeds abnormal threshold",
                        current_date=current_date,
                        action="cap_weight" if intraday_vol < 0.12 else "forbid_add",
                    )
                )

            if news_heat >= 0.95:
                signals.append(
                    GuardrailSignal(
                        meta_sector=sector,
                        trigger_type="ABNORMAL_HEAT",
                        severity=min(news_heat, 1.0),
                        reason=f"News heat percentile {news_heat:.2f} exceeds abnormal threshold",
                        current_date=current_date,
                        action="forbid_open",
                    )
                )

            if data_missing:
                signals.append(
                    GuardrailSignal(
                        meta_sector=sector,
                        trigger_type="DATA_MISSING",
                        severity=1.0,
                        reason="Critical data missing for sector decision inputs",
                        current_date=current_date,
                        action="forbid_open",
                    )
                )

            if hard_risk:
                signals.append(
                    GuardrailSignal(
                        meta_sector=sector,
                        trigger_type="HARD_RISK",
                        severity=1.0,
                        reason="Hard risk event detected, force flatten required",
                        current_date=current_date,
                        action="force_flat",
                    )
                )

        # Check for breaking news (placeholder)
        if news_df is not None:
            breaking_news_signals = self._check_breaking_news(current_date, news_df)
            signals.extend(breaking_news_signals)

        triggered = {signal.meta_sector for signal in signals}
        for sector in self.state_machine.get_forbidden_sectors(current_date):
            if sector in triggered:
                continue
            self.state_machine.record_clean_day(sector, current_date)
            self.state_machine.auto_release(sector, current_date)

        for signal in signals:
            self._event_history.append(signal)

        return signals

    def _check_breaking_news(
        self,
        current_date: str,
        news_df: Any,
    ) -> list[GuardrailSignal]:
        """Check for breaking news triggers.

        Args:
            current_date: Current date
            news_df: News DataFrame

        Returns:
            List of breaking news signals
        """
        # Placeholder: would search for specific keywords like "黑天鹅", "利空", etc.
        return []

    def _get_etf_for_sector(self, sector: str) -> str:
        """Get the representative ETF code for a sector.

        Args:
            sector: Meta sector name

        Returns:
            ETF code
        """
        # Placeholder mapping
        sector_etf_map = {
            "科技成长": "512000",
            "高端制造": "512660",
            "消费文娱": "159928",
            "医药健康": "512010",
            "资源材料": "512400",
            "金融地产": "512800",
            "基础设施/公共": "512580",
            "主题策略": "510300",
        }
        return sector_etf_map.get(sector, "510300")

    def emergency_exit(
        self,
        signal: GuardrailSignal,
        current_date: str,
    ) -> dict[str, Any]:
        """Handle emergency exit for a guardrail trigger.

        Args:
            signal: The guardrail signal
            current_date: Current date

        Returns:
            Dict with exit actions
        """
        # Mark sector as forbidden
        self.state_machine.mark_forbidden(
            sector=signal.meta_sector,
            reason=signal.reason,
            trigger_type=signal.trigger_type,
            current_date=current_date,
        )

        return {
            "action": "emergency_exit",
            "sector": signal.meta_sector,
            "reason": signal.reason,
            "trigger_type": signal.trigger_type,
            "guardrail_action": signal.action,
            "pnl_impact": -signal.severity * 0.05,  # Estimated impact
        }

    def apply_forbidden_zone(
        self,
        agent_plan: list[dict[str, Any]],
        current_date: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply forbidden zone restrictions to agent plan.

        Args:
            agent_plan: Agent's Level 1 plan [{meta_sector, action, weight, reason}, ...]
            current_date: Current date string

        Returns:
            Tuple of (adjusted_plan, overrides)
            - adjusted_plan: Plan with forbidden sectors downgraded to hold
            - overrides: List of override actions taken
        """
        forbidden_sectors = self.state_machine.get_forbidden_sectors(current_date)
        overrides = []

        adjusted_plan = []
        for item in agent_plan:
            sector = item.get("meta_sector", "")
            action = item.get("action", "")

            if sector in forbidden_sectors and action == "buy":
                adjusted_item = item.copy()
                info = self.state_machine.get_forbidden_info(sector) or {}
                trigger_type = info.get("trigger_type", "")
                adjusted_item["action"] = "hold"
                adjusted_item["weight"] = 0.0
                adjusted_item["reason"] = f"[FORBIDDEN_ZONE] {item.get('reason', '')}".strip()
                adjusted_item["original_action"] = action
                adjusted_item["guardrail_action"] = "forbid_open"
                if trigger_type == "ABNORMAL_VOLATILITY":
                    adjusted_item["guardrail_action"] = "forbid_add"
                if trigger_type == "HARD_RISK":
                    adjusted_item["guardrail_action"] = "force_flat"
                adjusted_plan.append(adjusted_item)
                overrides.append(
                    {
                        "sector": sector,
                        "original_action": action,
                        "adjusted_action": "hold",
                        "guardrail_action": adjusted_item["guardrail_action"],
                        "reason": info,
                    }
                )
            else:
                adjusted_plan.append(item)

        return adjusted_plan, overrides

    def get_last_guardrail_events(self) -> list[dict[str, Any]]:
        """Get history of guardrail events.

        Returns:
            List of event dictionaries
        """
        return [
            {
                "meta_sector": e.meta_sector,
                "trigger_type": e.trigger_type,
                "severity": e.severity,
                "reason": e.reason,
                "date": e.current_date,
            }
            for e in self._event_history[-10:]  # Last 10 events
        ]

    def reset(self) -> None:
        """Reset the monitor state (for testing)."""
        self.state_machine.clear_all()
        self._event_history.clear()
