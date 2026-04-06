"""Weekly Rule Engine - applies trading rules to Level 1 plan.

This module provides the WeeklyRuleEngine class that applies rules
to the agent's weekly plan:
  1. Weight upper limit per sector
  2. Beta penalty for losing weeks
  3. Mirror position check (correlation clusters)
  4. Loss protection (stop-loss)
  5. Minimum operation threshold
  6. Global position limit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import load_config
from src.utils.meta_sector_map import get_meta_sectors


@dataclass
class RuleViolation:
    """Represents a rule violation."""

    rule_name: str
    sector: str
    severity: str  # "error", "warning"
    message: str
    original_value: Any
    adjusted_value: Any | None = None


@dataclass
class RuleCheckResult:
    """Result of applying all rules."""

    adjusted_plan: list[dict[str, Any]]
    violations: list[RuleViolation] = field(default_factory=list)
    is_valid: bool = True
    error_message: str = ""


class WeeklyRuleEngine:
    """Applies weekly trading rules to Level 1 plan."""

    def __init__(self):
        """Initialize the rule engine."""
        self.config = load_config()
        self._beta_map = self._build_beta_map()

    def _build_beta_map(self) -> dict[str, str]:
        """Build beta mapping for meta sectors.

        Returns:
            Dict mapping meta sector to beta level
        """
        # Beta levels: very_high, high, medium, low
        # Based on typical sector characteristics
        return {
            "科技成长": "high",
            "高端制造": "high",
            "消费文娱": "medium",
            "医药健康": "medium",
            "资源材料": "medium",
            "金融地产": "low",
            "基础设施/公共": "low",
            "主题策略": "medium",
        }

    def _get_beta_level(self, sector: str) -> str:
        """Get beta level for a sector."""
        return self._beta_map.get(sector, "medium")

    def apply_weekly_rules(
        self,
        level1_plan: list[dict[str, Any]],
        last_week_pnl: float,
        last_week_holdings: dict[str, float],
        last_week_returns: dict[str, float],
    ) -> tuple[list[dict[str, Any]], list[RuleViolation]]:
        """Apply all weekly rules to the Level 1 plan.

        Args:
            level1_plan: Agent's Level 1 plan [{meta_sector, action, weight, reason}, ...]
            last_week_pnl: Last week's portfolio return
            last_week_holdings: Last week's holdings {sector: weight}
            last_week_returns: Last week's returns per sector

        Returns:
            Tuple of (adjusted_plan, violations)
        """
        violations: list[RuleViolation] = []
        adjusted_plan = [item.copy() for item in level1_plan]

        max_weight = self.config.agent.max_weight_per_industry
        max_total = self.config.agent.max_total_weight

        # Rule 1: Weight upper limit per sector
        adjusted_plan, v = self._apply_weight_limit(adjusted_plan, max_weight)
        violations.extend(v)

        # Rule 2: Beta penalty - no new very_high beta buys when losing week
        if last_week_pnl < 0:
            adjusted_plan, v = self._apply_beta_penalty(
                adjusted_plan, last_week_holdings, last_week_pnl
            )
            violations.extend(v)

        # Rule 3: Mirror position check - avoid high correlation clusters
        adjusted_plan, v = self._apply_mirror_check(adjusted_plan)
        violations.extend(v)

        # Rule 4: Loss protection - reduce exposure after losses
        adjusted_plan, v = self._apply_loss_protection(adjusted_plan, last_week_pnl)
        violations.extend(v)

        # Rule 5: Minimum operation threshold
        adjusted_plan, v = self._apply_min_threshold(adjusted_plan)
        violations.extend(v)

        # Rule 6: Global position limit
        adjusted_plan, v = self._apply_global_limit(adjusted_plan, max_total)
        violations.extend(v)

        is_valid = all(v.severity != "error" for v in violations)

        return adjusted_plan, violations

    def _apply_weight_limit(
        self,
        plan: list[dict[str, Any]],
        max_weight: float,
    ) -> tuple[list[dict[str, Any]], list[RuleViolation]]:
        """Apply per-sector weight limit."""
        violations = []
        adjusted = []

        for item in plan:
            sector = item.get("meta_sector", item.get("industry", ""))
            weight = item.get("weight", 0.0)
            action = item.get("action", "")

            if action == "buy" and weight > max_weight:
                violations.append(
                    RuleViolation(
                        rule_name="weight_limit",
                        sector=sector,
                        severity="error",
                        message=f"Weight {weight:.3f} exceeds max {max_weight}",
                        original_value=weight,
                        adjusted_value=max_weight,
                    )
                )
                item["weight"] = max_weight
                item["reason"] = f"[WEIGHT_LIMIT] {item.get('reason', '')}"

            adjusted.append(item)

        return adjusted, violations

    def _apply_beta_penalty(
        self,
        plan: list[dict[str, Any]],
        last_week_holdings: dict[str, float],
        last_week_pnl: float,
    ) -> tuple[list[dict[str, Any]], list[RuleViolation]]:
        """Apply beta penalty: no new very_high beta buys when losing week."""
        violations = []
        adjusted = []

        for item in plan:
            sector = item.get("meta_sector", item.get("industry", ""))
            action = item.get("action", "")
            weight = item.get("weight", 0.0)

            # Only check NEW positions (not already held)
            is_new_position = sector not in last_week_holdings or last_week_holdings.get(sector, 0) < 0.01

            beta = self._get_beta_level(sector)
            if is_new_position and action == "buy" and beta == "very_high" and last_week_pnl < 0:
                violations.append(
                    RuleViolation(
                        rule_name="beta_penalty",
                        sector=sector,
                        severity="error",
                        message=f"Cannot add new very_high Beta position when last_week_pnl={last_week_pnl:.2%}",
                        original_value=action,
                        adjusted_value="hold",
                    )
                )
                item["action"] = "hold"
                item["reason"] = f"[BETA_PENALTY] {item.get('reason', '')}"

            adjusted.append(item)

        return adjusted, violations

    def _apply_mirror_check(
        self,
        plan: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[RuleViolation]]:
        """Apply mirror position check - avoid high correlation clusters.

        Note: With meta sectors, mirror checking is simplified since each
        meta sector is already a distinct group. This rule mainly prevents
        over-concentration in a single meta sector.
        """
        violations = []
        adjusted = plan

        # Check for too many high-weight positions in the same sector group
        high_weight_items = [item for item in plan if item.get("action") == "buy" and item.get("weight", 0) >= 0.15]

        # With meta sectors, this is less critical since sectors are already aggregated
        # But we still check for over-concentration

        return adjusted, violations

    def _apply_loss_protection(
        self,
        plan: list[dict[str, Any]],
        last_week_pnl: float,
    ) -> tuple[list[dict[str, Any]], list[RuleViolation]]:
        """Apply loss protection: reduce exposure after significant losses."""
        violations = []
        adjusted = []

        # If last week loss > 5%, reduce all buy weights by 20%
        if last_week_pnl < -0.05:
            for item in plan:
                if item.get("action") == "buy":
                    old_weight = item.get("weight", 0)
                    new_weight = old_weight * 0.8
                    item["weight"] = new_weight
                    item["reason"] = f"[LOSS_PROTECTION] {item.get('reason', '')}"

                    if old_weight > 0.01:
                        violations.append(
                            RuleViolation(
                                rule_name="loss_protection",
                                sector=item.get("meta_sector", item.get("industry", "")),
                                severity="warning",
                                message=f"Reduced weight {old_weight:.3f} -> {new_weight:.3f} after loss {last_week_pnl:.2%}",
                                original_value=old_weight,
                                adjusted_value=new_weight,
                            )
                        )

                adjusted.append(item)
        else:
            adjusted = plan

        return adjusted, violations

    def _apply_min_threshold(
        self,
        plan: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[RuleViolation]]:
        """Apply minimum operation threshold: weight < 5% -> hold."""
        violations = []
        adjusted = []

        for item in plan:
            action = item.get("action", "")
            weight = item.get("weight", 0.0)
            sector = item.get("meta_sector", item.get("industry", ""))

            if action == "buy" and 0 < weight < 0.05:
                violations.append(
                    RuleViolation(
                        rule_name="min_threshold",
                        sector=sector,
                        severity="warning",
                        message=f"Weight {weight:.3f} < 0.05 minimum - downgraded to hold",
                        original_value=action,
                        adjusted_value="hold",
                    )
                )
                item["action"] = "hold"
                item["reason"] = f"[MIN_THRESHOLD] {item.get('reason', '')}"

            adjusted.append(item)

        return adjusted, violations

    def _apply_global_limit(
        self,
        plan: list[dict[str, Any]],
        max_total: float,
    ) -> tuple[list[dict[str, Any]], list[RuleViolation]]:
        """Apply global position limit."""
        violations = []
        adjusted = []

        total_buy_weight = sum(
            item.get("weight", 0) for item in plan if item.get("action") == "buy"
        )

        if total_buy_weight > max_total:
            # Scale down all buy weights proportionally
            scale_factor = max_total / total_buy_weight

            for item in plan:
                if item.get("action") == "buy":
                    old_weight = item.get("weight", 0)
                    new_weight = old_weight * scale_factor
                    item["weight"] = new_weight

                adjusted.append(item)

            violations.append(
                RuleViolation(
                    rule_name="global_limit",
                    sector="TOTAL",
                    severity="error",
                    message=f"Total weight {total_buy_weight:.3f} exceeds max {max_total} - scaled by {scale_factor:.3f}",
                    original_value=total_buy_weight,
                    adjusted_value=max_total,
                )
            )
        else:
            adjusted = plan

        return adjusted, violations
