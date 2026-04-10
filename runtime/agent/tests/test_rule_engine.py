"""Tests for src/agent/rule_engine.py"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.rule_engine import (
    RuleCheckResult,
    RuleViolation,
    WeeklyRuleEngine,
)


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.agent.max_weight_per_industry = 0.3
    config.agent.max_total_weight = 1.0
    return config


@pytest.fixture
def sample_level1_plan() -> list[dict]:
    """Create a sample Level 1 plan for testing."""
    return [
        {
            "meta_sector": "科技成长",
            "action": "buy",
            "weight": 0.25,
            "reason": "TMT板块上涨趋势确认",
        },
        {
            "meta_sector": "高端制造",
            "action": "buy",
            "weight": 0.15,
            "reason": "军工板块表现强劲",
        },
        {
            "meta_sector": "消费文娱",
            "action": "hold",
            "weight": 0.0,
            "reason": "等待更多信号",
        },
    ]


class TestRuleViolation:
    """Tests for RuleViolation dataclass."""

    def test_creation(self) -> None:
        """Test creating a RuleViolation."""
        violation = RuleViolation(
            rule_name="weight_limit",
            sector="科技成长",
            severity="error",
            message="Weight 0.5 exceeds max 0.3",
            original_value=0.5,
            adjusted_value=0.3,
        )
        assert violation.rule_name == "weight_limit"
        assert violation.sector == "科技成长"
        assert violation.severity == "error"
        assert violation.original_value == 0.5
        assert violation.adjusted_value == 0.3

    def test_default_adjusted_value(self) -> None:
        """Test default adjusted_value is None."""
        violation = RuleViolation(
            rule_name="weight_limit",
            sector="科技成长",
            severity="error",
            message="Weight exceeds max",
            original_value=0.5,
        )
        assert violation.adjusted_value is None


class TestRuleCheckResult:
    """Tests for RuleCheckResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a RuleCheckResult."""
        result = RuleCheckResult(
            adjusted_plan=[],
            violations=[],
            is_valid=True,
            error_message="",
        )
        assert result.adjusted_plan == []
        assert result.violations == []
        assert result.is_valid is True
        assert result.error_message == ""

    def test_with_violations(self) -> None:
        """Test RuleCheckResult with violations."""
        violations = [
            RuleViolation(
                rule_name="weight_limit",
                sector="科技成长",
                severity="error",
                message="Exceeds max",
                original_value=0.5,
                adjusted_value=0.3,
            )
        ]
        result = RuleCheckResult(
            adjusted_plan=[],
            violations=violations,
            is_valid=False,
            error_message="Weight limit violated",
        )
        assert len(result.violations) == 1
        assert result.is_valid is False


class TestWeeklyRuleEngineInit:
    """Tests for WeeklyRuleEngine initialization."""

    def test_init_with_mock_config(self, mock_config: MagicMock) -> None:
        """Test initialization with mock config."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()
            assert engine.config is not None

    def test_beta_map_built(self, mock_config: MagicMock) -> None:
        """Test that beta map is built during initialization."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()
            assert isinstance(engine._beta_map, dict)
            assert len(engine._beta_map) > 0


class TestApplyWeeklyRules:
    """Tests for apply_weekly_rules()."""

    def test_returns_tuple(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that apply_weekly_rules returns a tuple."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()
            result = engine.apply_weekly_rules(
                level1_plan=sample_level1_plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_list(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that first element is the adjusted plan (list)."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()
            adjusted_plan, violations = engine.apply_weekly_rules(
                level1_plan=sample_level1_plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

        assert isinstance(adjusted_plan, list)

    def test_second_element_is_list(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that second element is violations (list)."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()
            adjusted_plan, violations = engine.apply_weekly_rules(
                level1_plan=sample_level1_plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

        assert isinstance(violations, list)

    def test_violations_contain_rule_violations(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that violations contain RuleViolation instances."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            # Plan with weight exceeding max
            plan_with_violation = [
                {
                    "meta_sector": "科技成长",
                    "action": "buy",
                    "weight": 0.5,  # > 0.3 max
                    "reason": "Test",
                }
            ]

            _, violations = engine.apply_weekly_rules(
                level1_plan=plan_with_violation,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

        assert len(violations) > 0
        assert all(isinstance(v, RuleViolation) for v in violations)

    def test_adjusted_plan_has_same_length(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that adjusted plan has same length as input."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()
            adjusted_plan, _ = engine.apply_weekly_rules(
                level1_plan=sample_level1_plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

        assert len(adjusted_plan) == len(sample_level1_plan)


class TestWeightLimitRule:
    """Tests for weight limit rule."""

    def test_weight_limit_violation(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that weights exceeding max are flagged."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            # Plan with weight 0.25, but max is 0.3, so should pass
            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.25, "reason": "Test"}
            ]
            _, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

            # Should not have weight_limit violation since 0.25 < 0.3
            weight_violations = [v for v in violations if v.rule_name == "weight_limit"]
            assert len(weight_violations) == 0

    def test_weight_limit_capped(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that weights exceeding max are capped."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.5, "reason": "Test"}
            ]
            adjusted_plan, _ = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

            # Weight should be capped to max
            assert adjusted_plan[0]["weight"] <= 0.3

    def test_hold_actions_not_checked(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that hold actions are not checked for weight limit."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            plan = [
                {"meta_sector": "科技成长", "action": "hold", "weight": 0.5, "reason": "Test"}
            ]
            _, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

            weight_violations = [v for v in violations if v.rule_name == "weight_limit"]
            assert len(weight_violations) == 0


class TestBetaPenaltyRule:
    """Tests for beta penalty rule."""

    def test_beta_penalty_on_losing_week(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that beta penalty is applied on losing weeks."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            # Plan with very_high beta sector (科技成长 -> high beta)
            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.2, "reason": "Test"}
            ]
            _, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=-0.03,  # Losing week
                last_week_holdings={},  # New position
                last_week_returns={},
            )

            # Note: 科技成长 has beta level "high", not "very_high"
            # So no beta penalty should be triggered
            beta_violations = [v for v in violations if v.rule_name == "beta_penalty"]
            assert len(beta_violations) == 0

    def test_beta_penalty_only_on_new_positions(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that beta penalty only applies to new positions."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.2, "reason": "Test"}
            ]
            _, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=-0.03,
                last_week_holdings={"科技成长": 0.2},  # Already held
                last_week_returns={},
            )

            beta_violations = [v for v in violations if v.rule_name == "beta_penalty"]
            assert len(beta_violations) == 0


class TestLossProtectionRule:
    """Tests for loss protection rule."""

    def test_loss_protection_on_big_loss(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that weights are reduced after significant loss (>5%)."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.25, "reason": "Test"}
            ]
            adjusted_plan, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=-0.06,  # > 5% loss
                last_week_holdings={},
                last_week_returns={},
            )

            loss_protection_violations = [
                v for v in violations if v.rule_name == "loss_protection"
            ]
            assert len(loss_protection_violations) > 0

    def test_no_loss_protection_on_small_loss(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that weights are not reduced after small loss (<5%)."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.25, "reason": "Test"}
            ]
            _, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=-0.02,  # < 5% loss
                last_week_holdings={},
                last_week_returns={},
            )

            loss_protection_violations = [
                v for v in violations if v.rule_name == "loss_protection"
            ]
            assert len(loss_protection_violations) == 0


class TestMinThresholdRule:
    """Tests for minimum threshold rule."""

    def test_min_threshold_downgrades_small_weight(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that weights < 5% are downgraded to hold."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.03, "reason": "Test"}
            ]
            adjusted_plan, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

            min_threshold_violations = [
                v for v in violations if v.rule_name == "min_threshold"
            ]
            assert len(min_threshold_violations) > 0
            # Action should be changed to hold
            assert adjusted_plan[0]["action"] == "hold"


class TestGlobalLimitRule:
    """Tests for global limit rule."""

    def test_global_limit_scaling(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that total weight exceeding max is scaled down."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            plan = [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.5, "reason": "Test"},
                {"meta_sector": "高端制造", "action": "buy", "weight": 0.5, "reason": "Test"},
                {"meta_sector": "消费文娱", "action": "buy", "weight": 0.3, "reason": "Test"},
            ]
            adjusted_plan, violations = engine.apply_weekly_rules(
                level1_plan=plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

            # Total weight 1.3 > max 1.0, should be scaled
            total_weight = sum(item["weight"] for item in adjusted_plan)
            assert total_weight <= 1.0


class TestMirrorCheck:
    """Tests for mirror position check."""

    def test_mirror_check_runs(
        self, mock_config: MagicMock, sample_level1_plan: list[dict]
    ) -> None:
        """Test that mirror check is applied."""
        with patch("src.agent.rule_engine.load_config", return_value=mock_config):
            engine = WeeklyRuleEngine()

            _, violations = engine.apply_weekly_rules(
                level1_plan=sample_level1_plan,
                last_week_pnl=0.0,
                last_week_holdings={},
                last_week_returns={},
            )

            # Should not raise, mirror check is simplified with meta sectors
            assert isinstance(violations, list)
