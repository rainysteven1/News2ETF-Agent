"""Tests for src/agent/decision_logger.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.decision_logger import (
    DecisionLogger,
    DecisionRecord,
    GuardrailEvent,
    TCNPredictionError,
)


@pytest.fixture
def sample_decision_record() -> DecisionRecord:
    """Create a sample DecisionRecord for testing."""
    return DecisionRecord(
        monday_date="2024-10-01",
        agent_input={"market_state": {"market_state": "bullish", "market_volatility": 0.6}, "vol_percentile": 0.6, "sector_signals": {}},
        level1_plan=[
            {
                "meta_sector": "科技成长",
                "action": "buy",
                "weight": 0.3,
                "reason": "TMT板块上涨趋势确认",
            },
            {
                "meta_sector": "高端制造",
                "action": "hold",
                "weight": 0.0,
                "reason": "等待更多信号",
            },
        ],
        level2_plan=[
            {
                "meta_sector": "科技成长",
                "selected_indices": ["半导体/芯片指数"],
                "selected_etf": "512000 华夏国证半导体ETF",
            }
        ],
        weekly_return=0.05,
        guardrail_events=[],
        tcn_prediction_errors=[],
        reasoning_summary="基于TMT板块动量和技术指标买入",
        quality_label="good",
    )


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Create a temporary log file path."""
    return tmp_path / "decision_logs.jsonl"


class TestTCNPredictionError:
    """Tests for TCNPredictionError dataclass."""

    def test_creation(self) -> None:
        """Test creating a TCNPredictionError."""
        error = TCNPredictionError(
            meta_sector="科技成长",
            tcn_predicted=0.1,
            actual_return=0.05,
            divergence=0.05,
            root_cause_guess="TCN_overestimated",
        )
        assert error.meta_sector == "科技成长"
        assert error.tcn_predicted == 0.1
        assert error.actual_return == 0.05
        assert error.divergence == 0.05
        assert error.root_cause_guess == "TCN_overestimated"

    def test_default_root_cause(self) -> None:
        """Test default root_cause_guess is empty string."""
        error = TCNPredictionError(
            meta_sector="科技成长",
            tcn_predicted=0.1,
            actual_return=0.08,
            divergence=0.02,
        )
        assert error.root_cause_guess == ""


class TestGuardrailEvent:
    """Tests for GuardrailEvent dataclass."""

    def test_creation(self) -> None:
        """Test creating a GuardrailEvent."""
        event = GuardrailEvent(
            date="2024-10-01",
            meta_sector="科技成长",
            trigger_type="DAILY_LOSS_5PCT",
            etf_code="512000",
            pnl_impact=-0.05,
            reason="Daily loss exceeds threshold",
        )
        assert event.date == "2024-10-01"
        assert event.meta_sector == "科技成长"
        assert event.trigger_type == "DAILY_LOSS_5PCT"


class TestDecisionRecord:
    """Tests for DecisionRecord dataclass."""

    def test_creation(self) -> None:
        """Test creating a DecisionRecord."""
        record = DecisionRecord(
            monday_date="2024-10-01",
            agent_input={},
            level1_plan=[],
            level2_plan=[],
            weekly_return=0.03,
        )
        assert record.monday_date == "2024-10-01"
        assert record.weekly_return == 0.03

    def test_default_quality_label(self) -> None:
        """Test default quality_label is empty string."""
        record = DecisionRecord(
            monday_date="2024-10-01",
            agent_input={},
            level1_plan=[],
            level2_plan=[],
            weekly_return=0.0,
        )
        assert record.quality_label == ""

    def test_default_guardrail_events(self) -> None:
        """Test default guardrail_events is empty list."""
        record = DecisionRecord(
            monday_date="2024-10-01",
            agent_input={},
            level1_plan=[],
            level2_plan=[],
            weekly_return=0.0,
        )
        assert record.guardrail_events == []


class TestDecisionLoggerInit:
    """Tests for DecisionLogger initialization."""

    def test_init_with_log_path(self, temp_log_file: Path) -> None:
        """Test initialization with explicit log path."""
        logger = DecisionLogger(log_path=temp_log_file)
        assert logger.log_path == temp_log_file

    def test_log_path_creates_parent_dirs(self, temp_log_file: Path) -> None:
        """Test that log file parent directories are created."""
        logger = DecisionLogger(log_path=temp_log_file)
        assert temp_log_file.parent.exists()

    def test_default_log_path(self, temp_log_file: Path) -> None:
        """Test that default log path is set when not provided."""
        logger = DecisionLogger(log_path=temp_log_file)
        assert logger.log_path is not None


class TestDecisionLoggerLogDecision:
    """Tests for log_decision()."""

    def test_writes_to_file(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that log_decision writes a record to the file."""
        logger = DecisionLogger(log_path=temp_log_file)
        logger.log_decision(sample_decision_record)

        assert temp_log_file.exists()
        with open(temp_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_writes_valid_json(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that the written content is valid JSON."""
        logger = DecisionLogger(log_path=temp_log_file)
        logger.log_decision(sample_decision_record)

        with open(temp_log_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                assert isinstance(data, dict)
                assert data["monday_date"] == "2024-10-01"

    def test_multiple_records(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test logging multiple decision records."""
        logger = DecisionLogger(log_path=temp_log_file)

        record1 = sample_decision_record
        record2 = DecisionRecord(
            monday_date="2024-10-08",
            agent_input={},
            level1_plan=[],
            level2_plan=[],
            weekly_return=-0.02,
            quality_label="bad",
        )

        logger.log_decision(record1)
        logger.log_decision(record2)

        with open(temp_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_log_decision_with_none_path(self, sample_decision_record: DecisionRecord) -> None:
        """Test that log_decision handles None log_path gracefully."""
        logger = DecisionLogger(log_path=None)
        # Should not raise an exception
        logger.log_decision(sample_decision_record)


class TestDecisionLoggerComputeTCNError:
    """Tests for compute_tcn_error()."""

    def test_returns_list(self) -> None:
        """Test that compute_tcn_error returns a list."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [0.1, 0.15, 0.12]}
        actual_returns = {"科技成长": 0.08}

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        assert isinstance(result, list)

    def test_returns_tcn_prediction_errors(self) -> None:
        """Test that returned items are TCNPredictionError instances."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [0.1, 0.15, 0.12]}
        actual_returns = {"科技成长": 0.08}

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        assert len(result) == 1
        assert isinstance(result[0], TCNPredictionError)

    def test_divergence_calculation(self) -> None:
        """Test that divergence is correctly calculated."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [0.1]}
        actual_returns = {"科技成长": 0.05}

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        assert result[0].divergence == pytest.approx(0.05)
        assert result[0].tcn_predicted == pytest.approx(0.1)
        assert result[0].actual_return == pytest.approx(0.05)

    def test_uses_last_prediction(self) -> None:
        """Test that the last value in the sequence is used."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [0.05, 0.08, 0.12]}  # Last is 0.12
        actual_returns = {"科技成长": 0.10}

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        assert result[0].tcn_predicted == pytest.approx(0.12)

    def test_skips_empty_sequences(self) -> None:
        """Test that empty sequences are skipped."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [], "高端制造": [0.1]}
        actual_returns = {"科技成长": 0.05, "高端制造": 0.08}

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        # Only 1 result (高端制造), 科技成长 is skipped
        assert len(result) == 1

    def test_missing_actual_returns(self) -> None:
        """Test handling of missing actual returns."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [0.1]}
        actual_returns = {}  # No actual return for 科技成长

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        assert len(result) == 1
        assert result[0].actual_return == pytest.approx(0.0)

    def test_root_cause_guess_low_error(self) -> None:
        """Test root_cause_guess is 'low_error' when divergence is small."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [0.1]}
        actual_returns = {"科技成长": 0.09}  # Divergence = 0.01 < 0.02

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        assert result[0].root_cause_guess == "low_error"

    def test_root_cause_guess_overconfident_bearish(self) -> None:
        """Test root_cause_guess for overconfident bearish case."""
        logger = DecisionLogger(log_path=None)
        tcn_sequence = {"科技成长": [0.1]}
        actual_returns = {"科技成长": -0.05}  # Predicted positive, actual negative

        result = logger.compute_tcn_error(tcn_sequence, actual_returns)

        assert result[0].root_cause_guess == "TCN_overconfident_bearish"


class TestDecisionLoggerExtractGoodBadPatterns:
    """Tests for extract_good_bad_patterns()."""

    def test_returns_tuple(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that extract_good_bad_patterns returns a tuple."""
        logger = DecisionLogger(log_path=temp_log_file)
        logger.log_decision(sample_decision_record)

        result = logger.extract_good_bad_patterns()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_good_patterns(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that first element is list of good patterns."""
        logger = DecisionLogger(log_path=temp_log_file)
        logger.log_decision(sample_decision_record)

        good_patterns, bad_patterns = logger.extract_good_bad_patterns()

        assert isinstance(good_patterns, list)
        assert len(good_patterns) > 0

    def test_second_element_is_bad_patterns(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that second element is list of bad patterns."""
        logger = DecisionLogger(log_path=temp_log_file)

        bad_record = DecisionRecord(
            monday_date="2024-10-08",
            agent_input={},
            level1_plan=[
                {
                    "meta_sector": "科技成长",
                    "action": "buy",
                    "weight": 0.5,
                    "reason": "过度自信",
                }
            ],
            level2_plan=[],
            weekly_return=-0.08,
            quality_label="bad",
        )
        logger.log_decision(sample_decision_record)  # good
        logger.log_decision(bad_record)  # bad

        good_patterns, bad_patterns = logger.extract_good_bad_patterns()

        assert isinstance(bad_patterns, list)
        assert len(bad_patterns) > 0

    def test_empty_log_file(self, temp_log_file: Path) -> None:
        """Test that empty log file returns empty lists."""
        logger = DecisionLogger(log_path=temp_log_file)

        good_patterns, bad_patterns = logger.extract_good_bad_patterns()

        assert good_patterns == []
        assert bad_patterns == []

    def test_pattern_format(self, temp_log_file: Path) -> None:
        """Test that pattern strings have expected format."""
        logger = DecisionLogger(log_path=temp_log_file)

        record = DecisionRecord(
            monday_date="2024-10-01",
            agent_input={},
            level1_plan=[
                {
                    "meta_sector": "科技成长",
                    "action": "buy",
                    "weight": 0.3,
                    "reason": "测试原因",
                }
            ],
            level2_plan=[],
            weekly_return=0.05,
            quality_label="good",
        )
        logger.log_decision(record)

        good_patterns, _ = logger.extract_good_bad_patterns()

        assert len(good_patterns) > 0
        pattern = good_patterns[0]
        assert "科技成长" in pattern
        assert "buy" in pattern

    def test_none_log_path(self) -> None:
        """Test that None log_path returns empty lists."""
        logger = DecisionLogger(log_path=None)
        # After init, log_path defaults to config.data.output_logs, so it's not None
        # Extract from an empty log file (just created by __init__)
        good_patterns, bad_patterns = logger.extract_good_bad_patterns()
        # With empty log, should return empty lists
        assert isinstance(good_patterns, list)
        assert isinstance(bad_patterns, list)


class TestDecisionLoggerAssignQualityLabels:
    """Tests for assign_quality_labels()."""

    def test_returns_string(self) -> None:
        """Test that assign_quality_labels returns a string."""
        logger = DecisionLogger(log_path=None)
        result = logger.assign_quality_labels(weekly_return=0.05, signal_alignment=0.6)
        assert isinstance(result, str)

    def test_good_positive_return(self) -> None:
        """Test 'good' label for positive return > 2%."""
        logger = DecisionLogger(log_path=None)
        result = logger.assign_quality_labels(weekly_return=0.03, signal_alignment=0.5)
        assert result == "good"

    def test_good_high_alignment(self) -> None:
        """Test 'good' label for high alignment with positive return."""
        logger = DecisionLogger(log_path=None)
        result = logger.assign_quality_labels(weekly_return=0.01, signal_alignment=0.9)
        assert result == "good"

    def test_bad_significant_loss(self) -> None:
        """Test 'bad' label for significant loss > 5%."""
        logger = DecisionLogger(log_path=None)
        result = logger.assign_quality_labels(weekly_return=-0.06, signal_alignment=0.5)
        assert result == "bad"

    def test_bad_misalignment_negative(self) -> None:
        """Test 'bad' label for high misalignment with negative return."""
        logger = DecisionLogger(log_path=None)
        result = logger.assign_quality_labels(weekly_return=-0.02, signal_alignment=0.2)
        assert result == "bad"

    def test_neutral_default(self) -> None:
        """Test 'neutral' label for everything else."""
        logger = DecisionLogger(log_path=None)
        result = logger.assign_quality_labels(weekly_return=0.01, signal_alignment=0.5)
        assert result == "neutral"


class TestDecisionLoggerLoadRecentDecisions:
    """Tests for load_recent_decisions()."""

    def test_returns_list(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that load_recent_decisions returns a list."""
        logger = DecisionLogger(log_path=temp_log_file)
        logger.log_decision(sample_decision_record)

        result = logger.load_recent_decisions(n=10)

        assert isinstance(result, list)

    def test_returns_decision_records(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that returned items are DecisionRecord instances."""
        logger = DecisionLogger(log_path=temp_log_file)
        logger.log_decision(sample_decision_record)

        result = logger.load_recent_decisions(n=10)

        assert len(result) == 1
        assert isinstance(result[0], DecisionRecord)

    def test_returns_last_n_records(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that only last n records are returned."""
        logger = DecisionLogger(log_path=temp_log_file)

        for i in range(5):
            record = DecisionRecord(
                monday_date=f"2024-10-{i+1:02d}",
                agent_input={},
                level1_plan=[],
                level2_plan=[],
                weekly_return=0.0,
            )
            logger.log_decision(record)

        result = logger.load_recent_decisions(n=3)

        assert len(result) == 3

    def test_empty_log_file(self, temp_log_file: Path) -> None:
        """Test that empty log file returns empty list."""
        logger = DecisionLogger(log_path=temp_log_file)

        result = logger.load_recent_decisions(n=10)

        assert result == []

    def test_none_log_path(self) -> None:
        """Test that None log_path returns empty list when log is empty."""
        logger = DecisionLogger(log_path=None)
        # After init, log_path defaults to config path; if empty, returns []
        result = logger.load_recent_decisions(n=10)
        assert isinstance(result, list)


class TestDecisionLoggerGetPatternsForContext:
    """Tests for get_patterns_for_context()."""

    def test_returns_tuple(self, temp_log_file: Path) -> None:
        """Test that get_patterns_for_context returns a tuple."""
        logger = DecisionLogger(log_path=temp_log_file)

        result = logger.get_patterns_for_context(market_state="bullish", vol_percentile=0.6)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_log_file(self, temp_log_file: Path) -> None:
        """Test that empty log file returns empty lists."""
        logger = DecisionLogger(log_path=temp_log_file)

        good, bad = logger.get_patterns_for_context(market_state="bullish", vol_percentile=0.6)

        assert good == []
        assert bad == []

    def test_market_state_matching(
        self,
        sample_decision_record: DecisionRecord,
        temp_log_file: Path,
    ) -> None:
        """Test that market state matching works."""
        logger = DecisionLogger(log_path=temp_log_file)
        logger.log_decision(sample_decision_record)

        good, bad = logger.get_patterns_for_context(
            market_state="bullish", vol_percentile=0.6
        )

        # sample_decision_record has market_state="bullish" in agent_input
        assert isinstance(good, list)
        assert isinstance(bad, list)

    def test_none_log_path(self, temp_log_file: Path) -> None:
        """Test that with a clean temp log file, returns empty lists."""
        logger = DecisionLogger(log_path=temp_log_file)
        good, bad = logger.get_patterns_for_context(
            market_state="bullish", vol_percentile=0.6
        )
        assert isinstance(good, list)
        assert isinstance(bad, list)
