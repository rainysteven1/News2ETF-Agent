"""Decision Logger - tracks agent decisions, TCN errors, and patterns.

This module provides:
  - TCNPredictionError: Tracks TCN prediction divergence from actual returns
  - GuardrailEvent: Records guardrail trigger events
  - DecisionRecord: Complete record of agent decision and outcome
  - DecisionLogger: Logs decisions to JSONL and extracts good/bad patterns
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config


@dataclass
class TCNPredictionError:
    """Tracks TCN prediction divergence from actual returns."""

    meta_sector: str
    tcn_predicted: float
    actual_return: float
    divergence: float  # tcn_predicted - actual_return
    root_cause_guess: str = ""


@dataclass
class GuardrailEvent:
    """Records a guardrail trigger event."""

    date: str
    meta_sector: str
    trigger_type: str  # DAILY_LOSS_5PCT, BREAKING_NEWS, etc.
    etf_code: str = ""
    pnl_impact: float = 0.0
    reason: str = ""


@dataclass
class DecisionRecord:
    """Complete record of agent decision and outcome."""

    monday_date: str  # Week starting Monday
    agent_input: dict[str, Any]  # Full input context
    level1_plan: list[dict[str, Any]]  # Meta sector plans
    level2_plan: list[dict[str, Any]]  # ETF selections
    weekly_return: float  # Actual return for this week
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)
    tcn_prediction_errors: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str = ""
    quality_label: str = ""  # "good", "neutral", "bad"


class DecisionLogger:
    """Logs agent decisions to JSONL and extracts good/bad patterns."""

    def __init__(self, log_path: Path | None = None):
        """Initialize the decision logger.

        Args:
            log_path: Path to the JSONL log file
        """
        if log_path is None:
            config = load_config()
            log_path = config.data.output_logs

        self.log_path = Path(log_path) if log_path else None
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure the log file exists."""
        if self.log_path and not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.touch()

    def log_decision(self, record: DecisionRecord) -> None:
        """Log a decision record to the JSONL file.

        Args:
            record: DecisionRecord to log
        """
        if self.log_path is None:
            return

        record_dict = asdict(record)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

    def compute_tcn_error(
        self,
        tcn_sequence: dict[str, list[float]],
        actual_returns: dict[str, float],
    ) -> list[TCNPredictionError]:
        """Compute TCN prediction errors.

        Args:
            tcn_sequence: TCN predicted momentum per meta sector
            actual_returns: Actual returns per meta sector

        Returns:
            List of TCNPredictionError records
        """
        errors = []

        for meta_sector, predicted_list in tcn_sequence.items():
            if not predicted_list:
                continue

            # Use the last predicted value
            tcn_predicted = predicted_list[-1]
            actual = actual_returns.get(meta_sector, 0.0)
            divergence = tcn_predicted - actual

            # Root cause guess based on divergence pattern
            if abs(divergence) < 0.02:
                root_cause = "low_error"
            elif divergence > 0 and actual < 0:
                root_cause = "TCN_overconfident_bearish"
            elif divergence < 0 and actual > 0:
                root_cause = "TCN_underconfident_bullish"
            elif divergence > 0:
                root_cause = "TCN_overestimated"
            else:
                root_cause = "TCN_underestimated"

            errors.append(
                TCNPredictionError(
                    meta_sector=meta_sector,
                    tcn_predicted=float(tcn_predicted),
                    actual_return=float(actual),
                    divergence=float(divergence),
                    root_cause_guess=root_cause,
                )
            )

        return errors

    def extract_good_bad_patterns(self) -> tuple[list[str], list[str]]:
        """Extract good and bad decision patterns from logged decisions.

        Returns:
            Tuple of (good_patterns, bad_patterns) where each is a list
            of pattern description strings
        """
        if self.log_path is None or not self.log_path.exists():
            return [], []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        good_patterns: list[str] = []
        bad_patterns: list[str] = []

        for record in records:
            label = record.get("quality_label", "")
            level1_plan = record.get("level1_plan", [])

            if label == "good":
                # Extract successful patterns
                for plan in level1_plan:
                    action = plan.get("action", "")
                    meta_sector = plan.get("meta_sector", "")
                    reason = plan.get("reason", "")[:100]
                    pattern = f"{meta_sector}:{action} - {reason}"
                    if pattern not in good_patterns:
                        good_patterns.append(pattern)

            elif label == "bad":
                # Extract failure patterns
                for plan in level1_plan:
                    action = plan.get("action", "")
                    meta_sector = plan.get("meta_sector", "")
                    reason = plan.get("reason", "")[:100]
                    pattern = f"{meta_sector}:{action} - {reason}"
                    if pattern not in bad_patterns:
                        bad_patterns.append(pattern)

        # Limit to top 20 patterns each
        return good_patterns[:20], bad_patterns[:20]

    def assign_quality_labels(
        self,
        weekly_return: float,
        signal_alignment: float,
    ) -> str:
        """Assign quality label to a decision based on outcome.

        Args:
            weekly_return: Actual weekly return of the portfolio
            signal_alignment: How well the decision aligned with signals (0-1)

        Returns:
            Quality label: "good", "neutral", or "bad"
        """
        # Good: positive return OR high alignment with correct direction
        if weekly_return > 0.02 or (signal_alignment > 0.8 and weekly_return > 0):
            return "good"

        # Bad: significant loss or high misalignment
        if weekly_return < -0.05 or (signal_alignment < 0.3 and weekly_return < 0):
            return "bad"

        # Neutral: everything else
        return "neutral"

    def load_recent_decisions(self, n: int = 10) -> list[DecisionRecord]:
        """Load the n most recent decision records.

        Args:
            n: Number of recent decisions to load

        Returns:
            List of DecisionRecord objects
        """
        if self.log_path is None or not self.log_path.exists():
            return []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Return last n records
        return [self._dict_to_record(r) for r in records[-n:]]

    def _dict_to_record(self, d: dict[str, Any]) -> DecisionRecord:
        """Convert dictionary to DecisionRecord."""
        return DecisionRecord(
            monday_date=d.get("monday_date", ""),
            agent_input=d.get("agent_input", {}),
            level1_plan=d.get("level1_plan", []),
            level2_plan=d.get("level2_plan", []),
            weekly_return=d.get("weekly_return", 0.0),
            guardrail_events=d.get("guardrail_events", []),
            tcn_prediction_errors=d.get("tcn_prediction_errors", []),
            reasoning_summary=d.get("reasoning_summary", ""),
            quality_label=d.get("quality_label", ""),
        )

    def get_patterns_for_context(
        self,
        market_state: str,
        vol_percentile: float,
    ) -> tuple[list[str], list[str]]:
        """Get patterns relevant to current market context.

        Args:
            market_state: "bullish", "bearish", or "neutral"
            vol_percentile: Volume percentile (0-1)

        Returns:
            Tuple of (relevant_good_patterns, relevant_bad_patterns)
        """
        if self.log_path is None or not self.log_path.exists():
            return [], []

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        good_patterns: list[str] = []
        bad_patterns: list[str] = []

        for record in records:
            # Check if this record matches the market context
            agent_input = record.get("agent_input", {})
            if not isinstance(agent_input, dict):
                agent_input = {}
            market_state_info = agent_input.get("market_state", {})
            if not isinstance(market_state_info, dict):
                market_state_info = {}
            record_state = market_state_info.get("market_state", "neutral")

            if record_state == market_state:
                label = record.get("quality_label", "")
                level1_plan = record.get("level1_plan", [])

                if label == "good":
                    for plan in level1_plan:
                        pattern = f"{plan.get('meta_sector', '')}:{plan.get('action', '')}"
                        if pattern not in good_patterns:
                            good_patterns.append(pattern)
                elif label == "bad":
                    for plan in level1_plan:
                        pattern = f"{plan.get('meta_sector', '')}:{plan.get('action', '')}"
                        if pattern not in bad_patterns:
                            bad_patterns.append(pattern)

        return good_patterns[:10], bad_patterns[:10]
