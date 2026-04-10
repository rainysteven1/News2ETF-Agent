"""PromptManager — dynamic few-shot pattern injection for the trader agent.

Responsibilities:
  1. Load decision_logs.jsonl and find similar past decisions based on current context
  2. Extract good/bad patterns from similar decisions
  3. Inject patterns into the trader.md prompt template

The core idea: rather than static few-shot examples, dynamically召回 (recall)
the most relevant historical decisions for the current market state so the
LLM can learn from past successes/failures in similar conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.decision_logger import DecisionLogger, DecisionRecord
from src.config import load_config


class PromptManager:
    """Dynamically injects good/bad decision patterns into the trader prompt.

    Call `update_prompt(current_context)` before each trader decision to get
    a context-aware prompt with relevant patterns pre-injected.
    """

    def __init__(self, logger: DecisionLogger | None = None):
        """Initialize PromptManager.

        Args:
            logger: DecisionLogger instance. If None, creates one with default path.
        """
        self.logger = logger or DecisionLogger()
        self._good_patterns: list[str] = []
        self._bad_patterns: list[str] = []
        self._last_context: dict[str, Any] = {}

    @staticmethod
    def _safe_nested_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def recall_similar_decisions(
        self,
        current_context: dict[str, Any],
        n: int = 5,
    ) -> list[DecisionRecord]:
        """Find n most similar past decisions based on current market context.

        Matching rules:
          - High vol_percentile → recall high-volatility period decisions
          - Strong sector signal → recall same-sector decisions
          - FORBIDDEN_ZONE triggered → recall similar FORBIDDEN_ZONE recoveries
          - Market state (bullish/bearish/neutral) alignment

        Args:
            current_context: Dict with keys like vol_percentile, sector_signal,
                             market_state, forbidden_zones, etc.
            n: Number of similar decisions to recall

        Returns:
            List of DecisionRecord objects sorted by similarity
        """
        records = self.logger.load_recent_decisions(n=100)
        if not records:
            return []

        vol_pct = current_context.get("vol_percentile", 0.5)
        market_state = current_context.get("market_state", "neutral")
        sector_signals = current_context.get("sector_signals", {})
        forbidden = set(current_context.get("forbidden_zones", {}).keys())

        # Score each record by similarity
        scored: list[tuple[float, DecisionRecord]] = []
        for record in records:
            score = 0.0
            agent_input = record.agent_input if isinstance(record.agent_input, dict) else {}
            market_state_info = self._safe_nested_dict(agent_input.get("market_state"))

            # Market state match (most important)
            rec_state = market_state_info.get("market_state", "neutral")
            if rec_state == market_state:
                score += 3.0

            # Volatility band match (vol_percentile within same 0.2 band)
            rec_vol = market_state_info.get("market_volatility", 0.0)
            if abs(rec_vol - vol_pct) < 0.2:
                score += 2.0

            # Sector signal match
            rec_signals = agent_input.get("sector_signals", {})
            for sector, signal in sector_signals.items():
                if sector in rec_signals:
                    if rec_signals[sector] == signal:
                        score += 1.0
                    elif abs(rec_signals[sector] - signal) < 0.2:
                        score += 0.5

            # FORBIDDEN_ZONE pattern match
            rec_forbidden = set(
                r.get("meta_sector", "") for r in record.guardrail_events
            )
            if forbidden and rec_forbidden:
                overlap = len(forbidden & rec_forbidden)
                score += overlap * 0.5

            # Prefer decisions with quality labels
            if record.quality_label == "good":
                score += 1.0
            elif record.quality_label == "bad":
                score -= 0.5

            scored.append((score, record))

        # Sort by score descending and return top n
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:n]]

    def load_patterns_by_context(
        self,
        current_context: dict[str, Any],
        n: int = 5,
    ) -> tuple[list[str], list[str]]:
        """Load good/bad patterns relevant to current context.

        Args:
            current_context: Current market context dict
            n: Number of similar decisions to consider per category

        Returns:
            Tuple of (good_pattern_strings, bad_pattern_strings)
        """
        similar = self.recall_similar_decisions(current_context, n=n)
        if not similar:
            return self._load_default_patterns()

        good_patterns: list[str] = []
        bad_patterns: list[str] = []

        for record in similar:
            level1 = record.level1_plan or []
            weekly_return = record.weekly_return

            for plan in level1:
                meta_sector = plan.get("meta_sector", "unknown")
                action = plan.get("action", "hold")
                weight = plan.get("weight", 0.0)
                reason = plan.get("reason", "")[:80]

                pattern = f"[{record.monday_date}] {meta_sector}:{action} w={weight:.0%} | {reason}"

                # Quality-based filtering
                if record.quality_label == "good":
                    if pattern not in good_patterns:
                        good_patterns.append(pattern)
                elif record.quality_label == "bad":
                    if pattern not in bad_patterns:
                        bad_patterns.append(pattern)
                else:
                    # Neutral: include as good if weekly_return > 0, bad if < 0
                    if weekly_return > 0.02 and pattern not in good_patterns:
                        good_patterns.append(pattern)
                    elif weekly_return < -0.03 and pattern not in bad_patterns:
                        bad_patterns.append(pattern)

        # Also check get_patterns_for_context for additional matches
        market_state = current_context.get("market_state", "neutral")
        vol_pct = current_context.get("vol_percentile", 0.5)
        ctx_good, ctx_bad = self.logger.get_patterns_for_context(market_state, vol_pct)

        for p in ctx_good:
            if p not in good_patterns:
                good_patterns.append(p)
        for p in ctx_bad:
            if p not in bad_patterns:
                bad_patterns.append(p)

        return good_patterns[:10], bad_patterns[:10]

    def _load_default_patterns(self) -> tuple[list[str], list[str]]:
        """Return minimal default patterns when no history is available."""
        good = [
            "高情绪+正动量 → 买入科技成长",
            "低波动+趋势确认 → 持有高端制造",
            "情感-价格背离 → 买入资源材料",
        ]
        bad = [
            "高波动+高Beta → 避免追高",
            "负动量+空头信号 → 不应买入",
            "FORBIDDEN_ZONE触发 → 不得买入该板块",
        ]
        return good, bad

    def inject_patterns(
        self,
        good: list[str],
        bad: list[str],
    ) -> str:
        """Inject good/bad patterns into the trader.md template.

        Args:
            good: List of good pattern strings
            bad: List of bad pattern strings

        Returns:
            Formatted patterns string for template injection
        """
        good_block = "\n".join(f"  - {p}" for p in good) if good else "  (无成功案例参考)"
        bad_block = "\n".join(f"  - {p}" for p in bad) if bad else "  (无失败案例参考)"

        return f"## Good Patterns (成功案例)\n{good_block}\n\n## Bad Patterns (失败案例 - 应避免类似决策)\n{bad_block}"

    def update_prompt(self, current_context: dict[str, Any]) -> tuple[list[str], list[str], str]:
        """Update the trader prompt with context-aware patterns.

        This is the main entry point. Call it before each trader decision
        to get a context-aware prompt.

        Args:
            current_context: Dict with keys:
                - vol_percentile: float (0-1)
                - market_state: str ("bullish" | "bearish" | "neutral")
                - sector_signals: dict[str, float] (sector → signal strength)
                - forbidden_zones: dict[str, str] (sector → reason)
                - date: str (decision date)

        Returns:
            Tuple of (good_patterns, bad_patterns, reasoning_summary)
        """
        self._last_context = current_context

        good, bad = self.load_patterns_by_context(current_context, n=5)
        self._good_patterns = good
        self._bad_patterns = bad

        # Build a reasoning summary string
        similar = self.recall_similar_decisions(current_context, n=3)
        if similar:
            summaries = []
            for rec in similar:
                label = rec.quality_label or "neutral"
                summaries.append(
                    f"[{rec.monday_date}] {label}: "
                    f"return={rec.weekly_return:.2%} "
                    f"sectors={[p.get('meta_sector', '') for p in (rec.level1_plan or [])[:2]]}"
                )
            reasoning_summary = "相似历史决策: " + "; ".join(summaries)
        else:
            reasoning_summary = "无相似历史决策，使用默认策略。"

        good_str = self.inject_patterns(good, bad)
        return good, bad, reasoning_summary

    def get_forbidden_sector_checklist(self, forbidden_zones: dict[str, str]) -> str:
        """Generate a FORBIDDEN_ZONE checklist string for the trader prompt.

        Args:
            forbidden_zones: Dict mapping sector name to forbidden reason

        Returns:
            Formatted checklist string
        """
        if not forbidden_zones:
            return "  (无禁闭板块)"

        lines = []
        for sector, reason in forbidden_zones.items():
            lines.append(f"  - [{sector}]: {reason} → 禁闭中，禁止买入")
        return "\n".join(lines)

    def get_context_summary(self, current_context: dict[str, Any]) -> str:
        """Generate a brief summary of current context for logging.

        Args:
            current_context: Current context dict

        Returns:
            Human-readable summary string
        """
        parts = []
        vol = current_context.get("vol_percentile", 0.5)
        if vol > 0.8:
            parts.append("高波动市场")
        elif vol < 0.3:
            parts.append("低波动市场")

        state = current_context.get("market_state", "neutral")
        parts.append(f"市场状态: {state}")

        signals = current_context.get("sector_signals", {})
        top_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        if top_signals:
            signal_str = ", ".join(f"{s}:{v:.2f}" for s, v in top_signals)
            parts.append(f"强信号: {signal_str}")

        forbidden = current_context.get("forbidden_zones", {})
        if forbidden:
            parts.append(f"禁闭区: {list(forbidden.keys())}")

        return " | ".join(parts) if parts else "一般市场"
