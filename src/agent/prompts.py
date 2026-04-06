"""System prompts loaded from external config/prompts/ .md files.

All prompts are loaded via functions so callers must invoke them.
Files:
    researcher.md → researcher_prompt()  [takes env_context, date]
    trader.md → trader_prompt()      [takes research_summary, last_week_pnl, holdings, max_weight, max_total, date]
    tool_descriptions.md → tool_descriptions()
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def _load(name: str) -> str:
    return (_ROOT / f"{name}.md").read_text(encoding="utf-8")


# ── Loaded prompts ──────────────────────────────────────────────────────────────


def tool_descriptions() -> str:
    return _load("tool_descriptions")


def researcher_prompt(date: str, env_context: str) -> str:
    return _load("researcher").format(date=date, env_context=env_context)


def trader_prompt(
    date: str,
    research_summary: str,
    last_week_pnl: float,
    holdings: str,
    max_weight: float,
    max_total: float,
    tcn_sequence: str = "",
    news_summary: str = "",
    market_state: str = "",
    position_state: str = "",
    sent_p_divergence: str = "",
    forbidden_sectors: str = "",
    good_patterns: str = "",
    bad_patterns: str = "",
) -> str:
    """Build trader prompt with all context features.

    Args:
        date: Decision date
        research_summary: Researcher's findings
        last_week_pnl: Last week's return
        holdings: Current holdings string
        max_weight: Maximum weight per sector
        max_total: Maximum total weight
        tcn_sequence: TCN sequence feature (Feature A)
        news_summary: News summary feature (Feature B)
        market_state: Market state feature (Feature C)
        position_state: Position state feature (Feature D)
        sent_p_divergence: Sentiment vs price divergence (Feature E)
        forbidden_sectors: FORBIDDEN_ZONE sectors
        good_patterns: Good decision patterns
        bad_patterns: Bad decision patterns
    """
    template = _load("trader")
    return template.format(
        date=date,
        research_summary=research_summary,
        last_week_pnl=last_week_pnl,
        holdings=holdings,
        max_weight=max_weight,
        max_total=max_total,
        tcn_sequence=tcn_sequence or "(TCN序列未提供)",
        news_summary=news_summary or "(新闻摘要未提供)",
        market_state=market_state or "(市场状态未提供)",
        position_state=position_state or "(持仓状态未提供)",
        sent_p_divergence=sent_p_divergence or "(背离分析未提供)",
        forbidden_sectors=forbidden_sectors or "(无禁闭板块)",
        good_patterns=good_patterns or "(无成功案例参考)",
        bad_patterns=bad_patterns or "(无失败案例参考)",
    )
