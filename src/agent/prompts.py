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
) -> str:
    return _load("trader").format(
        date=date,
        research_summary=research_summary,
        last_week_pnl=last_week_pnl,
        holdings=holdings,
        max_weight=max_weight,
        max_total=max_total,
    )
