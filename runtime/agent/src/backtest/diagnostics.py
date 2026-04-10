"""Backtest result diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from src.backtest.metrics import calculate_metrics
from src.backtest.portfolio import Portfolio
from src.config import AgentRootConfig


@dataclass
class DiagnosticIssue:
    week_start: str
    severity: str
    code: str
    detail: str


def _parse_json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return fallback
    return value if value is not None else fallback


def diagnose_backtest(
    *,
    config: AgentRootConfig,
    backtest_path: Path,
    run_id: str | None = None,
    start_week: str | None = None,
    end_week: str | None = None,
) -> tuple[dict[str, Any], list[DiagnosticIssue], pl.DataFrame]:
    """Diagnose a backtest parquet and return summary plus issues."""
    if not backtest_path.exists():
        raise FileNotFoundError(f"Backtest parquet not found: {backtest_path}")

    df = pl.read_parquet(backtest_path).sort("week_start")
    if run_id and "run_id" in df.columns:
        df = df.filter(pl.col("run_id") == run_id)
    if start_week:
        df = df.filter(pl.col("week_start") >= start_week)
    if end_week:
        df = df.filter(pl.col("week_start") <= end_week)
    if len(df) == 0:
        raise ValueError("No rows matched the requested backtest slice")

    metrics = calculate_metrics(df, risk_free_rate=config.backtest.risk_free_rate)
    etf_prices = pl.read_parquet(config.data.etf_prices) if config.data.etf_prices.exists() else None

    issues: list[DiagnosticIssue] = []
    guardrail = float(config.backtest.max_abs_weekly_return_guardrail)

    for row in df.to_dicts():
        week_start = str(row.get("week_start", "unknown"))
        holdings = _parse_json_value(row.get("holdings"), {})
        selected_etfs = _parse_json_value(row.get("selected_etfs"), {})
        meta_sector_contributions = _parse_json_value(row.get("meta_sector_contributions"), {})
        agent_decisions = _parse_json_value(row.get("agent_decisions"), [])

        invested_weight = float(row.get("invested_weight", 0.0) or 0.0)
        cash_weight = float(row.get("cash_weight", 0.0) or 0.0)
        weekly_return = float(row.get("weekly_return", 0.0) or 0.0)
        nav = float(row.get("nav", 0.0) or 0.0)
        initial_capital = float(row.get("initial_capital", nav) or nav)
        cumulative_return = float(row.get("cumulative_return", 0.0) or 0.0)

        if nav <= 0:
            issues.append(DiagnosticIssue(week_start, "error", "nav_non_positive", f"nav={nav:.2f}"))

        if abs(weekly_return) > guardrail:
            issues.append(
                DiagnosticIssue(
                    week_start,
                    "error",
                    "weekly_return_guardrail",
                    f"weekly_return={weekly_return:.2%} threshold={guardrail:.2%}",
                )
            )

        holdings_sum = sum(float(v) for v in dict(holdings).values())
        if abs(holdings_sum - invested_weight) > 1e-6:
            issues.append(
                DiagnosticIssue(
                    week_start,
                    "warn",
                    "invested_weight_mismatch",
                    f"holdings_sum={holdings_sum:.6f} invested_weight={invested_weight:.6f}",
                )
            )
        if abs((invested_weight + cash_weight) - 1.0) > 1e-6:
            issues.append(
                DiagnosticIssue(
                    week_start,
                    "warn",
                    "cash_weight_mismatch",
                    f"invested+cash={invested_weight + cash_weight:.6f}",
                )
            )

        contribution_sum = sum(float(v) for v in dict(meta_sector_contributions).values())
        if abs(contribution_sum - weekly_return) > 1e-6:
            issues.append(
                DiagnosticIssue(
                    week_start,
                    "warn",
                    "contribution_mismatch",
                    f"contribution_sum={contribution_sum:.6f} weekly_return={weekly_return:.6f}",
                )
            )

        expected_cum = (nav - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        if abs(expected_cum - cumulative_return) > 1e-6:
            issues.append(
                DiagnosticIssue(
                    week_start,
                    "warn",
                    "cumulative_return_mismatch",
                    f"expected={expected_cum:.6f} stored={cumulative_return:.6f}",
                )
            )

        if holdings and not agent_decisions:
            issues.append(
                DiagnosticIssue(week_start, "info", "missing_decision_payload", "active holdings but agent_decisions empty")
            )

        portfolio = Portfolio(initial_capital=initial_capital)
        portfolio.holdings = {str(k): float(v) for k, v in dict(holdings).items()}
        portfolio.selected_etfs = {str(k): str(v) for k, v in dict(selected_etfs).items()}
        total_positions, covered_positions, missing_selected = portfolio.selected_etf_coverage()
        if missing_selected:
            issues.append(
                DiagnosticIssue(
                    week_start,
                    "error",
                    "missing_selected_etf",
                    f"coverage={covered_positions}/{total_positions} missing={','.join(missing_selected)}",
                )
            )

        if etf_prices is not None and portfolio.holdings:
            availability = portfolio.inspect_price_availability(etf_prices, week_start)
            missing_price_sectors = availability["missing_price_sectors"]
            if missing_price_sectors:
                issues.append(
                    DiagnosticIssue(
                        week_start,
                        "error",
                        "missing_price_data",
                        f"missing={','.join(missing_price_sectors)}",
                    )
                )

    summary = {
        "rows": len(df),
        "run_ids": ",".join(sorted(set(df["run_id"].cast(pl.Utf8).to_list()))) if "run_id" in df.columns else "-",
        "start_week": str(df["week_start"].min()),
        "end_week": str(df["week_start"].max()),
        "final_nav": metrics.final_nav,
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "issue_count": len(issues),
        "error_count": sum(1 for i in issues if i.severity == "error"),
        "warn_count": sum(1 for i in issues if i.severity == "warn"),
    }
    return summary, issues, df
