from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from src.backtest.diagnostics import diagnose_backtest


def test_diagnose_backtest_detects_missing_selected_etf(tmp_path: Path) -> None:
    backtest_path = tmp_path / "backtest.parquet"
    pl.DataFrame(
        {
            "run_id": ["bt_test"],
            "week_start": ["2024-01-08"],
            "initial_capital": [1_000_000.0],
            "nav": [1_010_000.0],
            "weekly_return": [0.01],
            "invested_weight": [0.2],
            "cash_weight": [0.8],
            "holdings": ['{"科技成长": 0.2}'],
            "selected_etfs": ['{}'],
            "meta_sector_contributions": ['{"科技成长": 0.01}'],
            "meta_sector_returns": ['{"科技成长": 0.05}'],
            "industry_contributions": ['{"科技成长": 0.01}'],
            "status": ["PROFIT"],
            "cumulative_return": [0.01],
            "observations": ['{}'],
            "agent_decisions": ['[]'],
        }
    ).write_parquet(backtest_path)

    etf_prices = tmp_path / "prices.parquet"
    pl.DataFrame({"Code": ["512480", "512480"], "trade_dt": [20240105, 20240112], "close": [1.0, 1.1]}).write_parquet(
        etf_prices
    )

    cfg = SimpleNamespace(
        data=SimpleNamespace(etf_prices=etf_prices),
        backtest=SimpleNamespace(risk_free_rate=0.03, max_abs_weekly_return_guardrail=0.30),
    )

    summary, issues, _ = diagnose_backtest(config=cfg, backtest_path=backtest_path)

    assert summary["issue_count"] >= 1
    assert any(issue.code == "missing_selected_etf" for issue in issues)


def test_diagnose_backtest_detects_contribution_mismatch(tmp_path: Path) -> None:
    backtest_path = tmp_path / "backtest.parquet"
    pl.DataFrame(
        {
            "run_id": ["bt_test"],
            "week_start": ["2024-01-08"],
            "initial_capital": [1_000_000.0],
            "nav": [1_010_000.0],
            "weekly_return": [0.01],
            "invested_weight": [0.2],
            "cash_weight": [0.8],
            "holdings": ['{"科技成长": 0.2}'],
            "selected_etfs": ['{"科技成长": "512480"}'],
            "meta_sector_contributions": ['{"科技成长": 0.02}'],
            "meta_sector_returns": ['{"科技成长": 0.05}'],
            "industry_contributions": ['{"科技成长": 0.02}'],
            "status": ["PROFIT"],
            "cumulative_return": [0.01],
            "observations": ['{}'],
            "agent_decisions": ['[]'],
        }
    ).write_parquet(backtest_path)

    etf_prices = tmp_path / "prices.parquet"
    pl.DataFrame({"Code": ["512480", "512480"], "trade_dt": [20240105, 20240112], "close": [1.0, 1.1]}).write_parquet(
        etf_prices
    )

    cfg = SimpleNamespace(
        data=SimpleNamespace(etf_prices=etf_prices),
        backtest=SimpleNamespace(risk_free_rate=0.03, max_abs_weekly_return_guardrail=0.30),
    )

    _, issues, _ = diagnose_backtest(config=cfg, backtest_path=backtest_path)

    assert any(issue.code == "contribution_mismatch" for issue in issues)
