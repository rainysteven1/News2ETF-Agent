from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from src.backtest.engine import WalkForwardEngine


def test_run_skips_execution_when_week_has_no_trading_rows(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine._etf_prices = pl.DataFrame(
        {
            "Code": ["512480", "512480"],
            "trade_dt": [20240208, 20240219],
            "close": [1.0, 1.1],
        }
    )
    engine._meta_sector_etf_code_map = {}
    engine._etf_universe = None
    engine.mapper = None
    engine.config = SimpleNamespace(
        data=SimpleNamespace(
            output_backtest=tmp_path / "backtest_results.parquet",
            output_backtest_metrics=tmp_path / "backtest_metrics.parquet",
        ),
        backtest=SimpleNamespace(
            initial_capital=1_000_000.0,
            transaction_fee=0.0,
            slippage=0.0,
            risk_free_rate=0.03,
            max_abs_weekly_return_guardrail=0.30,
        ),
    )

    class StubWorkflow:
        def invoke(self, state: dict) -> dict:
            return {
                "decisions": [
                    SimpleNamespace(
                        model_dump=lambda: {
                            "industry": "meta_allocation",
                            "action": "hold",
                            "weight": 0.0,
                            "reason": "",
                            "level1_plan": [{"meta_sector": "科技成长", "action": "buy", "weight": 1.0, "reason": "test"}],
                            "level2_plan": [{"meta_sector": "科技成长", "selected_etf": "512480 ETF"}],
                        }
                    )
                ],
                "observations": {"researcher_summary": f"week={state['date']}"},
                "last_error": "",
            }

    results = engine.run(
        "2024-02-12",
        "2024-02-12",
        run_id="bt_holiday",
        agent_workflow=StubWorkflow(),
    )

    assert results.shape[0] == 1
    row = results.to_dicts()[0]
    assert row["week_start"] == "2024-02-12"
    assert row["weekly_return"] == 0.0
    assert row["market_closed_week"] is True
    assert row["trading_day_count"] == 0
    assert row["first_trading_day"] == ""
    assert row["last_trading_day"] == ""
    assert row["cash_weight"] == 1.0
    assert row["holdings"] == "{}"
    assert "科技成长" in row["agent_decisions"]
