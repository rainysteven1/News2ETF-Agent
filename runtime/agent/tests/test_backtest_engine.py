from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest

from src.backtest.engine import WalkForwardEngine, _format_weekly_decisions, _result_to_wandb_row
from src.backtest.metrics import calculate_metrics


def test_format_weekly_decisions_for_meta_plan() -> None:
    decisions = [
        {
            "level1_plan": [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.2, "reason": "景气上行"},
                {"meta_sector": "高端制造", "action": "hold", "weight": 0.1, "reason": "趋势稳定"},
            ],
            "level2_plan": [
                {"meta_sector": "科技成长", "selected_etf": "512480 ETF-A"},
                {"meta_sector": "高端制造", "selected_etf": "159987 ETF-B"},
            ],
        }
    ]

    assert (
        _format_weekly_decisions(decisions)
        == "科技成长:buy 20.0% 512480 ETF-A | 高端制造:hold 10.0% 159987 ETF-B"
    )


def test_format_weekly_decisions_for_legacy_plan() -> None:
    decisions = [
        {
            "industry": "新能源",
            "action": "buy",
            "weight": 0.15,
            "selected_etf": "516160 新能源ETF",
            "reason": "盈利预期改善",
        },
        {"industry": "消费", "action": "sell", "weight": 0.0, "selected_etf": "", "reason": "动能转弱"},
    ]

    assert _format_weekly_decisions(decisions) == "新能源:buy 15.0% 516160 新能源ETF | 消费:sell 0.0% -"


def test_format_weekly_decisions_with_reason() -> None:
    decisions = [
        {
            "level1_plan": [
                {"meta_sector": "科技成长", "action": "buy", "weight": 0.2, "reason": "景气上行"},
            ],
            "level2_plan": [
                {"meta_sector": "科技成长", "selected_etf": "512480 ETF-A"},
            ],
        }
    ]

    assert _format_weekly_decisions(decisions, include_reason=True) == (
        "科技成长:buy 20.0% 512480 ETF-A reason=景气上行"
    )


def test_persist_backtest_snapshot_writes_results_and_metrics(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine.config = SimpleNamespace(
        data=SimpleNamespace(
            output_backtest=tmp_path / "backtest_results.parquet",
            output_backtest_metrics=tmp_path / "backtest_metrics.parquet",
        ),
        backtest=SimpleNamespace(risk_free_rate=0.03),
    )

    results = [
        {
            "run_id": "bt_test",
            "week_start": "2024-01-01",
            "initial_capital": 1_000_000.0,
            "nav": 1_000_000.0,
            "weekly_return": 0.0,
            "invested_weight": 0.0,
            "cash_weight": 1.0,
            "holdings": {},
            "selected_etfs": {},
            "meta_sector_contributions": {},
            "meta_sector_returns": {},
            "industry_contributions": {},
            "status": "LOSS",
            "cumulative_return": 0.0,
            "observations": {},
            "agent_decisions": [],
        },
        {
            "run_id": "bt_test",
            "week_start": "2024-01-08",
            "initial_capital": 1_000_000.0,
            "nav": 1_010_000.0,
            "weekly_return": 0.01,
            "invested_weight": 0.2,
            "cash_weight": 0.8,
            "holdings": {"科技成长": 0.2},
            "selected_etfs": {"科技成长": "512480"},
            "meta_sector_contributions": {"科技成长": 0.01},
            "meta_sector_returns": {"科技成长": 0.05},
            "industry_contributions": {"科技成长": 0.01},
            "status": "PROFIT",
            "cumulative_return": 0.01,
            "observations": {"researcher_summary": "ok"},
            "agent_decisions": [],
        },
    ]

    results_df, metrics = engine._persist_backtest_snapshot(
        results,
        run_id="bt_test",
        as_of_week="2024-01-08",
    )

    assert results_df.shape[0] == 2
    assert metrics["run_id"] == "bt_test"
    assert metrics["as_of_week"] == "2024-01-08"
    assert metrics["market_closed_weeks"] == 0

    stored_results = pl.read_parquet(engine._backtest_results_path("bt_test"))
    stored_metrics = pl.read_parquet(engine._backtest_metrics_path("bt_test"))
    assert stored_results.shape[0] == 2
    assert stored_metrics.shape[0] == 1
    assert stored_metrics["final_nav"][0] == 1_010_000
    assert stored_metrics["market_closed_weeks"][0] == 0
    assert stored_results["holdings"][0] == "{}"


def test_persist_backtest_snapshot_appends_metric_history_per_week(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine.config = SimpleNamespace(
        data=SimpleNamespace(
            output_backtest=tmp_path / "unused_backtest_results.parquet",
            output_backtest_metrics=tmp_path / "unused_backtest_metrics.parquet",
        ),
        backtest=SimpleNamespace(risk_free_rate=0.03),
    )

    first_results = [
        {
            "run_id": "bt_test",
            "week_start": "2024-01-01",
            "initial_capital": 1_000_000.0,
            "nav": 1_000_000.0,
            "weekly_return": 0.0,
            "invested_weight": 0.0,
            "cash_weight": 1.0,
            "holdings": {},
            "selected_etfs": {},
            "meta_sector_contributions": {},
            "meta_sector_returns": {},
            "industry_contributions": {},
            "status": "LOSS",
            "cumulative_return": 0.0,
            "observations": {},
            "agent_decisions": [],
        }
    ]
    second_results = [
        *first_results,
        {
            "run_id": "bt_test",
            "week_start": "2024-01-08",
            "initial_capital": 1_000_000.0,
            "nav": 1_020_000.0,
            "weekly_return": 0.02,
            "invested_weight": 0.2,
            "cash_weight": 0.8,
            "holdings": {"科技成长": 0.2},
            "selected_etfs": {"科技成长": "512480"},
            "meta_sector_contributions": {"科技成长": 0.02},
            "meta_sector_returns": {"科技成长": 0.10},
            "industry_contributions": {"科技成长": 0.02},
            "status": "PROFIT",
            "cumulative_return": 0.02,
            "observations": {},
            "agent_decisions": [],
        },
    ]

    engine._persist_backtest_snapshot(first_results, run_id="bt_test", as_of_week="2024-01-01")
    engine._persist_backtest_snapshot(second_results, run_id="bt_test", as_of_week="2024-01-08")
    engine._persist_backtest_snapshot(second_results, run_id="bt_test", as_of_week="2024-01-08")

    stored_metrics = pl.read_parquet(engine._backtest_metrics_path("bt_test"))
    stored_results = pl.read_parquet(engine._backtest_results_path("bt_test"))

    assert stored_results.shape[0] == 2
    assert stored_metrics.shape[0] == 2
    assert stored_metrics["as_of_week"].to_list() == ["2024-01-01", "2024-01-08"]


def test_checkpoint_round_trip_preserves_engine_state(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"

    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio(initial_capital=1_000_000.0, transaction_fee=0.0003, slippage=0.0005)
    portfolio.total_value = 1_012_345.0
    portfolio.holdings = {"科技成长": 0.2}
    portfolio.selected_etfs = {"科技成长": "512480"}

    checkpoint_path = engine._save_checkpoint(
        run_id="bt_test",
        completed_week="2024-01-08",
        results=[{"week_start": "2024-01-01", "nav": 1_000_000.0}],
        portfolio=portfolio,
        last_week_return=0.01,
        last_week_holdings={"科技成长": 0.2},
        last_week_returns={"科技成长": 0.05},
        prev_observations={"summary": "ok"},
        prev_agent_decisions=[{"industry": "meta_allocation"}],
    )

    assert checkpoint_path.exists()
    loaded = engine._load_checkpoint(run_id="bt_test", completed_week="2024-01-08")
    assert loaded["completed_week"] == "2024-01-08"
    assert loaded["portfolio"]["total_value"] == 1_012_345.0
    assert loaded["memory"]["last_week_return"] == 0.01
    assert loaded["results"][0]["week_start"] == "2024-01-01"

    run_meta = json.loads((engine._checkpoint_run_dir("bt_test") / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["run_id"] == "bt_test"
    assert run_meta["latest_completed_week"] == "2024-01-08"
    assert run_meta["latest_checkpoint_path"].endswith("2024-01-08.json")
    assert run_meta["latest_total_value"] == 1_012_345.0
    assert run_meta["latest_weekly_return"] == 0.01
    assert run_meta["latest_cash_weight"] == 0.8
    assert isinstance(run_meta["created_at"], str)
    assert isinstance(run_meta["updated_at"], str)


def test_load_latest_checkpoint_uses_latest_json(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"

    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio()
    engine._save_checkpoint(
        run_id="bt_test",
        completed_week="2024-01-15",
        results=[{"week_start": "2024-01-15", "nav": 1_000_000.0}],
        portfolio=portfolio,
        last_week_return=0.0,
        last_week_holdings={},
        last_week_returns={},
        prev_observations={},
        prev_agent_decisions=[],
    )

    latest = engine._load_latest_checkpoint(run_id="bt_test")
    assert latest["completed_week"] == "2024-01-15"


def test_repair_checkpoint_selected_etfs_updates_checkpoint_and_latest(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine._etf_prices = pl.DataFrame(
        {
            "Code": ["512480.SH", "512480.SH"],
            "trade_dt": [20240105, 20240112],
            "close": [1.0, 1.1],
        }
    )
    engine._meta_sector_etf_code_map = {"科技成长": ["512480.SH"]}
    engine._etf_universe = None
    engine.mapper = None

    payload = {
        "run_id": "bt_test",
        "completed_week": "2024-01-08",
        "portfolio": {
            "initial_capital": 1_000_000.0,
            "total_value": 1_000_000.0,
            "holdings": {"科技成长": 1.0},
            "selected_etfs": {},
            "transaction_fee": 0.0,
            "slippage": 0.0,
        },
        "memory": {
            "last_week_return": 0.0,
            "last_week_holdings": {"科技成长": 1.0},
            "last_week_returns": {},
            "prev_observations": {},
            "prev_agent_decisions": [],
        },
        "results": [],
    }
    engine._write_checkpoint_payload(
        run_id="bt_test",
        completed_week="2024-01-08",
        payload=payload,
        write_latest=True,
    )

    repaired = engine._repair_checkpoint_selected_etfs(
        run_id="bt_test",
        completed_week="2024-01-08",
        payload=payload,
        target_week="2024-01-08",
        write_latest=True,
    )

    assert repaired == ["科技成长->512480.SH"]
    checkpoint_payload = engine._load_checkpoint(run_id="bt_test", completed_week="2024-01-08")
    latest_payload = engine._load_latest_checkpoint(run_id="bt_test")
    assert checkpoint_payload["portfolio"]["selected_etfs"] == {"科技成长": "512480.SH"}
    assert latest_payload["portfolio"]["selected_etfs"] == {"科技成长": "512480.SH"}


def test_update_run_meta_preserves_created_at(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"

    first = engine._update_run_meta("bt_test", latest_completed_week="2024-01-08")
    payload1 = json.loads(first.read_text(encoding="utf-8"))
    second = engine._update_run_meta("bt_test", latest_completed_week="2024-01-15")
    payload2 = json.loads(second.read_text(encoding="utf-8"))

    assert payload1["created_at"] == payload2["created_at"]
    assert payload2["latest_completed_week"] == "2024-01-15"


def test_result_to_wandb_row_restores_trace_fields() -> None:
    row = _result_to_wandb_row(
        {
            "week_start": "2024-01-08",
            "weekly_return": 0.02,
            "nav": 1_020_000.0,
            "total_value": 1_020_000.0,
            "market_closed_week": False,
            "trading_day_count": 5,
            "first_trading_day": "2024-01-08",
            "last_trading_day": "2024-01-12",
            "invested_weight": 0.2,
            "cash_weight": 0.8,
            "last_error": "none",
            "observations": {"researcher_summary": "summary"},
            "agent_decisions": [
                {
                    "level1_plan": [{"meta_sector": "科技成长", "action": "buy", "weight": 0.2, "reason": "ok"}],
                    "level2_plan": [{"meta_sector": "科技成长", "selected_etf": "512480 ETF"}],
                }
            ],
        }
    )

    assert row["week_start"] == "2024-01-08"
    assert "nav" not in row
    assert row["total_value"] == 1_020_000.0
    assert row["market_closed_week"] is False
    assert row["trading_day_count"] == 5
    assert row["first_trading_day"] == "2024-01-08"
    assert row["last_trading_day"] == "2024-01-12"
    assert row["decision_count"] == 1
    assert "科技成长:buy 20.0% 512480 ETF" in row["decision_text"]
    assert row["researcher_summary"] == "summary"


def test_apply_decisions_requires_selected_etf_for_buy_sector() -> None:
    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio()

    try:
        portfolio.apply_decisions(
            [
                {
                    "level1_plan": [{"meta_sector": "科技成长", "action": "buy", "weight": 0.2}],
                    "level2_plan": [],
                }
            ]
        )
        raise AssertionError("Expected ValueError for missing selected_etf")
    except ValueError as exc:
        assert "Missing selected_etf" in str(exc)


def test_compute_weekly_return_raises_on_missing_price_data() -> None:
    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio()
    portfolio.holdings = {"科技成长": 0.2}
    portfolio.selected_etfs = {"科技成长": "512480"}
    prices = pl.DataFrame(
        {
            "Code": ["999999", "999999"],
            "trade_dt": [20240105, 20240112],
            "close": [1.0, 1.1],
        }
    )

    try:
        portfolio.compute_weekly_return(prices, "2024-01-08", {})
        raise AssertionError("Expected ValueError for missing ETF price data")
    except ValueError as exc:
        assert "Missing ETF price data" in str(exc)


def test_compute_weekly_return_uses_only_current_week_window() -> None:
    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio()
    portfolio.holdings = {"科技成长": 1.0}
    portfolio.selected_etfs = {"科技成长": "512480"}
    prices = pl.DataFrame(
        {
            "Code": ["512480", "512480", "512480"],
            "trade_dt": [20240105, 20240112, 20240202],
            "close": [1.0, 1.1, 2.0],
        }
    )

    weekly_return, contributions, sector_returns = portfolio.compute_weekly_return(prices, "2024-01-08", {})

    assert round(weekly_return, 6) == 0.1
    assert round(contributions["科技成长"], 6) == 0.1
    assert round(sector_returns["科技成长"], 6) == 0.1


def test_compute_weekly_return_normalizes_bare_selected_etf_code() -> None:
    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio()
    portfolio.holdings = {"科技成长": 1.0}
    portfolio.selected_etfs = {"科技成长": "512480"}
    prices = pl.DataFrame(
        {
            "Code": ["512480.SH", "512480.SH"],
            "trade_dt": [20240105, 20240112],
            "close": [1.0, 1.1],
        }
    )

    weekly_return, contributions, sector_returns = portfolio.compute_weekly_return(prices, "2024-01-08", {})

    assert round(weekly_return, 6) == 0.1
    assert round(contributions["科技成长"], 6) == 0.1
    assert round(sector_returns["科技成长"], 6) == 0.1


def test_compute_weekly_return_recovers_missing_selected_etf_from_sector_map() -> None:
    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio()
    portfolio.holdings = {"科技成长": 1.0}
    portfolio.selected_etfs = {}
    prices = pl.DataFrame(
        {
            "Code": ["512480.SH", "512480.SH"],
            "trade_dt": [20240105, 20240112],
            "close": [1.0, 1.1],
        }
    )

    weekly_return, contributions, sector_returns = portfolio.compute_weekly_return(
        prices,
        "2024-01-08",
        {"科技成长": ["512480.SH"]},
    )

    assert round(weekly_return, 6) == 0.1
    assert round(contributions["科技成长"], 6) == 0.1
    assert round(sector_returns["科技成长"], 6) == 0.1
    assert portfolio.selected_etfs["科技成长"] == "512480.SH"


def test_validate_weekly_accounting_raises_on_guardrail_breach() -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.config = SimpleNamespace(
        backtest=SimpleNamespace(max_abs_weekly_return_guardrail=0.30),
    )

    try:
        engine._validate_weekly_accounting(
            week_start="2024-01-08",
            prev_nav=1_000_000.0,
            weekly_return=0.50,
            sector_contributions={"科技成长": 0.50},
            post_nav=1_500_000.0,
        )
        raise AssertionError("Expected ValueError for guardrail breach")
    except ValueError as exc:
        assert "guardrail breached" in str(exc)


def test_settle_week_marks_holdings_to_market() -> None:
    from src.backtest.portfolio import Portfolio

    portfolio = Portfolio()
    portfolio.holdings = {"科技成长": 0.5, "高端制造": 0.3}
    portfolio.selected_etfs = {"科技成长": "512480", "高端制造": "159987"}

    portfolio.settle_week({"科技成长": 0.10, "高端制造": -0.10}, total_return=0.02)

    assert round(sum(portfolio.holdings.values()) + portfolio.cash_weight, 6) == 1.0
    assert portfolio.holdings["科技成长"] > 0.5
    assert portfolio.holdings["高端制造"] < 0.3


def test_repair_missing_selected_etfs_backfills_held_sector_mapping() -> None:
    from src.backtest.portfolio import Portfolio

    class DummyCandidate:
        def __init__(self, code: str) -> None:
            self.code = code

    class DummyResolver:
        def fallback_candidate_for_meta_sector(self, meta_sector: str, week_start: str, mapper) -> DummyCandidate | None:
            mapping = {
                "消费文娱": DummyCandidate("159928.SZ"),
                "资源材料": DummyCandidate("159870.SZ"),
            }
            return mapping.get(meta_sector)

    portfolio = Portfolio()
    portfolio.holdings = {"消费文娱": 0.1, "资源材料": 0.2}
    portfolio.selected_etfs = {}

    repaired = portfolio.repair_missing_selected_etfs(
        resolver=DummyResolver(),
        week_start="2024-01-22",
        mapper=None,
    )

    assert repaired == ["消费文娱->159928.SZ", "资源材料->159870.SZ"]
    assert portfolio.selected_etfs == {
        "消费文娱": "159928.SZ",
        "资源材料": "159870.SZ",
    }


def test_calculate_metrics_uses_initial_capital_column() -> None:
    df = pl.DataFrame(
        {
            "week_start": ["2024-01-01", "2024-01-08"],
            "initial_capital": [1_000_000.0, 1_000_000.0],
            "nav": [1_020_000.0, 1_050_000.0],
            "weekly_return": [0.02, 0.0294117647],
        }
    )

    metrics = calculate_metrics(df, risk_free_rate=0.03)

    assert round(metrics.total_return, 6) == 0.05
    assert metrics.initial_capital == 1_000_000.0


def test_run_records_same_week_decision_and_realized_return(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine._etf_prices = pl.DataFrame(
        {
            "Code": ["512480", "512480", "512480"],
            "trade_dt": [20240105, 20240112, 20240119],
            "close": [1.0, 1.1, 1.21],
        }
    )
    engine._meta_sector_etf_code_map = {}
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
        "2024-01-08",
        "2024-01-15",
        run_id="bt_test",
        agent_workflow=StubWorkflow(),
    )

    assert results.shape[0] == 2
    rows = results.to_dicts()
    assert rows[0]["week_start"] == "2024-01-08"
    assert "科技成长" in rows[0]["agent_decisions"]
    assert rows[0]["weekly_return"] > 0


def test_calculate_metrics_counts_market_closed_weeks() -> None:
    df = pl.DataFrame(
        {
            "week_start": ["2024-02-05", "2024-02-12", "2024-02-19"],
            "initial_capital": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "nav": [1_000_000.0, 1_000_000.0, 1_020_000.0],
            "weekly_return": [0.0, 0.0, 0.02],
            "market_closed_week": [False, True, False],
        }
    )

    metrics = calculate_metrics(df, risk_free_rate=0.03)

    assert metrics.weeks == 3
    assert metrics.market_closed_weeks == 1


def test_run_logs_weekly_total_value_to_wandb(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine._etf_prices = pl.DataFrame(
        {
            "Code": ["512480", "512480"],
            "trade_dt": [20240105, 20240112],
            "close": [1.0, 1.1],
        }
    )
    engine._meta_sector_etf_code_map = {}
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
                "observations": {},
                "last_error": "",
            }

    class StubWandbHandler:
        def __init__(self) -> None:
            self.metrics_calls: list[tuple[dict[str, float], int | None]] = []
            self.summary_calls: list[dict[str, float]] = []

        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            self.metrics_calls.append((metrics, step))

        def log_summary(self, metrics: dict[str, float]) -> None:
            self.summary_calls.append(metrics)

        def log_table_rows(self, name: str, rows: list[dict[str, object]], step: int | None = None) -> None:
            del name, rows, step

        def log_artifact(self, artifact_path, *, name: str, artifact_type: str = "dataset", metadata=None, aliases=None) -> None:
            del artifact_path, name, artifact_type, metadata, aliases

    stub_handler = StubWandbHandler()

    with patch("src.backtest.engine.WandbRegistry.get", return_value=stub_handler):
        engine.run(
            "2024-01-08",
            "2024-01-08",
            run_id="bt_test",
            agent_workflow=StubWorkflow(),
        )

    assert len(stub_handler.metrics_calls) == 1
    metrics, step = stub_handler.metrics_calls[0]
    assert step == 1
    assert metrics["week/total_value"] == metrics["week/nav"]
    assert metrics["week/total_value"] == 1_100_000.0
    assert len(stub_handler.summary_calls) == 1
    summary = stub_handler.summary_calls[0]
    assert summary["latest_total_value"] == 1_100_000.0
    assert summary["latest_weekly_return"] == pytest.approx(0.1)
    assert summary["latest_cash_weight"] == 0.0


def test_run_resume_latest_repairs_selected_etfs_before_processing(tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine._etf_prices = pl.DataFrame(
        {
            "Code": ["512480.SH", "512480.SH", "512480.SH"],
            "trade_dt": [20240105, 20240112, 20240119],
            "close": [1.0, 1.1, 1.21],
        }
    )
    engine._meta_sector_etf_code_map = {"科技成长": ["512480.SH"]}
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

    checkpoint_payload = {
        "run_id": "bt_test",
        "completed_week": "2024-01-08",
        "portfolio": {
            "initial_capital": 1_000_000.0,
            "total_value": 1_000_000.0,
            "holdings": {"科技成长": 1.0},
            "selected_etfs": {},
            "transaction_fee": 0.0,
            "slippage": 0.0,
        },
        "memory": {
            "last_week_return": 0.1,
            "last_week_holdings": {"科技成长": 1.0},
            "last_week_returns": {"科技成长": 0.1},
            "prev_observations": {},
            "prev_agent_decisions": [],
        },
        "results": [
            {
                "run_id": "bt_test",
                "week_start": "2024-01-08",
                "initial_capital": 1_000_000.0,
                "nav": 1_000_000.0,
                "total_value": 1_000_000.0,
                "weekly_return": 0.0,
                "invested_weight": 1.0,
                "cash_weight": 0.0,
                "holdings": {"科技成长": 1.0},
                "selected_etfs": {},
                "meta_sector_contributions": {},
                "meta_sector_returns": {},
                "industry_contributions": {},
                "status": "LOSS",
                "cumulative_return": 0.0,
                "observations": {},
                "agent_decisions": [],
            }
        ],
    }
    engine._write_checkpoint_payload(
        run_id="bt_test",
        completed_week="2024-01-08",
        payload=checkpoint_payload,
        write_latest=True,
    )
    engine._update_run_meta("bt_test", latest_completed_week="2024-01-08")

    results = engine.run(
        "2024-01-08",
        "2024-01-15",
        run_id="bt_test",
        resume_latest=True,
        agent_workflow=None,
    )

    latest_payload = engine._load_latest_checkpoint(run_id="bt_test")
    repaired_checkpoint = engine._load_checkpoint(run_id="bt_test", completed_week="2024-01-08")
    assert repaired_checkpoint["portfolio"]["selected_etfs"] == {"科技成长": "512480.SH"}
    assert results.shape[0] == 2
    assert latest_payload["portfolio"]["selected_etfs"] == {"科技成长": "512480.SH"}
