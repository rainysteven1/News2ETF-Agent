from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.backtest.engine import WalkForwardEngine
from src.backtest.visualization import visualize_backtest


def _write_sample_backtest(run_dir: Path, run_id: str = "bt_viz") -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "backtest_results.parquet"
    metrics_path = run_dir / "backtest_metrics.parquet"
    pl.DataFrame(
        {
            "run_id": [run_id, run_id, run_id],
            "week_start": ["2024-01-01", "2024-01-08", "2024-01-15"],
            "initial_capital": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "nav": [1_000_000.0, 1_020_000.0, 1_010_000.0],
            "weekly_return": [0.0, 0.02, -0.0098039216],
            "market_closed_week": [False, False, False],
            "invested_weight": [0.0, 0.2, 0.3],
            "cash_weight": [1.0, 0.8, 0.7],
            "holdings": [
                "{}",
                json.dumps({"科技成长": 0.2}, ensure_ascii=False),
                json.dumps({"科技成长": 0.18, "高端制造": 0.12}, ensure_ascii=False),
            ],
            "meta_sector_contributions": [
                "{}",
                json.dumps({"科技成长": 0.02}, ensure_ascii=False),
                json.dumps({"科技成长": -0.006, "高端制造": -0.0038039216}, ensure_ascii=False),
            ],
            "meta_sector_returns": [
                "{}",
                json.dumps({"科技成长": 0.10}, ensure_ascii=False),
                json.dumps({"科技成长": -0.0333, "高端制造": -0.0317}, ensure_ascii=False),
            ],
        }
    ).write_parquet(results_path)
    pl.DataFrame(
        {
            "run_id": [run_id],
            "as_of_week": ["2024-01-15"],
            "total_return": [0.01],
            "annual_return": [0.18],
            "max_drawdown": [-0.0098039216],
            "sharpe_ratio": [1.2],
            "sortino_ratio": [1.4],
            "calmar_ratio": [18.0],
            "win_rate": [1 / 3],
            "weeks": [3],
            "market_closed_weeks": [0],
            "final_nav": [1_010_000.0],
            "initial_capital": [1_000_000.0],
        }
    ).write_parquet(metrics_path)
    return results_path, metrics_path


def test_visualize_backtest_writes_plotly_dashboard(tmp_path: Path) -> None:
    results_path, metrics_path = _write_sample_backtest(tmp_path / "run")
    output_dir = tmp_path / "viz"

    result = visualize_backtest(
        results_path=results_path,
        metrics_path=metrics_path,
        output_dir=output_dir,
        run_id="bt_viz",
    )

    assert result.report_path.exists()
    assert result.summary_path.exists()
    assert len(result.chart_paths) >= 7
    assert len(result.image_paths) >= 7
    assert all(path.exists() for path in result.chart_paths)
    assert all(path.exists() and path.suffix == ".png" for path in result.image_paths)

    report = result.report_path.read_text(encoding="utf-8")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert "Interactive Performance Dashboard" in report
    assert "Cash vs Invested Weight" in report
    assert "Sector Contribution" in report
    assert "Sector Return Heatmap" in report
    assert "#dc2626" in report
    assert "#16a34a" in report
    assert "plotly" in report.lower()
    assert summary["run_id"] == "bt_viz"


def test_engine_auto_visualizes_at_completion(monkeypatch, tmp_path: Path) -> None:
    engine = WalkForwardEngine.__new__(WalkForwardEngine)
    engine.checkpoint_dir = tmp_path / "checkpoints"
    engine.config = type(
        "Config",
        (),
        {
            "backtest": type(
                "Backtest",
                (),
                {
                    "auto_visualize": True,
                    "initial_capital": 1_000_000.0,
                    "transaction_fee": 0.0003,
                    "slippage": 0.0005,
                },
            )(),
        },
    )()
    results_path, metrics_path = _write_sample_backtest(tmp_path / "checkpoints" / "bt_viz_engine", "bt_viz_engine")
    called: dict[str, Path] = {}

    def fake_persist(results, *, run_id: str, as_of_week: str):
        del results, as_of_week
        return pl.read_parquet(results_path), {"weeks": 3, "market_closed_weeks": 0}

    def fake_visualize_backtest(*, results_path, metrics_path, output_dir, run_id):
        called["results_path"] = Path(results_path)
        called["metrics_path"] = Path(metrics_path)
        called["output_dir"] = Path(output_dir)
        called["run_id"] = run_id

        class DummyResult:
            def __init__(self):
                self.output_dir = Path(output_dir)
                self.report_path = Path(output_dir) / "report.html"
                self.image_paths = []

        return DummyResult()

    monkeypatch.setattr(engine, "_get_week_starts", lambda start, end: [])
    monkeypatch.setattr(engine, "_load_etf_prices", lambda: None)
    monkeypatch.setattr(engine, "_persist_backtest_snapshot", fake_persist)
    monkeypatch.setattr("src.backtest.engine.visualize_backtest", fake_visualize_backtest)

    engine.run("2024-01-01", "2024-01-31", run_id="bt_viz_engine", auto_visualize=True)

    assert called["results_path"] == results_path
    assert called["metrics_path"] == metrics_path
    assert called["output_dir"] == tmp_path / "checkpoints" / "bt_viz_engine" / "visualizations"
    assert called["run_id"] == "bt_viz_engine"
