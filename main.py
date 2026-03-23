"""News2ETF Agent — unified CLI entry point.

Usage:
    python main.py train-signals
    python main.py backtest --start-date 2021-01-01 --end-date 2023-12-31
    python main.py decide --week 2023-06-12
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.agent.state import AgentState
from src.agent.workflow import build_workflow
from src.backtest.engine import WalkForwardEngine
from src.config import AgentRootConfig, load_config

app = typer.Typer(name="news2etf", add_completion=False, pretty_exceptions_show_locals=False)
console = Console()
_ROOT = Path(__file__).resolve().parent.parent


def _load(config_path: Path | None) -> AgentRootConfig:
    cfg_path = config_path or (_ROOT / "config.toml")
    return load_config(cfg_path)


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    t = Table(title=title, show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column(style="green")
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)


# ─── Commands ─────────────────────────────────────────────────────────────────


@app.command()
def backtest(
    start_date: Annotated[str, typer.Option("--start-date")],
    end_date: Annotated[str, typer.Option("--end-date")],
    train_end: Annotated[str | None, typer.Option("--train-end")] = None,
    test_start: Annotated[str | None, typer.Option("--test-start")] = None,
    config: Annotated[Path | None, typer.Option("-c", "--config")] = None,
) -> None:
    """Run weekly walk-forward backtest using ReAct agent."""
    cfg = _load(config)
    run_id = f"bt_{uuid.uuid4().hex[:8]}"

    console.print("[bold]Weekly Backtest[/bold]")
    _print_table(
        "",
        [
            ("Start", start_date),
            ("End", end_date),
            ("Train end", train_end or "N/A"),
            ("Test start", test_start or "N/A"),
            ("Run ID", run_id),
        ],
    )

    workflow = build_workflow(cfg)
    engine = WalkForwardEngine(cfg, checkpoint_dir=_ROOT / "checkpoints")
    engine.run(start_date, end_date, run_id=run_id, agent_workflow=workflow)
    console.print("[bold green]Backtest complete![/bold green]")


@app.command()
def decide(
    week: Annotated[str, typer.Option("--week", help="Monday date YYYY-MM-DD")],
    config: Annotated[Path | None, typer.Option("-c", "--config")] = None,
) -> None:
    """Run single-week agent decision (debug mode)."""
    cfg = _load(config)

    console.print(f"[bold cyan]Running agent for week of {week}...[/bold cyan]")

    # TypedDict access — use dict-style
    state: AgentState = {
        "date": week,
        "messages": [],
        "observations": {},
        "decisions": [],
        "is_risk_passed": False,
        "retry_count": 0,
        "last_error": "",
        "loop_step": 0,
        "last_week_pnl": 0.0,
        "last_week_holdings": {},
    }

    workflow = build_workflow(cfg)
    try:
        result = workflow.invoke(state)
        console.print("\n[bold]=== Decisions ({}) ===[/bold]".format(len(result.get("decisions", []))))
        for d in result.get("decisions", []):
            console.print(f"  {d.industry}: {d.action} {d.weight:.3f} — {d.reason}")
    except Exception as e:
        console.print(f"[red]Workflow failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
