"""News2ETF Agent — unified CLI entry point.

Usage:
    python runtime/agent/main.py backtest --start-date 2021-01-01 --end-date 2023-12-31
    python runtime/agent/main.py decide --week 2023-06-12
"""

from __future__ import annotations

import functools
import os
import random
import uuid
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from src.agent.state import AgentState
from src.agent.workflow import build_workflow
from src.backtest.diagnostics import diagnose_backtest
from src.backtest.engine import WalkForwardEngine
from src.config import AgentRootConfig, get_config, init_config, runtime_root
from src.env import load_project_env
from src.logger import init_logger
from src.runtime import init_runtime
from src.wandb_handler import WandbRegistry

app = typer.Typer(name="news2etf", add_completion=False, pretty_exceptions_show_locals=False)
console = Console()
_ROOT = runtime_root()

load_project_env(_ROOT)


def _print_table(title: str, rows: list[tuple[str, str]]) -> None:
    t = Table(title=title, show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column(style="green")
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)


def _init_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _init_src(
    config_path: Path | None,
    log_path: Path | None,
    *,
    run_id: str | None = None,
    checkpoint_dir: Path | None = None,
    init_wandb: bool = False,
    wandb_tags: list[str] | None = None,
) -> AgentRootConfig:
    _ROOT.mkdir(parents=True, exist_ok=True)
    (_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (_ROOT / "wandb").mkdir(parents=True, exist_ok=True)
    init_config(config_path)
    init_logger(log_path)
    cfg = get_config()
    _init_seed(cfg.seed)
    init_runtime(run_id=run_id, checkpoint_dir=checkpoint_dir)
    os.environ.setdefault("WANDB_DIR", str(_ROOT / "wandb"))
    if init_wandb:
        WandbRegistry.init(
            "backtest",
            run_name=run_id or "src-run",
            cfg_dict=cfg.model_dump(mode="json"),
            tags=wandb_tags or ["backtest"],
        )
    return cfg


def with_src_init(
    func: Callable | None = None,
    *,
    log_path_resolver: Callable[..., Path | None] | None = None,
    kwargs_preprocessor: Callable[[dict], dict] | None = None,
    init_wandb: bool = False,
    wandb_tags: list[str] | None = None,
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if kwargs_preprocessor is not None:
                kwargs = kwargs_preprocessor(dict(kwargs))
            resolved_log_file = (
                log_path_resolver(*args, **kwargs) if log_path_resolver is not None else kwargs.get("log_file")
            )
            checkpoint_dir = _ROOT / "checkpoints"
            _init_src(
                kwargs.get("config"),
                resolved_log_file,
                run_id=kwargs.get("run_id"),
                checkpoint_dir=checkpoint_dir,
                init_wandb=init_wandb,
                wandb_tags=wandb_tags,
            )
            try:
                return fn(*args, **kwargs)
            finally:
                if init_wandb:
                    WandbRegistry.finish_all()

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def _prepare_backtest_kwargs(kwargs: dict) -> dict:
    prepared = dict(kwargs)
    if not prepared.get("run_id"):
        prepared["run_id"] = f"bt_{uuid.uuid4().hex[:8]}"
    return prepared


def _resolve_backtest_log_path(*args, **kwargs) -> Path | None:
    log_file = kwargs.get("log_file")
    if log_file is not None:
        return Path(log_file)
    run_id = kwargs["run_id"]
    return _ROOT / "checkpoints" / run_id / "backtest.log"


# ─── Commands ─────────────────────────────────────────────────────────────────


@app.command()
@with_src_init(
    log_path_resolver=_resolve_backtest_log_path,
    kwargs_preprocessor=_prepare_backtest_kwargs,
    init_wandb=True,
    wandb_tags=["backtest"],
)
def backtest(
    start_date: Annotated[str, typer.Option("--start-date")] = (date.today() - timedelta(days=730)).isoformat(),
    end_date: Annotated[str, typer.Option("--end-date")] = date.today().isoformat(),
    train_end: Annotated[str | None, typer.Option("--train-end")] = None,
    test_start: Annotated[str | None, typer.Option("--test-start")] = None,
    config: Annotated[Path | None, typer.Option("-c", "--config")] = None,
    log_file: Annotated[Path | None, typer.Option("--log-file")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    resume_from_week: Annotated[str | None, typer.Option("--resume-from-week")] = None,
    resume_to_week: Annotated[str | None, typer.Option("--resume-to-week")] = None,
    resume_latest: Annotated[bool, typer.Option("--resume-latest")] = False,
) -> None:
    """Run weekly walk-forward backtest using ReAct agent."""
    if (resume_from_week or resume_latest) and not run_id:
        raise typer.BadParameter("--run-id is required when using resume options")
    if resume_from_week and resume_latest:
        raise typer.BadParameter("--resume-from-week cannot be used together with --resume-latest")

    cfg = get_config()
    resolved_log_file = log_file or (_ROOT / "checkpoints" / run_id / "backtest.log")

    console.print("[bold]Weekly Backtest[/bold]")
    _print_table(
        "",
        [
            ("Start", start_date),
            ("End", end_date),
            ("Train end", train_end or "N/A"),
            ("Test start", test_start or "N/A"),
            ("Run ID", run_id),
            ("Log file", str(resolved_log_file)),
            ("Resume from", resume_from_week or "N/A"),
            ("Resume to", resume_to_week or "N/A"),
            ("Resume latest", "Yes" if resume_latest else "No"),
        ],
    )

    workflow = build_workflow(cfg)
    engine = WalkForwardEngine(cfg, checkpoint_dir=_ROOT / "checkpoints")
    engine.run(
        start_date,
        end_date,
        run_id=run_id,
        agent_workflow=workflow,
        resume_from_week=resume_from_week,
        resume_to_week=resume_to_week,
        resume_latest=resume_latest,
    )
    console.print("[bold green]Backtest complete![/bold green]")


@app.command()
@with_src_init
def decide(
    week: Annotated[str, typer.Option("--week", help="Monday date YYYY-MM-DD")],
    config: Annotated[Path | None, typer.Option("-c", "--config")] = None,
    log_file: Annotated[Path | None, typer.Option("--log-file")] = None,
) -> None:
    """Run single-week agent decision (debug mode)."""
    cfg = get_config()

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


@app.command("diagnose-backtest")
@with_src_init
def diagnose_backtest_cmd(
    config: Annotated[Path | None, typer.Option("-c", "--config")] = None,
    path: Annotated[Path | None, typer.Option("--path")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    start_week: Annotated[str | None, typer.Option("--start-week")] = None,
    end_week: Annotated[str | None, typer.Option("--end-week")] = None,
    max_issues: Annotated[int, typer.Option("--max-issues")] = 20,
) -> None:
    """Diagnose a backtest parquet and print suspicious weeks."""
    cfg = get_config()
    if path is not None:
        backtest_path = path
    elif run_id:
        backtest_path = _ROOT / "checkpoints" / run_id / "backtest_results.parquet"
    else:
        backtest_path = cfg.data.output_backtest

    summary, issues, df = diagnose_backtest(
        config=cfg,
        backtest_path=backtest_path,
        run_id=run_id,
        start_week=start_week,
        end_week=end_week,
    )

    console.print("[bold]Backtest Diagnostics[/bold]")
    _print_table(
        "",
        [
            ("Path", str(backtest_path)),
            ("Run ID", summary["run_ids"]),
            ("Rows", str(summary["rows"])),
            ("Start week", summary["start_week"]),
            ("End week", summary["end_week"]),
            ("Final NAV", f"{summary['final_nav']:.2f}"),
            ("Total return", f"{summary['total_return']:.2%}"),
            ("Max drawdown", f"{summary['max_drawdown']:.2%}"),
            ("Issues", str(summary["issue_count"])),
        ],
    )

    best = df.sort("weekly_return", descending=True).select(["week_start", "weekly_return"]).head(3).to_dicts()
    worst = df.sort("weekly_return").select(["week_start", "weekly_return"]).head(3).to_dicts()

    best_table = Table(title="Best Weeks", show_header=True)
    best_table.add_column("Week", style="bold cyan")
    best_table.add_column("Return", style="green")
    for row in best:
        best_table.add_row(str(row["week_start"]), f"{float(row['weekly_return']):.2%}")
    console.print(best_table)

    worst_table = Table(title="Worst Weeks", show_header=True)
    worst_table.add_column("Week", style="bold cyan")
    worst_table.add_column("Return", style="red")
    for row in worst:
        worst_table.add_row(str(row["week_start"]), f"{float(row['weekly_return']):.2%}")
    console.print(worst_table)

    if issues:
        issue_table = Table(title=f"Issues (showing up to {max_issues})", show_header=True)
        issue_table.add_column("Week", style="bold cyan")
        issue_table.add_column("Severity")
        issue_table.add_column("Code")
        issue_table.add_column("Detail")
        for issue in issues[:max_issues]:
            issue_table.add_row(issue.week_start, issue.severity, issue.code, issue.detail)
        console.print(issue_table)
        raise typer.Exit(1 if summary["error_count"] > 0 else 0)

    console.print("[bold green]No issues detected.[/bold green]")


if __name__ == "__main__":
    app()
