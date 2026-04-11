"""News2ETF Agent — unified CLI entry point.

Usage:
    python runtime/agent/main.py backtest --start-date 2021-01-01 --end-date 2023-12-31
    python runtime/agent/main.py decide --week 2023-06-12
"""

from __future__ import annotations

import functools
import json
import os
import random
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from src.agent.state import AgentState
from src.agent.workflow import build_workflow
from src.backtest.diagnostics import diagnose_backtest
from src.backtest.engine import WalkForwardEngine
from src.backtest.visualization import visualize_backtest
from src.config import AgentRootConfig, get_config, init_config, runtime_root
from src.env import load_project_env
from src.logger import init_logger, logger
from src.runtime import init_runtime
from src.wandb_handler import WandbRegistry

app = typer.Typer(name="news2etf", add_completion=False, pretty_exceptions_show_locals=False)
console = Console()
_ROOT = runtime_root()

load_project_env(_ROOT)


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_meta_path(checkpoint_dir: Path | None, run_id: str | None) -> Path | None:
    if checkpoint_dir is None or not run_id:
        return None
    return Path(checkpoint_dir) / run_id / "run_meta.json"


def _load_run_meta(checkpoint_dir: Path | None, run_id: str | None) -> dict[str, object]:
    path = _run_meta_path(checkpoint_dir, run_id)
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_run_meta(checkpoint_dir: Path | None, run_id: str | None, payload: dict[str, object]) -> Path | None:
    path = _run_meta_path(checkpoint_dir, run_id)
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _validate_backtest_inputs(cfg: object) -> None:
    data = getattr(cfg, "data", None)
    predict = getattr(cfg, "predict", None)
    if data is None or predict is None:
        return

    feature_candidates = [
        _path_or_none(getattr(data, "output_agent_features_oof", None)),
        _path_or_none(getattr(data, "output_agent_features", None)),
    ]
    if any(path is not None and path.exists() for path in feature_candidates):
        return

    sentiment_path = _path_or_none(getattr(data, "output_sentiment", None))
    bundle_dir = _path_or_none(getattr(predict, "signals_onnx_dir", None))

    if sentiment_path is None or not sentiment_path.exists():
        raise FileNotFoundError(
            "Backtest requires either precomputed agent features or the signals sentiment dataset.\n"
            f"- Missing sentiment dataset: {sentiment_path}\n"
            f"- Expected feature caches: {feature_candidates[0]} or {feature_candidates[1]}\n"
            "Generate the sentiment dataset with `just signals-train-final-3y` "
            "or `./.venv/bin/python -m trainer.main signals train`.\n"
            "If you already have an exported ONNX bundle and only need the held-out features, run `just signals-infer-2024`."
        )

    if bundle_dir is None or not bundle_dir.exists():
        raise FileNotFoundError(
            "Backtest could not find precomputed agent features, so it needs the exported signals ONNX bundle "
            "to rebuild them on demand.\n"
            f"- Missing bundle dir: {bundle_dir}\n"
            f"- Available sentiment dataset: {sentiment_path}\n"
            "Export the bundle with `just signals-export-onnx-final-3y`, "
            "or run the full pipeline with `just signals-agent-pipeline-2024`."
        )


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
        run_meta = _load_run_meta(checkpoint_dir, run_id)
        existing_wandb_id = str(run_meta.get("wandb_run_id", "") or "") or None
        if existing_wandb_id:
            logger.info("Resuming W&B run: run_id={} wandb_run_id={}", run_id, existing_wandb_id)
        WandbRegistry.init(
            "backtest",
            run_name=run_id or "src-run",
            cfg_dict=cfg.model_dump(mode="json"),
            tags=wandb_tags or ["backtest"],
            existing_run_id=existing_wandb_id,
            resume="must" if existing_wandb_id else None,
        )
        handler = WandbRegistry.get("backtest")
        if handler is not None:
            now = _utc_now_iso()
            _save_run_meta(
                checkpoint_dir,
                run_id,
                {
                    "run_id": run_id,
                    "created_at": str(run_meta.get("created_at", "") or now),
                    "updated_at": now,
                    **handler.metadata(),
                },
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
    visualize: Annotated[bool, typer.Option("--visualize/--no-visualize")] = True,
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
            ("Visualize", "Yes" if visualize else "No"),
        ],
    )

    try:
        _validate_backtest_inputs(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
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
        auto_visualize=visualize,
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


@app.command("visualize-backtest")
@with_src_init
def visualize_backtest_cmd(
    config: Annotated[Path | None, typer.Option("-c", "--config")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    results_path: Annotated[Path | None, typer.Option("--results-path")] = None,
    metrics_path: Annotated[Path | None, typer.Option("--metrics-path")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    upload_wandb: Annotated[bool, typer.Option("--upload-wandb/--no-upload-wandb")] = False,
) -> None:
    """Generate a local Plotly dashboard from persisted backtest parquet files."""
    cfg = get_config()
    if run_id:
        default_run_dir = _ROOT / "checkpoints" / run_id
        resolved_results_path = results_path or default_run_dir / "backtest_results.parquet"
        resolved_metrics_path = metrics_path or default_run_dir / "backtest_metrics.parquet"
        resolved_output_dir = output_dir or default_run_dir / "visualizations"
    else:
        resolved_results_path = results_path or cfg.data.output_backtest
        resolved_metrics_path = metrics_path or cfg.data.output_backtest_metrics
        resolved_output_dir = output_dir or _ROOT / "data" / "visualizations" / "backtest"

    console.print("[bold]Backtest Visualization[/bold]")
    _print_table(
        "",
        [
            ("Run ID", run_id or "N/A"),
            ("Results", str(resolved_results_path)),
            ("Metrics", str(resolved_metrics_path)),
            ("Output", str(resolved_output_dir)),
            ("Upload W&B", "Yes" if upload_wandb else "No"),
        ],
    )

    wandb_key = "visualize-backtest"
    run_meta: dict[str, object] = {}
    if upload_wandb:
        if not run_id:
            console.print(
                "[bold red]--upload-wandb requires --run-id so the checkpoint run_meta.json can be used.[/bold red]"
            )
            raise typer.Exit(1)
        checkpoint_dir = _ROOT / "checkpoints"
        run_meta = _load_run_meta(checkpoint_dir, run_id)
        existing_wandb_id = str(run_meta.get("wandb_run_id", "") or "") or None
        if not existing_wandb_id:
            console.print(
                "[bold red]No wandb_run_id found in "
                f"{_run_meta_path(checkpoint_dir, run_id)}. "
                "Run the backtest with W&B enabled first.[/bold red]"
            )
            raise typer.Exit(1)
        logger.info("Resuming W&B run for visualization upload: run_id={} wandb_run_id={}", run_id, existing_wandb_id)
        raw_tags = run_meta.get("tags", [])
        tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else ["backtest"]
        tags = list(dict.fromkeys([*(tags or ["backtest"]), "visualized"]))
        WandbRegistry.init(
            wandb_key,
            run_name=str(run_meta.get("wandb_run_name", "") or run_id),
            cfg_dict=cfg.model_dump(mode="json"),
            tags=tags or ["backtest"],
            existing_run_id=existing_wandb_id,
            resume="must",
        )

    try:
        result = visualize_backtest(
            results_path=resolved_results_path,
            metrics_path=resolved_metrics_path,
            output_dir=resolved_output_dir,
            run_id=run_id,
        )
        if upload_wandb:
            handler = WandbRegistry.get(wandb_key)
            if handler is not None:
                handler.add_tags(["visualized"])
                images = {
                    f"backtest/visualizations/{path.stem}": path
                    for path in getattr(result, "image_paths", [])
                }
                captions = {
                    key: f"{result.run_id} {path.stem.replace('_', ' ')}"
                    for key, path in images.items()
                }
                handler.log_images(
                    images,
                    captions=captions,
                    gallery_key="backtest_visualizations",
                )
                now = _utc_now_iso()
                _save_run_meta(
                    _ROOT / "checkpoints",
                    run_id,
                    {
                        **run_meta,
                        "run_id": run_id,
                        "updated_at": now,
                        **handler.metadata(),
                    },
                )
        console.print(f"[bold green]Visualization saved:[/bold green] {result.report_path}")
    finally:
        if upload_wandb:
            WandbRegistry.finish_all()


if __name__ == "__main__":
    app()
