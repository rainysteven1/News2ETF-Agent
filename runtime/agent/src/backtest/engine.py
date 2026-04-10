"""Walk-forward backtesting engine — WEEKLY granularity.

Execution semantics (T+1, no look-ahead bias):
    Week T agent decision:
      - Uses news/signals through T-1 (lagged by 1 week)
      - Decision is APPLIED at week T open price (via apply_decisions friction)
      - Return is COMPUTED for week T based on T-1 close → T close

    Loop order (critical for accounting):
      1. Compute last week's return (based on holdings established last iteration)
      2. Agent decides for current week (gets real last_week_return + holdings)
      3. Apply decisions (deducts摩擦成本, updates holdings)
      4. Record state (stores the decision intent, not yet-realized return)
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.backtest.metrics import calculate_metrics
from src.backtest.portfolio import Portfolio
from src.config import AgentRootConfig, best_etf_by_index_path
from src.logger import logger
from src.wandb_handler import WandbRegistry
from src.utils.industry_map import IndustryMapper
from src.utils.etf_universe import get_etf_universe


def _normalize_log_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _truncate_wandb_text(value: Any, limit: int = 500) -> str:
    text = _normalize_log_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _jsonify_backtest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert nested payload fields to JSON strings for stable parquet writes."""
    json_columns = {
        "holdings",
        "selected_etfs",
        "meta_sector_contributions",
        "meta_sector_returns",
        "industry_contributions",
        "observations",
        "agent_decisions",
    }
    normalized: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        for key in json_columns:
            if key in item:
                item[key] = json.dumps(item.get(key, {} if key != "agent_decisions" else []), ensure_ascii=False)
        normalized.append(item)
    return normalized


def _sum_dict_values(values: dict[str, float]) -> float:
    return float(sum(float(v) for v in values.values()))


def _format_weekly_decisions(decisions: list[dict], include_reason: bool = False) -> str:
    if not decisions:
        return "-"

    meta_decision = decisions[0] if decisions else {}
    if meta_decision.get("level1_plan"):
        level2_map = {
            item.get("meta_sector", ""): _normalize_log_text(item.get("selected_etf", "")) or "-"
            for item in meta_decision.get("level2_plan", [])
        }
        parts = []
        for item in meta_decision.get("level1_plan", []):
            sector = item.get("meta_sector", "") or "unknown"
            action = item.get("action", "hold")
            weight = float(item.get("weight", 0.0) or 0.0)
            etf = level2_map.get(sector, "-")
            part = f"{sector}:{action} {weight:.1%} {etf}"
            if include_reason:
                reason = _normalize_log_text(item.get("reason", ""))
                if reason:
                    part = f"{part} reason={reason}"
            parts.append(part)
        return " | ".join(parts) if parts else "-"

    parts = []
    for item in decisions:
        sector = item.get("industry", "") or item.get("meta_sector", "") or "unknown"
        action = item.get("action", "hold")
        weight = float(item.get("weight", 0.0) or 0.0)
        etf = _normalize_log_text(item.get("selected_etf", "")) or "-"
        part = f"{sector}:{action} {weight:.1%} {etf}"
        if include_reason:
            reason = _normalize_log_text(item.get("reason", ""))
            if reason:
                part = f"{part} reason={reason}"
        parts.append(part)
    return " | ".join(parts) if parts else "-"


class WalkForwardEngine:
    """Walk-forward backtesting engine that runs weekly."""

    def __init__(self, config: AgentRootConfig, checkpoint_dir: Path | None = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.mapper = IndustryMapper(
            dict_path=config.data.industry_dict,
            etf_info=config.data.etf_info,
            best_etf_path=best_etf_by_index_path(config.data.etf_info),
        )
        self._etf_prices: pl.DataFrame | None = None
        self._etf_universe = get_etf_universe(str(config.data.etf_info), str(config.data.etf_prices))

        self._meta_sector_etf_code_map: dict[str, list[str]] = {}
        if config.data.meta_sector_mapping.exists():
            with open(config.data.meta_sector_mapping, encoding="utf-8") as f:
                meta_map = json.load(f)
            for meta_sector, info in meta_map.get("meta_sectors", {}).items():
                codes: list[str] = []
                seen: set[str] = set()
                for sub in info.get("sub_categories", []):
                    for industry in self.mapper.industries:
                        if sub not in self.mapper.get_small_cats(industry):
                            continue
                        for code in self.mapper.best_etf_codes(self.mapper.get_indices(industry, sub)):
                            if code and code not in seen:
                                seen.add(code)
                                codes.append(code)
                self._meta_sector_etf_code_map[meta_sector] = codes

    def _get_etf_universe(self):
        resolver = getattr(self, "_etf_universe", None)
        if resolver is None:
            data_cfg = getattr(self.config, "data", None)
            etf_info = getattr(data_cfg, "etf_info", None)
            etf_prices = getattr(data_cfg, "etf_prices", None)
            if not etf_info or not etf_prices:
                return None
            resolver = get_etf_universe(str(etf_info), str(etf_prices))
            self._etf_universe = resolver
        return resolver

    def _load_etf_prices(self) -> pl.DataFrame | None:
        if self._etf_prices is None:
            path = self.config.data.etf_prices
            if path.exists():
                self._etf_prices = pl.read_parquet(path)
            else:
                logger.warning("ETF prices not found at {}", path)
        return self._etf_prices

    def _get_week_starts(self, start_date: str, end_date: str) -> list[str]:
        """Return list of Monday date strings between start and end."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        days_to_monday = start.weekday()
        monday = start - timedelta(days=days_to_monday)

        weeks = []
        current = monday
        while current <= end:
            weeks.append(current.strftime("%Y-%m-%d"))
            current += timedelta(weeks=1)
        return weeks

    def _checkpoint_run_dir(self, run_id: str) -> Path:
        return self.checkpoint_dir / run_id

    def _checkpoint_path(self, run_id: str, completed_week: str) -> Path:
        return self._checkpoint_run_dir(run_id) / f"{completed_week}.json"

    def _backtest_results_path(self, run_id: str) -> Path:
        return self._checkpoint_run_dir(run_id) / "backtest_results.parquet"

    def _backtest_metrics_path(self, run_id: str) -> Path:
        return self._checkpoint_run_dir(run_id) / "backtest_metrics.parquet"

    def _save_checkpoint(
        self,
        *,
        run_id: str,
        completed_week: str,
        results: list[dict[str, Any]],
        portfolio: Portfolio,
        last_week_return: float,
        last_week_holdings: dict[str, float],
        last_week_returns: dict[str, float],
        prev_observations: dict[str, Any],
        prev_agent_decisions: list[dict[str, Any]],
    ) -> Path:
        """Save resumable weekly checkpoint after a week has been fully processed."""
        run_dir = self._checkpoint_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "completed_week": completed_week,
            "portfolio": portfolio.snapshot(),
            "memory": {
                "last_week_return": last_week_return,
                "last_week_holdings": last_week_holdings,
                "last_week_returns": last_week_returns,
                "prev_observations": prev_observations,
                "prev_agent_decisions": prev_agent_decisions,
            },
            "results": results,
        }
        checkpoint_path = self._checkpoint_path(run_id, completed_week)
        checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_path = run_dir / "latest.json"
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return checkpoint_path

    def _load_checkpoint(self, *, run_id: str, completed_week: str) -> dict[str, Any]:
        """Load a previously saved weekly checkpoint."""
        checkpoint_path = self._checkpoint_path(run_id, completed_week)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))

    def _load_latest_checkpoint(self, *, run_id: str) -> dict[str, Any]:
        """Load the latest checkpoint for a run."""
        latest_path = self._checkpoint_run_dir(run_id) / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"Latest checkpoint not found: {latest_path}")
        return json.loads(latest_path.read_text(encoding="utf-8"))

    def _validate_week_marker(self, week: str, available_weeks: list[str], label: str) -> None:
        if week not in available_weeks:
            raise ValueError(f"{label}={week} is not a valid Monday week in the selected range")

    def _persist_backtest_snapshot(
        self,
        results: list[dict],
        *,
        run_id: str,
        as_of_week: str,
    ) -> tuple[pl.DataFrame, dict]:
        """Persist current backtest results and cumulative metrics to parquet."""
        output_path = self._backtest_results_path(run_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        incoming_results_df = pl.DataFrame(_jsonify_backtest_records(results))
        if output_path.exists():
            existing_results_df = pl.read_parquet(output_path)
            results_df = (
                pl.concat([existing_results_df, incoming_results_df], how="diagonal_relaxed")
                .unique(subset=["run_id", "week_start"], keep="last")
                .sort("week_start")
            )
        else:
            results_df = incoming_results_df.sort("week_start")
        results_df.write_parquet(output_path)

        metrics_model = calculate_metrics(results_df, risk_free_rate=self.config.backtest.risk_free_rate)
        metrics_payload = {
            "run_id": run_id,
            "as_of_week": as_of_week,
            **metrics_model.model_dump(),
        }
        metrics_path = self._backtest_metrics_path(run_id)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        incoming_metrics_df = pl.DataFrame([metrics_payload])
        if metrics_path.exists():
            existing_metrics_df = pl.read_parquet(metrics_path)
            metrics_df = (
                pl.concat([existing_metrics_df, incoming_metrics_df], how="diagonal_relaxed")
                .unique(subset=["run_id", "as_of_week"], keep="last")
                .sort("as_of_week")
            )
        else:
            metrics_df = incoming_metrics_df
        metrics_df.write_parquet(metrics_path)
        return results_df, metrics_payload

    def _validate_weekly_accounting(
        self,
        *,
        week_start: str,
        prev_nav: float,
        weekly_return: float,
        sector_contributions: dict[str, float],
        post_nav: float,
    ) -> None:
        """Fail fast on obviously broken weekly accounting."""
        max_abs_weekly_return = float(self.config.backtest.max_abs_weekly_return_guardrail)
        if abs(weekly_return) > max_abs_weekly_return:
            raise ValueError(
                f"Weekly return guardrail breached for {week_start}: {weekly_return:.2%} "
                f"(threshold {max_abs_weekly_return:.2%})"
            )

        contribution_sum = _sum_dict_values(sector_contributions)
        if abs(contribution_sum - weekly_return) > 1e-8:
            raise ValueError(
                f"Contribution sum mismatch for {week_start}: sum={contribution_sum:.8f} "
                f"weekly_return={weekly_return:.8f}"
            )

        if prev_nav <= 0:
            raise ValueError(f"Invalid prev_nav for {week_start}: {prev_nav}")

        realized_return = (post_nav / prev_nav) - 1.0
        if abs(realized_return - weekly_return) > 1e-8:
            raise ValueError(
                f"NAV change mismatch for {week_start}: nav_return={realized_return:.8f} "
                f"weekly_return={weekly_return:.8f}"
            )

    def run(
        self,
        start_date: str,
        end_date: str,
        run_id: str | None = None,
        agent_workflow=None,
        resume_from_week: str | None = None,
        resume_to_week: str | None = None,
        resume_latest: bool = False,
    ) -> pl.DataFrame:
        """Run weekly backtest.

        Execution order each iteration:
          1. Compute last week's return (on holdings established by previous decision)
          2. Agent decides for current week (using real last_week_return + last_week_holdings)
          3. Apply decisions (deduct摩擦成本, update holdings — NO direct overwrite)
          4. Record state (decision intent for this week, return realized next iteration)
        """
        if run_id is None:
            run_id = f"bt_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting weekly backtest {} → {}, run_id={}, resume_from_week={}, resume_to_week={}, resume_latest={}",
            start_date,
            end_date,
            run_id,
            resume_from_week or "-",
            resume_to_week or "-",
            resume_latest,
        )

        portfolio = Portfolio(
            initial_capital=self.config.backtest.initial_capital,
            transaction_fee=self.config.backtest.transaction_fee,
            slippage=self.config.backtest.slippage,
        )

        all_week_starts = self._get_week_starts(start_date, end_date)
        logger.info("Total weeks in range: {}", len(all_week_starts))

        if resume_latest and resume_from_week:
            raise ValueError("resume_latest cannot be used together with resume_from_week")

        preloaded_checkpoint: dict[str, Any] | None = None
        if resume_latest:
            preloaded_checkpoint = self._load_latest_checkpoint(run_id=run_id)
            resume_from_week = str(preloaded_checkpoint.get("completed_week", "") or "")
            if not resume_from_week:
                raise ValueError("Latest checkpoint is missing completed_week")

        if resume_from_week:
            self._validate_week_marker(resume_from_week, all_week_starts, "resume_from_week")
        if resume_to_week:
            self._validate_week_marker(resume_to_week, all_week_starts, "resume_to_week")
        if resume_from_week and resume_to_week and resume_from_week > resume_to_week:
            raise ValueError("resume_from_week must be earlier than or equal to resume_to_week")

        etf_prices = self._load_etf_prices()

        # Persistent state passed to agent for behavioural memory
        last_week_return = 0.0
        last_week_holdings: dict[str, float] = {}
        last_week_returns: dict[str, float] = {}

        results: list[dict[str, Any]] = []
        if resume_from_week:
            checkpoint = preloaded_checkpoint or self._load_checkpoint(run_id=run_id, completed_week=resume_from_week)
            portfolio.restore(checkpoint.get("portfolio", {}))
            memory = checkpoint.get("memory", {})
            last_week_return = float(memory.get("last_week_return", 0.0) or 0.0)
            last_week_holdings = {
                str(k): float(v) for k, v in dict(memory.get("last_week_holdings", {})).items()
            }
            last_week_returns = {
                str(k): float(v) for k, v in dict(memory.get("last_week_returns", {})).items()
            }
            results = list(checkpoint.get("results", []))
            logger.info(
                "Loaded checkpoint for run_id={} completed_week={} results={} nav={:.2f}",
                run_id,
                resume_from_week,
                len(results),
                portfolio.total_value,
            )

        week_starts = [
            week
            for week in all_week_starts
            if (resume_from_week is None or week > resume_from_week)
            and (resume_to_week is None or week <= resume_to_week)
        ]
        logger.info("Weeks to process in this invocation: {}", len(week_starts))

        for week_start in tqdm(week_starts, desc="Backtesting weeks"):
            # ── Step 1: Agent decides for THIS week ─────────────────────────────
            # Uses last week's realized return and closing holdings as memory.
            decisions = []
            decision_payload: list[dict[str, Any]] = []
            current_observations: dict = {}
            current_error = ""
            if agent_workflow is not None:
                from src.agent.state import AgentState

                state: AgentState = {
                    "date": week_start,
                    "messages": [],
                    "observations": {},
                    "decisions": [],
                    "is_risk_passed": False,
                    "retry_count": 0,
                    "last_error": "",
                    "loop_step": 0,
                    "last_week_pnl": last_week_return,
                    "last_week_holdings": dict(last_week_holdings),
                    "last_week_returns": dict(last_week_returns),
                }
                try:
                    result = agent_workflow.invoke(state)
                    decisions = result.get("decisions", [])
                    current_observations = result.get("observations", {})
                    current_error = result.get("last_error", "")
                    decision_payload = [
                        d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in decisions
                    ]
                    logger.debug(
                        "[Agent] week={} decisions={} observations_keys={} last_error={}",
                        week_start,
                        len(decisions),
                        list(current_observations.keys()),
                        current_error,
                    )
                    logger.info(
                        "[Week Decisions] week={} {}",
                        week_start,
                        _format_weekly_decisions(decision_payload),
                    )
                    logger.debug(
                        "[Week Decision Reasons] week={} {}",
                        week_start,
                        _format_weekly_decisions(decision_payload, include_reason=True),
                    )
                except Exception as e:
                    logger.error("Agent workflow failed for week {}: {}", week_start, e)

            # ── Step 2: Apply decisions for THIS week ───────────────────────────
            # NO direct overwrite of portfolio.holdings or portfolio.total_value.
            # apply_decisions() handles 摩擦成本, 滑点, and target normalization.
            if decisions:
                formatted = [d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in decisions]
                portfolio.apply_decisions(formatted)
                repaired = portfolio.repair_missing_selected_etfs(
                    resolver=self._get_etf_universe(),
                    week_start=week_start,
                    mapper=getattr(self, "mapper", None),
                )
                if repaired:
                    logger.warning(
                        "[Week ETF Repair] week={} repaired_selected_etfs={}",
                        week_start,
                        ",".join(repaired),
                    )
                total_positions, covered_positions, missing_selected = portfolio.selected_etf_coverage()
                logger.info(
                    "[Week Target ETF Coverage] week={} coverage={}/{} missing_selected_etfs={}",
                    week_start,
                    covered_positions,
                    total_positions,
                    ",".join(missing_selected) if missing_selected else "-",
                )

            # ── Step 3: Realize THIS week's return on the updated holdings ─────
            nav_before_return = portfolio.total_value
            weekly_return = 0.0
            sector_contributions: dict[str, float] = {}
            sector_returns: dict[str, float] = {}
            if etf_prices is not None and portfolio.invested_weight > 0:
                repaired = portfolio.repair_missing_selected_etfs(
                    resolver=self._get_etf_universe(),
                    week_start=week_start,
                    mapper=getattr(self, "mapper", None),
                )
                if repaired:
                    logger.warning(
                        "[Week ETF Repair] week={} repaired_selected_etfs={}",
                        week_start,
                        ",".join(repaired),
                    )
                diagnostics = portfolio.inspect_price_availability(etf_prices, week_start)
                total_positions, covered_positions, missing_selected = portfolio.selected_etf_coverage()
                missing_price = diagnostics["missing_price_sectors"]
                logger.info(
                    "[Week ETF Coverage] week={} coverage={}/{} missing_selected_etfs={} missing_price_sectors={}",
                    week_start,
                    covered_positions,
                    total_positions,
                    ",".join(missing_selected) if missing_selected else "-",
                    ",".join(missing_price) if missing_price else "-",
                )
                weekly_return, sector_contributions, sector_returns = portfolio.compute_weekly_return(
                    etf_prices, week_start, self._meta_sector_etf_code_map
                )
                portfolio.update_nav(weekly_return)
                self._validate_weekly_accounting(
                    week_start=week_start,
                    prev_nav=nav_before_return,
                    weekly_return=weekly_return,
                    sector_contributions=sector_contributions,
                    post_nav=portfolio.total_value,
                )
                portfolio.settle_week(sector_returns, weekly_return)
            logger.info(
                "[Week Result] week={} weekly_return={:.2%} nav={:.2f} decisions={} last_error={}",
                week_start,
                weekly_return,
                portfolio.total_value,
                len(decisions),
                current_error or "-",
            )
            wandb_handler = WandbRegistry.get("backtest")
            if wandb_handler is not None:
                wandb_handler.log_metrics(
                    {
                        "week_index": len(results) + 1,
                        "week/weekly_return": weekly_return,
                        "week/nav": portfolio.total_value,
                        "week/invested_weight": portfolio.invested_weight,
                        "week/cash_weight": portfolio.cash_weight,
                        "week/decision_count": len(decisions),
                    },
                    step=len(results) + 1,
                )
                wandb_handler.log_table_row(
                    "backtest/weekly_trace",
                    {
                        "week_start": week_start,
                        "weekly_return": float(weekly_return),
                        "nav": float(portfolio.total_value),
                        "invested_weight": float(portfolio.invested_weight),
                        "cash_weight": float(portfolio.cash_weight),
                        "decision_count": int(len(decisions)),
                        "last_error": _truncate_wandb_text(current_error or "-"),
                        "decision_text": _truncate_wandb_text(_format_weekly_decisions(decision_payload)),
                        "decision_reasons": _truncate_wandb_text(
                            _format_weekly_decisions(decision_payload, include_reason=True),
                            limit=1200,
                        ),
                        "researcher_summary": _truncate_wandb_text(current_observations.get("researcher_summary", "")),
                        "tool_build_decision_context": _truncate_wandb_text(
                            current_observations.get("tool_build_decision_context", ""),
                            limit=800,
                        ),
                        "tool_read_market_news": _truncate_wandb_text(
                            current_observations.get("tool_read_market_news", ""),
                            limit=800,
                        ),
                        "tool_compute_ml_signals": _truncate_wandb_text(
                            current_observations.get("tool_compute_ml_signals", ""),
                            limit=800,
                        ),
                        "tool_check_last_week_pnl": _truncate_wandb_text(
                            current_observations.get("tool_check_last_week_pnl", ""),
                            limit=400,
                        ),
                        "tool_get_industry_top_news": _truncate_wandb_text(
                            current_observations.get("tool_get_industry_top_news", ""),
                            limit=800,
                        ),
                    },
                    step=len(results) + 1,
                )

            # Update memory for next iteration
            last_week_return = weekly_return
            last_week_holdings = dict(portfolio.holdings)
            last_week_returns = dict(sector_returns)

            current_agent_decisions = [
                d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in decisions
            ]
            record = portfolio.record_state(
                week_start,
                weekly_return,
                sector_contributions,
                run_id=run_id,
                observations=current_observations,
                agent_decisions=current_agent_decisions,
                sector_returns=sector_returns,
            )
            results.append(record)
            self._persist_backtest_snapshot(results, run_id=run_id, as_of_week=week_start)
            checkpoint_path = self._save_checkpoint(
                run_id=run_id,
                completed_week=week_start,
                results=results,
                portfolio=portfolio,
                last_week_return=last_week_return,
                last_week_holdings=last_week_holdings,
                last_week_returns=last_week_returns,
                prev_observations=current_observations,
                prev_agent_decisions=current_agent_decisions,
            )
            logger.debug("Saved checkpoint: {}", checkpoint_path)

        results_df, metrics = self._persist_backtest_snapshot(
            results,
            run_id=run_id,
            as_of_week=(week_starts[-1] if week_starts else (resume_from_week or end_date)),
        )
        logger.info("Backtest saved to {}", self._backtest_results_path(run_id))
        logger.info("Backtest metrics saved to {}", self._backtest_metrics_path(run_id))
        wandb_handler = WandbRegistry.get("backtest")
        if wandb_handler is not None:
            wandb_handler.log_summary(metrics)
            wandb_handler.log_artifact(
                self._backtest_results_path(run_id),
                name=f"{run_id}_backtest_results",
                artifact_type="dataset",
                aliases=["latest"],
            )
            wandb_handler.log_artifact(
                self._backtest_metrics_path(run_id),
                name=f"{run_id}_backtest_metrics",
                artifact_type="dataset",
                aliases=["latest"],
            )
        logger.info("=" * 60)
        logger.info("Backtest Results run_id={}", run_id)
        for k, v in metrics.items():
            logger.info("  {}: {}", k, v)
        logger.info("=" * 60)

        return results_df
