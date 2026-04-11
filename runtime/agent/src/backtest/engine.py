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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from tqdm import tqdm

from src.backtest.metrics import calculate_metrics
from src.backtest.portfolio import Portfolio
from src.backtest.visualization import visualize_backtest
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


def _result_to_wandb_row(result: dict[str, Any]) -> dict[str, Any]:
    observations = result.get("observations", {}) or {}
    decision_payload = result.get("agent_decisions", []) or []
    total_value = float(result.get("total_value", result.get("nav", 0.0)) or 0.0)
    return {
        "week_start": str(result.get("week_start", "")),
        "weekly_return": float(result.get("weekly_return", 0.0) or 0.0),
        "total_value": total_value,
        "market_closed_week": bool(result.get("market_closed_week", False)),
        "trading_day_count": int(result.get("trading_day_count", 0) or 0),
        "first_trading_day": str(result.get("first_trading_day", "") or ""),
        "last_trading_day": str(result.get("last_trading_day", "") or ""),
        "invested_weight": float(result.get("invested_weight", 0.0) or 0.0),
        "cash_weight": float(result.get("cash_weight", 0.0) or 0.0),
        "decision_count": int(len(decision_payload)),
        "last_error": _truncate_wandb_text(result.get("last_error", "-") or "-"),
        "decision_text": _truncate_wandb_text(_format_weekly_decisions(decision_payload)),
        "decision_reasons": _truncate_wandb_text(
            _format_weekly_decisions(decision_payload, include_reason=True),
            limit=1200,
        ),
        "researcher_summary": _truncate_wandb_text(observations.get("researcher_summary", "")),
        "tool_build_decision_context": _truncate_wandb_text(
            observations.get("tool_build_decision_context", ""),
            limit=800,
        ),
        "tool_read_market_news": _truncate_wandb_text(
            observations.get("tool_read_market_news", ""),
            limit=800,
        ),
        "tool_compute_ml_signals": _truncate_wandb_text(
            observations.get("tool_compute_ml_signals", ""),
            limit=800,
        ),
        "tool_check_last_week_pnl": _truncate_wandb_text(
            observations.get("tool_check_last_week_pnl", ""),
            limit=400,
        ),
        "tool_get_industry_top_news": _truncate_wandb_text(
            observations.get("tool_get_industry_top_news", ""),
            limit=800,
        ),
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_trade_dt(value: int | str | None) -> str:
    if value in (None, ""):
        return ""
    digits = str(value)
    if len(digits) == 8 and digits.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return digits


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
            config = getattr(self, "config", None)
            data_cfg = getattr(config, "data", None)
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

    def _get_week_trading_window(self, etf_prices: pl.DataFrame | None, week_start: str) -> dict[str, Any]:
        if etf_prices is None:
            return {
                "market_closed_week": True,
                "trading_day_count": 0,
                "first_trading_day": "",
                "last_trading_day": "",
                "trade_dates": [],
            }
        week_start_dt = datetime.strptime(week_start, "%Y-%m-%d")
        next_week_int = int((week_start_dt + timedelta(days=7)).strftime("%Y%m%d"))
        week_start_int = int(week_start_dt.strftime("%Y%m%d"))
        week_df = etf_prices.filter((pl.col("trade_dt") >= week_start_int) & (pl.col("trade_dt") < next_week_int))
        if len(week_df) == 0:
            return {
                "market_closed_week": True,
                "trading_day_count": 0,
                "first_trading_day": "",
                "last_trading_day": "",
                "trade_dates": [],
            }

        trade_dates = sorted(set(int(value) for value in week_df["trade_dt"].to_list()))
        return {
            "market_closed_week": False,
            "trading_day_count": len(trade_dates),
            "first_trading_day": _format_trade_dt(trade_dates[0]),
            "last_trading_day": _format_trade_dt(trade_dates[-1]),
            "trade_dates": trade_dates,
        }

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

    def _run_meta_path(self, run_id: str) -> Path:
        return self._checkpoint_run_dir(run_id) / "run_meta.json"

    def _load_run_meta(self, run_id: str) -> dict[str, Any]:
        path = self._run_meta_path(run_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _update_run_meta(self, run_id: str, **fields: Any) -> Path:
        path = self._run_meta_path(run_id)
        payload = self._load_run_meta(run_id)
        now = _utc_now_iso()
        payload.setdefault("run_id", run_id)
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        payload.update(fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _checkpoint_path(self, run_id: str, completed_week: str) -> Path:
        return self._checkpoint_run_dir(run_id) / f"{completed_week}.json"

    def _backtest_results_path(self, run_id: str) -> Path:
        return self._checkpoint_run_dir(run_id) / "backtest_results.parquet"

    def _backtest_metrics_path(self, run_id: str) -> Path:
        return self._checkpoint_run_dir(run_id) / "backtest_metrics.parquet"

    def _write_checkpoint_payload(
        self,
        *,
        run_id: str,
        completed_week: str,
        payload: dict[str, Any],
        write_latest: bool,
    ) -> Path:
        checkpoint_path = self._checkpoint_path(run_id, completed_week)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        checkpoint_path.write_text(serialized, encoding="utf-8")
        if write_latest:
            latest_path = self._checkpoint_run_dir(run_id) / "latest.json"
            latest_path.write_text(serialized, encoding="utf-8")
        return checkpoint_path

    def _repair_checkpoint_selected_etfs(
        self,
        *,
        run_id: str,
        completed_week: str,
        payload: dict[str, Any],
        target_week: str,
        write_latest: bool,
    ) -> list[str]:
        portfolio = Portfolio()
        portfolio.restore(payload.get("portfolio", {}))
        if portfolio.invested_weight <= 0:
            return []

        before_selected = dict(portfolio.selected_etfs)
        repaired: list[str] = []
        etf_prices = self._load_etf_prices()
        if etf_prices is not None:
            repaired.extend(
                portfolio.repair_selected_etfs_from_price_map(
                    etf_prices,
                    target_week,
                    self._meta_sector_etf_code_map,
                )
            )
        repaired.extend(
            portfolio.repair_missing_selected_etfs(
                resolver=self._get_etf_universe(),
                week_start=target_week,
                mapper=getattr(self, "mapper", None),
            )
        )
        repaired = list(dict.fromkeys(repaired))
        if portfolio.selected_etfs == before_selected:
            return repaired

        payload["portfolio"] = portfolio.snapshot()
        self._write_checkpoint_payload(
            run_id=run_id,
            completed_week=completed_week,
            payload=payload,
            write_latest=write_latest,
        )
        return repaired

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
        checkpoint_path = self._write_checkpoint_payload(
            run_id=run_id,
            completed_week=completed_week,
            payload=payload,
            write_latest=True,
        )
        self._update_run_meta(
            run_id,
            latest_completed_week=completed_week,
            latest_checkpoint_path=str(checkpoint_path),
            latest_total_value=float(portfolio.total_value),
            latest_weekly_return=float(last_week_return),
            latest_cash_weight=float(portfolio.cash_weight),
        )
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
        auto_visualize: bool | None = None,
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
        wandb_trace_rows: list[dict[str, Any]] = []
        if resume_from_week:
            checkpoint = preloaded_checkpoint or self._load_checkpoint(run_id=run_id, completed_week=resume_from_week)
            run_meta = self._load_run_meta(run_id)
            next_resume_week = next(
                (
                    week
                    for week in all_week_starts
                    if week > resume_from_week and (resume_to_week is None or week <= resume_to_week)
                ),
                resume_from_week,
            )
            should_refresh_latest = resume_latest or (
                str(run_meta.get("latest_completed_week", "") or "") == resume_from_week
            )
            repaired = self._repair_checkpoint_selected_etfs(
                run_id=run_id,
                completed_week=resume_from_week,
                payload=checkpoint,
                target_week=next_resume_week,
                write_latest=should_refresh_latest,
            )
            if repaired:
                logger.warning(
                    "[Resume ETF Repair] run_id={} completed_week={} target_week={} repaired_selected_etfs={}",
                    run_id,
                    resume_from_week,
                    next_resume_week,
                    ",".join(repaired),
                )
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
            wandb_trace_rows = [_result_to_wandb_row(item) for item in results]
            logger.info(
                "Loaded checkpoint for run_id={} completed_week={} results={} nav={:.2f}",
                run_id,
                resume_from_week,
                len(results),
                portfolio.total_value,
            )
            logger.info(
                "Resume summary: run_id={} local_checkpoint={} wandb_run_id={} prior_results={} latest_completed_week={} latest_total_value={} latest_weekly_return={} latest_cash_weight={}",
                run_id,
                self._checkpoint_path(run_id, resume_from_week),
                run_meta.get("wandb_run_id", "-"),
                len(results),
                run_meta.get("latest_completed_week", resume_from_week),
                run_meta.get("latest_total_value", portfolio.total_value),
                run_meta.get("latest_weekly_return", last_week_return),
                run_meta.get("latest_cash_weight", portfolio.cash_weight),
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
            trading_window = self._get_week_trading_window(etf_prices, week_start)
            week_has_trading = not trading_window["market_closed_week"]
            logger.info(
                "[Week Trading Window] week={} trading_day_count={} first_trading_day={} last_trading_day={} market_closed_week={}",
                week_start,
                trading_window["trading_day_count"],
                trading_window["first_trading_day"] or "-",
                trading_window["last_trading_day"] or "-",
                trading_window["market_closed_week"],
            )
            if trading_window["market_closed_week"]:
                logger.warning(
                    "[Week Holiday Window] week={} has no trading sessions in the Monday-Friday bucket; treating as market-closed week",
                    week_start,
                )
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
            if decisions and not week_has_trading:
                week_end = (datetime.strptime(week_start, "%Y-%m-%d") + timedelta(days=6)).strftime("%Y-%m-%d")
                logger.warning(
                    "[Week Market Closed] week={} no ETF trading rows between {} and {}; skipping decision execution",
                    week_start,
                    week_start,
                    week_end,
                )
            elif decisions:
                formatted = [d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in decisions]
                portfolio.apply_decisions(formatted)
                if etf_prices is not None:
                    repaired = portfolio.repair_selected_etfs_from_price_map(
                        etf_prices,
                        week_start,
                        self._meta_sector_etf_code_map,
                    )
                    if repaired:
                        logger.warning(
                            "[Week ETF Repair] week={} repaired_selected_etfs_from_prices={}",
                            week_start,
                            ",".join(repaired),
                        )
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
            if week_has_trading and etf_prices is not None and portfolio.invested_weight > 0:
                repaired = portfolio.repair_selected_etfs_from_price_map(
                    etf_prices,
                    week_start,
                    self._meta_sector_etf_code_map,
                )
                if repaired:
                    logger.warning(
                        "[Week ETF Repair] week={} repaired_selected_etfs_from_prices={}",
                        week_start,
                        ",".join(repaired),
                    )
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
                        "week/total_value": portfolio.total_value,
                        "week/market_closed_week": int(trading_window["market_closed_week"]),
                        "week/trading_day_count": int(trading_window["trading_day_count"]),
                        "week/invested_weight": portfolio.invested_weight,
                        "week/cash_weight": portfolio.cash_weight,
                        "week/decision_count": len(decisions),
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
                last_error=current_error,
                market_closed_week=bool(trading_window["market_closed_week"]),
                trading_day_count=int(trading_window["trading_day_count"]),
                first_trading_day=str(trading_window["first_trading_day"]),
                last_trading_day=str(trading_window["last_trading_day"]),
            )
            results.append(record)
            wandb_trace_rows.append(_result_to_wandb_row(record))
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
        visualization_result = None
        should_visualize = (
            bool(getattr(self.config.backtest, "auto_visualize", False))
            if auto_visualize is None
            else auto_visualize
        )
        if should_visualize:
            try:
                visualization_result = visualize_backtest(
                    results_path=self._backtest_results_path(run_id),
                    metrics_path=self._backtest_metrics_path(run_id),
                    output_dir=self._checkpoint_run_dir(run_id) / "visualizations",
                    run_id=run_id,
                )
                logger.info("Backtest visualization report saved to {}", visualization_result.report_path)
            except Exception as exc:
                logger.warning("Backtest visualization failed for run_id={}: {}", run_id, exc)
        wandb_handler = WandbRegistry.get("backtest")
        if wandb_handler is not None:
            summary_payload = {
                **metrics,
                "latest_total_value": float(portfolio.total_value),
                "latest_weekly_return": float(last_week_return),
                "latest_cash_weight": float(portfolio.cash_weight),
            }
            wandb_handler.log_summary(summary_payload)
            wandb_handler.log_table_rows("backtest/weekly_trace", wandb_trace_rows, step=len(wandb_trace_rows))
            wandb_handler.log_artifact(
                self._backtest_results_path(run_id),
                name=f"{run_id}_backtest_results",
                artifact_type="dataset",
                aliases=["latest"],
            )
            if visualization_result is not None:
                if hasattr(wandb_handler, "add_tags"):
                    wandb_handler.add_tags(["visualized"])
                images = {
                    f"backtest/visualizations/{path.stem}": path
                    for path in getattr(visualization_result, "image_paths", [])
                }
                captions = {
                    key: f"{run_id} {path.stem.replace('_', ' ')}"
                    for key, path in images.items()
                }
                wandb_handler.log_images(
                    images,
                    captions=captions,
                    gallery_key="backtest_visualizations",
                    step=len(wandb_trace_rows),
                )
                if hasattr(wandb_handler, "metadata"):
                    self._update_run_meta(run_id, **wandb_handler.metadata())
            wandb_handler.log_artifact(
                self._backtest_metrics_path(run_id),
                name=f"{run_id}_backtest_metrics",
                artifact_type="dataset",
                aliases=["latest"],
            )
        logger.info("=" * 60)
        logger.info("Backtest Results run_id={}", run_id)
        logger.info(
            "  trading_weeks: {} / {} (market_closed_weeks={})",
            int(metrics.get("weeks", 0)) - int(metrics.get("market_closed_weeks", 0)),
            int(metrics.get("weeks", 0)),
            int(metrics.get("market_closed_weeks", 0)),
        )
        for k, v in metrics.items():
            logger.info("  {}: {}", k, v)
        logger.info("=" * 60)

        return results_df
