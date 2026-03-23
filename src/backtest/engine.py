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

import uuid
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm import tqdm

from src.backtest.metrics import calculate_metrics
from src.backtest.portfolio import Portfolio
from src.config import AgentRootConfig
from src.utils.industry_map import IndustryMapper


class WalkForwardEngine:
    """Walk-forward backtesting engine that runs weekly."""

    def __init__(self, config: AgentRootConfig, checkpoint_dir: Path | None = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.mapper = IndustryMapper(
            dict_path=config.data.industry_dict,
            etf_info=config.data.etf_info,
            best_etf_path=config.data.etf_info.parent.parent / "best_etf_by_index.parquet",
        )
        self._etf_prices: pl.DataFrame | None = None

        # _industry_etf_code_map: large_cat → list of ETF codes (not tracking index names)
        # Built by resolving each tracking index → best_etf_code via mapper
        self._industry_etf_code_map: dict[str, list[str]] = {}
        for industry in self.mapper.industries:
            tracking_indices = self.mapper.industry_etfs(industry)
            codes = []
            for idx in tracking_indices:
                code = self.mapper.best_etf_code(idx)
                if code:
                    codes.append(code)
            self._industry_etf_code_map[industry] = codes

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

    def run(
        self,
        start_date: str,
        end_date: str,
        run_id: str | None = None,
        agent_workflow=None,
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

        logger.info("Starting weekly backtest {} → {}, run_id={}", start_date, end_date, run_id)

        portfolio = Portfolio(
            initial_capital=self.config.backtest.initial_capital,
            transaction_fee=self.config.backtest.transaction_fee,
            slippage=self.config.backtest.slippage,
        )

        week_starts = self._get_week_starts(start_date, end_date)
        logger.info("Total weeks: {}", len(week_starts))

        etf_prices = self._load_etf_prices()

        # Persistent state passed to agent for behavioural memory
        last_week_return = 0.0
        last_week_holdings: dict[str, float] = {}

        # Agent reasoning / decisions for the PREVIOUS week (recorded with THIS week's return)
        prev_observations: dict = {}
        prev_agent_decisions: list[dict] = []

        results = []
        for week_start in tqdm(week_starts, desc="Backtesting weeks"):
            # ── Step 1: Compute last week's return on EXISTING holdings ──────────
            # This return is based on the PREVIOUS iteration's applied decisions.
            # For week 1, holdings are empty → return = 0.
            weekly_return = 0.0
            industry_contributions: dict[str, float] = {}
            if etf_prices is not None and portfolio.invested_weight > 0:
                weekly_return, industry_contributions = portfolio.compute_weekly_return(
                    etf_prices, week_start, self._industry_etf_code_map
                )
                portfolio.update_nav(weekly_return)

            # Record last week's result (before new decisions overwrite holdings)
            # observations + agent_decisions come from the PREVIOUS iteration's agent call
            record = portfolio.record_state(
                week_start,
                weekly_return,
                industry_contributions,
                run_id=run_id,
                observations=prev_observations,
                agent_decisions=prev_agent_decisions,
            )
            results.append(record)

            # ── Step 2: Agent decides for THIS week ─────────────────────────────
            # Uses last_week_return + last_week_holdings as behavioural memory.
            # News/signals for week T are based on T-1 data (T+1 execution, no look-ahead).
            decisions = []
            current_observations: dict = {}
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
                }
                try:
                    result = agent_workflow.invoke(state)
                    decisions = result.get("decisions", [])
                    current_observations = result.get("observations", {})
                    logger.debug(
                        "[Agent] week={} decisions={} observations_keys={}",
                        week_start,
                        len(decisions),
                        list(current_observations.keys()),
                    )
                except Exception as e:
                    logger.error("Agent workflow failed for week {}: {}", week_start, e)

            # ── Step 3: Apply decisions — ONLY via apply_decisions() ───────────
            # NO direct overwrite of portfolio.holdings or portfolio.total_value.
            # apply_decisions() handles 摩擦成本, 滑点, and target normalization.
            if decisions:
                formatted = [d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in decisions]
                portfolio.apply_decisions(formatted)

            # Update memory for next iteration
            last_week_return = weekly_return
            last_week_holdings = dict(portfolio.holdings)

            # Save THIS week's agent output to be recorded NEXT iteration
            prev_observations = current_observations
            prev_agent_decisions = [
                d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in decisions
            ]

        # Final flush: compute return for the last applied holdings (no agent call after)
        if etf_prices is not None and portfolio.invested_weight > 0 and week_starts:
            last_ws = week_starts[-1]
            # Advance to next week to capture the return on last week's holdings
            next_week = datetime.strptime(last_ws, "%Y-%m-%d") + timedelta(weeks=1)
            next_week_int = int(next_week.strftime("%Y%m%d"))
            tail = etf_prices.filter(pl.col("trade_dt") >= next_week_int)
            if len(tail) > 0:
                # Compute return on last holdings using next available week prices
                final_return, final_contrib = portfolio.compute_weekly_return(
                    etf_prices,
                    next_week.strftime("%Y-%m-%d"),
                    self._industry_etf_code_map,
                )
                portfolio.update_nav(final_return)
                # Patch last record with final return + last week's agent output
                if results:
                    last_record = results[-1].copy()
                    last_record["weekly_return"] = final_return
                    last_record["industry_contributions"] = final_contrib
                    last_record["nav"] = portfolio.total_value
                    # Last week's agent reasoning was stored after the loop iteration
                    last_record["observations"] = prev_observations
                    last_record["agent_decisions"] = prev_agent_decisions
                    results[-1] = last_record

        results_df = pl.DataFrame(results)
        output_path = self.config.data.output_backtest
        results_df.write_parquet(output_path)
        logger.info("Backtest saved to {}", output_path)

        metrics = calculate_metrics(results_df, risk_free_rate=self.config.backtest.risk_free_rate)
        logger.info("=" * 60)
        logger.info("Backtest Results run_id={}", run_id)
        for k, v in metrics.items():
            logger.info("  {}: {}", k, v)
        logger.info("=" * 60)

        return results_df
