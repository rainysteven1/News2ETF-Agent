"""Portfolio management for weekly backtesting — with explicit cash and behavioural memory."""

from __future__ import annotations

from typing import Any

import polars as pl


class Portfolio:
    """Portfolio manager with explicit total_value, cash weight, and per-industry attribution."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        transaction_fee: float = 0.0003,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.total_value = initial_capital  # 总资产净值 = cash + holdings市值
        self.holdings: dict[str, float] = {}  # industry -> weight (sum <= 1.0)
        self.transaction_fee = transaction_fee
        self.slippage = slippage

    @property
    def nav(self) -> float:
        return self.total_value

    @property
    def invested_weight(self) -> float:
        return sum(self.holdings.values())

    @property
    def cash_weight(self) -> float:
        return 1.0 - self.invested_weight

    def apply_decisions(self, decisions: list[dict[str, Any]]) -> float:
        """更新持仓，直接在 total_value 上扣除摩擦成本。

        Returns:
            本次换手的总交易成本
        """
        # Build target holdings from decisions
        target: dict[str, float] = {}
        total_w = 0.0

        for d in decisions:
            action = d["action"]
            industry = d["industry"]
            w = max(0.0, float(d.get("weight", 0.0)))

            if action == "sell":
                continue  # sell → weight 0
            if action in ("buy", "hold") and w > 0:
                target[industry] = w
                total_w += w

        # 归一化，防止开杠杆
        if total_w > 1.0:
            target = {k: v / total_w for k, v in target.items()}

        # 计算换手率
        all_industries = set(self.holdings) | set(target)
        turnover = sum(
            abs(target.get(i, 0.0) - self.holdings.get(i, 0.0))
            for i in all_industries
        )

        # 摩擦成本直接在净值上扣除
        trade_cost = turnover * (self.transaction_fee + self.slippage)
        self.total_value *= (1.0 - trade_cost)

        # 更新持仓
        self.holdings = {k: v for k, v in target.items() if v > 0.001}

        return trade_cost

    def compute_weekly_return(
        self,
        etf_prices: pl.DataFrame,
        week_start: str,
        industry_etf_code_map: dict[str, list[str]],
    ) -> tuple[float, dict[str, float]]:
        """计算本周收益率及每个行业的贡献度。

        ETF price columns: Code (ETF code), trade_dt (int YYYYMMDD), close

        Returns:
            (total_return, industry_contributions)
            industry_contributions = {industry: weight * industry_return}
        """
        # Convert YYYY-MM-DD string to int YYYYMMDD for trade_dt comparison
        week_start_int = int(week_start.replace("-", ""))

        week_df = etf_prices.filter(pl.col("trade_dt") >= week_start_int)
        if len(week_df) == 0:
            return 0.0, {}

        last_day = week_df["trade_dt"].max()
        prev_day_df = etf_prices.filter(pl.col("trade_dt") < week_start_int)
        prev_last = prev_day_df["trade_dt"].max() if len(prev_day_df) > 0 else None
        if prev_last is None:
            return 0.0, {}

        last_prices = week_df.filter(pl.col("trade_dt") == last_day).rename({"close": "close_curr"})
        prev_prices = etf_prices.filter(pl.col("trade_dt") == prev_last).rename({"close": "close_prev"})
        # Join on ETF code column (Code in raw data)
        merged = last_prices.join(prev_prices, on="Code", how="inner")

        if len(merged) == 0:
            return 0.0, {}

        merged = merged.with_columns(
            ((pl.col("close_curr") - pl.col("close_prev")) / pl.col("close_prev")).alias("etf_return")
        )

        industry_contributions: dict[str, float] = {}
        invested_return = 0.0

        for industry, weight in self.holdings.items():
            etf_codes = industry_etf_code_map.get(industry, [])
            etf_rets = merged.filter(pl.col("Code").is_in(etf_codes))["etf_return"].to_list()
            if etf_rets:
                ind_return = sum(etf_rets) / len(etf_rets)
                contribution = weight * ind_return
                industry_contributions[industry] = contribution
                invested_return += contribution

        # 总收益 = 持仓收益 + 现金收益(通常为0)
        cash_return = self.cash_weight * 0.0
        total_return = invested_return + cash_return

        return total_return, industry_contributions

    def update_nav(self, weekly_return: float) -> None:
        self.total_value *= (1.0 + weekly_return)

    def record_state(
        self,
        week_start: str,
        weekly_return: float,
        industry_contributions: dict[str, float],
        run_id: str = "default",
        observations: dict | None = None,
        agent_decisions: list[dict] | None = None,
    ) -> dict:
        """记录本周状态，供 Agent 后续复盘（行为记忆）。

        observations: agent reasoning / tool outputs from the research loop.
        agent_decisions: the raw trade decisions output by the trader.
        """
        return {
            "run_id": run_id,
            "week_start": week_start,
            "nav": self.total_value,
            "weekly_return": weekly_return,
            "invested_weight": self.invested_weight,
            "cash_weight": self.cash_weight,
            "holdings": self.holdings.copy(),
            "industry_contributions": industry_contributions.copy(),
            "status": "PROFIT" if weekly_return > 0 else "LOSS",
            "cumulative_return": (self.total_value - self.initial_capital) / self.initial_capital,
            "observations": observations or {},
            "agent_decisions": agent_decisions or [],
        }
