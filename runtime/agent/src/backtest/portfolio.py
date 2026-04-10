"""Portfolio management for weekly backtesting — with explicit cash and behavioural memory."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl


class Portfolio:
    """Portfolio manager with explicit total_value, cash weight, and meta-sector attribution."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        transaction_fee: float = 0.0003,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.total_value = initial_capital  # 总资产净值 = cash + holdings市值
        self.holdings: dict[str, float] = {}  # meta_sector -> weight (sum <= 1.0)
        self.selected_etfs: dict[str, str] = {}  # meta_sector -> ETF code
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

    def _extract_etf_code(self, raw_value: str) -> str:
        if not raw_value:
            return ""
        normalized = str(raw_value).replace("|", " ").replace(",", " ").replace("，", " ")
        for token in normalized.split():
            token = token.strip()
            if not token:
                continue
            if any(ch.isdigit() for ch in token):
                return token
        return normalized.split()[0] if normalized.split() else ""

    def _resolve_price_code(self, raw_value: str, available_codes: set[str]) -> str:
        token = self._extract_etf_code(raw_value)
        if not token:
            return ""
        if token in available_codes:
            return token
        base = token.split(".", 1)[0]
        matches = sorted(code for code in available_codes if code.split(".", 1)[0] == base)
        if len(matches) == 1:
            return matches[0]
        if f"{base}.SH" in available_codes:
            return f"{base}.SH"
        if f"{base}.SZ" in available_codes:
            return f"{base}.SZ"
        return ""

    def selected_etf_coverage(self) -> tuple[int, int, list[str]]:
        """Return active holding count, covered count, and sectors missing ETF mapping."""
        active_sectors = sorted(self.holdings)
        missing = [sector for sector in active_sectors if not self.selected_etfs.get(sector)]
        return len(active_sectors), len(active_sectors) - len(missing), missing

    def repair_selected_etfs_from_price_map(
        self,
        etf_prices: pl.DataFrame,
        week_start: str,
        sector_etf_code_map: dict[str, list[str]],
    ) -> list[str]:
        """Recover or normalize selected ETF codes using tradable weekly price data."""
        week_start_dt = datetime.strptime(week_start, "%Y-%m-%d")
        next_week_int = int((week_start_dt + timedelta(days=7)).strftime("%Y%m%d"))
        week_start_int = int(week_start_dt.strftime("%Y%m%d"))
        week_df = etf_prices.filter((pl.col("trade_dt") >= week_start_int) & (pl.col("trade_dt") < next_week_int))
        prev_day_df = etf_prices.filter(pl.col("trade_dt") < week_start_int)
        if len(week_df) == 0 or len(prev_day_df) == 0:
            return []

        last_day = week_df["trade_dt"].max()
        prev_last = prev_day_df["trade_dt"].max()
        if last_day is None or prev_last is None:
            return []

        last_prices = week_df.filter(pl.col("trade_dt") == last_day).rename({"close": "close_curr"})
        prev_prices = etf_prices.filter(pl.col("trade_dt") == prev_last).rename({"close": "close_prev"})
        merged = last_prices.join(prev_prices, on="Code", how="inner")
        if len(merged) == 0:
            return []

        available_codes = set(merged["Code"].to_list())
        repaired: list[str] = []
        for sector in sorted(self.holdings):
            existing_raw = self.selected_etfs.get(sector, "")
            resolved_existing = self._resolve_price_code(existing_raw, available_codes)
            if resolved_existing:
                if existing_raw != resolved_existing:
                    self.selected_etfs[sector] = resolved_existing
                    repaired.append(f"{sector}->{resolved_existing}")
                continue

            for code in sector_etf_code_map.get(sector, []):
                resolved = self._resolve_price_code(code, available_codes)
                if not resolved:
                    continue
                self.selected_etfs[sector] = resolved
                repaired.append(f"{sector}->{resolved}")
                break
        return repaired

    def repair_missing_selected_etfs(
        self,
        *,
        resolver,
        week_start: str,
        mapper,
    ) -> list[str]:
        """Backfill missing selected_etf mappings for currently held sectors.

        This primarily protects resumed runs / legacy checkpoints where holdings
        may exist but the ETF mapping was absent or previously malformed.
        """
        if resolver is None:
            return []
        repaired: list[str] = []
        for sector in sorted(self.holdings):
            existing = self.selected_etfs.get(sector, "")
            if existing:
                continue
            candidate = resolver.fallback_candidate_for_meta_sector(
                sector,
                week_start=week_start,
                mapper=mapper,
            )
            if candidate is None:
                continue
            self.selected_etfs[sector] = candidate.code
            repaired.append(f"{sector}->{candidate.code}")
        return repaired

    def inspect_price_availability(
        self,
        etf_prices: pl.DataFrame,
        week_start: str,
    ) -> dict[str, object]:
        """Inspect current holdings for ETF mapping and price availability issues."""
        week_start_dt = datetime.strptime(week_start, "%Y-%m-%d")
        next_week_int = int((week_start_dt + timedelta(days=7)).strftime("%Y%m%d"))
        week_start_int = int(week_start_dt.strftime("%Y%m%d"))
        week_df = etf_prices.filter((pl.col("trade_dt") >= week_start_int) & (pl.col("trade_dt") < next_week_int))
        prev_day_df = etf_prices.filter(pl.col("trade_dt") < week_start_int)

        if len(week_df) == 0:
            return {
                "missing_selected_etfs": sorted(self.holdings),
                "missing_price_sectors": sorted(self.holdings),
                "week_trade_dt": None,
                "prev_trade_dt": None,
            }

        prev_last = prev_day_df["trade_dt"].max() if len(prev_day_df) > 0 else None
        if prev_last is None:
            return {
                "missing_selected_etfs": sorted(self.holdings),
                "missing_price_sectors": sorted(self.holdings),
                "week_trade_dt": int(week_df["trade_dt"].max()),
                "prev_trade_dt": None,
            }

        last_day = int(week_df["trade_dt"].max())
        last_prices = week_df.filter(pl.col("trade_dt") == last_day).rename({"close": "close_curr"})
        prev_prices = etf_prices.filter(pl.col("trade_dt") == prev_last).rename({"close": "close_prev"})
        merged = last_prices.join(prev_prices, on="Code", how="inner")
        available_codes = set(merged["Code"].to_list())

        missing_selected_etfs: list[str] = []
        missing_price_sectors: list[str] = []
        for sector in sorted(self.holdings):
            chosen_code = self.selected_etfs.get(sector, "")
            if not chosen_code:
                missing_selected_etfs.append(sector)
                continue
            resolved_code = self._resolve_price_code(chosen_code, available_codes)
            matched = merged.filter(pl.col("Code").cast(pl.Utf8) == resolved_code)
            if len(matched) == 0:
                missing_price_sectors.append(sector)

        return {
            "missing_selected_etfs": missing_selected_etfs,
            "missing_price_sectors": missing_price_sectors,
            "week_trade_dt": last_day,
            "prev_trade_dt": int(prev_last),
        }

    def apply_decisions(self, decisions: list[dict[str, Any]]) -> float:
        """更新持仓，直接在 total_value 上扣除摩擦成本。

        Returns:
            本次换手的总交易成本
        """
        target: dict[str, float] = {}
        total_w = 0.0
        target_selected_etfs: dict[str, str] = {}

        meta_decision = decisions[0] if decisions else {}
        if meta_decision.get("level1_plan"):
            level2_map = {
                item.get("meta_sector", ""): self._extract_etf_code(item.get("selected_etf", ""))
                for item in meta_decision.get("level2_plan", [])
            }
            for item in meta_decision.get("level1_plan", []):
                action = item.get("action", "hold")
                meta_sector = item.get("meta_sector", "")
                weight = max(0.0, float(item.get("weight", 0.0)))
                if action == "sell" or not meta_sector:
                    continue
                if action in ("buy", "hold") and weight > 0:
                    target[meta_sector] = weight
                    total_w += weight
                    selected = level2_map.get(meta_sector) or self.selected_etfs.get(meta_sector, "")
                    if action == "buy" and not selected:
                        raise ValueError(f"Missing selected_etf for buy sector '{meta_sector}'")
                    if action == "hold" and not selected:
                        raise ValueError(f"Missing selected_etf for hold sector '{meta_sector}'")
                    if selected:
                        target_selected_etfs[meta_sector] = selected
        else:
            for d in decisions:
                action = d["action"]
                industry = d["industry"]
                w = max(0.0, float(d.get("weight", 0.0)))

                if action == "sell":
                    continue
                if action in ("buy", "hold") and w > 0:
                    target[industry] = w
                    total_w += w
                    selected = self._extract_etf_code(d.get("selected_etf", "")) or self.selected_etfs.get(industry, "")
                    if action == "buy" and not selected:
                        raise ValueError(f"Missing selected_etf for buy sector '{industry}'")
                    if action == "hold" and not selected:
                        raise ValueError(f"Missing selected_etf for hold sector '{industry}'")
                    if selected:
                        target_selected_etfs[industry] = selected

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
        self.selected_etfs = {
            k: v for k, v in target_selected_etfs.items() if k in self.holdings and v
        }

        return trade_cost

    def compute_weekly_return(
        self,
        etf_prices: pl.DataFrame,
        week_start: str,
        sector_etf_code_map: dict[str, list[str]],
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """计算本周收益率及每个元板块的贡献度。

        ETF price columns: Code (ETF code), trade_dt (int YYYYMMDD), close

        Returns:
            (total_return, sector_contributions, sector_returns)
        """
        self.repair_selected_etfs_from_price_map(etf_prices, week_start, sector_etf_code_map)
        diagnostics = self.inspect_price_availability(etf_prices, week_start)
        missing_selected_etfs = diagnostics["missing_selected_etfs"]
        missing_price_sectors = diagnostics["missing_price_sectors"]
        if missing_selected_etfs:
            raise ValueError(f"Missing selected_etf for held sectors: {', '.join(missing_selected_etfs)}")
        if missing_price_sectors:
            raise ValueError(f"Missing ETF price data for held sectors: {', '.join(missing_price_sectors)}")

        # Convert YYYY-MM-DD string to int YYYYMMDD for trade_dt comparison
        week_start_dt = datetime.strptime(week_start, "%Y-%m-%d")
        next_week_int = int((week_start_dt + timedelta(days=7)).strftime("%Y%m%d"))
        week_start_int = int(week_start_dt.strftime("%Y%m%d"))

        week_df = etf_prices.filter((pl.col("trade_dt") >= week_start_int) & (pl.col("trade_dt") < next_week_int))
        if len(week_df) == 0:
            raise ValueError(f"No ETF price rows found on or after {week_start}")

        last_day = week_df["trade_dt"].max()
        prev_day_df = etf_prices.filter(pl.col("trade_dt") < week_start_int)
        prev_last = prev_day_df["trade_dt"].max() if len(prev_day_df) > 0 else None
        if prev_last is None:
            raise ValueError(f"No ETF price rows found before {week_start}")

        last_prices = week_df.filter(pl.col("trade_dt") == last_day).rename({"close": "close_curr"})
        prev_prices = etf_prices.filter(pl.col("trade_dt") == prev_last).rename({"close": "close_prev"})
        # Join on ETF code column (Code in raw data)
        merged = last_prices.join(prev_prices, on="Code", how="inner")
        available_codes = set(merged["Code"].to_list())

        if len(merged) == 0:
            return 0.0, {}, {}

        merged = merged.with_columns(
            ((pl.col("close_curr") - pl.col("close_prev")) / pl.col("close_prev")).alias("etf_return")
        )

        sector_contributions: dict[str, float] = {}
        sector_returns: dict[str, float] = {}
        invested_return = 0.0

        for sector, weight in self.holdings.items():
            chosen_code = self.selected_etfs.get(sector)
            resolved_selected = self._resolve_price_code(chosen_code or "", available_codes)
            fallback_codes = [
                resolved
                for resolved in (self._resolve_price_code(code, available_codes) for code in sector_etf_code_map.get(sector, []))
                if resolved
            ]
            if not resolved_selected and fallback_codes:
                resolved_selected = fallback_codes[0]
                self.selected_etfs[sector] = resolved_selected
            etf_codes = [resolved_selected] if resolved_selected else []
            etf_rets = merged.filter(pl.col("Code").cast(pl.Utf8).is_in(etf_codes))["etf_return"].to_list()
            if etf_rets:
                sector_return = sum(etf_rets) / len(etf_rets)
                contribution = weight * sector_return
                sector_returns[sector] = sector_return
                sector_contributions[sector] = contribution
                invested_return += contribution
            else:
                raise ValueError(f"Missing joined ETF prices for held sector '{sector}'")

        # 总收益 = 持仓收益 + 现金收益(通常为0)
        cash_return = self.cash_weight * 0.0
        total_return = invested_return + cash_return

        return total_return, sector_contributions, sector_returns

    def update_nav(self, weekly_return: float) -> None:
        self.total_value *= (1.0 + weekly_return)

    def settle_week(self, sector_returns: dict[str, float], total_return: float) -> None:
        """Mark holdings to market so next week's rebalance starts from drifted weights."""
        denominator = 1.0 + total_return
        if denominator <= 0:
            raise ValueError(f"Invalid weekly return leading to non-positive NAV multiplier: {total_return}")

        updated_holdings: dict[str, float] = {}
        for sector, weight in self.holdings.items():
            sector_return = float(sector_returns.get(sector, 0.0))
            new_weight = weight * (1.0 + sector_return) / denominator
            if new_weight > 0.001:
                updated_holdings[sector] = new_weight

        self.holdings = updated_holdings
        self.selected_etfs = {
            sector: code for sector, code in self.selected_etfs.items() if sector in self.holdings and code
        }

    def record_state(
        self,
        week_start: str,
        weekly_return: float,
        sector_contributions: dict[str, float],
        run_id: str = "default",
        observations: dict | None = None,
        agent_decisions: list[dict] | None = None,
        sector_returns: dict[str, float] | None = None,
        last_error: str = "",
        market_closed_week: bool = False,
        trading_day_count: int = 0,
        first_trading_day: str = "",
        last_trading_day: str = "",
    ) -> dict:
        """记录本周状态，供 Agent 后续复盘（行为记忆）。

        observations: agent reasoning / tool outputs from the research loop.
        agent_decisions: the raw trade decisions output by the trader.
        """
        return {
            "run_id": run_id,
            "week_start": week_start,
            "initial_capital": self.initial_capital,
            "nav": self.total_value,
            "total_value": self.total_value,
            "weekly_return": weekly_return,
            "market_closed_week": market_closed_week,
            "trading_day_count": int(trading_day_count),
            "first_trading_day": first_trading_day,
            "last_trading_day": last_trading_day,
            "invested_weight": self.invested_weight,
            "cash_weight": self.cash_weight,
            "holdings": self.holdings.copy(),
            "selected_etfs": self.selected_etfs.copy(),
            "meta_sector_contributions": sector_contributions.copy(),
            "meta_sector_returns": (sector_returns or {}).copy(),
            "industry_contributions": sector_contributions.copy(),
            "status": "PROFIT" if weekly_return > 0 else "LOSS",
            "cumulative_return": (self.total_value - self.initial_capital) / self.initial_capital,
            "observations": observations or {},
            "agent_decisions": agent_decisions or [],
            "last_error": last_error,
        }

    def snapshot(self) -> dict[str, Any]:
        """Serialize portfolio state for checkpointing."""
        return {
            "initial_capital": self.initial_capital,
            "total_value": self.total_value,
            "holdings": self.holdings.copy(),
            "selected_etfs": self.selected_etfs.copy(),
            "transaction_fee": self.transaction_fee,
            "slippage": self.slippage,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        """Restore portfolio state from checkpoint payload."""
        self.initial_capital = float(payload.get("initial_capital", self.initial_capital))
        self.total_value = float(payload.get("total_value", self.total_value))
        self.holdings = {str(k): float(v) for k, v in dict(payload.get("holdings", {})).items()}
        self.selected_etfs = {str(k): str(v) for k, v in dict(payload.get("selected_etfs", {})).items()}
        self.transaction_fee = float(payload.get("transaction_fee", self.transaction_fee))
        self.slippage = float(payload.get("slippage", self.slippage))
