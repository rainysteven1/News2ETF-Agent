"""Agent Feature Builder - builds A/B/C/D/E feature sets for agent decision making.

This module provides the AgentFeatureBuilder class that constructs all features
needed for the agent to make decisions:
  - Feature A: TCN sequence (8 meta sectors × 5 days momentum)
  - Feature B: News summary (top-k news per meta sector)
  - Feature C: Market state (price momentum, volume, volatility)
  - Feature D: Position state (current holdings, weekly returns, agent performance)
  - Feature E: Sentiment vs price divergence

All features are computed using only data available before `current_time` to
prevent look-ahead bias.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from src.config import load_config


class AgentFeatureBuilder:
    """Builds all features needed for agent decision making."""

    def __init__(self, sentiment_df: pl.DataFrame | None = None, price_df: pl.DataFrame | None = None):
        """Initialize the feature builder.

        Args:
            sentiment_df: Optional pre-loaded sentiment DataFrame
            price_df: Optional pre-loaded price DataFrame
        """
        self.config = load_config()
        self._sentiment_df = sentiment_df
        self._price_df = price_df
        self._meta_sector_map = None
        self._agent_feature_df = None

    @property
    def sentiment_df(self) -> pl.DataFrame:
        """Load sentiment DataFrame if not cached."""
        if self._sentiment_df is None:
            path = self.config.data.output_sentiment
            if path and path.exists():
                self._sentiment_df = pl.read_parquet(path)
            else:
                self._sentiment_df = pl.DataFrame()
        return self._sentiment_df

    @property
    def price_df(self) -> pl.DataFrame:
        """Load price DataFrame if not cached."""
        if self._price_df is None:
            path = self.config.data.etf_prices
            if path and path.exists():
                self._price_df = pl.read_parquet(path)
                if "date" not in self._price_df.columns and "trade_dt" in self._price_df.columns:
                    self._price_df = self._price_df.with_columns(
                        pl.col("trade_dt").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").cast(pl.Utf8).alias("date")
                    )
            else:
                self._price_df = pl.DataFrame()
        return self._price_df

    @property
    def meta_sector_map(self) -> dict[str, Any]:
        """Load meta sector mapping if not cached."""
        if self._meta_sector_map is None:
            path = self.config.data.meta_sector_mapping
            if path and path.exists():
                import json

                with open(path, encoding="utf-8") as f:
                    self._meta_sector_map = json.load(f)
            else:
                self._meta_sector_map = {"meta_sectors": {}, "global_leader_map": {}}
        return self._meta_sector_map

    @property
    def agent_feature_df(self) -> pl.DataFrame:
        """Load exported agent feature parquet if available."""
        if self._agent_feature_df is None:
            path = self.config.data.output_agent_features
            if path and path.exists():
                self._agent_feature_df = pl.read_parquet(path)
                if "date" in self._agent_feature_df.columns:
                    self._agent_feature_df = self._agent_feature_df.with_columns(pl.col("date").cast(pl.Utf8))
            else:
                self._agent_feature_df = pl.DataFrame()
        return self._agent_feature_df

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        if isinstance(date_str, datetime):
            return date_str
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _get_sub_weight(self, sub_category: str) -> float:
        notes = self.meta_sector_map.get("notes", {})
        if sub_category in notes.get("核心驱动（×1.5）", []):
            return 1.5
        if sub_category in notes.get("边缘平滑（×0.5）", []):
            return 0.5
        return 1.0

    def _get_trading_days_before(self, date: str | datetime, lookback: int = 5) -> list[str]:
        """Get the last `lookback` trading days before the given date.

        Args:
            date: Reference date
            lookback: Number of trading days to look back

        Returns:
            List of date strings in YYYY-MM-DD format
        """
        if isinstance(date, str):
            date = self._parse_date(date)

        df = self.sentiment_df
        if len(df) == 0:
            return []

        df = df.with_columns(pl.col("date").cast(str))
        dates = df["date"].unique().sort().to_list()

        # Filter dates before the reference date
        date_str = date.strftime("%Y-%m-%d")
        valid_dates = [d for d in dates if d < date_str]

        # Return last `lookback` dates
        return valid_dates[-lookback:] if len(valid_dates) >= lookback else valid_dates

    def build_tcn_sequence(self, date: str | datetime, lookback: int = 5) -> dict[str, list[float]]:
        """Build TCN sequence: 8 meta sectors × 5 days momentum.

        Args:
            date: Current decision date
            lookback: Number of days to look back (default 5)

        Returns:
            Dict mapping meta sector name to list of 5 daily momentum values
        """
        meta_sector_map = self.meta_sector_map
        meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
        agent_df = self.agent_feature_df

        if len(agent_df) > 0:
            date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
            hist = agent_df.filter(pl.col("date") < date_str).sort("date").tail(lookback)
            if len(hist) > 0:
                result = {ms: [] for ms in meta_sectors}
                for row in hist.iter_rows(named=True):
                    for ms in meta_sectors:
                        result[ms].append(float(row.get(f"tcn_reg_{ms}", 0.0) or 0.0))
                for ms in meta_sectors:
                    if len(result[ms]) < lookback:
                        result[ms] = [0.0] * (lookback - len(result[ms])) + result[ms]
                return result

        trading_days = self._get_trading_days_before(date, lookback)

        result: dict[str, list[float]] = {ms: [] for ms in meta_sectors}

        for day in trading_days:
            day_data = self.sentiment_df.filter(pl.col("date") == day)
            if len(day_data) == 0:
                for ms in meta_sectors:
                    result[ms].append(0.0)
                continue

            for ms in meta_sectors:
                ms_info = meta_sector_map.get("meta_sectors", {}).get(ms, {})
                subs = ms_info.get("sub_categories", [])
                total_sent = 0.0
                total_weight = 0.0

                for sub in subs:
                    sub_rows = day_data.filter(
                        (pl.col("sub_category") == sub)
                        if "sub_category" in day_data.columns
                        else (pl.col("industry") == sub)
                    )
                    if len(sub_rows) > 0:
                        sent_col = "sentiment_mean" if "sentiment_mean" in sub_rows.columns else "sentiment_weighted"
                        sent = float(sub_rows[sent_col][0])
                        sub_weight = self._get_sub_weight(sub)
                        total_sent += sent * sub_weight
                        total_weight += sub_weight

                momentum = total_sent / total_weight if total_weight > 0 else 0.0
                result[ms].append(momentum)

        return result

    def build_news_summary(self, date: str | datetime, top_k: int = 1) -> dict[str, list[str]]:
        """Build news summary: top-k news titles per meta sector.

        Args:
            date: Current decision date
            top_k: Number of top news to include per sector

        Returns:
            Dict mapping meta sector name to list of news title strings
        """
        if isinstance(date, str):
            date = self._parse_date(date)

        # Get news for the week leading up to this date
        week_start = date - timedelta(days=7)

        # Load raw news data
        news_path = self.config.data.input_news_raw
        if not news_path or not news_path.exists():
            return {ms: [] for ms in self.meta_sector_map.get("meta_sectors", {}).keys()}

        news_df = pl.read_parquet(news_path)
        news_df = news_df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))

        # Filter to the week before the decision date
        news_df = news_df.filter((pl.col("date") >= week_start.date()) & (pl.col("date") < date.date()))

        meta_sector_map = self.meta_sector_map
        meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
        result: dict[str, list[str]] = {ms: [] for ms in meta_sectors}

        for ms in meta_sectors:
            ms_info = meta_sector_map.get("meta_sectors", {}).get(ms, {})
            subs = ms_info.get("sub_categories", [])

            for sub in subs:
                sub_news = news_df.filter(pl.col("sub_category") == sub)
                if len(sub_news) > 0:
                    # Sort by sentiment confidence and take top-k
                    sub_news = sub_news.sort(
                        "sentiment_confidence" if "sentiment_confidence" in sub_news.columns else "datetime",
                        descending=True,
                    )
                    titles = sub_news["title"].head(top_k).to_list() if "title" in sub_news.columns else []
                    result[ms].extend([str(t)[:100] for t in titles[:top_k]])

            # Keep only top_k unique titles
            result[ms] = list(dict.fromkeys(result[ms]))[:top_k]

        return result

    def build_market_state(self, date: str | datetime) -> dict[str, Any]:
        """Build market state: price momentum, volume, volatility.

        Args:
            date: Current decision date

        Returns:
            Dict with market state features
        """
        if isinstance(date, str):
            date = self._parse_date(date)

        trading_days = self._get_trading_days_before(date, 20)
        if len(trading_days) < 5:
            return {
                "market_return_1w": 0.0,
                "market_return_2w": 0.0,
                "market_volatility": 0.0,
                "volume_ratio": 1.0,
                "market_state": "neutral",
            }

        # Compute returns
        returns = []
        for i in range(1, len(trading_days)):
            prev_day = self._get_price_on_date(trading_days[i - 1])
            curr_day = self._get_price_on_date(trading_days[i])
            if prev_day > 0:
                ret = (curr_day - prev_day) / prev_day
            else:
                ret = 0.0
            returns.append(ret)

        returns = np.array(returns)

        # 1-week and 2-week return
        market_return_1w = float(np.sum(returns[-1:])) if len(returns) >= 1 else 0.0
        market_return_2w = float(np.sum(returns[-2:])) if len(returns) >= 2 else 0.0

        # Volatility (annualized)
        market_volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.0

        # Volume ratio (recent avg / historical avg)
        volume_ratio = 1.0  # Placeholder

        # Determine market state
        if market_return_1w > 0.02:
            market_state = "bullish"
        elif market_return_1w < -0.02:
            market_state = "bearish"
        else:
            market_state = "neutral"

        return {
            "market_return_1w": market_return_1w,
            "market_return_2w": market_return_2w,
            "market_volatility": market_volatility,
            "volume_ratio": volume_ratio,
            "market_state": market_state,
        }

    def _get_price_on_date(self, date_str: str) -> float:
        """Get the price on a specific date."""
        df = self.price_df
        if len(df) == 0:
            return 0.0

        day_data = df.filter(pl.col("date").cast(str) == date_str)
        if len(day_data) == 0:
            return 0.0

        # Try close price
        price_cols = [c for c in df.columns if "close" in c.lower() or "CLOSE" in c.upper()]
        if price_cols:
            return float(day_data[price_cols[0]][0])
        return 0.0

    def _get_index_data(self, dates: list[str]) -> dict[str, float]:
        """Get index data for given dates."""
        return {d: self._get_price_on_date(d) for d in dates}

    def build_position_state(
        self,
        current_holdings: dict[str, float],
        weekly_returns: dict[str, float],
        agent_perf_1w: float,
        agent_perf_4w: float,
    ) -> dict[str, Any]:
        """Build position state: holdings, returns, agent performance.

        Args:
            current_holdings: Dict mapping sector to weight
            weekly_returns: Dict mapping sector to weekly return
            agent_perf_1w: Agent performance over past 1 week
            agent_perf_4w: Agent performance over past 4 weeks

        Returns:
            Dict with position state features
        """
        # Sort holdings by weight
        sorted_holdings = sorted(current_holdings.items(), key=lambda x: x[1], reverse=True)
        top_holdings = sorted_holdings[:5]  # Top 5 by weight

        # Compute portfolio metrics
        total_weight = sum(current_holdings.values())
        invested_weight = sum(w for w in current_holdings.values() if w > 0.01)

        # Weekly return of portfolio
        portfolio_return_1w = sum(
            weekly_returns.get(sector, 0.0) * weight for sector, weight in current_holdings.items()
        )

        return {
            "total_weight": total_weight,
            "invested_weight": invested_weight,
            "num_positions": len([w for w in current_holdings.values() if w > 0.01]),
            "top_holdings": top_holdings,
            "portfolio_return_1w": portfolio_return_1w,
            "agent_perf_1w": agent_perf_1w,
            "agent_perf_4w": agent_perf_4w,
        }

    def build_sent_p_divergence(self, date: str | datetime) -> dict[str, float]:
        """Build sentiment vs price divergence for each meta sector.

        Args:
            date: Current decision date

        Returns:
            Dict mapping meta sector name to divergence score
        """
        if isinstance(date, str):
            date = self._parse_date(date)

        trading_days = self._get_trading_days_before(date, 10)
        meta_sector_map = self.meta_sector_map
        meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
        agent_df = self.agent_feature_df

        if len(agent_df) > 0:
            date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
            hist = agent_df.filter(pl.col("date") < date_str).sort("date").tail(1)
            if len(hist) > 0:
                row = hist.row(0, named=True)
                return {ms: float(row.get(f"sentiment_vs_price_residual_{ms}", 0.0) or 0.0) for ms in meta_sectors}

        result: dict[str, float] = {}

        for ms in meta_sectors:
            ms_info = meta_sector_map.get("meta_sectors", {}).get(ms, {})
            subs = ms_info.get("sub_categories", [])

            # Compute average sentiment over window
            sent_values = []
            price_values = []

            for day in trading_days[-5:]:
                day_data = self.sentiment_df.filter(pl.col("date") == day)
                if len(day_data) == 0:
                    continue

                day_sent = 0.0
                count = 0
                for sub in subs:
                    sub_rows = day_data.filter(
                        (pl.col("sub_category") == sub)
                        if "sub_category" in day_data.columns
                        else (pl.col("industry") == sub)
                    )
                    if len(sub_rows) > 0:
                        sent_col = "sentiment_mean" if "sentiment_mean" in sub_rows.columns else "sentiment_weighted"
                        day_sent += float(sub_rows[sent_col][0])
                        count += 1

                if count > 0:
                    sent_values.append(day_sent / count)

                # Get price data for this sector
                price = self._get_sector_price(ms, day)
                if price > 0:
                    price_values.append(price)

            # Compute divergence
            if len(sent_values) >= 3 and len(price_values) >= 3:
                sent_trend = sent_values[-1] - sent_values[0]
                price_trend = (price_values[-1] - price_values[0]) / price_values[0] if price_values[0] > 0 else 0.0

                # Sentiment up but price down = positive divergence (opportunity)
                divergence = sent_trend - price_trend
            else:
                divergence = 0.0

            result[ms] = float(divergence)

        return result

    def _get_sector_price(self, meta_sector: str, date: str) -> float:
        """Get the price for a meta sector on a specific date."""
        # Placeholder: would need to map meta sector to ETF/index
        return 0.0

    def build_agent_features(
        self,
        date: str | datetime,
        current_holdings: dict[str, float],
        current_time: str | None = None,
    ) -> dict[str, Any]:
        """Build complete agent features for decision making.

        This is the main entry point that constructs all feature sets A/B/C/D/E.

        Args:
            date: Current decision date
            current_holdings: Current portfolio holdings
            current_time: Decision timestamp (e.g., "2024-10-07 08:30:00")

        Returns:
            Dict containing all features:
              - tcn_sequence: Feature A
              - news_summary: Feature B
              - market_state: Feature C
              - position_state: Feature D
              - sent_p_divergence: Feature E
        """
        if isinstance(date, str):
            date = self._parse_date(date)

        # Parse current_time if provided
        if current_time:
            try:
                decision_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                decision_dt = date
        else:
            decision_dt = date

        # Feature A: TCN sequence
        tcn_sequence = self.build_tcn_sequence(decision_dt, lookback=5)

        # Feature B: News summary
        news_summary = self.build_news_summary(decision_dt, top_k=1)

        # Feature C: Market state
        market_state = self.build_market_state(decision_dt)

        # Feature D: Position state
        # Compute weekly returns from price data (placeholder)
        weekly_returns = {sector: 0.0 for sector in current_holdings.keys()}
        agent_perf_1w = 0.0
        agent_perf_4w = 0.0

        position_state = self.build_position_state(current_holdings, weekly_returns, agent_perf_1w, agent_perf_4w)

        # Feature E: Sentiment vs price divergence
        sent_p_divergence = self.build_sent_p_divergence(decision_dt)

        return {
            "tcn_sequence": tcn_sequence,
            "news_summary": news_summary,
            "market_state": market_state,
            "position_state": position_state,
            "sent_p_divergence": sent_p_divergence,
        }
