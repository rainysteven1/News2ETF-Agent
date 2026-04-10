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

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import best_etf_by_index_path, load_config
from src.logger import logger
from src.utils.industry_map import IndustryMapper
from src.utils.meta_sector_map import meta_to_subs
from src.utils.news_loader import load_raw_news_df


class _NullIndustryMapper:
    def get_large_cats(self) -> list[str]:
        return []

    def get_small_cats(self, large_cat: str) -> list[str]:
        return []

    def get_indices(self, large_cat: str, small_cat: str) -> list[str]:
        return []

    @property
    def industries(self) -> list[str]:
        return []

    def best_etf_codes(self, index_names: list[str]) -> list[str]:
        return []


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
        self._market_daily_df = None
        self._backtest_df = None
        self._mapper = None
        self._meta_sector_etf_codes = None

    def _ensure_string_date_column(self, df: pl.DataFrame, column: str = "date") -> pl.DataFrame:
        if len(df) == 0 or column not in df.columns:
            return df
        if df.schema.get(column) == pl.Utf8:
            return df
        return df.with_columns(pl.col(column).cast(pl.Utf8))

    def _coerce_path(self, value: Any) -> Path | None:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        return None

    @property
    def sentiment_df(self) -> pl.DataFrame:
        """Load sentiment DataFrame if not cached."""
        if self._sentiment_df is None:
            path = self.config.data.output_sentiment
            if path and path.exists():
                self._sentiment_df = pl.read_parquet(path)
            else:
                self._sentiment_df = pl.DataFrame()
        self._sentiment_df = self._ensure_string_date_column(self._sentiment_df)
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
    def mapper(self) -> IndustryMapper:
        """Load industry mapper for ETF resolution."""
        if self._mapper is None:
            dict_path = self._coerce_path(getattr(self.config.data, "industry_dict", None))
            etf_info_path = self._coerce_path(getattr(self.config.data, "etf_info", None))
            if dict_path is None or not dict_path.exists():
                self._mapper = _NullIndustryMapper()
            else:
                self._mapper = IndustryMapper(
                    dict_path=dict_path,
                    etf_info=etf_info_path,
                    best_etf_path=best_etf_by_index_path(etf_info_path),
                )
        return self._mapper

    @property
    def market_daily_df(self) -> pl.DataFrame:
        """Aggregate ETF prices into a daily market proxy."""
        if self._market_daily_df is None:
            df = self.price_df
            if len(df) == 0:
                self._market_daily_df = pl.DataFrame()
            else:
                date_col = "date" if "date" in df.columns else "trade_dt"
                close_col = next(
                    (c for c in df.columns if c.lower() == "close" or "close" in c.lower()),
                    None,
                )
                volume_col = next(
                    (
                        c
                        for c in df.columns
                        if c.lower() in {"volume", "vol", "amount"} or "成交量" in c or "成交额" in c
                    ),
                    None,
                )
                if close_col is None:
                    self._market_daily_df = pl.DataFrame()
                else:
                    agg_exprs: list[pl.Expr] = [pl.col(close_col).cast(pl.Float64).mean().alias("market_close")]
                    if volume_col is not None:
                        agg_exprs.append(pl.col(volume_col).cast(pl.Float64).sum().alias("market_volume"))
                    daily = (
                        df.with_columns(pl.col(date_col).cast(pl.Utf8).alias("date"))
                        .group_by("date")
                        .agg(agg_exprs)
                        .sort("date")
                    )
                    if "market_volume" not in daily.columns:
                        daily = daily.with_columns(pl.lit(0.0).alias("market_volume"))
                    self._market_daily_df = daily
        return self._market_daily_df

    @property
    def backtest_df(self) -> pl.DataFrame:
        """Load backtest state history if available."""
        if self._backtest_df is None:
            path = self._coerce_path(getattr(self.config.data, "output_backtest", None))
            if path and path.exists():
                self._backtest_df = pl.read_parquet(path)
                if "week_start" in self._backtest_df.columns:
                    self._backtest_df = self._backtest_df.with_columns(pl.col("week_start").cast(pl.Utf8))
            else:
                self._backtest_df = pl.DataFrame()
        return self._backtest_df

    @property
    def meta_sector_etf_codes(self) -> dict[str, list[str]]:
        """Resolve representative ETF codes for each meta sector."""
        if self._meta_sector_etf_codes is None:
            mapping: dict[str, list[str]] = {}
            for meta_sector in self.meta_sector_map.get("meta_sectors", {}).keys():
                codes: list[str] = []
                seen: set[str] = set()
                for sub in meta_to_subs(meta_sector):
                    for large_cat in self.mapper.get_large_cats():
                        if sub not in self.mapper.get_small_cats(large_cat):
                            continue
                        for code in self.mapper.best_etf_codes(self.mapper.get_indices(large_cat, sub)):
                            if code and code not in seen:
                                seen.add(code)
                                codes.append(code)
                mapping[meta_sector] = codes
            self._meta_sector_etf_codes = mapping
        return self._meta_sector_etf_codes

    def _preferred_agent_feature_path(self) -> Path | None:
        oof_path = self._coerce_path(getattr(self.config.data, "output_agent_features_oof", None))
        full_path = self._coerce_path(getattr(self.config.data, "output_agent_features", None))
        if oof_path and oof_path.exists():
            return oof_path
        if full_path and full_path.exists():
            return full_path
        return oof_path or full_path

    def _ensure_agent_feature_cache_upto(self, date: str | datetime) -> None:
        """Incrementally refresh the signals inference cache up to `date` when possible."""
        target_date = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
        path = self._preferred_agent_feature_path()
        signals_onnx_dir = self._coerce_path(getattr(self.config.predict, "signals_onnx_dir", None))
        meta_sector_mapping_path = self._coerce_path(getattr(self.config.data, "meta_sector_mapping", None))
        if path is None or signals_onnx_dir is None or not signals_onnx_dir.exists() or meta_sector_mapping_path is None:
            return

        existing = self._agent_feature_df
        if existing is None and path.exists():
            existing = pl.read_parquet(path).with_columns(pl.col("date").cast(pl.Utf8))
            self._agent_feature_df = existing

        if existing is not None and len(existing) > 0:
            max_cached = str(existing["date"].max())
            if max_cached >= target_date:
                return
            start_date = (datetime.strptime(max_cached, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            start_date = None

        try:
            from src.signals.signals_inference import SignalsONNXInferencePipeline

            pipeline = SignalsONNXInferencePipeline(
                bundle_dir=signals_onnx_dir,
                meta_sector_mapping_path=meta_sector_mapping_path,
            )
            inferred = pipeline.infer_feature_frame(
                self.sentiment_df,
                start_date=start_date,
                end_date=target_date,
            )
            if len(inferred) == 0:
                return

            if existing is not None and len(existing) > 0:
                combined = pl.concat([existing, inferred], how="diagonal_relaxed").unique(
                    subset=["date"], keep="last"
                ).sort("date")
            else:
                combined = inferred.sort("date")
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.write_parquet(path)
            self._agent_feature_df = combined
            logger.info(
                "[AgentFeatureBuilder] Refreshed agent feature cache to {} ({} rows)",
                target_date,
                len(combined),
            )
        except Exception as exc:
            logger.warning(f"[AgentFeatureBuilder] Failed to refresh agent feature cache to {target_date}: {exc}")

    @property
    def agent_feature_df(self) -> pl.DataFrame:
        """Load exported agent feature parquet if available."""
        if self._agent_feature_df is None:
            path = self._preferred_agent_feature_path()
            signals_onnx_dir = self._coerce_path(getattr(self.config.predict, "signals_onnx_dir", None))
            meta_sector_mapping_path = self._coerce_path(getattr(self.config.data, "meta_sector_mapping", None))
            if (not path or not path.exists()) and signals_onnx_dir is not None and signals_onnx_dir.exists() and meta_sector_mapping_path is not None:
                try:
                    from src.signals.signals_inference import SignalsONNXInferencePipeline

                    pipeline = SignalsONNXInferencePipeline(
                        bundle_dir=signals_onnx_dir,
                        meta_sector_mapping_path=meta_sector_mapping_path,
                    )
                    inferred = pipeline.infer_feature_frame(
                        self.sentiment_df,
                        output_path=path,
                    )
                    logger.info("[AgentFeatureBuilder] Generated agent feature cache from signals ONNX bundle")
                    self._agent_feature_df = inferred
                except Exception as exc:
                    logger.warning(f"[AgentFeatureBuilder] Failed to auto-build agent_features.parquet: {exc}")
            if path and path.exists():
                self._agent_feature_df = pl.read_parquet(path)
                self._agent_feature_df = self._ensure_string_date_column(self._agent_feature_df)
            else:
                self._agent_feature_df = pl.DataFrame()
        return self._agent_feature_df

    def build_signal_snapshot(self, date: str | datetime) -> dict[str, dict[str, float]]:
        """Build the latest per-meta-sector signal snapshot before `date`."""
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        self._ensure_agent_feature_cache_upto(date)
        agent_df = self.agent_feature_df
        meta_sectors = list(self.meta_sector_map.get("meta_sectors", {}).keys())
        if len(agent_df) == 0:
            return {ms: {} for ms in meta_sectors}

        hist = agent_df.filter(pl.col("date") < str(date)).sort("date").tail(1)
        if len(hist) == 0:
            return {ms: {} for ms in meta_sectors}

        row = hist.row(0, named=True)
        snapshot: dict[str, dict[str, float]] = {}
        for ms in meta_sectors:
            snapshot[ms] = {
                "tcn_reg": float(row.get(f"tcn_reg_{ms}", 0.0) or 0.0),
                "tcn_reg_delta": float(row.get(f"tcn_reg_delta_{ms}", 0.0) or 0.0),
                "tcn_prediction_stability": float(row.get(f"tcn_prediction_stability_{ms}", 0.0) or 0.0),
                "lgbm_score": float(row.get(f"lgbm_score_{ms}", 0.0) or 0.0),
                "news_heat": float(row.get(f"news_heat_{ms}", 0.0) or 0.0),
                "meta_sentiment": float(row.get(f"meta_sentiment_{ms}", 0.0) or 0.0),
                "global_leader_sentiment": float(row.get(f"global_leader_sentiment_{ms}", 0.0) or 0.0),
                "sentiment_vs_price_residual": float(row.get(f"sentiment_vs_price_residual_{ms}", 0.0) or 0.0),
            }
        return snapshot

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

    def _sub_keywords(self, sub_category: str) -> list[str]:
        keywords: list[str] = []
        for token in re.split(r"[/\s,，]+", str(sub_category or "")):
            token = token.strip()
            if not token:
                continue
            keywords.append(token)
            if token == "半导体":
                keywords.append("芯片")
            elif token == "芯片":
                keywords.append("半导体")
            elif token == "通信":
                keywords.append("5G")
        return list(dict.fromkeys(keywords))

    def _build_news_summary_from_raw_keywords(
        self,
        news_df: pl.DataFrame,
        meta_sectors: list[str],
        top_k: int,
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {ms: [] for ms in meta_sectors}
        scored: dict[str, list[tuple[int, str]]] = defaultdict(list)

        for row in news_df.sort("datetime", descending=True).iter_rows(named=True):
            title = str(row.get("title") or row.get("content") or "").strip()
            if not title:
                continue
            haystack = f"{row.get('title', '')} {row.get('content', '')}"
            for ms in meta_sectors:
                subs = self.meta_sector_map.get("meta_sectors", {}).get(ms, {}).get("sub_categories", [])
                score = 0
                for sub in subs:
                    keywords = self._sub_keywords(sub)
                    score += sum(1 for keyword in keywords if keyword and keyword in haystack)
                if score > 0:
                    scored[ms].append((score, title[:100]))

        for ms in meta_sectors:
            seen: set[str] = set()
            for _, title in sorted(scored.get(ms, []), key=lambda item: item[0], reverse=True):
                if title in seen:
                    continue
                seen.add(title)
                result[ms].append(title)
                if len(result[ms]) >= top_k:
                    break
        return result

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
        self._ensure_agent_feature_cache_upto(date)

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
        if not news_path:
            return {ms: [] for ms in self.meta_sector_map.get("meta_sectors", {}).keys()}

        news_df = load_raw_news_df(news_path)
        if len(news_df) == 0:
            return {ms: [] for ms in self.meta_sector_map.get("meta_sectors", {}).keys()}
        dt_dtype = news_df.schema.get("datetime")
        if dt_dtype == pl.Utf8:
            news_df = news_df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))
        else:
            news_df = news_df.with_columns(pl.col("datetime").cast(pl.Datetime).dt.date().alias("date"))

        # Filter to the week before the decision date
        news_df = news_df.filter((pl.col("date") >= week_start.date()) & (pl.col("date") < date.date()))

        meta_sector_map = self.meta_sector_map
        meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
        result: dict[str, list[str]] = {ms: [] for ms in meta_sectors}

        if "sub_category" not in news_df.columns and "industry" not in news_df.columns:
            return self._build_news_summary_from_raw_keywords(news_df, meta_sectors, top_k)

        for ms in meta_sectors:
            ms_info = meta_sector_map.get("meta_sectors", {}).get(ms, {})
            subs = ms_info.get("sub_categories", [])

            for sub in subs:
                sub_news = news_df.filter(
                    (pl.col("sub_category") == sub)
                    if "sub_category" in news_df.columns
                    else (pl.col("industry") == sub)
                )
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

        market_df = self.market_daily_df
        if len(market_df) == 0:
            return {
                "market_return_1w": 0.0,
                "market_return_2w": 0.0,
                "market_volatility": 0.0,
                "volume_ratio": 1.0,
                "market_state": "neutral",
            }

        trading_days = [d for d in market_df["date"].to_list() if d < date.strftime("%Y-%m-%d")][-20:]
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
        market_return_1w = float(np.sum(returns[-5:])) if len(returns) >= 5 else float(np.sum(returns))
        market_return_2w = float(np.sum(returns[-10:])) if len(returns) >= 10 else float(np.sum(returns))

        # Volatility (annualized)
        market_volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.0

        # Volume ratio (recent avg / historical avg)
        recent_days = trading_days[-5:]
        hist_days = trading_days[-20:-5]
        recent_volume = float(
            market_df.filter(pl.col("date").is_in(recent_days))["market_volume"].mean()
        ) if recent_days else 0.0
        hist_volume = float(
            market_df.filter(pl.col("date").is_in(hist_days))["market_volume"].mean()
        ) if hist_days else 0.0
        volume_ratio = recent_volume / (hist_volume + 1e-9) if hist_volume > 0 else 1.0

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
        df = self.market_daily_df
        if len(df) == 0:
            return 0.0

        day_data = df.filter(pl.col("date") == date_str)
        if len(day_data) == 0:
            return 0.0

        if "market_close" in day_data.columns:
            return float(day_data["market_close"][0])
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

        self._ensure_agent_feature_cache_upto(date)
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
        df = self.price_df
        if len(df) == 0:
            return 0.0

        etf_codes = self.meta_sector_etf_codes.get(meta_sector, [])
        if not etf_codes:
            return 0.0

        code_col = "Code" if "Code" in df.columns else "code" if "code" in df.columns else None
        close_col = next((c for c in df.columns if c.lower() == "close" or "close" in c.lower()), None)
        if code_col is None or close_col is None:
            return 0.0

        day_df = df.filter((pl.col("date").cast(pl.Utf8) == date) & pl.col(code_col).cast(pl.Utf8).is_in(etf_codes))
        if len(day_df) == 0:
            return 0.0
        return float(day_df[close_col].cast(pl.Float64).mean())

    def _load_recent_portfolio_state(self, date: str | datetime) -> tuple[dict[str, float], dict[str, float], float, float]:
        """Load the latest realized portfolio state before `date` from backtest output."""
        date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
        backtest_df = self.backtest_df
        if len(backtest_df) == 0:
            return {}, {}, 0.0, 0.0

        hist = backtest_df.filter(pl.col("week_start") < date_str).sort("week_start")
        if len(hist) == 0:
            return {}, {}, 0.0, 0.0

        latest = hist.tail(1).row(0, named=True)
        raw_holdings = latest.get("holdings", {}) or {}
        if isinstance(raw_holdings, str):
            try:
                holdings = json.loads(raw_holdings)
            except Exception:
                holdings = {}
        else:
            holdings = raw_holdings

        raw_weekly_returns = latest.get("meta_sector_returns", latest.get("industry_contributions", {})) or {}
        if isinstance(raw_weekly_returns, str):
            try:
                weekly_returns = json.loads(raw_weekly_returns)
            except Exception:
                weekly_returns = {}
        else:
            weekly_returns = raw_weekly_returns
        perf_1w = float(latest.get("weekly_return", 0.0) or 0.0)
        perf_4w = float(hist.tail(4)["weekly_return"].mean()) if "weekly_return" in hist.columns else 0.0
        return dict(holdings), dict(weekly_returns), perf_1w, perf_4w

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
        ml_signal_snapshot = self.build_signal_snapshot(decision_dt)

        # Feature B: News summary
        news_summary = self.build_news_summary(decision_dt, top_k=1)

        # Feature C: Market state
        market_state = self.build_market_state(decision_dt)

        # Feature D: Position state
        hist_holdings, weekly_returns, agent_perf_1w, agent_perf_4w = self._load_recent_portfolio_state(decision_dt)
        if not current_holdings:
            current_holdings = hist_holdings
        if not weekly_returns:
            weekly_returns = {sector: 0.0 for sector in current_holdings.keys()}
        position_state = self.build_position_state(current_holdings, weekly_returns, agent_perf_1w, agent_perf_4w)

        # Feature E: Sentiment vs price divergence
        sent_p_divergence = self.build_sent_p_divergence(decision_dt)

        return {
            "tcn_sequence": tcn_sequence,
            "ml_signal_snapshot": ml_signal_snapshot,
            "news_summary": news_summary,
            "market_state": market_state,
            "position_state": position_state,
            "sent_p_divergence": sent_p_divergence,
        }
