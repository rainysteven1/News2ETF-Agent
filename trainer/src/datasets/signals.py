"""Weekly/daily signal dataset builder — aggregates news sentiment & volume by industry-(week/day).

Supports two training strategies:
  - cross_industry=True: pool all industries into one shared TCN (104 wks × 8 industries = 832 samples)
  - cross_industry=False: train one TCN per industry

Compatible with weekly backtest frequency.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from trainer.src.config.signals import SignalsDatasetConfig

EVENT_BUCKET_WEIGHTS: dict[str, float] = {
    "policy_macro": 1.0,
    "earnings_fundamental": 0.6,
    "product_industry": 0.3,
    "risk_negative": -1.0,
}

SUBCATEGORY_EVENT_BUCKET: dict[str, str] = {
    "央企/国企/国资改革": "policy_macro",
    "区域经济": "policy_macro",
    "ESG/可持续": "policy_macro",
    "交通运输/物流": "policy_macro",
    "地产/建筑/基建": "policy_macro",
    "金融/银行/证券": "policy_macro",
    "生物医药/创新药": "earnings_fundamental",
    "新能源/光伏": "earnings_fundamental",
    "新能源车/锂电": "earnings_fundamental",
    "消费电子/家电": "earnings_fundamental",
    "食品饮料/消费": "earnings_fundamental",
    "半导体/芯片": "product_industry",
    "人工智能": "product_industry",
    "云计算/大数据": "product_industry",
    "软件/信创": "product_industry",
    "TMT": "product_industry",
    "物联网/车联网": "product_industry",
}

GLOBAL_LEADER_BASKET: dict[str, float] = {
    "SPY": 0.35,
    "QQQ": 0.30,
    "SOXX": 0.20,
    "TLT": 0.15,
}

# ─── OHLCV Aggregation ─────────────────────────────────────────────────────────


def build_ohlcv_by_industry(
    ohlcv_path: str | Path,
    industry_dict_path: str | Path,
    etf_info_path: str | Path,
    freq: str = "daily",
) -> pl.DataFrame:
    """Aggregate daily OHLCV data by industry from ETF price data.

    Data flow: 大类 → 小类 → indices → ETF info table → ETF codes → OHLCV

    Returns unpivoted DataFrame:
        date, industry, close, volume, amount, high, low, open
    """
    ohlcv = pl.read_parquet(ohlcv_path)
    etf_info = pl.read_parquet(etf_info_path)

    # Parse trade_dt int → date
    ohlcv = ohlcv.with_columns(pl.col("trade_dt").cast(str).str.to_date("%Y%m%d").alias("date"))

    # Build index_name → ETF code mapping
    index_to_codes: dict[str, list[str]] = {}
    for row in etf_info.iter_rows(named=True):
        code = row["代码"]
        index_name = row["跟踪指数名称"]
        if code and index_name:
            index_to_codes.setdefault(index_name, []).append(code)

    # Flatten industry_dict: major → sub → indices → ETF codes
    with open(industry_dict_path, encoding="utf-8") as f:
        industry_dict = json.load(f)

    industry_etf_codes: dict[str, list[str]] = {}  # industry → codes
    for major, subs in industry_dict.items():
        for sub_data in subs.values():
            for idx_name in sub_data.get("indices", []):
                if idx_name in index_to_codes:
                    industry_etf_codes.setdefault(major, []).extend(index_to_codes[idx_name])

    # Deduplicate
    for k in industry_etf_codes:
        industry_etf_codes[k] = list(set(industry_etf_codes[k]))

    # Aggregate OHLCV by industry and date
    rows = []
    for industry, codes in industry_etf_codes.items():
        ind_ohlcv = ohlcv.filter(pl.col("Code").is_in(codes))
        if ind_ohlcv.is_empty():
            continue

        # Group by date — average across ETFs
        agg = (
            ind_ohlcv.group_by("date")
            .agg(
                pl.col("close").mean().alias("close"),
                pl.col("open").mean().alias("open"),
                pl.col("high").mean().alias("high"),
                pl.col("low").mean().alias("low"),
                pl.col("volume").mean().alias("volume"),
                pl.col("amount").mean().alias("amount"),
            )
            .with_columns(pl.lit(industry).alias("industry"))
        )
        rows.append(agg)

    if not rows:
        return pl.DataFrame()

    df = pl.concat(rows).sort(["industry", "date"])
    return df


class WeeklySignalDataset:
    """Build per-industry weekly/daily sentiment & volume DataFrames for ML signal training."""

    SENTIMENT_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

    def __init__(
        self,
        cfg: SignalsDatasetConfig,
        force: bool = False,
        ohlcv_cfg=None,  # SignalsOhlcvConfig | None
    ):
        assert cfg.raw_data_path is not None, "raw_data_path must be set"
        self.raw_path = Path(cfg.raw_data_path)
        self.output_sentiment = Path(cfg.output_sentiment) if cfg.output_sentiment else None
        self.train_end_week = datetime.fromisoformat(cfg.train_end_week)
        self.freq = cfg.freq  # "weekly" or "daily"
        self.cross_industry = cfg.cross_industry
        self.ohlcv_cfg = ohlcv_cfg
        self.lf: pl.LazyFrame | None = None
        self.sentiment_df: pl.DataFrame | None = None
        self.volume_df: pl.DataFrame | None = None
        self.ohlcv_df: pl.DataFrame | None = None

        # Cache logic: if processed file exists and not forced, load directly
        if self.output_sentiment and self.output_sentiment.exists() and not force:
            self._load_cached(self.output_sentiment)
        else:
            self._load_raw()
            if self.output_sentiment:
                self._save_cached(self.output_sentiment)
                # Reload so sentiment_df matches the saved unpivoted format
                self._load_cached(self.output_sentiment)

    def _load_cached(self, path: Path) -> None:
        """Load pre-aggregated parquet (may include OHLCV columns if present)."""
        df = pl.read_parquet(path)
        self.sentiment_df = df.sort(["industry", "date"])
        self.lf = df.lazy()
        # OHLCV columns are in sentiment_df if they were joined during save

    def _save_cached(self, path: Path) -> None:
        """Save aggregated data as unpivoted parquet for reuse (sentiment + volume + OHLCV)."""
        self._ensure_weekly("major_category")
        assert self.sentiment_df is not None and self.volume_df is not None

        sent_long = self.sentiment_df.unpivot(index="period", variable_name="industry", value_name="sentiment_mean")
        vol_long = self.volume_df.unpivot(index="period", variable_name="industry", value_name="news_count")
        merged = (
            sent_long.join(vol_long, on=["period", "industry"], how="left")
            .rename({"period": "date"})
            .sort(["industry", "date"])
        )

        # Join OHLCV data if configured
        if self.ohlcv_cfg and self.ohlcv_cfg.ohlcv_path:
            ohlcv = build_ohlcv_by_industry(
                self.ohlcv_cfg.ohlcv_path,
                self.ohlcv_cfg.industry_dict_path,
                self.ohlcv_cfg.etf_info_path,
            )
            if not ohlcv.is_empty():
                self.ohlcv_df = ohlcv
                # Cast date types to match: merged.date is datetime[μs], ohlcv.date is date
                ohlcv_sel = ohlcv.select(
                    [
                        pl.col("date").cast(pl.Date).alias("date"),
                        "industry",
                        "amount",
                        "high",
                        "low",
                        "close",
                        "open",
                        "volume",
                    ]
                )
                merged = merged.with_columns(pl.col("date").cast(pl.Date)).join(
                    ohlcv_sel,
                    on=["date", "industry"],
                    how="left",
                )
                # Fill OHLCV NaN: forward-fill then backward-fill per industry, finally 0
                ohlcv_cols = ["amount", "high", "low", "close", "open", "volume"]
                merged = merged.sort(["industry", "date"])
                for col in ohlcv_cols:
                    if col in merged.columns:
                        merged = merged.with_columns(
                            pl.col(col).forward_fill().over("industry").fill_null(0).alias(col)
                        )
                # Compute daily return: (close - open) / open, safe for zero open
                if "close" in merged.columns and "open" in merged.columns:
                    merged = merged.with_columns(
                        ((pl.col("close") - pl.col("open")) / pl.col("open").clip(lower_bound=1e-9)).alias("return")
                    )

        # Join sentiment_std and avg_confidence
        if hasattr(self, "_sentiment_std_df") and self._sentiment_std_df is not None:
            std_long = self._sentiment_std_df.unpivot(
                index="period", variable_name="industry", value_name="sentiment_std"
            )
            std_long = std_long.with_columns(pl.col("period").cast(pl.Date).alias("date")).drop("period")
            merged = merged.join(std_long, on=["date", "industry"], how="left")
            merged = merged.with_columns(pl.col("sentiment_std").fill_null(0))
        if hasattr(self, "_avg_conf_df") and self._avg_conf_df is not None:
            conf_long = self._avg_conf_df.unpivot(index="period", variable_name="industry", value_name="avg_confidence")
            conf_long = conf_long.with_columns(pl.col("period").cast(pl.Date).alias("date")).drop("period")
            merged = merged.join(conf_long, on=["date", "industry"], how="left")
            merged = merged.with_columns(pl.col("avg_confidence").fill_null(0))

        # Fill any remaining nulls in core sentiment columns
        for col in ["sentiment_mean", "news_count"]:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).fill_null(0))

        # Ensure return column exists (0 if no OHLCV data)
        if "return" not in merged.columns:
            merged = merged.with_columns(pl.lit(0.0).alias("return"))

        merged.write_parquet(path)

    def _load_raw(self) -> None:
        df = pl.read_parquet(self.raw_path)
        df = df.with_columns(
            pl.col("datetime").str.to_datetime(),
            pl.col("sentiment").replace(self.SENTIMENT_MAP).cast(pl.Float64).alias("sentiment_score"),
        )
        df = df.with_columns(
            (pl.col("sentiment_score") * pl.col("sentiment_confidence")).alias("sentiment_weighted"),
            pl.col("sentiment_confidence").alias("confidence_sum"),
        )
        self.lf = df.lazy()

    def build_weekly(self, industry_col: str = "major_category") -> tuple[pl.DataFrame, pl.DataFrame]:
        """Aggregate by (period, industry) → weighted sentiment + news count + std + avg_conf."""
        if self.freq == "daily":
            period_expr = pl.col("datetime").dt.truncate("1d")
        else:
            period_expr = pl.col("datetime").dt.truncate("1w")

        # Sentiment per (period, industry): weighted mean + std + avg_confidence
        sent_lf = (
            self.lf.with_columns(period_expr.alias("period"))
            .group_by(["period", industry_col])
            .agg(
                (pl.col("sentiment_weighted").sum() / (pl.col("confidence_sum").sum() + 1e-9)).alias(
                    "sentiment_weighted"
                ),
                pl.col("sentiment_score").std().alias("sentiment_std"),
                pl.col("sentiment_confidence").mean().alias("avg_confidence"),
                pl.len().alias("news_count"),
            )
        )
        self.sentiment_df = (
            sent_lf.collect().pivot(values="sentiment_weighted", index="period", on=industry_col).sort("period")
        )
        self.volume_df = (
            sent_lf.collect().pivot(values="news_count", index="period", on=industry_col).sort("period").fill_null(0)
        )
        # Also keep sentiment_std and avg_confidence per industry per period
        self._sentiment_std_df = (
            sent_lf.collect().pivot(values="sentiment_std", index="period", on=industry_col).sort("period")
        )
        self._avg_conf_df = (
            sent_lf.collect().pivot(values="avg_confidence", index="period", on=industry_col).sort("period")
        )

        return self.sentiment_df, self.volume_df

    def _ensure_weekly(self, industry_col: str) -> None:
        if self.sentiment_df is None:
            self.build_weekly(industry_col)

    def _df_to_arrays(self, industry_col: str) -> tuple[np.ndarray, list, list[str]]:
        """Convert (periods x industries) DFs to (n_periods, n_industries) arrays."""
        self._ensure_weekly(industry_col)
        assert self.sentiment_df is not None and self.volume_df is not None

        sent_sorted = self.sentiment_df.sort("period")
        vol_sorted = self.volume_df.sort("period")

        periods: list[str | datetime] = sent_sorted["period"].to_list()
        industries: list[str] = [c for c in sent_sorted.columns if c != "period"]

        sent_arr = sent_sorted.drop("period").to_numpy()
        vol_arr = vol_sorted.drop("period").to_numpy()
        return sent_arr, vol_arr, periods, industries

    def build_tcn_sequences(
        self,
        lookback_weeks: int = 8,
        industry_col: str = "major_category",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build (samples, lookback, n_industries) tensor for cross-industry TCN + momentum labels.

        If cross_industry=True: pool all industries into one tensor (each sample is one period,
        features = lookback × n_industries). Labels are per-industry momentum signals.
        """
        sent_arr, vol_arr, periods, industries = self._df_to_arrays(industry_col)

        if not self.cross_industry:
            # Per-industry mode: return stacked sequences per industry
            return self._build_per_industry_sequences(sent_arr, vol_arr, periods, industries, lookback_weeks)

        # ── Cross-industry pooled mode ──────────────────────────────────────────
        n_periods = len(periods)

        sequences, labels = [], []
        for i in range(lookback_weeks, n_periods):
            seq_sent = sent_arr[i - lookback_weeks : i]  # (lookback, n_industries)
            vol_delta = np.clip((vol_arr[i] - vol_arr[i - 1]) / (vol_arr[i - 1] + 1), -1, 1)
            # Stack sentiment + volume delta as features: (lookback, n_industries * 2)
            feat = np.concatenate([seq_sent, vol_delta.reshape(1, -1).repeat(lookback_weeks, axis=0)], axis=1)
            sequences.append(feat)

            # Per-industry momentum labels
            mom = np.clip((sent_arr[i] - sent_arr[i - 1]) / (np.abs(sent_arr[i - 1]) + 1e-9), -1, 1)
            labels.append(mom)

        X = np.stack(sequences).astype(np.float32)  # (samples, lookback, n_industries * 2)
        y = np.stack(labels).astype(np.float32)  # (samples, n_industries)

        split_idx = periods.index(self.train_end_week) if self.train_end_week in periods else len(periods)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, y_train, X_test, y_test

    def _build_per_industry_sequences(
        self,
        sent_arr: np.ndarray,
        vol_arr: np.ndarray,
        periods: list,
        industries: list[str],
        lookback: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stack per-industry sequences into one tensor (legacy per-industry mode)."""
        all_X, all_y = [], []
        for idx_industry in range(len(industries)):
            seqs, lbls = [], []
            for i in range(lookback, len(periods)):
                seq_sent = sent_arr[i - lookback : i, idx_industry : idx_industry + 1]
                vol_delta = np.clip(
                    (vol_arr[i, idx_industry] - vol_arr[i - 1, idx_industry]) / (vol_arr[i - 1, idx_industry] + 1),
                    -1,
                    1,
                )
                feat = np.concatenate([seq_sent, vol_delta.reshape(1, 1).repeat(lookback, axis=0)], axis=1)
                seqs.append(feat)
                mom = np.clip(
                    (sent_arr[i, idx_industry] - sent_arr[i - 1, idx_industry])
                    / (np.abs(sent_arr[i - 1, idx_industry]) + 1e-9),
                    -1,
                    1,
                )
                lbls.append(mom)
            all_X.append(np.stack(seqs))
            all_y.append(np.stack(lbls))

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        split_idx = periods.index(self.train_end_week) if self.train_end_week in periods else len(periods)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, y_train, X_test, y_test

    def build_isolation_forest_dataset(
        self,
        industry_col: str = "major_category",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (n_periods, n_industries * 3) features: [vol_norm, sent_diff, sent_cur]."""
        sent_arr, vol_arr, periods, _ = self._df_to_arrays(industry_col)

        features = []
        for i in range(len(periods)):
            vol_norm = vol_arr[i] / (vol_arr[i - 1] + 1) if i > 0 else vol_arr[i]
            sent_diff = sent_arr[i] - (sent_arr[i - 1] if i > 0 else sent_arr[i])
            sent_cur = sent_arr[i]
            feat = np.concatenate([vol_norm, sent_diff, sent_cur])
            features.append(feat)

        X = np.stack(features).astype(np.float32)
        split_idx = periods.index(self.train_end_week) if self.train_end_week in periods else len(periods)
        return X[:split_idx], X[split_idx:]

    def build_lgbm_dataset(
        self,
        lookback_weeks: int = 4,
        industry_col: str = "major_category",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build rolling-window momentum + heat features for LightGBM (cross-industry pooled)."""
        sent_arr, vol_arr, periods, industries = self._df_to_arrays(industry_col)
        n_periods, n_industries = sent_arr.shape

        # Momentum & heat time series: (n_periods, n_industries)
        momentum = np.zeros_like(sent_arr)
        for i in range(1, n_periods):
            momentum[i] = np.clip((sent_arr[i] - sent_arr[i - 1]) / (np.abs(sent_arr[i - 1]) + 1e-9), -1, 1)
        heat = np.zeros_like(vol_arr, dtype=float)
        for i in range(1, n_periods):
            heat[i] = np.clip((vol_arr[i] - vol_arr[i - 1]) / (vol_arr[i - 1] + 1), -1, 1)

        # Label: sign of next-period sentiment change: (n_periods - 1, n_industries)
        y_raw = np.sign(sent_arr[1:] - sent_arr[:-1])

        momentum_in = momentum[:-1]  # (n_periods-1, n_industries)
        heat_in = heat[:-1]

        def stack_rolling(arr: np.ndarray, n: int) -> np.ndarray:
            rows = [arr[i - n : i].flatten() for i in range(n, len(arr))]
            return np.stack(rows)

        X_feat = np.concatenate(
            [
                stack_rolling(momentum_in, lookback_weeks),
                stack_rolling(heat_in, lookback_weeks),
            ],
            axis=1,
        ).astype(np.float32)

        y = y_raw[lookback_weeks:].flatten().astype(np.int32)

        split_idx = (
            periods.index(self.train_end_week) - lookback_weeks
            if self.train_end_week in periods
            else len(periods) - lookback_weeks
        )

        X_train, X_test = X_feat[:split_idx], X_feat[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, y_train, X_test, y_test

    def summary(self) -> dict:
        self._ensure_weekly("major_category")
        assert self.sentiment_df is not None and self.volume_df is not None
        return {
            "n_periods": self.sentiment_df.height,
            "n_industries": self.sentiment_df.width - 1,
            "freq": self.freq,
            "cross_industry": self.cross_industry,
            "industries": [c for c in self.sentiment_df.columns if c != "period"],
            "periods": [str(p) for p in self.sentiment_df.sort("period")["period"]],
            "train_end": str(self.train_end_week.date()),
        }


# ─── Data Preparation ───────────────────────────────────────────────────────────


def build_sequences(
    sentiment_df: pl.DataFrame,
    industries: list[str],
    seq_len: int,
    anomaly_threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build TCN training data from sentiment + OHLCV time series.

    Returns:
        X:     (N, seq_len, 6) — 6-channel sequences:
               [sentiment_mean, sentiment_std, news_count, avg_confidence,
                volume_ratio, intraday_vol]
        y_reg: (N, 1) — continuous sentiment delta at next step, clipped to [-1, 1]
        y_cls: (N, 1) — 1 if |return| > threshold else 0
    """
    X_list, y_reg_list, y_cls_list = [], [], []

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue

        vals = ind_df["sentiment_mean"].to_numpy()
        vals_std = ind_df["sentiment_std"].to_numpy() if "sentiment_std" in ind_df.columns else np.zeros_like(vals)
        nc = ind_df["news_count"].to_numpy()
        conf = ind_df["avg_confidence"].to_numpy() if "avg_confidence" in ind_df.columns else np.zeros_like(nc)
        rets = ind_df["return"].to_numpy() if "return" in ind_df.columns else np.zeros_like(vals)

        # OHLCV features (fillna → 0 if missing)
        vol_arr = ind_df["volume"].to_numpy() if "volume" in ind_df.columns else np.zeros_like(vals)
        high_arr = ind_df["high"].to_numpy() if "high" in ind_df.columns else np.zeros_like(vals)
        low_arr = ind_df["low"].to_numpy() if "low" in ind_df.columns else np.zeros_like(vals)
        close_arr = ind_df["close"].to_numpy() if "close" in ind_df.columns else np.zeros_like(vals)

        # Rolling volume MA5
        vol_ma5 = np.zeros_like(vol_arr)
        for i in range(1, len(vol_arr)):
            window = vol_arr[max(0, i - 4) : i + 1]
            vol_ma5[i] = np.mean(window) if window.size > 0 else 0
        volume_ratio = np.where(vol_arr > 0, vol_arr / (vol_ma5 + 1e-9), 0.0)
        volume_ratio = np.clip(volume_ratio, 0.0, 100.0)  # cap extreme ratios

        # Intraday volatility
        close_safe = np.where(close_arr != 0, close_arr, 1.0)
        intraday_vol = (high_arr - low_arr) / close_safe

        # Global normalization stats (per channel)
        n = len(vals)
        ch0 = vals.astype(np.float32)
        ch1 = vals_std.astype(np.float32)
        ch2 = nc.astype(np.float32)
        ch3 = conf.astype(np.float32)
        ch4 = volume_ratio.astype(np.float32)
        ch5 = intraday_vol.astype(np.float32)

        for i in range(n - seq_len - 1):
            window = np.zeros((seq_len, 6), dtype=np.float32)
            window[:, 0] = ch0[i : i + seq_len]
            window[:, 1] = ch1[i : i + seq_len]
            window[:, 2] = ch2[i : i + seq_len]
            window[:, 3] = ch3[i : i + seq_len]
            window[:, 4] = ch4[i : i + seq_len]
            window[:, 5] = ch5[i : i + seq_len]
            X_list.append(window)

            sent_delta = ch0[i + seq_len] - ch0[i + seq_len - 1]
            target = np.clip(sent_delta / (np.abs(ch0[i + seq_len - 1]) + 1e-9), -1, 1)
            y_reg_list.append(target)
            next_ret = rets[i + seq_len]
            y_cls_list.append(1 if abs(next_ret) > anomaly_threshold else 0)

    X = np.array(X_list, dtype=np.float32)  # (N, seq_len, 6)
    y_reg = np.array(y_reg_list, dtype=np.float32).reshape(-1, 1)
    y_cls = np.array(y_cls_list, dtype=np.float32).reshape(-1, 1)
    return X, y_reg, y_cls


def build_lgbm_features(
    sentiment_df: pl.DataFrame,
    industries: list[str],
    seq_len: int,
    tcn_model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build LightGBM feature matrix from TCN outputs + raw signals.

    Features (13 dims):
        [delta_sentiment_1w, delta_sentiment_2w, news_count, news_heat,
         tcn_reg, tcn_cls, tcn_reg_delta, news_count_std_5d,
         sentiment_volatility_5d, tcn_reg * news_heat,
         volume_ratio, intraday_vol, avg_price]

    Returns X, y, dates, industries (for time-based split and per-industry IC).
    """
    has_ohlcv = all(c in sentiment_df.columns for c in ["volume", "high", "low", "close", "open"])

    feat_rows, label_rows, date_rows, industry_rows = [], [], [], []

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        nc = ind_df["news_count"].to_numpy()
        nh = ind_df["news_heat"].to_numpy() if "news_heat" in ind_df.columns else np.zeros_like(nc)
        dates = ind_df["date"].to_list()

        # OHLCV arrays (all same length as vals)
        if has_ohlcv:
            vol_arr = ind_df["volume"].to_numpy()
            high_arr = ind_df["high"].to_numpy()
            low_arr = ind_df["low"].to_numpy()
            close_arr = ind_df["close"].to_numpy()
            open_arr = ind_df["open"].to_numpy()
        else:
            vol_arr = high_arr = low_arr = close_arr = open_arr = np.zeros_like(vals)

        for i in range(seq_len + 1, len(vals) - 1):
            delta1 = vals[i] - vals[i - 1]
            delta2 = vals[i] - vals[i - 2]
            news_count = nc[i]
            news_heat = nh[i]

            # Build 6-channel TCN input: (seq_len, 6)
            ch0 = vals[i - seq_len : i].copy()  # sentiment_mean
            ch1 = (
                ind_df["sentiment_std"][i - seq_len : i].to_numpy()
                if "sentiment_std" in ind_df.columns
                else np.zeros(seq_len)
            )
            ch2 = nc[i - seq_len : i].copy()  # news_count
            ch3 = (
                ind_df["avg_confidence"][i - seq_len : i].to_numpy()
                if "avg_confidence" in ind_df.columns
                else np.zeros(seq_len)
            )
            # volume_ratio and intraday_vol per timestep
            vol_ma5 = np.array([np.mean(vol_arr[max(0, j - 4) : j + 1]) for j in range(i - seq_len, i)])
            close_safe = np.where(close_arr[i - seq_len : i] != 0, close_arr[i - seq_len : i], 1.0)
            ch4 = np.where(vol_arr[i - seq_len : i] > 0, vol_arr[i - seq_len : i] / (vol_ma5 + 1e-9), 0.0)
            ch5 = (high_arr[i - seq_len : i] - low_arr[i - seq_len : i]) / close_safe

            x_t = (
                np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=1)  # (seq_len, 6)
                .reshape(1, seq_len, 6)
                .astype(np.float32)
            )
            x_t = torch.from_numpy(x_t).float().to(device)
            with torch.no_grad():
                reg_out, cls_out = tcn_model(x_t)
            tcn_reg = reg_out.item()
            tcn_cls = cls_out.item()

            # Previous step TCN reg for delta (6-channel)
            prev_ch0 = vals[i - seq_len - 1 : i - 1].copy()
            prev_ch1 = (
                ind_df["sentiment_std"][i - seq_len - 1 : i - 1].to_numpy()
                if "sentiment_std" in ind_df.columns
                else np.zeros(seq_len)
            )
            prev_ch2 = nc[i - seq_len - 1 : i - 1].copy()
            prev_ch3 = (
                ind_df["avg_confidence"][i - seq_len - 1 : i - 1].to_numpy()
                if "avg_confidence" in ind_df.columns
                else np.zeros(seq_len)
            )
            prev_vol_ma5 = np.array([np.mean(vol_arr[max(0, j - 4) : j + 1]) for j in range(i - seq_len - 1, i - 1)])
            prev_close_safe = np.where(close_arr[i - seq_len - 1 : i - 1] != 0, close_arr[i - seq_len - 1 : i - 1], 1.0)
            prev_ch4 = np.where(
                vol_arr[i - seq_len - 1 : i - 1] > 0,
                vol_arr[i - seq_len - 1 : i - 1] / (prev_vol_ma5 + 1e-9),
                0.0,
            )
            prev_ch5 = (high_arr[i - seq_len - 1 : i - 1] - low_arr[i - seq_len - 1 : i - 1]) / prev_close_safe
            x_prev = (
                np.stack([prev_ch0, prev_ch1, prev_ch2, prev_ch3, prev_ch4, prev_ch5], axis=1)
                .reshape(1, seq_len, 6)
                .astype(np.float32)
            )
            x_prev = torch.from_numpy(x_prev).float().to(device)
            with torch.no_grad():
                reg_prev, _ = tcn_model(x_prev)
            tcn_reg_prev = reg_prev.item()
            tcn_reg_delta = tcn_reg - tcn_reg_prev

            # Rolling stats
            news_count_std = np.std(nc[i - 4 : i + 1])
            sent_vol = np.std(vals[i - 4 : i + 1])

            # Interaction
            tcn_heat_interact = tcn_reg * news_heat

            # OHLCV features for LightGBM
            if has_ohlcv:
                vr = np.where(vol_arr[i] > 0, vol_arr[i] / (vol_ma5[-1] + 1e-9), 0.0) if i > 0 else 0.0
                iv = (high_arr[i] - low_arr[i]) / max(close_arr[i], 1e-9) if close_arr[i] != 0 else 0.0
                ap = (high_arr[i] + low_arr[i] + close_arr[i] + open_arr[i]) / 4.0
            else:
                vr, iv, ap = 0.0, 0.0, 0.0

            next_dir = 1 if vals[i + 1] > vals[i] else (-1 if vals[i + 1] < vals[i] else 0)

            feat_rows.append(
                [
                    delta1,
                    delta2,
                    news_count,
                    news_heat,
                    tcn_reg,
                    tcn_cls,
                    tcn_reg_delta,
                    news_count_std,
                    sent_vol,
                    tcn_heat_interact,
                    vr,
                    iv,
                    ap,
                ]
            )
            label_rows.append(next_dir)
            date_rows.append(dates[i])
            industry_rows.append(ind)

    return (
        np.array(feat_rows, dtype=np.float32),
        np.array(label_rows, dtype=np.int32),
        np.array(date_rows),
        np.array(industry_rows),
    )


def build_sub_category_sequences(
    sentiment_df: pl.DataFrame,
    meta_sector_map: dict,
    lookback_days: int = 5,
    forecast_days: int = 5,
    price_df: pl.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list, list[str]]:
    """构建扇入式 TCN 训练数据。

    标签采用:
      raw_momentum -> winsorize -> z-score -> tanh
    并使用 t+forecast_days 与 t 的元板块加权情感差构造目标。
    """
    df = sentiment_df.sort("date")
    if df.is_empty():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), [], []

    sector_col = "sub_category" if "sub_category" in df.columns else "industry"
    sent_col = "sentiment_mean" if "sentiment_mean" in df.columns else "sentiment_weighted"
    std_col = "sentiment_std" if "sentiment_std" in df.columns else None
    dates = df["date"].unique().sort().to_list()
    sub_industries = sorted(df[sector_col].unique().to_list())
    meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())

    if len(dates) <= lookback_days + forecast_days:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), [], sub_industries

    sub_to_idx = {sub: idx for idx, sub in enumerate(sub_industries)}
    date_to_idx = {d: idx for idx, d in enumerate(dates)}
    n_sub = len(sub_industries)
    n_meta = len(meta_sectors)
    n_dates = len(dates)

    sent_matrix = np.zeros((n_dates, n_sub), dtype=np.float32)
    news_matrix = np.zeros((n_dates, n_sub), dtype=np.float32)
    std_matrix = np.zeros((n_dates, n_sub), dtype=np.float32)

    for row in df.iter_rows(named=True):
        d_idx = date_to_idx[row["date"]]
        s_idx = sub_to_idx[row[sector_col]]
        sent_matrix[d_idx, s_idx] = float(row.get(sent_col, 0.0) or 0.0)
        news_matrix[d_idx, s_idx] = float(row.get("news_count", 0.0) or 0.0)
        if std_col is not None:
            std_matrix[d_idx, s_idx] = float(row.get(std_col, 0.0) or 0.0)

    ema_matrix = np.zeros_like(sent_matrix)
    alpha = 0.2
    ema_matrix[0] = sent_matrix[0]
    for i in range(1, n_dates):
        ema_matrix[i] = alpha * sent_matrix[i] + (1 - alpha) * ema_matrix[i - 1]

    velocity = np.zeros_like(sent_matrix)
    velocity[1:] = sent_matrix[1:] - sent_matrix[:-1]
    acceleration = np.zeros_like(sent_matrix)
    acceleration[2:] = velocity[2:] - velocity[1:-1]
    acceleration = np.clip(acceleration / 2.0, -1.0, 1.0)

    rolling_sent_std = np.zeros_like(sent_matrix)
    for i in range(n_dates):
        start = max(0, i - lookback_days + 1)
        rolling_sent_std[i] = np.std(sent_matrix[start : i + 1], axis=0)
    if std_col is not None:
        rolling_sent_std = np.where(std_matrix > 0, std_matrix, rolling_sent_std)

    log_news = np.log1p(news_matrix)
    event_type_matrix = np.zeros((n_dates, n_sub), dtype=np.float32)
    for row in df.iter_rows(named=True):
        d_idx = date_to_idx[row["date"]]
        s_idx = sub_to_idx[row[sector_col]]
        event_type_matrix[d_idx, s_idx] = _compute_event_type_score_for_row(
            row=row,
            sub_category=row[sector_col],
        )

    price_matrix = None
    if price_df is not None and not price_df.is_empty() and "close" in price_df.columns:
        price_df = price_df.sort("date")
        price_sector_col = "sub_category" if "sub_category" in price_df.columns else "industry"
        price_matrix = np.zeros((n_dates, n_sub), dtype=np.float32)
        price_date_to_idx = {d: idx for idx, d in enumerate(dates)}
        for row in price_df.iter_rows(named=True):
            row_date = row.get("date")
            row_sector = row.get(price_sector_col)
            if row_date in price_date_to_idx and row_sector in sub_to_idx:
                price_matrix[price_date_to_idx[row_date], sub_to_idx[row_sector]] = float(row.get("close", 0.0) or 0.0)

    residual_matrix = np.zeros_like(sent_matrix)
    if price_matrix is not None:
        returns_matrix = np.zeros_like(price_matrix)
        prev_prices = price_matrix[:-1]
        valid_prev = np.abs(prev_prices) > 1e-9
        returns_matrix[1:] = np.where(
            valid_prev,
            (price_matrix[1:] - prev_prices) / (prev_prices + 1e-9),
            0.0,
        )

        residual_history = np.zeros_like(sent_matrix)
        for i in range(1, n_dates):
            start = max(1, i - 59)
            sent_hist = ema_matrix[max(0, start - 1) : i, :]
            ret_hist = returns_matrix[start : i + 1, :]
            hist_len = min(len(sent_hist), len(ret_hist))
            if hist_len <= 1:
                continue

            sent_hist = sent_hist[-hist_len:]
            ret_hist = ret_hist[-hist_len:]
            sent_curr = ema_matrix[i - 1]
            beta = np.zeros(n_sub, dtype=np.float32)

            sent_var = np.var(sent_hist, axis=0)
            valid = sent_var > 1e-9
            if np.any(valid):
                sent_mean = np.mean(sent_hist[:, valid], axis=0)
                ret_mean = np.mean(ret_hist[:, valid], axis=0)
                cov = np.mean(
                    (sent_hist[:, valid] - sent_mean) * (ret_hist[:, valid] - ret_mean),
                    axis=0,
                )
                beta[valid] = cov / (sent_var[valid] + 1e-9)

            residual_history[i] = returns_matrix[i] - beta * sent_curr
            hist_res = residual_history[start : i + 1]
            res_mean = np.mean(hist_res, axis=0)
            res_std = np.std(hist_res, axis=0) + 1e-9
            residual_matrix[i] = (residual_history[i] - res_mean) / res_std

    meta_weights = np.zeros((n_meta, n_sub), dtype=np.float32)
    for m_idx, meta_sector in enumerate(meta_sectors):
        for sub in meta_sector_map.get("meta_sectors", {}).get(meta_sector, {}).get("sub_categories", []):
            if sub in sub_to_idx:
                meta_weights[m_idx, sub_to_idx[sub]] = _get_market_cap_weight(sub, meta_sector_map)

    raw_targets: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    sample_dates: list = []

    for current_idx in range(lookback_days - 1, n_dates - forecast_days):
        start_idx = current_idx - lookback_days + 1
        seq = np.zeros((lookback_days, n_sub, 6), dtype=np.float32)
        for offset, day_idx in enumerate(range(start_idx, current_idx + 1)):
            seq[offset, :, 0] = ema_matrix[day_idx]
            seq[offset, :, 1] = acceleration[day_idx]
            seq[offset, :, 2] = rolling_sent_std[day_idx]
            seq[offset, :, 3] = log_news[day_idx]
            seq[offset, :, 4] = event_type_matrix[day_idx]
            seq[offset, :, 5] = residual_matrix[day_idx]
        sequences.append(seq)
        sample_dates.append(dates[current_idx])

        future_idx = current_idx + forecast_days
        cur_meta = np.zeros(n_meta, dtype=np.float32)
        future_meta = np.zeros(n_meta, dtype=np.float32)
        for m_idx in range(n_meta):
            weights = meta_weights[m_idx]
            denom = float(weights.sum())
            if denom > 0:
                cur_meta[m_idx] = float((sent_matrix[current_idx] * weights).sum() / denom)
                future_meta[m_idx] = float((sent_matrix[future_idx] * weights).sum() / denom)
        raw_targets.append((future_meta - cur_meta) / (np.abs(cur_meta) + 1e-9))

    if not sequences:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), [], sub_industries

    raw_target_matrix = np.stack(raw_targets).astype(np.float32)
    flat = raw_target_matrix.reshape(-1)
    p1, p99 = np.percentile(flat, 1), np.percentile(flat, 99)
    clipped = np.clip(raw_target_matrix, p1, p99)
    mu = float(clipped.mean())
    sigma = float(clipped.std()) + 1e-9
    y = np.tanh((clipped - mu) / sigma).astype(np.float32)

    X = np.stack(sequences).astype(np.float32)
    return X, y, sample_dates, sub_industries


def _normalize_event_bucket(raw_value: object | None) -> str | None:
    if raw_value is None:
        return None

    value = str(raw_value).strip().lower()
    aliases = {
        "policy": "policy_macro",
        "macro": "policy_macro",
        "policy_macro": "policy_macro",
        "earnings": "earnings_fundamental",
        "fundamental": "earnings_fundamental",
        "earnings_fundamental": "earnings_fundamental",
        "product": "product_industry",
        "industry": "product_industry",
        "product_industry": "product_industry",
        "risk": "risk_negative",
        "negative": "risk_negative",
        "risk_negative": "risk_negative",
    }
    return aliases.get(value)


def _compute_event_type_score_for_row(row: dict[str, object], sub_category: str) -> float:
    explicit_score = row.get("event_type_score")
    if explicit_score is not None:
        return float(np.clip(float(explicit_score), -1.0, 1.0))

    weighted_sum = 0.0
    total_count = 0.0
    for bucket, weight in EVENT_BUCKET_WEIGHTS.items():
        count_val = row.get(f"{bucket}_count")
        if count_val is None:
            count_val = row.get(f"event_count_{bucket}")
        if count_val is None:
            continue
        count = float(count_val or 0.0)
        weighted_sum += weight * count
        total_count += count

    if total_count > 0:
        return float(np.clip(weighted_sum / (total_count + 1e-9), -1.0, 1.0))

    bucket = _normalize_event_bucket(row.get("event_type_bucket") or row.get("event_type") or row.get("major_category"))
    if bucket is not None:
        return float(EVENT_BUCKET_WEIGHTS[bucket])

    fallback_bucket = SUBCATEGORY_EVENT_BUCKET.get(sub_category, "product_industry")
    return float(EVENT_BUCKET_WEIGHTS[fallback_bucket])


def _get_market_cap_weight(sub: str, meta_sector_map: dict) -> float:
    """Get market cap weight for a sub-category."""
    notes = meta_sector_map.get("notes", {})
    core_driver = notes.get("核心驱动（×1.5）", [])
    important = notes.get("重要辅助（×1.0）", [])
    edge_smoothing = notes.get("边缘平滑（×0.5）", [])

    if sub in core_driver:
        return 1.5
    elif sub in important:
        return 1.0
    elif sub in edge_smoothing:
        return 0.5
    else:
        return 1.0


def compute_global_leader_sentiment(
    sentiment_df: pl.DataFrame,
    meta_sector_map: dict,
    lookback: int = 5,
) -> pl.DataFrame:
    """Build global leader sentiment using the most recent fully closed session."""
    df = sentiment_df.sort("date")
    meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
    if df.is_empty():
        return pl.DataFrame({"date": [], **{f"global_leader_{ms}": [] for ms in meta_sectors}})

    dates = df["date"].unique().sort().to_list()
    sector_col = (
        "symbol" if "symbol" in df.columns else ("sub_category" if "sub_category" in df.columns else "industry")
    )
    sent_col = "sentiment_mean" if "sentiment_mean" in df.columns else "sentiment_weighted"

    available_symbols = set(df[sector_col].unique().to_list())
    symbol_history: dict[str, list[float]] = {}
    for symbol in GLOBAL_LEADER_BASKET:
        if symbol not in available_symbols:
            continue
        series = []
        for date in dates:
            day_rows = df.filter((pl.col("date") == date) & (pl.col(sector_col) == symbol))
            series.append(float(day_rows[sent_col][0]) if len(day_rows) > 0 else 0.0)
        symbol_history[symbol] = series

    fallback_series = np.zeros(len(dates), dtype=np.float32)
    if not symbol_history:
        leader_map = meta_sector_map.get("global_leader_map", {})
        all_leaders = sorted({leader for leaders in leader_map.values() for leader in leaders})
        if all_leaders:
            leader_histories = []
            for leader in all_leaders:
                history = []
                for date in dates:
                    day_rows = df.filter((pl.col("date") == date) & (pl.col(sector_col) == leader))
                    history.append(float(day_rows[sent_col][0]) if len(day_rows) > 0 else 0.0)
                leader_histories.append(np.array(history, dtype=np.float32))
            fallback_series = np.mean(np.stack(leader_histories), axis=0).astype(np.float32)

    result_rows = []
    for i, date in enumerate(dates):
        start_idx = max(0, i - lookback + 1)
        if symbol_history:
            gl_value = 0.0
            for symbol, weight in GLOBAL_LEADER_BASKET.items():
                history = symbol_history.get(symbol)
                if history is None:
                    continue
                gl_value += float(np.mean(history[start_idx : i + 1])) * weight
        else:
            gl_value = float(np.mean(fallback_series[start_idx : i + 1])) if len(fallback_series) else 0.0

        row = {"date": date}
        for sector in meta_sectors:
            row[f"global_leader_{sector}"] = gl_value
        result_rows.append(row)

    return pl.DataFrame(result_rows)


def compute_market_beta(
    price_df: pl.DataFrame,
    index_df: pl.DataFrame,
    meta_sector_map: dict,
    window: int = 20,
) -> pl.DataFrame:
    """滚动 20 日 Beta: beta[sector][t] = rolling_correlation(returns[sector], index_returns)

    Beta measures each sector's sensitivity to market movements.
    A beta > 1 means the sector is more volatile than the market.
    """
    price_df = price_df.sort("date")
    index_df = index_df.sort("date")

    meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
    dates = price_df["date"].unique().sort().to_list()

    # Pre-compute returns for price_df columns (sectors)
    sector_cols = [c for c in price_df.columns if c != "date" and c != "datetime"]
    sector_returns: dict[str, list[float]] = {c: [] for c in sector_cols}

    for i, date in enumerate(dates):
        for col in sector_cols:
            day_data = price_df.filter(pl.col("date") == date)
            if len(day_data) > 0 and col in day_data.columns:
                price = float(day_data[col][0])
            else:
                price = 0.0

            if i > 0:
                prev_day_data = price_df.filter(pl.col("date") == dates[i - 1])
                if len(prev_day_data) > 0 and col in prev_day_data.columns:
                    prev_price = float(prev_day_data[col][0])
                    if prev_price != 0:
                        ret = (price - prev_price) / prev_price
                    else:
                        ret = 0.0
                else:
                    ret = 0.0
            else:
                ret = 0.0
            sector_returns[col].append(ret)

    # Compute index returns
    index_returns = []
    for i, date in enumerate(dates):
        day_data = index_df.filter(pl.col("date") == date)
        if len(day_data) > 0 and len(index_df.columns) > 1:
            price = float(list(index_df.columns)[1].__class__(day_data[list(index_df.columns)[1]][0]))
            price = float(day_data[list(index_df.columns)[1]][0])
        else:
            price = 0.0

        if i > 0:
            prev_day_data = index_df.filter(pl.col("date") == dates[i - 1])
            if len(prev_day_data) > 0 and len(index_df.columns) > 1:
                prev_price = float(prev_day_data[list(index_df.columns)[1]][0])
                if prev_price != 0:
                    ret = (price - prev_price) / prev_price
                else:
                    ret = 0.0
            else:
                ret = 0.0
        else:
            ret = 0.0
        index_returns.append(ret)

    result_rows = []
    for i, date in enumerate(dates):
        row = {"date": date}
        start_idx = max(0, i - window + 1)
        index_window = index_returns[start_idx : i + 1]

        for ms in meta_sectors:
            # Map meta sector to sub-categories and average their returns
            ms_info = meta_sector_map.get("meta_sectors", {}).get(ms, {})
            subs = ms_info.get("sub_categories", [])
            sector_rets = []
            for sub in subs:
                if sub in sector_returns:
                    sector_rets.append(sector_returns[sub][i])

            if sector_rets and len(index_window) > 1:
                # Compute rolling correlation (beta approximation)
                sector_arr = np.array(sector_rets)
                index_arr = np.array(index_window)
                if len(sector_arr) > 0 and len(index_arr) > 0:
                    sector_mean = np.mean(sector_arr)
                    index_mean = np.mean(index_arr)
                    cov = np.mean((sector_arr - sector_mean) * (index_arr - index_mean))
                    index_var = np.var(index_arr)
                    if index_var > 0:
                        beta = cov / index_var
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0

            row[f"beta_{ms}"] = float(beta)

        result_rows.append(row)

    return pl.DataFrame(result_rows)


def export_phase2_dataset(
    sentiment_df: pl.DataFrame,
    price_df: pl.DataFrame,
    index_df: pl.DataFrame,
    meta_sector_map: dict,
    tcn_model,
    lgbm_models: dict,
    iforest_model,
    device: torch.device,
    output_path: Path,
) -> None:
    """导出每日特征用于 Phase 2 Agent 训练（向量化批量推理）。

    只导出训练和推理时都可用的字段，不导出任何依赖未来标签的特征。
    """
    # Build sub-category sequences
    X_all, _, dates, sub_industries = build_sub_category_sequences(
        sentiment_df, meta_sector_map, lookback_days=5, price_df=price_df
    )

    if len(X_all) == 0:
        return

    # Batch TCN inference
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_all).to(device)
        tcn_reg, _ = tcn_model(X_tensor)
        tcn_reg_np = tcn_reg.cpu().numpy()
    meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
    reg_delta = np.zeros_like(tcn_reg_np)
    reg_delta[1:] = tcn_reg_np[1:] - tcn_reg_np[:-1]

    stability = np.zeros_like(tcn_reg_np, dtype=np.float32)
    for i in range(len(tcn_reg_np)):
        start = max(0, i - 4)
        window = tcn_reg_np[start : i + 1]
        signs = np.sign(window)
        dir_consistency = np.abs(np.sum(signs, axis=0)) / float(len(window))
        dispersion = np.std(window, axis=0) / (np.mean(np.abs(window), axis=0) + 1e-9)
        stability[i] = np.clip(dir_consistency - 0.5 * dispersion, 0.0, 1.0)

    flat_iforest = None
    if iforest_model is not None:
        try:
            flat_iforest = -iforest_model.score_samples(X_all.reshape(len(X_all), -1)).astype(np.float32)
        except Exception:
            flat_iforest = np.zeros(len(X_all), dtype=np.float32)

    if flat_iforest is None:
        flat_iforest = np.zeros(len(X_all), dtype=np.float32)

    news_heat = np.zeros(len(flat_iforest), dtype=np.float32)
    for i in range(len(flat_iforest)):
        start = max(0, i - 251)
        window = flat_iforest[start : i + 1]
        news_heat[i] = float(np.mean(window <= flat_iforest[i])) if len(window) > 1 else 0.5

    sector_col = "sub_category" if "sub_category" in sentiment_df.columns else "industry"
    sent_col = "sentiment_mean" if "sentiment_mean" in sentiment_df.columns else "sentiment_weighted"
    sub_to_idx = {sub: idx for idx, sub in enumerate(sub_industries)}
    gl_df = compute_global_leader_sentiment(sentiment_df, meta_sector_map)
    gl_map = {row["date"]: row for row in gl_df.iter_rows(named=True)} if not gl_df.is_empty() else {}

    feature_rows = []
    for i, date in enumerate(dates):
        row = {"date": date}
        row["iforest_score"] = float(flat_iforest[i])
        for m_idx, ms in enumerate(meta_sectors):
            ms_subs = meta_sector_map.get("meta_sectors", {}).get(ms, {}).get("sub_categories", [])
            day_df = sentiment_df.filter((pl.col("date") == date) & pl.col(sector_col).is_in(ms_subs))
            weights = (
                np.array(
                    [_get_market_cap_weight(sub, meta_sector_map) for sub in day_df[sector_col].to_list()],
                    dtype=np.float32,
                )
                if len(day_df) > 0
                else np.array([], dtype=np.float32)
            )
            sent_values = (
                np.array(day_df[sent_col].to_list(), dtype=np.float32)
                if len(day_df) > 0
                else np.array([], dtype=np.float32)
            )
            residuals = [float(X_all[i, -1, sub_to_idx[sub], 5]) for sub in ms_subs if sub in sub_to_idx]

            row[f"tcn_reg_{ms}"] = float(tcn_reg_np[i, m_idx])
            row[f"tcn_reg_delta_{ms}"] = float(reg_delta[i, m_idx])
            row[f"tcn_prediction_stability_{ms}"] = float(stability[i, m_idx])
            row[f"news_heat_{ms}"] = float(news_heat[i])
            row[f"global_leader_sentiment_{ms}"] = float(gl_map.get(date, {}).get(f"global_leader_{ms}", 0.0))
            row[f"meta_sentiment_{ms}"] = (
                float(np.average(sent_values, weights=weights))
                if len(sent_values) and weights.sum() > 0
                else float(np.mean(sent_values))
                if len(sent_values)
                else 0.0
            )
            row[f"sentiment_vs_price_residual_{ms}"] = float(np.mean(residuals)) if residuals else 0.0
            if ms in lgbm_models and hasattr(lgbm_models[ms], "predict"):
                # Leave placeholder at export time unless the caller provides full LightGBM features.
                row[f"lgbm_score_{ms}"] = 0.0
        feature_rows.append(row)

    feature_df = pl.DataFrame(feature_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.write_parquet(output_path)
