"""Weekly/daily signal dataset builder — aggregates news sentiment & volume by industry-(week/day).

Supports two training strategies:
  - cross_industry=True: pool all industries into one shared TCN (104 wks × 8 industries = 832 samples)
  - cross_industry=False: train one TCN per industry

Compatible with weekly backtest frequency.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch

from trainer.config import SignalsDatasetConfig


class WeeklySignalDataset:
    """Build per-industry weekly/daily sentiment & volume DataFrames for ML signal training."""

    SENTIMENT_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

    def __init__(self, cfg: SignalsDatasetConfig, force: bool = False):
        assert cfg.raw_data_path is not None, "raw_data_path must be set"
        self.raw_path = Path(cfg.raw_data_path)
        self.output_sentiment = Path(cfg.output_sentiment) if cfg.output_sentiment else None
        self.train_end_week = datetime.fromisoformat(cfg.train_end_week)
        self.freq = cfg.freq  # "weekly" or "daily"
        self.cross_industry = cfg.cross_industry
        self.lf: pl.LazyFrame | None = None
        self.sentiment_df: pl.DataFrame | None = None
        self.volume_df: pl.DataFrame | None = None

        # Cache logic: if processed file exists and not forced, load directly
        if self.output_sentiment and self.output_sentiment.exists() and not force:
            self._load_cached(self.output_sentiment)
        else:
            self._load_raw()
            if self.output_sentiment:
                self._save_cached(self.output_sentiment)

    def _load_cached(self, path: Path) -> None:
        """Load pre-aggregated sentiment parquet (unpivoted: date, industry, sentiment_mean, news_count)."""
        df = pl.read_parquet(path)
        self.sentiment_df = df.sort(["industry", "date"])
        self.lf = df.lazy()

    def _save_cached(self, path: Path) -> None:
        """Save aggregated data as unpivoted parquet for reuse."""
        self._ensure_weekly("major_category")
        assert self.sentiment_df is not None and self.volume_df is not None
        sent_long = self.sentiment_df.unpivot(index="period", variable_name="industry", value_name="sentiment_mean")
        vol_long = self.volume_df.unpivot(index="period", variable_name="industry", value_name="news_count")
        merged = (
            sent_long.join(vol_long, on=["period", "industry"], how="left")
            .rename({"period": "date"})
            .sort(["industry", "date"])
        )
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
        """Aggregate by (period, industry) → weighted sentiment + news count."""
        if self.freq == "daily":
            period_expr = pl.col("datetime").dt.truncate("1d")
        else:
            period_expr = pl.col("datetime").dt.truncate("1w")

        # Weighted sentiment per (period, industry)
        sent_lf = (
            self.lf.with_columns(period_expr.alias("period"))
            .group_by(["period", industry_col])
            .agg(
                (pl.col("sentiment_weighted").sum() / (pl.col("confidence_sum").sum() + 1e-9)).alias(
                    "sentiment_weighted"
                )
            )
        )
        self.sentiment_df = (
            sent_lf.collect().pivot(values="sentiment_weighted", index="period", on=industry_col).sort("period")
        )

        # News volume per (period, industry)
        vol_lf = (
            self.lf.with_columns(period_expr.alias("period"))
            .group_by(["period", industry_col])
            .agg(pl.len().alias("news_count"))
        )
        self.volume_df = (
            vol_lf.collect().pivot(values="news_count", index="period", on=industry_col).sort("period").fill_null(0)
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
    """Build TCN training data from sentiment time series.

    Returns:
        X:     (N, seq_len, 1) — sentiment_mean sequences
        y_reg: (N, 1) — continuous sentiment delta at next step, clipped to [-1, 1]
               (not discrete direction, so tanh output is meaningful)
        y_cls: (N, 1) — 1 if |return| > threshold else 0
    """
    X_list, y_reg_list, y_cls_list = [], [], []

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        rets = ind_df["return"].to_numpy() if "return" in ind_df.columns else np.zeros_like(vals)

        for i in range(len(vals) - seq_len - 1):
            X_list.append(vals[i : i + seq_len])
            # Continuous target: normalized sentiment change (tanh-compatible)
            sent_delta = vals[i + seq_len] - vals[i + seq_len - 1]
            target = np.clip(sent_delta / (np.abs(vals[i + seq_len - 1]) + 1e-9), -1, 1)
            y_reg_list.append(target)
            next_ret = rets[i + seq_len]
            y_cls_list.append(1 if abs(next_ret) > anomaly_threshold else 0)

    X = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, 1)
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

    Features (10 dims):
        [delta_sentiment_1w, delta_sentiment_2w, news_count, news_heat,
         tcn_reg, tcn_cls, tcn_reg_delta,  news_count_std_5d,
         sentiment_volatility_5d, tcn_reg * news_heat]

    Returns X, y, dates, industries (for time-based split and per-industry IC).
    """
    feat_rows, label_rows, date_rows, industry_rows = [], [], [], []

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        nc = ind_df["news_count"].to_numpy()
        nh = ind_df["news_heat"].to_numpy() if "news_heat" in ind_df.columns else np.zeros_like(nc)
        dates = ind_df["date"].to_list()

        for i in range(seq_len + 1, len(vals) - 1):
            delta1 = vals[i] - vals[i - 1]
            delta2 = vals[i] - vals[i - 2]
            news_count = nc[i]
            news_heat = nh[i]

            x_t = torch.from_numpy(vals[i - seq_len : i].copy()).reshape(1, seq_len, 1).float().to(device)
            with torch.no_grad():
                reg_out, cls_out = tcn_model(x_t)
            tcn_reg = reg_out.item()
            tcn_cls = cls_out.item()

            # Previous step TCN reg for delta
            x_prev = torch.from_numpy(vals[i - seq_len - 1 : i - 1].copy()).reshape(1, seq_len, 1).float().to(device)
            with torch.no_grad():
                reg_prev, _ = tcn_model(x_prev)
            tcn_reg_prev = reg_prev.item()
            tcn_reg_delta = tcn_reg - tcn_reg_prev

            # Rolling stats
            news_count_std = np.std(nc[i - 4 : i + 1])
            sent_vol = np.std(vals[i - 4 : i + 1])

            # Interaction
            tcn_heat_interact = tcn_reg * news_heat

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
