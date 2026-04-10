"""ETF price feature extraction utilities."""

from __future__ import annotations

import polars as pl


def compute_price_momentum(etf_df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Compute rolling price momentum per ETF."""
    return (
        etf_df
        .sort(["etf", "date"])
        .with_columns(
            (pl.col("close") - pl.col("close").shift(window).over("etf"))
            .alias("momentum")
        )
    )


def compute_weekly_return(etf_df: pl.DataFrame) -> pl.DataFrame:
    """Compute weekly return per ETF.

    Returns DataFrame with [week_start, etf, weekly_return].
    """
    return (
        etf_df
        .sort(["etf", "date"])
        .with_columns(pl.col("date").dt.truncate("1w").alias("week_start"))
        .group_by(["etf", "week_start"])
        .agg(
            pl.col("close").first().alias("week_open"),
            pl.col("close").last().alias("week_close"),
        )
        .with_columns(
            ((pl.col("week_close") - pl.col("week_open")) / pl.col("week_open")).alias("weekly_return")
        )
        .drop(["week_open", "week_close"])
    )
