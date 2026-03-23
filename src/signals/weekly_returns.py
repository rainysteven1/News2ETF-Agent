"""Weekly returns calculation and storage."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def compute_weekly_industry_returns(
    etf_prices: pl.DataFrame,
    industry_etf_map: dict[str, list[str]],
    week_start: str,
) -> dict[str, float]:
    """Compute weekly return per industry for a given week.

    Args:
        etf_prices: DataFrame with [date, etf, close]
        industry_etf_map: industry -> list of ETF names
        week_start: Monday date string (YYYY-MM-DD)

    Returns:
        Dict of industry -> weekly return (float)
    """
    week_df = etf_prices.filter(pl.col("date") >= week_start)
    if len(week_df) == 0:
        return {}

    last_day = week_df["date"].max()
    prev_day_df = etf_prices.filter(pl.col("date") < week_start)
    prev_last = prev_day_df["date"].max() if len(prev_day_df) > 0 else None
    if prev_last is None:
        return {}

    last_prices = week_df.filter(pl.col("date") == last_day).rename({"close": "close_curr"})
    prev_prices = etf_prices.filter(pl.col("date") == prev_last).rename({"close": "close_prev"})
    merged = last_prices.join(prev_prices, on="etf", how="inner")

    if len(merged) == 0:
        return {}

    merged = merged.with_columns(
        ((pl.col("close_curr") - pl.col("close_prev")) / pl.col("close_prev")).alias("etf_return")
    )

    industry_returns = {}
    for industry, etfs in industry_etf_map.items():
        etf_rets = merged.filter(pl.col("etf").is_in(etfs))["etf_return"].to_list()
        if etf_rets:
            industry_returns[industry] = sum(etf_rets) / len(etf_rets)
        else:
            industry_returns[industry] = 0.0

    return industry_returns


def save_weekly_returns(
    returns: dict[str, float],
    week_start: str,
    output_path: Path,
) -> None:
    """Append weekly industry returns to parquet file."""
    row = {"week_start": week_start, **returns}
    df = pl.DataFrame([row])

    if output_path.exists():
        existing = pl.read_parquet(output_path)
        df = pl.concat([existing, df])
    df.write_parquet(output_path)
