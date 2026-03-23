"""Performance metrics calculation for backtest results."""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel


class Metrics(BaseModel):
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    weeks: int
    final_nav: float
    initial_capital: float


def calculate_metrics(
    backtest_df: pl.DataFrame,
    risk_free_rate: float = 0.03,
) -> Metrics:
    """Calculate performance metrics from weekly backtest results.

    Args:
        backtest_df: DataFrame with [week_start, nav, weekly_return]
        risk_free_rate: Annual risk-free rate

    Returns:
        Metrics Pydantic model with key metrics.
    """
    if len(backtest_df) == 0:
        return Metrics(
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            weeks=0,
            final_nav=0.0,
            initial_capital=0.0,
        )

    nav_series = backtest_df["nav"].to_numpy()
    weekly_returns = backtest_df["weekly_return"].to_numpy()

    # Total return
    total_return = (nav_series[-1] - nav_series[0]) / nav_series[0]

    # Annualized return (52 weeks/year)
    n_weeks = len(nav_series)
    n_years = n_weeks / 52
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Cumulative max drawdown
    cummax = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series - cummax) / cummax
    max_drawdown = float(drawdowns.min())

    # Sharpe ratio
    weekly_rf = risk_free_rate / 52
    if weekly_returns.std() > 0:
        excess = weekly_returns - weekly_rf
        sharpe_ratio = np.sqrt(52) * excess.mean() / excess.std()
    else:
        sharpe_ratio = 0.0

    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Win rate (positive weeks)
    win_rate = float((weekly_returns > 0).sum() / n_weeks) if n_weeks > 0 else 0.0

    # Sortino ratio
    downside = weekly_returns[weekly_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        excess = weekly_returns - weekly_rf
        sortino_ratio = np.sqrt(52) * excess.mean() / downside.std()
    else:
        sortino_ratio = 0.0

    return Metrics(
        total_return=float(total_return),
        annual_return=float(annual_return),
        max_drawdown=max_drawdown,
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        calmar_ratio=float(calmar_ratio),
        win_rate=win_rate,
        weeks=int(n_weeks),
        final_nav=float(nav_series[-1]),
        initial_capital=float(nav_series[0]),
    )
