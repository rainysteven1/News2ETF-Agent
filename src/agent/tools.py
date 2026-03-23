"""Research tools for LangGraph agent — ONE file.

TOOL_REGISTRY: maps name -> tool function.
Researcher tools: read_market_news, compute_ml_signals, check_last_week_pnl, retrieve_history
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from langchain_core.tools import tool

from src.signals.knowledge_retrieval import KnowledgeRetrieval
from src.signals.raw_scorer import RawScorer
from src.utils.industry_map import IndustryMapper


# ─── read_market_news ─────────────────────────────────────────────────────────


@tool
def read_market_news(date: str) -> str:
    """Read raw market news articles for the week starting on `date`.

    Returns a list of news items (title + source + date) for the agent to analyse.
    The LLM researcher is responsible for classifying each article into an industry.
    """
    from src.config import load_config

    config = load_config()
    news_path = config.data.input_news_raw
    if not news_path.exists():
        return "No news data available."

    df = pl.read_parquet(news_path)
    week_start = datetime.strptime(date, "%Y-%m-%d")
    week_end = week_start + timedelta(days=6)

    # Parse datetime, filter to this week
    df = df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))
    df = df.filter((pl.col("date") >= week_start.date()) & (pl.col("date") <= week_end.date()))

    if len(df) == 0:
        return f"No news found for week of {date}."

    # Return raw news items — LLM classifies into industries
    lines = [f"## Week of {date} News ({len(df)} articles)"]
    for row in df.sort("datetime", descending=True).iter_rows(named=True):
        lines.append(
            f"- [{row['date']}] {row.get('title', 'N/A')} ({row.get('source', 'unknown')})"
        )
    return "\n".join(lines)


# ─── compute_ml_signals ────────────────────────────────────────────────────────


@tool
def compute_ml_signals(date: str) -> str:
    """Compute ML signals (momentum, heat, composite, trend) per industry for `date`."""
    from src.config import load_config

    config = load_config()
    sentiment_path = config.data.output_sentiment
    if not sentiment_path.exists():
        return "{}"

    sentiment_df = pl.read_parquet(sentiment_path)
    scorer = RawScorer(config, Path("checkpoints"))
    mapper = IndustryMapper(
        dict_path=config.data.industry_dict,
        etf_info=config.data.etf_info,
    )

    signals = scorer.score_all(sentiment_df, mapper.industries, date)
    if len(signals) == 0:
        return "{}"

    lines = ["## ML Signals Per Industry"]
    for row in signals.iter_rows(named=True):
        lines.append(
            f"- {row['industry']}: "
            f"momentum={row['momentum_score']:.3f} "
            f"heat={row['heat_anomaly']:.3f} "
            f"composite={row['composite_score']:.3f} "
            f"trend={row['trend_direction']}"
        )
    return "\n".join(lines)


# ─── check_last_week_pnl ──────────────────────────────────────────────────────


@tool
def check_last_week_pnl() -> str:
    """Return last week's portfolio return and holdings (behavioural finance memory)."""
    from src.config import load_config

    config = load_config()
    backtest_path = config.data.output_backtest

    if not backtest_path.exists():
        return json.dumps({"note": "No backtest results yet."}, ensure_ascii=False)

    df = pl.read_parquet(backtest_path)
    if len(df) == 0:
        return "{}"

    last_row = df.tail(1).row(0, named=True)

    return json.dumps(
        {
            "week_start": last_row.get("week_start", "unknown"),
            "weekly_return": last_row.get("weekly_return", 0.0),
            "nav": last_row.get("nav", 0.0),
            "holdings": last_row.get("holdings", {}),
            "invested_weight": last_row.get("invested_weight", 0.0),
        },
        ensure_ascii=False,
        indent=2,
    )


# ─── retrieve_history ──────────────────────────────────────────────────────────


@tool
def retrieve_history(date: str, query: str) -> str:
    """Retrieve similar historical news cases using TF-IDF."""
    from src.config import load_config

    config = load_config()
    news_path = config.data.input_news_raw

    if not news_path.exists():
        return "No historical news data available."

    news_df = pl.read_parquet(news_path)
    news_df = news_df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))
    news_df = news_df.with_columns(
        (pl.col("title").fill_null("") + " " + pl.col("content").fill_null("")).alias("text")
    )

    retrieval = KnowledgeRetrieval(news_df, text_column="text")
    results = retrieval.retrieve(query, top_k=5)

    if not results:
        return f"No similar cases found for: {query}"

    lines = [f"## Similar Cases (query: '{query}')"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n{i}. [{r['date']}]")
        lines.append(f"   Content: {r['content'][:200]}...")
        lines.append(f"   Similarity: {r['similarity']:.3f}")
    return "\n".join(lines)


# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "read_market_news": read_market_news,
    "compute_ml_signals": compute_ml_signals,
    "check_last_week_pnl": check_last_week_pnl,
    "retrieve_history": retrieve_history,
}
