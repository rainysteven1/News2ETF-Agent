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

from src.signals.memos_retrieval import MemosRetrieval
from src.signals.onnx_inference import get_onnx_predictions
from src.signals.raw_scorer import RawScorer
from src.utils.industry_map import IndustryMapper

# ─── read_market_news ─────────────────────────────────────────────────────────


@tool
def read_market_news(date: str) -> str:
    """Read raw market news articles for the week starting on `date`.

    Runs FinBERT + SetFit ONNX inference to classify each article's
    major category, sentiment, and sub-category. Results are cached per week.
    """
    from src.config import load_config

    config = load_config()
    news_path = config.data.input_news_raw
    if not news_path.exists():
        return "No news data available."

    df = pl.read_parquet(news_path)
    week_start_dt = datetime.strptime(date, "%Y-%m-%d")
    week_end_dt = week_start_dt + timedelta(days=6)

    # Run ONNX inference on the full parquet for this week (cache handles it)
    preds = get_onnx_predictions(date, df, config)

    # Now filter BOTH df and preds to this week
    df = df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))
    df = df.filter((pl.col("date") >= week_start_dt.date()) & (pl.col("date") <= week_end_dt.date()))

    if len(df) == 0:
        return f"No news found for week of {date}."

    # preds has same row order as the raw df (unfiltered) — filter it too
    # by matching datetime values
    pred_dates = preds["datetime"].str.to_datetime().dt.date()
    week_mask = (pred_dates >= week_start_dt.date()) & (pred_dates <= week_end_dt.date())
    preds = preds.filter(week_mask)

    # Merge predictions with filtered df (same row order)
    df = df.with_columns([
        preds["major_category"],
        preds["sentiment"],
        preds["l1_confidence"],
        preds["sub_category"],
        preds["sub_category_confidence"],
    ])

    lines = [f"## Week of {date} News ({len(df)} articles)"]
    for row in df.sort("datetime", descending=True).iter_rows(named=True):
        lines.append(
            f"- [{row['date']}] {row.get('title', 'N/A')} ({row.get('source', 'unknown')})"
            f" | 行业: {row.get('major_category', 'N/A')}/{row.get('sub_category', 'N/A')}"
            f" | 情感: {row.get('sentiment', 'N/A')}"
            f" (conf={row.get('l1_confidence', 0.0):.2f})"
            f" | 子行业置信: {row.get('sub_category_confidence', 0.0):.2f}"
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

    # holdings may be stored as a JSON string or dict depending on backtest engine
    raw_holdings = last_row.get("holdings", {})
    if isinstance(raw_holdings, str):
        try:
            holdings = json.loads(raw_holdings)
        except Exception:
            holdings = {}
    else:
        holdings = raw_holdings or {}

    return json.dumps(
        {
            "week_start": last_row.get("week_start", "unknown"),
            "weekly_return": last_row.get("weekly_return", 0.0),
            "nav": last_row.get("nav", 0.0),
            "holdings": holdings,
            "invested_weight": last_row.get("invested_weight", 0.0),
        },
        ensure_ascii=False,
        indent=2,
    )


# ─── retrieve_history ──────────────────────────────────────────────────────────


@tool
def retrieve_history(date: str, query: str) -> str:
    """Retrieve similar historical investment cases via Memos vector search.

    Uses MemOS /search/memory API for efficient embedding-based retrieval,
    much faster than TF-IDF on raw news.
    """
    from src.config import load_config

    config = load_config()

    memos_api_key = getattr(config.memos, "api_key", None) if hasattr(config, "memos") else None
    memos_base_url = getattr(config.memos, "base_url", None) if hasattr(config, "memos") else None

    if not memos_api_key:
        return "Memos API key not configured. Set MEMOS_API_KEY environment variable."

    retrieval = MemosRetrieval(api_key=memos_api_key, base_url=memos_base_url)
    results = retrieval.retrieve(query, conversation_id=date, top_k=5)

    if not results or results[0].get("error"):
        return f"No similar cases found for: {query}"

    lines = [f"## Similar Historical Cases (query: '{query}')"]
    for i, r in enumerate(results, 1):
        content = r.get("content", "")[:200]
        sim = r.get("similarity", 0.0)
        lines.append(f"\n{i}. [similarity={sim:.3f}]")
        lines.append(f"   {content}")
    return "\n".join(lines)


# ─── get_industry_top_news ─────────────────────────────────────────────────────


@tool
def get_industry_top_news(date: str, industry: str, top_k: int = 3) -> str:
    """Get top-k most confident news for a specific sub_category industry.

    Returns compressed news summaries sorted by confidence, suitable for LLM input.
    """
    from src.config import load_config

    config = load_config()
    news_path = config.data.input_news_raw
    if not news_path.exists():
        return "No news data available."

    df = pl.read_parquet(news_path)
    week_start_dt = datetime.strptime(date, "%Y-%m-%d")
    week_end_dt = week_start_dt + timedelta(days=6)

    # Get predictions (cache handles the full week)
    preds = get_onnx_predictions(date, df, config)

    # Filter to this week
    df = df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))
    df = df.filter((pl.col("date") >= week_start_dt.date()) & (pl.col("date") <= week_end_dt.date()))

    if len(df) == 0:
        return f"No news found for week of {date}."

    # preds aligned with raw df — filter it too
    pred_dates = preds["datetime"].str.to_datetime().dt.date()
    week_mask = (pred_dates >= week_start_dt.date()) & (pred_dates <= week_end_dt.date())
    preds = preds.filter(week_mask)

    df = df.with_columns([
        preds["major_category"],
        preds["sentiment"],
        preds["l1_confidence"],
        preds["sub_category"],
        preds["sub_category_confidence"],
    ])

    # Filter to the requested sub_category
    ind_df = df.filter(pl.col("sub_category") == industry)
    if len(ind_df) == 0:
        # Try major_category match if sub_category not found
        ind_df = df.filter(pl.col("major_category") == industry)

    if len(ind_df) == 0:
        return f"No news for industry: {industry}"

    # Sort by composite confidence and take top-k
    ind_df = ind_df.with_columns(
        (pl.col("l1_confidence") + pl.col("sub_category_confidence")).alias("composite_conf")
    )
    top = ind_df.sort("composite_conf", descending=True).head(top_k)

    lines = []
    for row in top.iter_rows(named=True):
        title = str(row.get("title", ""))[:50]
        sentiment = row.get("sentiment", "neutral")
        conf = row.get("l1_confidence", 0.0)
        lines.append(
            f"[{sentiment}] {conf:.2f} | {title}"
        )
    return "\n".join(lines) if lines else f"No news for industry: {industry}"


# ─── get_etf_candidates ───────────────────────────────────────────────────────


@tool
def get_etf_candidates(industry: str) -> str:
    """Get candidate ETFs for a sub_category industry.

    Returns ETF list with code, name, tracking index, AUM, for LLM to select from.
    """
    from src.config import load_config

    config = load_config()
    mapper = IndustryMapper(
        dict_path=config.data.industry_dict,
        etf_info=config.data.etf_info,
    )

    # Find the small_cat across all large_cats
    all_small_cats: list[tuple[str, str]] = []  # (large_cat, small_cat)
    for large_cat in mapper.get_large_cats():
        for small_cat in mapper.get_small_cats(large_cat):
            all_small_cats.append((large_cat, small_cat))

    matched_large = None
    for large_cat, small_cat in all_small_cats:
        if small_cat == industry or small_cat in industry:
            matched_large = large_cat
            break
    if matched_large is None:
        return f"Industry not found: {industry}"

    indices = mapper.get_indices(matched_large, industry)
    if not indices:
        return f"No tracking indices for industry: {industry}"

    # Get ETF info from etf_info parquet
    etf_df = pl.read_parquet(config.data.etf_info)
    aum_col = [c for c in etf_df.columns if "基金规模" in c][0]

    lines = [f"## ETF Candidates for {industry}"]
    for idx_name in indices:
        # Find ETFs tracking this index
        idx_etfs = etf_df.filter(pl.col("跟踪指数名称") == idx_name)
        if len(idx_etfs) == 0:
            continue
        # Sort by AUM and take top 3
        idx_etfs = idx_etfs.sort(aum_col, descending=True).head(3)
        for row in idx_etfs.iter_rows(named=True):
            code = row.get("代码", "")
            name = row.get("名称", "")
            aum_val = row.get(aum_col, 0)
            lines.append(
                f"- {code} {name} | 跟踪:{idx_name} | 规模:{aum_val:.1f}亿"
            )

    return "\n".join(lines) if len(lines) > 1 else f"No ETFs found for industry: {industry}"


# ─── store_decision ──────────────────────────────────────────────────────────


@tool
def store_decision(date: str, decision: str, context: str = "") -> str:
    """Store an investment decision in Memos for future retrieval.

    This is called after the agent makes a decision, to build up
    historical memory for future similar situations.
    """
    from src.config import load_config

    config = load_config()

    memos_api_key = getattr(config.memos, "api_key", None) if hasattr(config, "memos") else None
    memos_base_url = getattr(config.memos, "base_url", None) if hasattr(config, "memos") else None

    if not memos_api_key:
        return "Memos API key not configured."

    retrieval = MemosRetrieval(api_key=memos_api_key, base_url=memos_base_url)
    success = retrieval.add_decision(
        conversation_id=date,
        decision=decision,
        context=context,
        date=date,
    )

    if success:
        return f"Decision stored in Memos for {date}."
    return "Failed to store decision in Memos."


# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "read_market_news": read_market_news,
    "compute_ml_signals": compute_ml_signals,
    "check_last_week_pnl": check_last_week_pnl,
    "retrieve_history": retrieve_history,
    "get_industry_top_news": get_industry_top_news,
    "get_etf_candidates": get_etf_candidates,
    "store_decision": store_decision,
}
