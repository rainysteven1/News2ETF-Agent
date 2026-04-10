"""Research tools for LangGraph agent — ONE file.

TOOL_REGISTRY: maps name -> tool function.
Researcher tools: read_market_news, compute_ml_signals, check_last_week_pnl, retrieve_history
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from langchain_core.tools import tool

from src.agent.memos_retrieval import MemosRetrieval
from src.runtime import get_runtime
from src.utils.etf_universe import get_etf_universe
from src.utils.industry_map import IndustryMapper
from src.utils.meta_sector_map import meta_to_subs
from src.utils.news_loader import load_raw_news_df


_HOT_TERM_STOPWORDS = {
    "公司", "市场", "板块", "行业", "今日", "本周", "数据", "经济", "中国", "全球", "相关", "持续",
    "预计", "同比", "环比", "上涨", "下跌", "消息", "新闻", "亿元", "万元", "发布", "恢复", "实现",
    "新增", "继续", "推动", "政策", "企业", "项目", "价格", "机构", "基金", "证券", "投资", "交易",
}


def _current_config():
    from src.config import get_config, load_config

    try:
        return get_config()
    except Exception:
        return load_config()


def _extract_hot_terms_from_titles(titles: list[str], top_k: int = 8) -> list[str]:
    doc_freq: Counter[str] = Counter()
    for title in titles:
        normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(title or ""))
        if not normalized:
            continue
        seen: set[str] = set()
        for size in (2, 3, 4):
            for idx in range(0, max(0, len(normalized) - size + 1)):
                token = normalized[idx : idx + size]
                if len(token) < 2 or token in _HOT_TERM_STOPWORDS:
                    continue
                if token.isdigit() or any(ch.isdigit() for ch in token):
                    continue
                seen.add(token)
        for token in seen:
            doc_freq[token] += 1

    ranked_strict = [
        token
        for token, freq in doc_freq.most_common()
        if freq >= 3 and token not in _HOT_TERM_STOPWORDS
    ]
    if ranked_strict:
        return ranked_strict[:top_k]

    ranked_relaxed = [
        token
        for token, freq in doc_freq.most_common()
        if freq >= 2 and token not in _HOT_TERM_STOPWORDS
    ]
    return ranked_relaxed[:top_k]


def _summarize_raw_weekly_news(df: pl.DataFrame, date: str, max_headlines: int = 18) -> str:
    deduped = df.unique(subset=["title"], keep="first").sort("datetime", descending=True)
    source_counts = Counter(str(src or "unknown") for src in deduped.get_column("source").to_list())
    titles = [str(title or "") for title in deduped.get_column("title").to_list() if str(title or "").strip()]
    hot_terms = _extract_hot_terms_from_titles(titles)

    lines = [f"## Week of {date} News Summary ({len(df)} articles, raw fallback)"]
    if source_counts:
        lines.append("Top sources: " + ", ".join(f"{src}:{cnt}" for src, cnt in source_counts.most_common(5)))
    lines.append("Hot terms: " + (", ".join(hot_terms) if hot_terms else "none"))

    lines.append("Recent headlines:")
    for row in deduped.head(max_headlines).iter_rows(named=True):
        headline = str(row.get("title") or row.get("content") or "")[:60]
        lines.append(f"- [{row.get('date', 'N/A')}] {headline} ({row.get('source', 'unknown')})")
    return "\n".join(lines)


def _industry_keywords(industry: str) -> list[str]:
    keywords: list[str] = []
    for token in re.split(r"[/\s,，]+", str(industry or "")):
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


def _fallback_industry_top_news(df: pl.DataFrame, industry: str, top_k: int) -> str:
    keywords = _industry_keywords(industry)
    if not keywords:
        return f"No news for industry: {industry}"

    scored_rows: list[tuple[int, str, str, str]] = []
    for row in df.sort("datetime", descending=True).iter_rows(named=True):
        title = str(row.get("title") or "")
        content = str(row.get("content") or "")
        haystack = f"{title} {content}"
        score = sum(1 for keyword in keywords if keyword and keyword in haystack)
        if score <= 0:
            continue
        scored_rows.append((
            score,
            str(row.get("datetime", "")),
            title[:50] if title else content[:50],
            str(row.get("source", "unknown")),
        ))

    if not scored_rows:
        return f"No news for industry: {industry}"

    lines = [f"## Top News for {industry} (raw keyword fallback)"]
    for score, dt, title, source in scored_rows[:top_k]:
        lines.append(f"[raw score={score}] {dt} | {title} ({source})")
    return "\n".join(lines)


def _news_has_precomputed_labels(df: pl.DataFrame) -> bool:
    required = {"major_category", "sub_category", "sentiment"}
    return required.issubset(set(df.columns))


def _materialize_news_labels(
    *,
    date: str,
    weekly_df: pl.DataFrame,
    raw_news_df: pl.DataFrame,
    config,
) -> tuple[pl.DataFrame, str | None]:
    """Return weekly news with labels attached.

    Preference order:
    1. Pre-labeled parquet columns already present in the news data
    2. ONNX inference over raw news as a fallback
    """
    if _news_has_precomputed_labels(weekly_df):
        df = weekly_df
        if "l1_confidence" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("l1_confidence"))
        if "sub_category_confidence" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("sub_category_confidence"))
        if "sentiment_confidence" not in df.columns:
            df = df.with_columns(pl.lit(1.0).alias("sentiment_confidence"))
        return df, None

    from src.signals.onnx_inference import get_onnx_predictions

    preds = get_onnx_predictions(date, raw_news_df, config)
    pred_dates = preds["datetime"].str.to_datetime().dt.date()
    week_start_dt = datetime.strptime(date, "%Y-%m-%d")
    week_end_dt = week_start_dt + timedelta(days=6)
    week_mask = (pred_dates >= week_start_dt.date()) & (pred_dates <= week_end_dt.date())
    preds = preds.filter(week_mask)

    df = weekly_df.with_columns([
        preds["major_category"],
        preds["sentiment"],
        preds["l1_confidence"],
        preds["sub_category"],
        preds["sub_category_confidence"],
    ])
    return df, "onnx_fallback"

# ─── read_market_news ─────────────────────────────────────────────────────────


@tool
def read_market_news(date: str) -> str:
    """Read raw market news articles for the week starting on `date`.

    Runs FinBERT + SetFit ONNX inference to classify each article's
    major category, sentiment, and sub-category. Results are cached per week.
    """
    config = _current_config()
    news_path = config.data.input_news_raw
    raw_news_df = load_raw_news_df(news_path)
    if len(raw_news_df) == 0:
        return "No news data available."

    df = raw_news_df
    week_start_dt = datetime.strptime(date, "%Y-%m-%d")
    week_end_dt = week_start_dt + timedelta(days=6)

    df = df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))
    df = df.filter((pl.col("date") >= week_start_dt.date()) & (pl.col("date") <= week_end_dt.date()))

    if len(df) == 0:
        return f"No news found for week of {date}."

    try:
        df, _ = _materialize_news_labels(
            date=date,
            weekly_df=df,
            raw_news_df=raw_news_df,
            config=config,
        )
    except ModuleNotFoundError as exc:
        return f"{_summarize_raw_weekly_news(df, date)}\n\n[Fallback] {exc}"
    except Exception as exc:
        return f"{_summarize_raw_weekly_news(df, date)}\n\n[Fallback] ONNX news classification unavailable: {exc}"

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
    """Compute meta-sector ML signals from the exported signals inference cache."""
    from src.agent.features import AgentFeatureBuilder

    builder = AgentFeatureBuilder()
    snapshot = builder.build_signal_snapshot(date)
    if not snapshot:
        return "{}"

    lines = ["## ML Signals Per Meta Sector"]
    for sector, values in snapshot.items():
        tcn_reg = float(values.get("tcn_reg", 0.0) or 0.0)
        lgbm_score = float(values.get("lgbm_score", 0.0) or 0.0)
        news_heat = float(values.get("news_heat", 0.0) or 0.0)
        meta_sent = float(values.get("meta_sentiment", 0.0) or 0.0)
        stability = float(values.get("tcn_prediction_stability", 0.0) or 0.0)
        lines.append(
            f"- {sector}: "
            f"tcn={tcn_reg:.3f} "
            f"lgbm={lgbm_score:.3f} "
            f"heat={news_heat:.3f} "
            f"meta_sent={meta_sent:.3f} "
            f"stability={stability:.3f}"
        )
    return "\n".join(lines)


# ─── check_last_week_pnl ──────────────────────────────────────────────────────


@tool
def check_last_week_pnl() -> str:
    """Return last week's portfolio return and holdings (behavioural finance memory)."""
    config = _current_config()
    runtime = get_runtime()

    backtest_path = config.data.output_backtest
    if runtime.run_id and runtime.checkpoint_dir is not None:
        candidate = Path(runtime.checkpoint_dir) / runtime.run_id / "backtest_results.parquet"
        if candidate.exists():
            backtest_path = candidate

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
    config = _current_config()

    if not getattr(config.agent, "enable_history_retrieval", False):
        return "History retrieval disabled by config."

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
    config = _current_config()
    news_path = config.data.input_news_raw
    raw_news_df = load_raw_news_df(news_path)
    if len(raw_news_df) == 0:
        return "No news data available."

    df = raw_news_df
    week_start_dt = datetime.strptime(date, "%Y-%m-%d")
    week_end_dt = week_start_dt + timedelta(days=6)

    df = df.with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("date"))
    df = df.filter((pl.col("date") >= week_start_dt.date()) & (pl.col("date") <= week_end_dt.date()))

    if len(df) == 0:
        return f"No news found for week of {date}."

    try:
        df, _ = _materialize_news_labels(
            date=date,
            weekly_df=df,
            raw_news_df=raw_news_df,
            config=config,
        )
    except ModuleNotFoundError as exc:
        fallback = _fallback_industry_top_news(df, industry, top_k)
        return f"{fallback}\n\n[Fallback] {exc}"
    except Exception as exc:
        fallback = _fallback_industry_top_news(df, industry, top_k)
        return f"{fallback}\n\n[Fallback] ONNX industry classification unavailable: {exc}"

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
def get_etf_candidates(industry: str, date: str = "") -> str:
    """Get candidate ETFs for a sub_category industry.

    Returns only ETFs that exist in the backtest price universe.
    If `date` is provided, further restricts to ETFs tradable for that week.
    """
    from src.config import load_config

    config = load_config()
    mapper = IndustryMapper(
        dict_path=config.data.industry_dict,
        etf_info=config.data.etf_info,
    )
    resolver = get_etf_universe(str(config.data.etf_info), str(config.data.etf_prices))

    # Find the small_cat across all large_cats
    all_small_cats: list[tuple[str, str]] = []  # (large_cat, small_cat)
    for large_cat in mapper.get_large_cats():
        for small_cat in mapper.get_small_cats(large_cat):
            all_small_cats.append((large_cat, small_cat))

    matched_large = None
    search_groups: list[tuple[str, str]] = []

    if industry in mapper.get_large_cats():
        matched_large = industry
        search_groups = [(industry, small_cat) for small_cat in mapper.get_small_cats(industry)]
    else:
        # Accept meta-sector inputs such as "医药健康" even when the ETF universe
        # still uses legacy large-category labels.
        for sub in meta_to_subs(industry):
            for large_cat, small_cat in all_small_cats:
                if small_cat != sub:
                    continue
                matched_large = large_cat
                if (large_cat, small_cat) not in search_groups:
                    search_groups.append((large_cat, small_cat))

    if not search_groups:
        for large_cat, small_cat in all_small_cats:
            if small_cat == industry or small_cat in industry or industry in small_cat:
                matched_large = large_cat
                search_groups = [(large_cat, small_cat)]
                break
    if matched_large is None:
        return f"Industry not found: {industry}"

    indices: list[str] = []
    for large_cat, small_cat in search_groups:
        indices.extend(mapper.get_indices(large_cat, small_cat))
    indices = list(dict.fromkeys(indices))
    if not indices:
        return f"No tracking indices for industry: {industry}"

    header = f"## ETF Candidates for {industry}"
    if date:
        header = f"{header} (tradable on week {date})"
    lines = [header]
    seen_codes: set[str] = set()
    for idx_name in indices:
        idx_etfs = resolver.candidates_for_index(idx_name, week_start=(date or None))
        if not idx_etfs:
            continue
        for candidate in idx_etfs[:3]:
            if candidate.code in seen_codes:
                continue
            seen_codes.add(candidate.code)
            lines.append(
                f"- {candidate.code} {candidate.name} | 跟踪:{candidate.tracking_index} | 规模:{candidate.aum:.1f}亿"
            )

    if len(lines) > 1:
        return "\n".join(lines)
    if date:
        return f"No tradable ETFs found for industry: {industry} on week {date}"
    return f"No ETFs found in price universe for industry: {industry}"


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


# ─── build_decision_context ───────────────────────────────────────────────────


@tool
def build_decision_context(date: str) -> str:
    """Build complete decision context by calling AgentFeatureBuilder.

    This tool constructs all features (A/B/C/D/E) needed for agent decision making:
    - Feature A: TCN sequence (8 meta sectors × 5 days momentum)
    - Feature B: News summary (top news per meta sector)
    - Feature C: Market state (price momentum, volume, volatility)
    - Feature D: Position state (current holdings, weekly returns)
    - Feature E: Sentiment vs price divergence

    Returns a formatted string with all features for the agent.
    """
    from src.agent.features import AgentFeatureBuilder
    from src.agent.prompt_manager import PromptManager
    from src.config import load_config

    builder = AgentFeatureBuilder()
    config = load_config()

    # Build all features
    features = builder.build_agent_features(
        date=date,
        current_holdings={},
        current_time=f"{date} 08:30:00",
    )

    ml_snapshot = features.get("ml_signal_snapshot", {})
    sector_signals = {
        sector: float(values.get("lgbm_score", values.get("tcn_reg", 0.0)) or 0.0)
        for sector, values in ml_snapshot.items()
    }
    current_context = {
        "date": date,
        "market_state": features.get("market_state", {}).get("market_state", "neutral"),
        "vol_percentile": float(max((v.get("news_heat", 0.0) for v in ml_snapshot.values()), default=0.0)),
        "sector_signals": sector_signals,
        "forbidden_zones": {},
    }
    prompt_manager = PromptManager()
    good_patterns, bad_patterns, reasoning_summary = prompt_manager.update_prompt(current_context)

    historical_memory: dict[str, object] = {
        "good_patterns_summary": good_patterns,
        "bad_patterns_summary": bad_patterns,
        "reasoning_summary": reasoning_summary,
    }

    memos_api_key = getattr(config.memos, "api_key", None) if hasattr(config, "memos") else None
    memos_base_url = getattr(config.memos, "base_url", None) if hasattr(config, "memos") else None
    if memos_api_key:
        top_query_parts = sorted(sector_signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        query = " | ".join(f"{k}:{v:.3f}" for k, v in top_query_parts)
        retrieval = MemosRetrieval(api_key=memos_api_key, base_url=memos_base_url)
        historical_memory["memos_cases"] = retrieval.retrieve(query=query, conversation_id=date, top_k=3)

    top_signals = sorted(sector_signals.items(), key=lambda x: x[1], reverse=True)[:3]
    weakest_signals = sorted(sector_signals.items(), key=lambda x: x[1])[:2]
    top_holdings = features.get("position_state", {}).get("top_holdings", [])
    human_summary_lines = [
        f"date={date}",
        f"market_state={features.get('market_state', {}).get('market_state', 'neutral')}",
        "top_meta_signals=" + ", ".join(f"{k}:{v:.3f}" for k, v in top_signals) if top_signals else "top_meta_signals=none",
        "weak_meta_signals=" + ", ".join(f"{k}:{v:.3f}" for k, v in weakest_signals) if weakest_signals else "weak_meta_signals=none",
        "top_holdings="
        + (", ".join(f"{k}:{v:.2%}" for k, v in top_holdings) if top_holdings else "empty"),
        f"memos_cases={len(historical_memory.get('memos_cases', [])) if isinstance(historical_memory.get('memos_cases'), list) else 0}",
    ]
    human_summary = "\n".join(human_summary_lines)

    context = {
        "schema_version": "decision_context.v2",
        **features,
        "good_patterns": good_patterns,
        "bad_patterns": bad_patterns,
        "historical_memory": historical_memory,
        "human_summary": human_summary,
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "read_market_news": read_market_news,
    "compute_ml_signals": compute_ml_signals,
    "check_last_week_pnl": check_last_week_pnl,
    "retrieve_history": retrieve_history,
    "get_industry_top_news": get_industry_top_news,
    "get_etf_candidates": get_etf_candidates,
    "store_decision": store_decision,
    "build_decision_context": build_decision_context,
}
