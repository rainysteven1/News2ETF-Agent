"""Runtime-safe signal helper functions used by ONNX inference.

These helpers are copied from trainer-side dataset preparation logic, but kept
free of torch so the runtime app can stay lightweight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.config import shared_data_root

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


def load_signal_subcategories_from_label_stats(
    label_stats_path: str | Path | None = None,
) -> list[str] | None:
    """Load canonical signal sub-category order from label_stats.json."""
    path = Path(label_stats_path) if label_stats_path is not None else shared_data_root() / "label_stats.json"
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    by_sub_category = payload.get("by_sub_category", {})
    ordered: list[str] = []
    for full_name in by_sub_category.keys():
        if " / " in full_name:
            _, sub_name = full_name.split(" / ", 1)
        else:
            sub_name = full_name
        if sub_name not in ordered:
            ordered.append(sub_name)
    return ordered


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


def get_market_cap_weight(sub: str, meta_sector_map: dict[str, Any]) -> float:
    """Get market cap weight for a sub-category."""
    notes = meta_sector_map.get("notes", {})
    core_driver = notes.get("核心驱动（×1.5）", [])
    important = notes.get("重要辅助（×1.0）", [])
    edge_smoothing = notes.get("边缘平滑（×0.5）", [])

    if sub in core_driver:
        return 1.5
    if sub in important:
        return 1.0
    if sub in edge_smoothing:
        return 0.5
    return 1.0


def compute_global_leader_sentiment(
    sentiment_df: pl.DataFrame,
    meta_sector_map: dict[str, Any],
    lookback: int = 5,
) -> pl.DataFrame:
    """Build global leader sentiment using only closed historical sessions."""
    df = sentiment_df.sort("date")
    meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
    if df.is_empty():
        return pl.DataFrame({"date": [], **{f"global_leader_{ms}": [] for ms in meta_sectors}})

    dates = df["date"].unique().sort().to_list()
    sector_col = "symbol" if "symbol" in df.columns else ("sub_category" if "sub_category" in df.columns else "industry")
    sent_col = "sentiment_mean" if "sentiment_mean" in df.columns else "sentiment_weighted"

    available_symbols = set(df[sector_col].unique().to_list())
    symbol_history: dict[str, list[float]] = {}
    for symbol in GLOBAL_LEADER_BASKET:
        if symbol not in available_symbols:
            continue
        series: list[float] = []
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


def build_sub_category_sequences(
    sentiment_df: pl.DataFrame,
    meta_sector_map: dict[str, Any],
    lookback_days: int = 5,
    forecast_days: int = 5,
    price_df: pl.DataFrame | None = None,
    label_stats_path: str | Path | None = None,
    target_mode: str = "meta_excess_return",
) -> tuple[np.ndarray, np.ndarray, list[object], list[str]]:
    """Build the runtime ONNX sequence tensor without trainer/torch imports."""
    df = sentiment_df.sort("date")
    if df.is_empty():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), [], []

    sector_col = "sub_category" if "sub_category" in df.columns else "industry"
    sent_col = "sentiment_mean" if "sentiment_mean" in df.columns else "sentiment_weighted"
    std_col = "sentiment_std" if "sentiment_std" in df.columns else None
    dates = df["date"].unique().sort().to_list()
    canonical_subs = load_signal_subcategories_from_label_stats(label_stats_path)
    observed_subs = set(df[sector_col].drop_nulls().unique().to_list())
    if canonical_subs:
        sub_industries = canonical_subs
        df = df.filter(pl.col(sector_col).is_in(sub_industries))
    else:
        sub_industries = sorted(observed_subs)
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
        event_type_matrix[d_idx, s_idx] = _compute_event_type_score_for_row(row=row, sub_category=row[sector_col])

    price_source = price_df
    if (price_source is None or price_source.is_empty()) and "close" in df.columns:
        price_source = df.select(["date", sector_col, "close"]).rename({sector_col: "sub_category"})

    price_matrix = None
    if price_source is not None and not price_source.is_empty() and "close" in price_source.columns:
        price_source = price_source.sort("date")
        price_sector_col = "sub_category" if "sub_category" in price_source.columns else "industry"
        price_matrix = np.zeros((n_dates, n_sub), dtype=np.float32)
        for row in price_source.iter_rows(named=True):
            row_date = row.get("date")
            row_sector = row.get(price_sector_col)
            if row_date in date_to_idx and row_sector in sub_to_idx:
                price_matrix[date_to_idx[row_date], sub_to_idx[row_sector]] = float(row.get("close", 0.0) or 0.0)

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
                cov = np.mean((sent_hist[:, valid] - sent_mean) * (ret_hist[:, valid] - ret_mean), axis=0)
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
                meta_weights[m_idx, sub_to_idx[sub]] = get_market_cap_weight(sub, meta_sector_map)

    raw_targets: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    sample_dates: list[object] = []
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
        if price_matrix is not None and target_mode in {"meta_excess_return", "meta_return"}:
            valid_now = np.abs(price_matrix[current_idx]) > 1e-9
            valid_future = np.abs(price_matrix[future_idx]) > 1e-9
            valid_mask = valid_now & valid_future
            sub_returns = np.zeros(n_sub, dtype=np.float32)
            if np.any(valid_mask):
                sub_returns[valid_mask] = (
                    (price_matrix[future_idx, valid_mask] - price_matrix[current_idx, valid_mask])
                    / (price_matrix[current_idx, valid_mask] + 1e-9)
                )
            benchmark = float(np.mean(sub_returns[valid_mask])) if np.any(valid_mask) else 0.0
            target = np.zeros(n_meta, dtype=np.float32)
            for m_idx in range(n_meta):
                weights = meta_weights[m_idx]
                sector_valid = valid_mask & (weights > 0)
                if np.any(sector_valid):
                    target[m_idx] = float(np.average(sub_returns[sector_valid], weights=weights[sector_valid]))
                if target_mode == "meta_excess_return":
                    target[m_idx] -= benchmark
            raw_targets.append(target)
        else:
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
    sigma = float(clipped.std()) + 1e-9
    y = np.tanh(clipped / sigma).astype(np.float32)

    X = np.stack(sequences).astype(np.float32)
    return X, y, sample_dates, sub_industries
