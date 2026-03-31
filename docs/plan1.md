# Plan: 重构 `RawScorer` — 基于 `get_onnx_predictions` 缓存做情感动量分析

## Context

`RawScorer` 当前依赖 PyTorch LSTM + IsolationForest + LightGBM，需要区分 CPU/GPU 模式。用户已确认：
1. **情感部分已通过 `get_onnx_predictions` 完成**，FinBERT + SetFit ONNX 推理结果已缓存
2. **`RawScorer` 不再需要管 CPU/GPU**，彻底移除 PyTorch 依赖
3. `RawScorer` 只需：**拿 `get_onnx_predictions` 的缓存结果** → 按 `major_category` 聚合 sentiment → 计算 momentum/heat/composite/trend

## 关键文件

| 文件 | 作用 |
|------|------|
| [src/signals/raw_scorer.py](src/signals/raw_scorer.py) | 重构：移除 PyTorch/LSTM/IF/LGBM，新增基于缓存的情感聚合 |
| [src/signals/onnx_inference.py](src/signals/onnx_inference.py) | 已完成：提供 `get_onnx_predictions` 缓存入口 |
| [src/agent/tools.py](src/agent/tools.py) | 已完成：集成 `get_onnx_predictions` |

## 重构 `RawScorer` 逻辑

### 新的输入输出

**输入**: `get_onnx_predictions(week_start, raw_news_df, config)` 返回的缓存 DataFrame
- `major_category`: 行业大类
- `sentiment`: "negative" / "neutral" / "positive"
- `l1_confidence`: FinBERT 置信度

**输出**: `score_all()` 返回每行业（major_category）的信号:
```
momentum_score: float   # 基于情感均值 * 置信度的滚动动量
heat_anomaly: float     # 新闻数量和置信度离散度
composite_score: float   # momentum * heat
trend_direction: int     # 1 (bullish) / 0 (neutral) / -1 (bearish)
```

### 情感 → 数值转换

```
sentiment = "negative"  → score = -1
sentiment = "neutral"   → score =  0
sentiment = "positive"  → score = +1
weighted_score = score * l1_confidence
```

### 实现步骤

#### Step 1: 重写 `src/signals/raw_scorer.py`

```python
"""ML signal scorer — 基于 ONNX 缓存的情感聚合。

不再依赖 PyTorch / LSTM / IsolationForest / LightGBM。
情感数据来自 get_onnx_predictions() 缓存。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.config import AgentRootConfig
from src.signals.onnx_inference import get_onnx_predictions


# ─── 工具函数 ────────────────────────────────────────────────────────────────


SENTIMENT_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def _sentiment_score(sentiment: str, confidence: float) -> float:
    return SENTIMENT_MAP.get(sentiment, 0.0) * confidence


# ─── 行业情感聚合 ────────────────────────────────────────────────────────────


def aggregate_industry_sentiment(
    week_start: str,
    raw_news_df: pl.DataFrame,
    config: AgentRootConfig,
) -> pl.DataFrame:
    """对给定周的新闻按 major_category 聚合情感。

    返回 DataFrame:
        industry, date, sentiment_mean, sentiment_std, news_count, avg_confidence
    """
    # 获取 ONNX 推理结果（从缓存）
    preds = get_onnx_predictions(week_start, raw_news_df, config)

    # 解析 datetime 为 date
    preds = preds.with_columns(
        pl.col("datetime").str.to_datetime().dt.date().alias("date")
    )

    # 加权情感分数
    preds = preds.with_columns(
        pl.struct(["sentiment", "l1_confidence"])
        .map_elements(
            lambda s: _sentiment_score(s["sentiment"], s["l1_confidence"]),
            return_dtype=pl.Float64,
        )
        .alias("weighted_sentiment")
    )

    # 按 major_category + date 聚合
    agg = preds.group_by(["major_category", "date"]).agg(
        pl.mean("weighted_sentiment").alias("sentiment_mean"),
        pl.std("weighted_sentiment").alias("sentiment_std"),
        pl.len().alias("news_count"),
        pl.mean("l1_confidence").alias("avg_confidence"),
    )

    return agg.rename({"major_category": "industry"})


# ─── RawScorer ────────────────────────────────────────────────────────────────


class RawScorer:
    """基于 ONNX 缓存的情感动量评分器。"""

    def __init__(self, config: AgentRootConfig, checkpoint_dir: Path | None = None):
        self.config = config
        self.sequence_length = config.model.lstm.sequence_length

    def _get_lookback(
        self,
        sentiment_history: pl.DataFrame,
        industry: str,
        date: str,
        window: int,
    ) -> list[float]:
        """取该行业截至 date 的最近 window 天情感均值序列。"""
        ind_df = (
            sentiment_history.filter(pl.col("industry") == industry)
            .filter(pl.col("date") <= date)
            .sort("date")
            .tail(window)
        )
        if len(ind_df) < window:
            return []
        return ind_df["sentiment_mean"].to_numpy().tolist()

    def score_industry(
        self,
        sentiment_history: pl.DataFrame,
        industry: str,
        date: str,
    ) -> dict:
        """计算某行业在 date 的所有信号。"""
        seq = self._get_lookback(sentiment_history, industry, date, self.sequence_length)

        if len(seq) < self.sequence_length:
            return {
                "momentum_score": 0.0,
                "heat_anomaly": 0.0,
                "composite_score": 0.0,
                "trend_direction": 0,
            }

        momentum = float(np.mean(seq[-self.sequence_length:]))
        heat = float(np.clip(np.mean(seq[-self.sequence_length:]) / self.sequence_length, 0, 1))

        # trend: 看最近2周对比
        if len(seq) >= self.sequence_length * 2:
            recent = np.mean(seq[-self.sequence_length:])
            prior = np.mean(seq[-self.sequence_length * 2 : -self.sequence_length])
            delta = recent - prior
        else:
            delta = momentum

        return {
            "momentum_score": float(np.clip(momentum, -1, 1)),
            "heat_anomaly": heat,
            "composite_score": float(np.clip(momentum * heat, -1, 1)),
            "trend_direction": 1 if delta > 0.05 else (-1 if delta < -0.05 else 0),
        }

    def score_all(
        self,
        sentiment_history: pl.DataFrame,
        industries: list[str],
        date: str,
    ) -> pl.DataFrame:
        rows = []
        for industry in industries:
            s = self.score_industry(sentiment_history, industry, date)
            rows.append({"industry": industry, "date": date, **s})
        return pl.DataFrame(rows)
```

#### Step 2: 确认 `score_all` 调用方适配

`compute_ml_signals` 工具（`tools.py`）调用 `RawScorer.score_all()`。改动后需要先构建 `sentiment_history` DataFrame（包含历史各周的聚合情感），传给 `score_all`。

新增 `build_sentiment_history` 函数：

```python
def build_sentiment_history(
    end_week_start: str,
    n_weeks: int,
    config: AgentRootConfig,
) -> pl.DataFrame:
    """从缓存中构建最近 n_weeks 的行业情感历史。"""
    news_path = config.data.input_news_raw
    raw_df = pl.read_parquet(news_path)

    end = datetime.strptime(end_week_start, "%Y-%m-%d")
    history_rows = []
    for i in range(n_weeks):
        week_start = (end - timedelta(weeks=i)).strftime("%Y-%m-%d")
        week_df = raw_df.filter(/* 过滤该周 */)
        agg = aggregate_industry_sentiment(week_start, week_df, config)
        history_rows.append(agg)

    return pl.concat(history_rows)
```

#### Step 3: 更新 `compute_ml_signals` 工具

```python
@tool
def compute_ml_signals(date: str) -> str:
    """Compute ML signals (momentum, heat, composite, trend) per industry for `date`."""
    from src.config import load_config

    config = load_config()
    news_path = config.data.input_news_raw
    if not news_path.exists():
        return "{}"

    raw_df = pl.read_parquet(news_path)

    # 构建历史情感（从缓存，含当前周）
    n_weeks = config.model.lstm.sequence_length + 2
    sentiment_history = build_sentiment_history(date, n_weeks, config)

    mapper = IndustryMapper(
        dict_path=config.data.industry_dict,
        etf_info=config.data.etf_info,
    )

    scorer = RawScorer(config)
    signals = scorer.score_all(sentiment_history, mapper.industries, date)
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
```

## 验证方式

1. 调用 `compute_ml_signals("2024-01-01")`，确认输出 momentum/heat/composite/trend
2. 确认 `RawScorer` 不再 import torch / sklearn / lightgbm
3. 确认情感数据直接来自 `get_onnx_predictions` 缓存，无重复推理
