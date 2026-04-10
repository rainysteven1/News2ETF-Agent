"""Keyword-based sentiment rules — CPU fallback when GPU/ML model is unavailable."""

from __future__ import annotations

POSITIVE_KEYWORDS = [
    "上涨", "大涨", "涨停", "突破", "牛市", "利好", "增长", "盈利",
    "超预期", "创新高", "买入", "看多", "推荐", "增持", "景气", "回暖",
]
NEGATIVE_KEYWORDS = [
    "下跌", "大跌", "跌停", "熊市", "利空", "亏损", "不及预期",
    "创新低", "卖出", "看空", "减持", "萧条", "衰退", "风险",
]


def keyword_sentiment(text: str) -> float:
    """Return a rough sentiment score in [-1, 1] based on keyword matching."""
    if not text:
        return 0.0
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
