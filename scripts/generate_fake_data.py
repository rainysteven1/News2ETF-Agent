"""Generate fake data for testing the agent pipeline.

Creates:
- data/fake_news.parquet          # fake news for 1 week
- data/industry_sentiment.parquet  # fake sentiment per sub_category per week
- data/backtest_results.parquet    # fake backtest history
- data/fake_etf_info.parquet       # fake ETF info subset
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)


# ── Load industry dict ──────────────────────────────────────────────────────────

with open(ROOT / "data" / "industry_dict.json", encoding="utf-8") as f:
    industry_dict = json.load(f)

# Flatten to list of (large_cat, small_cat) — 47 total
sub_cats = []
for large_cat, small_cats in industry_dict.items():
    for small_cat in small_cats:
        sub_cats.append((large_cat, small_cat))

print(f"Found {len(sub_cats)} sub-categories")


# ── 1. Fake news: 1 week, ~100 articles across industries ──────────────────────

SENTIMENTS = ["positive", "neutral", "negative"]
MAJOR_CATS = list(industry_dict.keys())

# Generate news ONLY for week of 2024-01-01 to 2024-01-07 (7 days = 168 hours)
# 100 articles at 96min intervals = 160h, all within the week
week_start = datetime(2024, 1, 1)
dates = [(week_start + timedelta(minutes=i * 96)) for i in range(100)]

news_rows = []
for i, dt in enumerate(dates):
    major = random.choice(MAJOR_CATS)
    # pick a sub_cat from this major
    sub = random.choice(list(industry_dict[major].keys()))
    sentiment = random.choice(SENTIMENTS)
    news_rows.append({
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "title": f"测试新闻{i} — {sub[:4]} {random.choice(['利好','中性','利空'])}",
        "content": f"这是关于{sub}的新闻内容摘要。{random.choice(['行业上涨','行业下跌','行业平稳'])}。",
        "source": random.choice(["yuncaijing", "eastmoney", "sina"]),
    })

pl.DataFrame(news_rows).write_parquet(ROOT / "data" / "fake_news.parquet")
print(f"Written: data/fake_news.parquet ({len(news_rows)} rows)")


# ── 2. Fake sentiment: 8 weeks × 47 sub_cats ──────────────────────────────────

# Build a realistic momentum trajectory for each sub_cat
# Start from random base, walk with momentum
sub_cat_momentum = {}
for large_cat, small_cat in sub_cats:
    base = random.uniform(-0.3, 0.3)
    sub_cat_momentum[small_cat] = [base]  # week 0

# Walk 7 more weeks
for week in range(1, 8):
    for large_cat, small_cat in sub_cats:
        prev = sub_cat_momentum[small_cat][-1]
        # Random walk with mean reversion
        delta = random.uniform(-0.15, 0.18)
        new_val = prev + delta
        new_val = max(-1.0, min(1.0, new_val))
        sub_cat_momentum[small_cat].append(new_val)

week_starts = [(datetime(2024, 1, 1) + timedelta(weeks=w)) for w in range(8)]

sentiment_rows = []
for week_idx, week_start in enumerate(week_starts):
    for large_cat, small_cat in sub_cats:
        mom = sub_cat_momentum[small_cat][week_idx]
        sentiment_mean = mom  # close to momentum
        sentiment_rows.append({
            "date": week_start.strftime("%Y-%m-%d"),
            "industry": small_cat,  # sub_category
            "sentiment_mean": sentiment_mean,
            "sentiment_std": random.uniform(0.05, 0.25),
            "news_count": random.randint(5, 40),
            "avg_confidence": random.uniform(0.55, 0.92),
            # RawScorer CPU mode needs these:
            "sentiment_trend": round(random.uniform(-0.2, 0.2), 3),
            "news_heat": random.uniform(0.1, 0.9),
        })

pl.DataFrame(sentiment_rows).write_parquet(ROOT / "data" / "industry_sentiment.parquet")
print(f"Written: data/industry_sentiment.parquet ({len(sentiment_rows)} rows)")


# ── 3. Fake backtest results: 7 weeks of history ─────────────────────────────

nav = 1_000_000.0
bt_rows = []
for week_idx, week_start in enumerate(week_starts[1:], 1):  # skip week 0
    weekly_return = random.uniform(-0.03, 0.05)
    nav = nav * (1 + weekly_return)
    holdings = {}
    if week_idx >= 2:
        # From week 2 onwards, have some holdings
        picked = random.sample(sub_cats, min(3, len(sub_cats)))
        for (_, sc) in picked:
            holdings[sc] = round(random.uniform(0.1, 0.3), 3)
    invested_weight = sum(holdings.values())
    bt_rows.append({
        "week_start": week_start.strftime("%Y-%m-%d"),
        "weekly_return": weekly_return,
        "nav": nav,
        "holdings": json.dumps(holdings, ensure_ascii=False),
        "invested_weight": invested_weight,
    })

pl.DataFrame(bt_rows).write_parquet(ROOT / "data" / "backtest_results.parquet")
print(f"Written: data/backtest_results.parquet ({len(bt_rows)} rows)")


# ── 4. Fake ETF info subset: only ETFs relevant to a few sub-categories ────────

# Load real ETF info to get column names
real_etf = pl.read_parquet(ROOT / "data" / "converted" / "主题ETF信息表-快照1_主题ETF.parquet")
aum_col = [c for c in real_etf.columns if "基金规模" in c][0]

# Pick a few small_cats and their indices
picked_sub_cats = random.sample([s for _, s in sub_cats], 5)
picked_indices = set()
for large_cat, small_cat in sub_cats:
    if small_cat in picked_sub_cats:
        picked_indices.update(industry_dict[large_cat][small_cat]["indices"])

# Filter ETFs
etf_rows = []
real_etf_sorted = real_etf.sort(aum_col, descending=True)
for idx_name in picked_indices:
    idx_etfs = real_etf_sorted.filter(pl.col("跟踪指数名称") == idx_name)
    for row in idx_etfs.iter_rows(named=True):
        etf_rows.append(row)

fake_etf = pl.DataFrame(etf_rows)
fake_etf.write_parquet(ROOT / "data" / "fake_etf_info.parquet")
print(f"Written: data/fake_etf_info.parquet ({len(etf_rows)} rows)")


# ── 5. ONNX cache: fake predictions for the fake news week ────────────────────

onnx_cache_dir = ROOT / "data" / "onnx_cache"
onnx_cache_dir.mkdir(exist_ok=True)

# Read the fake news we just wrote
fake_news = pl.read_parquet(ROOT / "data" / "fake_news.parquet")

# Fake ONNX predictions — for each row, assign random major/sub/sentiment/conf
pred_rows = []
for row in fake_news.iter_rows(named=True):
    major = random.choice(MAJOR_CATS)
    sub = random.choice(list(industry_dict[major].keys()))
    sentiment = random.choice(SENTIMENTS)
    l1_conf = random.uniform(0.55, 0.95)
    sub_cat_conf = random.uniform(0.50, 0.90)
    pred_rows.append({
        "datetime": row["datetime"],
        "major_category": major,
        "sentiment": sentiment,
        "l1_confidence": l1_conf,
        "sub_category": sub,
        "sub_category_confidence": sub_cat_conf,
    })

pred_df = pl.DataFrame(pred_rows)
pred_df.write_parquet(onnx_cache_dir / "2024-01-01.parquet")
print(f"Written: {onnx_cache_dir / '2024-01-01.parquet'} ({len(pred_rows)} rows)")


print("\nDone! To test with fake data, update config.toml:")
print('  input_news_raw = "data/fake_news.parquet"')
print('  etf_info = "data/fake_etf_info.parquet"')
print('  output_sentiment = "data/industry_sentiment.parquet"')
print('  output_backtest = "data/backtest_results.parquet"')
