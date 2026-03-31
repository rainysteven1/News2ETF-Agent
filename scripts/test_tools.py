"""Test the agent tools with fake data."""

import sys
sys.path.insert(0, ".")

from src.agent.tools import (
    read_market_news,
    compute_ml_signals,
    check_last_week_pnl,
    get_industry_top_news,
    get_etf_candidates,
)
from src.config import load_config

config = load_config()
print("Config loaded OK")
print(f"  input_news_raw: {config.data.input_news_raw}")
print(f"  etf_info: {config.data.etf_info}")
print(f"  output_sentiment: {config.data.output_sentiment}")
print()

print("=== Test 1: check_last_week_pnl ===")
result = check_last_week_pnl.invoke({})
print(result[:300])
print()

print("=== Test 2: compute_ml_signals ===")
result = compute_ml_signals.invoke({"date": "2024-01-01"})
print(result[:500])
print()

print("=== Test 3: read_market_news ===")
result = read_market_news.invoke({"date": "2024-01-01"})
print(result[:500])
print()

print("=== Test 4: get_industry_top_news ===")
# Try to find a sub_category from the fake data
result = get_industry_top_news.invoke({"date": "2024-01-01", "industry": "半导体/芯片", "top_k": 3})
print(result[:300])
print()

print("=== Test 5: get_etf_candidates ===")
result = get_etf_candidates.invoke({"industry": "半导体/芯片"})
print(result[:300])
print()

print("All tools OK!")
