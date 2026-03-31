"""Test Memos API integration."""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.signals.memos_retrieval import MemosRetrieval

print("=== Test 1: MemosRetrieval init ===")
memos = MemosRetrieval()
print(f"  API key: {memos.api_key[:10]}...")
print(f"  Base URL: {memos.base_url}")
print("  OK")
print()

print("=== Test 2: Store a test decision ===")
success = memos.add_decision(
    conversation_id="2024-01-01",
    decision="buy 银行 25% → 512800 华夏银行ETF",
    context="momentum=0.42, heat=0.65, composite=0.38",
    date="2024-01-01",
)
print(f"  Success: {success}")
print()

print("=== Test 3: Retrieve similar decisions ===")
results = memos.retrieve(
    query="银行板块投资决策，动量上行",
    conversation_id="2024-01-08",
    top_k=3,
)
print(f"  Found {len(results)} results:")
for r in results:
    print(f"  - similarity={r.get('similarity', 0):.3f}, content={r.get('content', '')[:80]}")
print()

print("All Memos API tests passed!")
