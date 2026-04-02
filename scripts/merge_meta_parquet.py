"""
Merge two parquet files under data/meta/ (level1 + level2) on news_id (inner join).
Output fields required by FinBERT: news_id, title, major_category, sentiment

NOTE: title and major_category appear in both files; values from level2 are used.
"""

import pandas as pd

# ── Path configuration ───────────────────────────────────────────────────────
META_DIR = "data/meta"
LEVEL1 = f"{META_DIR}/level1_122428cb-3437-4578-a947-164144e8898c.parquet"
LEVEL2 = f"{META_DIR}/level2_1e246a55-d528-487f-872a-1ede1d660158.parquet"
OUT = "trainer/data/labeled/finbert/raw.parquet"

# ── Load parquet files ────────────────────────────────────────────────────────
df1 = pd.read_parquet(LEVEL1)
df2 = pd.read_parquet(LEVEL2)

print(f"level1 rows: {len(df1)}")
print(f"level2 rows: {len(df2)}")

# ── Inner join on news_id (only keeps records present in both files) ──────────
merged = df1.merge(df2, on="news_id", how="inner", suffixes=("_l1", "_l2"))
print(f"merged rows (inner): {len(merged)}")

# ── Resolve duplicate columns: use level2 values ─────────────────────────────
# title: level2 title is typically more complete
# major_category: level2 uses a finer-grained taxonomy
merged["title"] = merged["title_l2"]
merged["major_category"] = merged["major_category_l2"]

# ── Select only FinBERT-required columns ─────────────────────────────────────
finbert_cols = ["news_id", "title", "major_category", "sentiment"]
out = merged[finbert_cols].copy()

# ── Drop rows where sentiment is null (sentiment only exists in level2;
#    null values indicate stale / invalid records) ────────────────────────────
before = len(out)
out = out.dropna(subset=["sentiment"])
print(f"dropped {before - len(out)} rows with null sentiment")

# ── Write output ─────────────────────────────────────────────────────────────
out.to_parquet(OUT, index=False)
print(f"saved → {OUT}  ({len(out)} rows)")
print(out.head(3)[finbert_cols].to_string())
