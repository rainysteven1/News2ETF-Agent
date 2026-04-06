"""Merge sub-categories in setfit raw.parquet.

Usage:
    python scripts/merge_setfit_categories.py
"""

import polars as pl
from pathlib import Path

RAW_PATH = Path("trainer/data/labeled/setfit/raw.parquet")
OUTPUT_PATH = Path("trainer/data/labeled/setfit/raw_merged.parquet")

# 定义合并规则：{原类别: 目标类别}，空目标表示删除该行
MERGE_RULES: dict[str, dict[str, str]] = {
    "金融地产": {
        "金融地产": "",  # 删除（样本太少，合并到金融/银行/证券语义不符）
    },
    "科技信息": {
        "电子/元件": "电子",
        "科技综合": "综合科技",
        "科技龙头": "",  # 删除（样本太少，21条）
    },
}

def main():
    df = pl.read_parquet(RAW_PATH)
    print(f"原始行数: {len(df)}")

    # 应用合并规则
    for major, rules in MERGE_RULES.items():
        for old_label, new_label in rules.items():
            mask = (pl.col("major_category") == major) & (pl.col("sub_category") == old_label)
            if new_label == "":
                # 删除该类别
                count = df.filter(mask).height
                df = df.filter(~mask)
                print(f"  删除 {major} -> {old_label}: {count}行")
            else:
                # 合并到新类别
                count = df.filter(mask).height
                df = df.with_columns(
                    pl.when(mask).then(pl.lit(new_label)).otherwise(pl.col("sub_category")).alias("sub_category")
                )
                print(f"  合并 {major} -> {old_label} -> {new_label}: {count}行")

    # 统计合并后结果
    print(f"\n合并后行数: {len(df)}")

    for major in df["major_category"].unique().sort():
        sub_counts = df.filter(pl.col("major_category") == major).group_by("sub_category").len().sort("len", descending=True)
        print(f"\n{major}:")
        for row in sub_counts.iter_rows():
            print(f"  {row[0]}: {row[1]}")

    # 保存
    df.write_parquet(OUTPUT_PATH)
    print(f"\n已保存到: {OUTPUT_PATH}")

    # 同步更新 label_stats.json
    stats_path = RAW_PATH.parent / "label_stats.json"
    major_counts: dict[str, int] = {}
    for row in df.group_by("major_category", maintain_order=True).len().iter_rows():
        major_counts[row[0]] = row[1]

    sub_by_major: dict[str, dict[str, int]] = {}
    for major in major_counts:
        sub_df = df.filter(pl.col("major_category") == major)
        sub_counts: dict[str, int] = {}
        for row in sub_df.group_by("sub_category", maintain_order=True).len().iter_rows():
            sub_counts[row[0]] = row[1]
        sub_by_major[major] = sub_counts

    import json
    stats = {
        "major_category": major_counts,
        "sub_category_by_major": sub_by_major,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"已更新: {stats_path}")

if __name__ == "__main__":
    main()
