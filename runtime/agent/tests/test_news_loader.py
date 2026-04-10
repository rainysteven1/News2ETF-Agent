from __future__ import annotations

from pathlib import Path

import polars as pl

from src.utils.news_loader import load_raw_news_df


def test_load_raw_news_df_falls_back_to_split_parts(tmp_path: Path) -> None:
    merged_path = tmp_path / "tushare_news_2021_today_merged.parquet"
    part1_path = tmp_path / "tushare_news_2021_today_part1.parquet"
    part2_path = tmp_path / "tushare_news_2021_today_part2.parquet"

    pl.DataFrame(
        {
            "datetime": ["2024-01-01 09:00:00"],
            "content": ["a"],
            "title": ["t1"],
            "source": ["s1"],
        }
    ).write_parquet(part1_path)
    pl.DataFrame(
        {
            "datetime": ["2024-01-02 09:00:00"],
            "content": ["b"],
            "title": ["t2"],
            "source": ["s2"],
        }
    ).write_parquet(part2_path)

    df = load_raw_news_df(merged_path)

    assert df.shape == (2, 4)
    assert sorted(df["title"].to_list()) == ["t1", "t2"]
