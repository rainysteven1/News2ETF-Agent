"""Helpers to load raw news parquet with fallback to split parts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl


def _part_paths(news_path: Path) -> list[Path]:
    stem = news_path.stem
    if stem.endswith("_merged"):
        base = stem[: -len("_merged")]
        return [
            news_path.with_name(f"{base}_part1.parquet"),
            news_path.with_name(f"{base}_part2.parquet"),
        ]
    return []


@lru_cache(maxsize=8)
def _load_news_cached(news_path_str: str) -> pl.DataFrame:
    news_path = Path(news_path_str)
    if news_path.exists():
        return pl.read_parquet(news_path)

    part_paths = [p for p in _part_paths(news_path) if p.exists()]
    if not part_paths:
        return pl.DataFrame()

    parts = [pl.read_parquet(path) for path in part_paths]
    return pl.concat(parts, how="diagonal_relaxed")


def load_raw_news_df(news_path: Path | str) -> pl.DataFrame:
    """Load merged news parquet, or fallback to known split parts."""
    return _load_news_cached(str(Path(news_path)))
