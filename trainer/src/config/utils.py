"""Shared utilities: safe_name + LabelStats singleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_name(name: str) -> str:
    """Make a string safe for use as a directory name."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


class LabelStats:
    """Load and access label_stats.json (singleton)."""

    _instance: LabelStats | None = None

    def __init__(self, stats: dict[str, Any]) -> None:
        self._stats = stats

    @classmethod
    def load(cls) -> LabelStats:
        """Factory: compute stats from raw.parquet or load from label_stats.json."""
        if cls._instance is not None:
            return cls._instance

        label_stats_path = (
            Path(__file__).resolve().parent.parent.parent / "data" / "labeled" / "setfit" / "label_stats.json"
        )
        if label_stats_path.exists():
            with open(label_stats_path, encoding="utf-8") as f:
                stats = json.load(f)
        else:
            import polars as pl

            raw_path = Path(__file__).resolve().parent.parent.parent / "data" / "labeled" / "setfit" / "raw.parquet"
            df = pl.read_parquet(raw_path)

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

            stats = {
                "major_category": major_counts,
                "sub_category_by_major": sub_by_major,
            }
            label_stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

        cls._instance = cls(stats)
        return cls._instance

    def get_major_categories(self) -> list[str]:
        return sorted(self._stats["major_category"].keys())

    def get_sub_categories(self, major: str) -> list[str]:
        return sorted(self._stats["sub_category_by_major"][major].keys())
