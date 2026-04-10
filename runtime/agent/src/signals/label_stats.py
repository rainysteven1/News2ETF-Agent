"""Runtime-local label stats helpers.

This keeps runtime ONNX inference independent from trainer-side config modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.config import shared_data_root


def safe_name(name: str) -> str:
    """Make a string safe for use as a directory name."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def default_label_stats_path() -> Path:
    return shared_data_root() / "labeled" / "setfit" / "label_stats.json"


def default_setfit_raw_path() -> Path:
    return shared_data_root() / "labeled" / "setfit" / "raw.parquet"


class LabelStats:
    """Load and access setfit label_stats.json with a small in-process cache."""

    _instances: dict[Path, "LabelStats"] = {}

    def __init__(self, stats: dict[str, Any]) -> None:
        self._stats = stats

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LabelStats":
        stats_path = Path(path) if path is not None else default_label_stats_path()
        stats_path = stats_path.resolve()
        cached = cls._instances.get(stats_path)
        if cached is not None:
            return cached

        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f)
        else:
            raw_path = default_setfit_raw_path()
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
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

        instance = cls(stats)
        cls._instances[stats_path] = instance
        return instance

    def get_major_categories(self) -> list[str]:
        return sorted(self._stats["major_category"].keys())

    def get_sub_categories(self, major: str) -> list[str]:
        return sorted(self._stats["sub_category_by_major"][major].keys())
