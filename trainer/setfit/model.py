"""SetFit sub-category classifier — one SetFit model per major category."""

from __future__ import annotations

import json
from pathlib import Path

from setfit import SetFitModel


def load_label_stats(stats_path: Path | str | None = None) -> dict:
    """Load label_stats.json to get major -> sub_category mapping."""
    if stats_path is None:
        stats_path = Path("label_stats.json")
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def get_major_categories(stats: dict | None = None) -> list[str]:
    """Return sorted list of major categories from label_stats.json."""
    if stats is None:
        stats = load_label_stats()
    return sorted(stats["sub_category_by_major"].keys())


def get_sub_categories(major: str, stats: dict | None = None) -> list[str]:
    """Return sorted list of sub-categories for a major from label_stats.json."""
    if stats is None:
        stats = load_label_stats()
    return sorted(stats["sub_category_by_major"][major].keys())


class SetFitSubCategoryClassifier:
    """Holds one SetFit model per major category for sub-category classification."""

    def __init__(self):
        self.models: dict[str, SetFitModel] = {}

    def set_model(self, major: str, model: SetFitModel) -> None:
        self.models[major] = model

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Save all models to output_dir."""
        output_dir = Path(output_dir)
        paths: dict[str, Path] = {}
        for major, model in self.models.items():
            major_dir = output_dir / _safe_name(major)
            model.save(str(major_dir))
            paths[major] = major_dir
        return paths

    @classmethod
    def load(cls, input_dir: Path) -> SetFitSubCategoryClassifier:
        """Load all models from input_dir."""
        clf = cls()
        for major in get_major_categories():
            major_dir = Path(input_dir) / _safe_name(major)
            if not major_dir.exists():
                continue
            clf.models[major] = SetFitModel.from_pretrained(str(major_dir))
        return clf


def _safe_name(name: str) -> str:
    """Make a string safe for use as a directory name."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")
