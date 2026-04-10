"""Meta Sector Mapping Utilities.

Provides functions for mapping between sub-categories and meta sectors,
as well as global leader sentiment tracking for cross-industry transmission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import shared_data_root

_META_SECTOR_MAPPING_PATH = shared_data_root() / "meta_sector_mapping.json"

# Global cache for meta sector mapping
_META_SECTOR_MAP: dict[str, Any] | None = None


def load_meta_sector_mapping(path: Path | str | None = None) -> dict[str, Any]:
    """Load the meta sector mapping JSON.

    Args:
        path: Optional path to the JSON file. Defaults to data/meta_sector_mapping.json

    Returns:
        The meta sector mapping dictionary
    """
    global _META_SECTOR_MAP

    if path is None:
        path = _META_SECTOR_MAPPING_PATH
    path = Path(path)

    if _META_SECTOR_MAP is None:
        with open(path, "r", encoding="utf-8") as f:
            _META_SECTOR_MAP = json.load(f)

    return _META_SECTOR_MAP


def sub_to_meta(sub: str) -> str:
    """Map a sub-category to its meta sector.

    Args:
        sub: The sub-category name (e.g., "半导体/芯片", "军工/国防")

    Returns:
        The meta sector name (e.g., "科技成长", "高端制造")
    """
    meta_map = load_meta_sector_mapping()
    meta_sectors = meta_map.get("meta_sectors", {})

    # Direct lookup
    for meta_sector, info in meta_sectors.items():
        if sub in info.get("sub_categories", []):
            return meta_sector

    # Partial match (sub-category name contained in sub_categories)
    for meta_sector, info in meta_sectors.items():
        for cat in info.get("sub_categories", []):
            if sub in cat or cat in sub:
                return meta_sector

    return "主题策略"  # Default fallback


def meta_to_subs(meta: str) -> list[str]:
    """Get all sub-categories belonging to a meta sector.

    Args:
        meta: The meta sector name (e.g., "科技成长", "高端制造")

    Returns:
        List of sub-category names
    """
    meta_map = load_meta_sector_mapping()
    meta_sectors = meta_map.get("meta_sectors", {})

    if meta not in meta_sectors:
        return []

    return meta_sectors[meta].get("sub_categories", [])


def get_meta_sectors() -> list[str]:
    """Get all meta sector names.

    Returns:
        List of meta sector names
    """
    meta_map = load_meta_sector_mapping()
    return list(meta_map.get("meta_sectors", {}).keys())


def get_upstream_sentiment(sector: str) -> list[str]:
    """Get all upstream leader sectors that influence the given sector.

    This is the core of "cross-industry transmission" logic:
    When making decisions about a sector, the Agent should also look
    at the past 5 days sentiment of its upstream leader sectors.

    Args:
        sector: The target sector name (can be meta sector or sub-category)

    Returns:
        List of upstream leader sector names
    """
    meta_map = load_meta_sector_mapping()
    leader_map = meta_map.get("global_leader_map", {})

    # If sector is a sub-category, convert to meta sector first
    if sector not in leader_map:
        meta = sub_to_meta(sector)
        if meta in leader_map:
            return leader_map[meta]
        return []

    return leader_map.get(sector, [])


def get_global_leader_map() -> dict[str, list[str]]:
    """Get the full global leader mapping.

    Returns:
        Dictionary mapping sectors to their upstream leaders
    """
    meta_map = load_meta_sector_mapping()
    return meta_map.get("global_leader_map", {})


def get_market_cap_weight(sub_category: str) -> float:
    """Get the market cap weight for a sub-category.

    Args:
        sub_category: The sub-category name

    Returns:
        The market cap weight (1.5 for core driver, 1.0 for important, 0.5 for edge)
    """
    meta_map = load_meta_sector_mapping()
    notes = meta_map.get("notes", {})

    core_driver = notes.get("核心驱动（×1.5）", [])
    important = notes.get("重要辅助（×1.0）", [])
    edge_smoothing = notes.get("边缘平滑（×0.5）", [])

    if sub_category in core_driver:
        return 1.5
    elif sub_category in important:
        return 1.0
    elif sub_category in edge_smoothing:
        return 0.5
    else:
        return 1.0  # Default weight


def get_meta_sector_weight(meta_sector: str) -> float:
    """Get the base market cap weight for a meta sector.

    Args:
        meta_sector: The meta sector name

    Returns:
        The meta sector base weight
    """
    meta_map = load_meta_sector_mapping()
    meta_sectors = meta_map.get("meta_sectors", {})

    if meta_sector not in meta_sectors:
        return 1.0

    return meta_sectors[meta_sector].get("market_cap_weight", 1.0)


def get_meta_sector_description(meta_sector: str) -> str:
    """Get the description for a meta sector.

    Args:
        meta_sector: The meta sector name

    Returns:
        The meta sector description
    """
    meta_map = load_meta_sector_mapping()
    meta_sectors = meta_map.get("meta_sectors", {})

    if meta_sector not in meta_sectors:
        return ""

    return meta_sectors[meta_sector].get("description", "")


def get_all_sub_categories() -> list[str]:
    """Get all sub-category names across all meta sectors.

    Returns:
        Sorted list of all sub-category names
    """
    meta_map = load_meta_sector_mapping()
    meta_sectors = meta_map.get("meta_sectors", {})

    all_subs = set()
    for info in meta_sectors.values():
        all_subs.update(info.get("sub_categories", []))

    return sorted(list(all_subs))


def is_valid_sub_category(sub: str) -> bool:
    """Check if a sub-category name is valid.

    Args:
        sub: The sub-category name to check

    Returns:
        True if valid, False otherwise
    """
    return sub in get_all_sub_categories()


def reload_mapping(path: Path | str | None = None) -> dict[str, Any]:
    """Force reload the meta sector mapping from disk.

    Args:
        path: Optional path to the JSON file

    Returns:
        The reloaded meta sector mapping dictionary
    """
    global _META_SECTOR_MAP
    _META_SECTOR_MAP = None
    return load_meta_sector_mapping(path)
