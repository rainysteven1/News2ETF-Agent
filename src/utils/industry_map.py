"""Industry mapping: large_cat → small_cat → indices, with beta & correlation metadata.

ETF best lookup (best_etf_by_index.parquet) is built once from raw etf_info,
sorted by AUM descending, first entry per tracking_index_name wins.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import polars as pl


@lru_cache(maxsize=200)
def _best_etf_code_cached(etf_map_tuple: tuple[tuple[str, str], ...], index_name: str) -> str | None:
    """Cached best_etf_code lookup: (index_name, code) tuples are fully hashable."""
    for k, code in etf_map_tuple:
        if k == index_name:
            return code
    return None


def _build_best_etf_parquet(etf_info_path: Path, output_path: Path) -> None:
    """Build best_etf_by_index.parquet from raw etf_info.

    Columns: tracking_index, code, name, aum
    One row per tracking_index (highest AUM wins).
    """
    df = pl.read_parquet(etf_info_path)

    # Rename Chinese columns to English aliases
    # 代码=ETF code, 名称=ETF name, 跟踪指数名称=tracking_index, 基金规模合计=AUM
    aum_col = [c for c in df.columns if "基金规模" in c][0]
    idx_name_col = "跟踪指数名称"

    df2 = df.rename(
        {
            "代码": "code",
            "名称": "name",
            idx_name_col: "tracking_index",
            aum_col: "aum",
        }
    ).select(["code", "name", "tracking_index", "aum"])

    # Parse AUM numeric (some values may have units embedded)
    df2 = df2.with_columns(pl.col("aum").cast(pl.Float64))

    # Sort by AUM descending, keep first per tracking_index
    best = df2.sort("aum", descending=True).unique(subset="tracking_index", keep="first")

    best.write_parquet(output_path)


class IndustryMapper:
    """Maps industries + tracking indices with beta labels and correlation clusters.

    Data sources:
      - industry_dict.json: large_cat → small_cat → {indices, beta, correlation_cluster}
      - etf_info: ETF metadata table (code, name, tracking_index, aum)
      - best_etf_by_index.parquet: built once from etf_info, cached on disk

    Beta levels: very_high / high / medium / low
    Correlation clusters: indices in the same cluster should not both be满仓
      (e.g. ev_battery + new_energy are mirrors of each other)
    """

    def __init__(
        self,
        dict_path: Path | str,
        etf_info: Path | str | None = None,
        best_etf_path: Path | str | None = None,
    ):
        self._dict_path = Path(dict_path)

        # Load industry dict with metadata
        with open(dict_path, encoding="utf-8") as f:
            self._raw: dict = json.load(f)

        # Reverse index: tracking_index_name → {large_cat, small_cat, beta, correlation_cluster}
        self._index_info: dict[str, dict] = {}
        # Small cat → beta (inherited by all its indices)
        self._small_cat_beta: dict[str, str] = {}
        # Small cat → correlation_cluster
        self._small_cat_cluster: dict[str, str] = {}
        # Large cat → list of small cats
        self._large_cats: dict[str, list[str]] = {}

        for large_cat, small_cats in self._raw.items():
            self._large_cats[large_cat] = []
            for small_cat, meta in small_cats.items():
                indices: list[str] = meta["indices"]
                beta: str = meta.get("beta", "medium")
                cluster: str = meta.get("correlation_cluster", small_cat)

                self._small_cat_beta[small_cat] = beta
                self._small_cat_cluster[small_cat] = cluster
                self._large_cats[large_cat].append(small_cat)

                for idx in indices:
                    self._index_info[idx] = {
                        "large_cat": large_cat,
                        "small_cat": small_cat,
                        "beta": beta,
                        "correlation_cluster": cluster,
                    }

        # ── ETF mapping: build or load best_etf_by_index.parquet ──────────────
        self._etf_map: dict[str, dict] = {}

        if etf_info is not None:
            # Determine output path: same dir as etf_info, or explicit best_etf_path
            if best_etf_path is not None:
                cache_path = Path(best_etf_path)
            else:
                cache_path = Path(etf_info).parent / "best_etf_by_index.parquet"

            if not cache_path.exists():
                _build_best_etf_parquet(Path(etf_info), cache_path)

            df = pl.read_parquet(cache_path)
            for row in df.iter_rows(named=True):
                self._etf_map[row["tracking_index"]] = {
                    "code": row["code"],
                    "name": row["name"],
                    "aum": row["aum"],
                }
        elif best_etf_path is not None and Path(best_etf_path).exists():
            df = pl.read_parquet(best_etf_path)
            for row in df.iter_rows(named=True):
                self._etf_map[row["tracking_index"]] = {
                    "code": row["code"],
                    "name": row["name"],
                    "aum": row["aum"],
                }

        # Hashable form for lru_cache on best_etf_code
        self._etf_code_tuple: tuple[tuple[str, str], ...] = tuple((k, v["code"]) for k, v in self._etf_map.items())

    # ── Structural accessors ────────────────────────────────────────────────────

    def get_large_cats(self) -> list[str]:
        return list(self._large_cats.keys())

    def get_small_cats(self, large_cat: str) -> list[str]:
        return self._large_cats.get(large_cat, [])

    def get_indices(self, large_cat: str, small_cat: str) -> list[str]:
        meta = self._raw.get(large_cat, {}).get(small_cat, {})
        return meta.get("indices", [])

    def get_all_indices(self) -> list[str]:
        return list(self._index_info.keys())

    @property
    def industries(self) -> list[str]:
        """Alias: list of large categories."""
        return self.get_large_cats()

    def industry_etfs(self, large_cat: str, small_cat: str | None = None) -> list[str]:
        """Return tracking index names for a large_cat (optionally filtered by small_cat).

        Kept for backward compat — returns tracking index names, not ETF codes.
        """
        if small_cat:
            return self.get_indices(large_cat, small_cat)
        cats = self.get_small_cats(large_cat)
        indices = []
        for sc in cats:
            indices.extend(self.get_indices(large_cat, sc))
        return indices

    # ── Metadata accessors ──────────────────────────────────────────────────────

    def info(self, index_name: str) -> dict:
        """Return {large_cat, small_cat, beta, correlation_cluster} for a tracking index."""
        return self._index_info.get(
            index_name,
            {"large_cat": "未知", "small_cat": "未知", "beta": "medium", "correlation_cluster": "unknown"},
        )

    def beta(self, index_name: str) -> str:
        return self.info(index_name).get("beta", "medium")

    def correlation_cluster(self, index_name: str) -> str:
        return self.info(index_name).get("correlation_cluster", "unknown")

    def small_cat_beta(self, small_cat: str) -> str:
        return self._small_cat_beta.get(small_cat, "medium")

    def small_cat_cluster(self, small_cat: str) -> str:
        return self._small_cat_cluster.get(small_cat, small_cat)

    # ── ETF mapping ────────────────────────────────────────────────────────────

    def best_etf(self, index_name: str) -> dict | None:
        """Return {code, name, aum} for the best ETF tracking this index."""
        return self._etf_map.get(index_name)

    def best_etf_code(self, index_name: str) -> str | None:
        """Return ETF code for the best ETF tracking this index (cached)."""
        return _best_etf_code_cached(self._etf_code_tuple, index_name)

    def best_etf_codes(self, index_names: list[str]) -> list[str]:
        """Return ETF codes for a list of tracking index names, skipping None."""
        codes = []
        for name in index_names:
            code = self.best_etf_code(name)
            if code:
                codes.append(code)
        return codes

    # ── Correlation cluster helpers ─────────────────────────────────────────────

    def is_same_cluster(self, index_a: str, index_b: str) -> bool:
        """Return True if both indices are in the same correlation cluster."""
        ca = self.correlation_cluster(index_a)
        cb = self.correlation_cluster(index_b)
        return ca == cb and ca != "unknown"

    def conflict_indices(self, index_names: list[str]) -> list[tuple[str, str]]:
        """Return all pairs of indices in the same correlation cluster.

        Used by Risk Guard to detect mirror positions.
        """
        clusters: dict[str, list[str]] = {}
        for idx in index_names:
            c = self.correlation_cluster(idx)
            if c not in clusters:
                clusters[c] = []
            clusters[c].append(idx)

        conflicts = []
        for c, indices in clusters.items():
            if c == "unknown" or len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    conflicts.append((indices[i], indices[j]))
        return conflicts
