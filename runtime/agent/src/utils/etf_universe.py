"""Executable ETF universe helpers for backtest/runtime selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ETFCandidate:
    code: str
    name: str
    tracking_index: str
    aum: float

    @property
    def display(self) -> str:
        return f"{self.code} {self.name}".strip()


def _extract_code_token(raw_value: str) -> str:
    if not raw_value:
        return ""
    normalized = str(raw_value).replace("|", " ").replace(",", " ").replace("，", " ")
    for token in normalized.split():
        token = token.strip()
        if not token:
            continue
        if any(ch.isdigit() for ch in token):
            return token
    return normalized.split()[0] if normalized.split() else ""


def _week_bounds(week_start: str) -> tuple[int, int]:
    week_start_dt = datetime.strptime(week_start, "%Y-%m-%d")
    week_start_int = int(week_start_dt.strftime("%Y%m%d"))
    next_week_int = int((week_start_dt + timedelta(days=7)).strftime("%Y%m%d"))
    return week_start_int, next_week_int


class ETFUniverseResolver:
    """Resolve ETF selections against the real price universe used in backtest."""

    def __init__(self, etf_info_path: Path | str, etf_prices_path: Path | str):
        self.etf_info_path = Path(etf_info_path)
        self.etf_prices_path = Path(etf_prices_path)
        self._prices = pl.read_parquet(self.etf_prices_path).select(["Code", "trade_dt"])
        self._price_codes = set(self._prices["Code"].unique().to_list())
        self._base_code_map: dict[str, list[str]] = {}
        for code in sorted(self._price_codes):
            base = code.split(".", 1)[0]
            self._base_code_map.setdefault(base, []).append(code)

        raw_info = pl.read_parquet(self.etf_info_path)
        aum_col = next((c for c in raw_info.columns if "基金规模" in c), None)
        rename_map = {
            "代码": "code",
            "名称": "name",
            "跟踪指数名称": "tracking_index",
        }
        if aum_col is not None:
            rename_map[aum_col] = "aum"

        info = raw_info.rename(rename_map).with_columns([
            pl.col("code").cast(pl.Utf8, strict=False).str.strip_chars(),
            pl.col("name").cast(pl.Utf8, strict=False).fill_null(""),
            pl.col("tracking_index").cast(pl.Utf8, strict=False).fill_null(""),
            (
                pl.col("aum").cast(pl.Float64, strict=False).fill_null(0.0)
                if "aum" in raw_info.columns or aum_col is not None
                else pl.lit(0.0).alias("aum")
            ),
        ])

        self._info = (
            info.filter(
                pl.col("code").is_not_null()
                & (pl.col("code") != "")
                & (pl.col("code") != "Code")
                & pl.col("tracking_index").is_not_null()
                & (pl.col("tracking_index") != "")
                & (pl.col("tracking_index") != "fund_trackindexname")
                & pl.col("code").is_in(list(self._price_codes))
            )
            .select(["code", "name", "tracking_index", "aum"])
            .unique(subset=["code", "tracking_index"], keep="first")
        )

    def normalize_code(self, raw_value: str) -> str:
        token = _extract_code_token(raw_value)
        if not token:
            return ""
        if token in self._price_codes:
            return token
        base = token.split(".", 1)[0]
        candidates = self._base_code_map.get(base, [])
        if len(candidates) == 1:
            return candidates[0]
        if f"{base}.SH" in self._price_codes:
            return f"{base}.SH"
        if f"{base}.SZ" in self._price_codes:
            return f"{base}.SZ"
        return ""

    def lookup(self, code: str) -> ETFCandidate | None:
        matched = self._info.filter(pl.col("code") == code).sort("aum", descending=True).head(1)
        if matched.is_empty():
            return None
        row = matched.row(0, named=True)
        return ETFCandidate(
            code=str(row.get("code", "")),
            name=str(row.get("name", "")),
            tracking_index=str(row.get("tracking_index", "")),
            aum=float(row.get("aum", 0.0) or 0.0),
        )

    def is_tradable(self, code: str, week_start: str) -> bool:
        week_start_int, next_week_int = _week_bounds(week_start)
        code_df = self._prices.filter(pl.col("Code") == code)
        if code_df.is_empty():
            return False
        has_prev = code_df.filter(pl.col("trade_dt") < week_start_int).height > 0
        has_curr = code_df.filter(
            (pl.col("trade_dt") >= week_start_int) & (pl.col("trade_dt") < next_week_int)
        ).height > 0
        return has_prev and has_curr

    def candidates_for_index(self, index_name: str, week_start: str | None = None) -> list[ETFCandidate]:
        matched = self._info.filter(pl.col("tracking_index") == index_name).sort("aum", descending=True)
        candidates: list[ETFCandidate] = []
        for row in matched.iter_rows(named=True):
            candidate = ETFCandidate(
                code=str(row.get("code", "")),
                name=str(row.get("name", "")),
                tracking_index=str(row.get("tracking_index", "")),
                aum=float(row.get("aum", 0.0) or 0.0),
            )
            if week_start and not self.is_tradable(candidate.code, week_start):
                continue
            candidates.append(candidate)
        return candidates

    def meta_sector_indices(self, meta_sector: str, mapper) -> list[str]:
        from src.utils.meta_sector_map import meta_to_subs

        indices: list[str] = []
        seen: set[str] = set()
        for sub in meta_to_subs(meta_sector):
            for large_cat in mapper.get_large_cats():
                if sub not in mapper.get_small_cats(large_cat):
                    continue
                for index_name in mapper.get_indices(large_cat, sub):
                    if index_name and index_name not in seen:
                        seen.add(index_name)
                        indices.append(index_name)
        return indices

    def fallback_candidate_for_meta_sector(self, meta_sector: str, week_start: str, mapper) -> ETFCandidate | None:
        best: ETFCandidate | None = None
        for index_name in self.meta_sector_indices(meta_sector, mapper):
            candidates = self.candidates_for_index(index_name, week_start=week_start)
            if not candidates:
                continue
            candidate = candidates[0]
            if best is None or candidate.aum > best.aum:
                best = candidate
        return best

    def resolve_selection(
        self,
        *,
        meta_sector: str,
        selected_indices: list[str],
        raw_selected_etf: str,
        week_start: str,
        mapper,
    ) -> tuple[ETFCandidate | None, str]:
        normalized = self.normalize_code(raw_selected_etf)
        if normalized and self.is_tradable(normalized, week_start):
            candidate = self.lookup(normalized)
            if candidate is not None:
                return candidate, "exact_match"

        for index_name in selected_indices:
            candidates = self.candidates_for_index(index_name, week_start=week_start)
            if candidates:
                return candidates[0], f"selected_index:{index_name}"

        fallback = self.fallback_candidate_for_meta_sector(meta_sector, week_start=week_start, mapper=mapper)
        if fallback is not None:
            return fallback, "meta_sector_fallback"

        return None, "no_tradable_candidate"


@lru_cache(maxsize=8)
def get_etf_universe(etf_info_path: str, etf_prices_path: str) -> ETFUniverseResolver:
    return ETFUniverseResolver(etf_info_path=etf_info_path, etf_prices_path=etf_prices_path)
