from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

from src.agent.single_agent import risk_check_node
from src.agent.tools import get_etf_candidates
from src.agent.state import ETFSelections, MetaSectorPlan, TradeDecision
from src.utils.etf_universe import ETFUniverseResolver


def _write_test_universe(tmp_path: Path) -> tuple[Path, Path]:
    etf_info_path = tmp_path / "etf_info.parquet"
    etf_prices_path = tmp_path / "etf_prices.parquet"

    pl.DataFrame(
        {
            "代码": ["512480.SH", "588000.SH", "159999.SZ", "Code"],
            "名称": ["半导体ETF", "科创50ETF", "未来半导体ETF", "Name"],
            "跟踪指数名称": ["中证全指半导体", "科创50", "中证全指半导体", "fund_trackindexname"],
            "基金规模合计\n[单位]亿元\n[交易日期]2025-12-31\n[币种]原始币种": [120.0, 90.0, 150.0, 0.0],
        }
    ).write_parquet(etf_info_path)

    pl.DataFrame(
        {
            "Code": [
                "512480.SH",
                "512480.SH",
                "159999.SZ",
                "159999.SZ",
            ],
            "trade_dt": [
                20240105,
                20240112,
                20250103,
                20250110,
            ],
            "close": [1.0, 1.1, 1.0, 1.1],
        }
    ).write_parquet(etf_prices_path)

    return etf_info_path, etf_prices_path


class _FakeMapper:
    def get_large_cats(self) -> list[str]:
        return ["科技信息"]

    def get_small_cats(self, large_cat: str) -> list[str]:
        assert large_cat == "科技信息"
        return ["半导体/芯片"]

    def get_indices(self, large_cat: str, small_cat: str) -> list[str]:
        assert large_cat == "科技信息"
        assert small_cat == "半导体/芯片"
        return ["中证全指半导体"]

    def small_cat_beta(self, small_cat: str) -> str:
        return "high"

    def small_cat_cluster(self, small_cat: str) -> str:
        return small_cat


class _FakeIndustryMapper:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_large_cats(self) -> list[str]:
        return ["科技信息"]

    def get_small_cats(self, large_cat: str) -> list[str]:
        assert large_cat == "科技信息"
        return ["半导体/芯片"]

    def get_indices(self, large_cat: str, small_cat: str) -> list[str]:
        assert large_cat == "科技信息"
        assert small_cat == "半导体/芯片"
        return ["中证全指半导体"]


def test_etf_universe_only_returns_tradable_candidates_for_week(tmp_path: Path) -> None:
    etf_info_path, etf_prices_path = _write_test_universe(tmp_path)
    resolver = ETFUniverseResolver(etf_info_path=etf_info_path, etf_prices_path=etf_prices_path)

    candidates = resolver.candidates_for_index("中证全指半导体", week_start="2024-01-08")

    assert [candidate.code for candidate in candidates] == ["512480.SH"]
    assert candidates[0].display == "512480.SH 半导体ETF"


def test_risk_check_node_normalizes_invalid_selected_etf_to_tradable_code(tmp_path: Path) -> None:
    etf_info_path, etf_prices_path = _write_test_universe(tmp_path)
    config = SimpleNamespace(
        agent=SimpleNamespace(
            max_weight_per_industry=0.3,
            max_total_weight=1.0,
        ),
        data=SimpleNamespace(
            etf_info=etf_info_path,
            etf_prices=etf_prices_path,
        ),
    )
    state = {
        "date": "2024-01-08",
        "decisions": [
            TradeDecision(
                industry="meta_allocation",
                action="hold",
                weight=0.0,
                level1_plan=[
                    MetaSectorPlan(meta_sector="科技成长", action="buy", weight=0.2, reason="动量改善"),
                ],
                level2_plan=[
                    ETFSelections(
                        meta_sector="科技成长",
                        selected_indices=["中证全指半导体"],
                        selected_etf="588000 科创50ETF",
                    )
                ],
            )
        ],
        "retry_count": 0,
        "last_error": "",
        "messages": [],
        "last_week_pnl": 0.0,
        "last_week_holdings": {},
        "forbidden_sectors": {},
    }

    result = risk_check_node(state=state, config=config, mapper=_FakeMapper())

    decision = result["decisions"][0]
    assert decision.level2_plan[0].selected_etf == "512480.SH 半导体ETF"
    assert "normalized selected_etf" in result["last_error"]


def test_get_etf_candidates_only_returns_real_price_universe_codes(tmp_path: Path) -> None:
    etf_info_path, etf_prices_path = _write_test_universe(tmp_path)
    config = SimpleNamespace(
        data=SimpleNamespace(
            etf_info=etf_info_path,
            etf_prices=etf_prices_path,
            industry_dict=tmp_path / "industry_dict.json",
        )
    )

    with patch("src.config.load_config", return_value=config), patch("src.agent.tools.IndustryMapper", _FakeIndustryMapper):
        out = get_etf_candidates.invoke({"industry": "半导体/芯片", "date": "2024-01-08"})

    assert "512480.SH 半导体ETF" in out
    assert "588000.SH" not in out
    assert "159999.SZ" not in out


def test_get_etf_candidates_accepts_meta_sector_name(tmp_path: Path) -> None:
    etf_info_path, etf_prices_path = _write_test_universe(tmp_path)
    config = SimpleNamespace(
        data=SimpleNamespace(
            etf_info=etf_info_path,
            etf_prices=etf_prices_path,
            industry_dict=tmp_path / "industry_dict.json",
        )
    )

    with patch("src.config.load_config", return_value=config), patch("src.agent.tools.IndustryMapper", _FakeIndustryMapper):
        out = get_etf_candidates.invoke({"industry": "科技信息", "date": "2024-01-08"})

    assert "512480.SH 半导体ETF" in out


def test_get_etf_candidates_accepts_meta_sector_mapping(tmp_path: Path) -> None:
    etf_info_path, etf_prices_path = _write_test_universe(tmp_path)
    config = SimpleNamespace(
        data=SimpleNamespace(
            etf_info=etf_info_path,
            etf_prices=etf_prices_path,
            industry_dict=tmp_path / "industry_dict.json",
        )
    )

    with (
        patch("src.config.load_config", return_value=config),
        patch("src.agent.tools.IndustryMapper", _FakeIndustryMapper),
        patch("src.agent.tools.meta_to_subs", return_value=["半导体/芯片"]),
    ):
        out = get_etf_candidates.invoke({"industry": "科技成长", "date": "2024-01-08"})

    assert "512480.SH 半导体ETF" in out
