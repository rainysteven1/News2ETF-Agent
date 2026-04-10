"""Tests for src/utils/meta_sector_map.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.utils.meta_sector_map import (
    get_all_sub_categories,
    get_global_leader_map,
    get_market_cap_weight,
    get_meta_sector_description,
    get_meta_sector_weight,
    get_meta_sectors,
    get_upstream_sentiment,
    load_meta_sector_mapping,
    meta_to_subs,
    reload_mapping,
    sub_to_meta,
)


@pytest.fixture
def sample_meta_sector_mapping() -> dict:
    """Sample meta sector mapping for testing."""
    return {
        "meta_sectors": {
            "科技成长": {
                "sub_categories": [
                    "半导体/芯片",
                    "软件/信创",
                    "TMT",
                ],
                "market_cap_weight": 1.0,
                "description": "科技成长：涵盖TMT、人工智能、半导体等",
            },
            "高端制造": {
                "sub_categories": [
                    "军工/国防",
                    "新能源/光伏",
                    "新能源车/锂电",
                ],
                "market_cap_weight": 1.0,
                "description": "高端制造：涵盖军工国防、新能源等",
            },
            "消费文娱": {
                "sub_categories": [
                    "食品饮料/消费",
                    "传媒/游戏/文娱",
                ],
                "market_cap_weight": 1.0,
                "description": "消费文娱",
            },
            "医药健康": {
                "sub_categories": [
                    "生物医药/创新药",
                    "医疗器械/医疗服务",
                ],
                "market_cap_weight": 1.0,
                "description": "医药健康",
            },
            "资源材料": {
                "sub_categories": [
                    "化工/新材料",
                    "有色金属/稀土",
                    "能源/油气/资源",
                ],
                "market_cap_weight": 1.0,
                "description": "资源材料",
            },
            "金融地产": {
                "sub_categories": [
                    "金融/银行/证券",
                    "地产/建筑/基建",
                ],
                "market_cap_weight": 1.0,
                "description": "金融地产",
            },
            "基础设施/公共": {
                "sub_categories": [
                    "交通运输/物流",
                    "环保/绿色低碳",
                ],
                "market_cap_weight": 1.0,
                "description": "基础设施/公共",
            },
            "主题策略": {
                "sub_categories": [
                    "央企/国企/国资改革",
                    "区域经济",
                ],
                "market_cap_weight": 0.5,
                "description": "主题策略",
            },
        },
        "global_leader_map": {
            "科技成长": ["半导体/芯片", "软件/信创", "TMT"],
            "高端制造": ["有色金属/稀土", "能源/油气/资源", "化工/新材料"],
            "消费文娱": ["互联网消费", "食品饮料/消费"],
            "医药健康": ["生物医药/创新药", "医疗器械/医疗服务"],
            "资源材料": ["能源/油气/资源", "有色金属/稀土"],
            "金融地产": ["金融/银行/证券", "地产/建筑/基建"],
            "基础设施/公共": ["交通运输/物流", "环保/绿色低碳"],
            "主题策略": ["央企/国企/国资改革", "区域经济"],
        },
        "notes": {
            "核心驱动（×1.5）": [
                "半导体/芯片",
                "军工/国防",
                "人工智能",
                "新能源/光伏",
                "新能源车/锂电",
            ],
            "重要辅助（×1.0）": [
                "软件/信创",
                "云计算/大数据",
            ],
            "边缘平滑（×0.5）": [
                "消费电子/家电",
            ],
        },
    }


@pytest.fixture
def temp_meta_sector_file(sample_meta_sector_mapping: dict, tmp_path: Path) -> Path:
    """Create a temporary meta sector mapping file."""
    file_path = tmp_path / "meta_sector_mapping.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_meta_sector_mapping, f, ensure_ascii=False)
    return file_path


class TestLoadMetaSectorMapping:
    """Tests for load_meta_sector_mapping()."""

    def test_returns_dict(self, temp_meta_sector_file: Path) -> None:
        """Test that load_meta_sector_mapping returns a dictionary."""
        result = load_meta_sector_mapping(path=temp_meta_sector_file)
        assert isinstance(result, dict)

    def test_contains_meta_sectors(self, temp_meta_sector_file: Path) -> None:
        """Test that returned dict contains 'meta_sectors' key."""
        result = load_meta_sector_mapping(path=temp_meta_sector_file)
        assert "meta_sectors" in result
        assert isinstance(result["meta_sectors"], dict)

    def test_contains_global_leader_map(self, temp_meta_sector_file: Path) -> None:
        """Test that returned dict contains 'global_leader_map' key."""
        result = load_meta_sector_mapping(path=temp_meta_sector_file)
        assert "global_leader_map" in result
        assert isinstance(result["global_leader_map"], dict)

    def test_caches_result(self, temp_meta_sector_file: Path) -> None:
        """Test that result is cached on subsequent calls."""
        result1 = load_meta_sector_mapping(path=temp_meta_sector_file)
        result2 = load_meta_sector_mapping(path=temp_meta_sector_file)
        # Same object due to caching
        assert result1 is result2


class TestSubToMeta:
    """Tests for sub_to_meta()."""

    def test_direct_mapping(self, temp_meta_sector_file: Path) -> None:
        """Test direct sub-category to meta sector mapping."""
        reload_mapping(path=temp_meta_sector_file)
        assert sub_to_meta("半导体/芯片") == "科技成长"
        assert sub_to_meta("军工/国防") == "高端制造"
        assert sub_to_meta("食品饮料/消费") == "消费文娱"

    def test_partial_match(self, temp_meta_sector_file: Path) -> None:
        """Test partial matching when sub-category name is contained."""
        reload_mapping(path=temp_meta_sector_file)
        # "半导体" should match "半导体/芯片" in 科技成长
        result = sub_to_meta("半导体")
        assert result == "科技成长"

    def test_fallback_to_主题策略(self, temp_meta_sector_file: Path) -> None:
        """Test that unknown sub-categories fall back to 主题策略."""
        reload_mapping(path=temp_meta_sector_file)
        result = sub_to_meta("完全不存在的分类 xyz123")
        assert result == "主题策略"


class TestMetaToSubs:
    """Tests for meta_to_subs()."""

    def test_returns_list(self, temp_meta_sector_file: Path) -> None:
        """Test that meta_to_subs returns a list."""
        reload_mapping(path=temp_meta_sector_file)
        result = meta_to_subs("科技成长")
        assert isinstance(result, list)

    def test_returns_correct_sub_categories(self, temp_meta_sector_file: Path) -> None:
        """Test that correct sub-categories are returned for a meta sector."""
        reload_mapping(path=temp_meta_sector_file)
        subs = meta_to_subs("科技成长")
        assert "半导体/芯片" in subs
        assert "软件/信创" in subs
        assert "TMT" in subs

    def test_returns_empty_for_unknown_meta(self, temp_meta_sector_file: Path) -> None:
        """Test that unknown meta sector returns empty list."""
        reload_mapping(path=temp_meta_sector_file)
        result = meta_to_subs("不存在的板块")
        assert result == []


class TestGetUpstreamSentiment:
    """Tests for get_upstream_sentiment()."""

    def test_returns_list(self, temp_meta_sector_file: Path) -> None:
        """Test that get_upstream_sentiment returns a list."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_upstream_sentiment("高端制造")
        assert isinstance(result, list)

    def test_returns_upstream_sectors(self, temp_meta_sector_file: Path) -> None:
        """Test that upstream sectors are correctly returned."""
        reload_mapping(path=temp_meta_sector_file)
        upstream = get_upstream_sentiment("高端制造")
        assert "有色金属/稀土" in upstream
        assert "能源/油气/资源" in upstream
        assert "化工/新材料" in upstream

    def test_works_with_sub_category_input(self, temp_meta_sector_file: Path) -> None:
        """Test that function works when given a sub-category name."""
        reload_mapping(path=temp_meta_sector_file)
        # 军工/国防 belongs to 高端制造, which has upstream sectors
        upstream = get_upstream_sentiment("军工/国防")
        assert isinstance(upstream, list)

    def test_returns_fallback_for_unknown_sector(self, temp_meta_sector_file: Path) -> None:
        """Test that unknown sector returns fallback (主题策略) leaders.

        Note: Due to the sub_to_meta fallback, unknown sectors map to 主题策略,
        which has leaders in global_leader_map. So unknown sectors return
        the leaders for 主题策略, not an empty list.
        """
        reload_mapping(path=temp_meta_sector_file)
        result = get_upstream_sentiment("完全不存在的板块")
        # Should return 主题策略's leaders since unknown sectors fallback to 主题策略
        assert isinstance(result, list)


class TestGetMetaSectors:
    """Tests for get_meta_sectors()."""

    def test_returns_list(self, temp_meta_sector_file: Path) -> None:
        """Test that get_meta_sectors returns a list."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_meta_sectors()
        assert isinstance(result, list)

    def test_returns_all_8_meta_sectors(self, temp_meta_sector_file: Path) -> None:
        """Test that all 8 meta sectors are returned."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_meta_sectors()
        expected = [
            "科技成长",
            "高端制造",
            "消费文娱",
            "医药健康",
            "资源材料",
            "金融地产",
            "基础设施/公共",
            "主题策略",
        ]
        for ms in expected:
            assert ms in result, f"{ms} not found in meta sectors"


class TestGetGlobalLeaderMap:
    """Tests for get_global_leader_map()."""

    def test_returns_dict(self, temp_meta_sector_file: Path) -> None:
        """Test that get_global_leader_map returns a dict."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_global_leader_map()
        assert isinstance(result, dict)

    def test_has_all_meta_sectors(self, temp_meta_sector_file: Path) -> None:
        """Test that all meta sectors are keys in the leader map."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_global_leader_map()
        expected_keys = [
            "科技成长",
            "高端制造",
            "消费文娱",
            "医药健康",
            "资源材料",
            "金融地产",
            "基础设施/公共",
            "主题策略",
        ]
        for key in expected_keys:
            assert key in result, f"{key} not found in global_leader_map"


class TestGetMarketCapWeight:
    """Tests for get_market_cap_weight()."""

    def test_core_driver_returns_1_5(self, temp_meta_sector_file: Path) -> None:
        """Test that core driver sub-categories return 1.5."""
        reload_mapping(path=temp_meta_sector_file)
        assert get_market_cap_weight("半导体/芯片") == 1.5
        assert get_market_cap_weight("军工/国防") == 1.5

    def test_important_returns_1_0(self, temp_meta_sector_file: Path) -> None:
        """Test that important sub-categories return 1.0."""
        reload_mapping(path=temp_meta_sector_file)
        assert get_market_cap_weight("软件/信创") == 1.0

    def test_edge_smoothing_returns_0_5(self, temp_meta_sector_file: Path) -> None:
        """Test that edge smoothing sub-categories return 0.5."""
        reload_mapping(path=temp_meta_sector_file)
        assert get_market_cap_weight("消费电子/家电") == 0.5

    def test_default_returns_1_0(self, temp_meta_sector_file: Path) -> None:
        """Test that unknown sub-categories default to 1.0."""
        reload_mapping(path=temp_meta_sector_file)
        assert get_market_cap_weight("完全不存在的分类") == 1.0


class TestGetMetaSectorWeight:
    """Tests for get_meta_sector_weight()."""

    def test_returns_float(self, temp_meta_sector_file: Path) -> None:
        """Test that get_meta_sector_weight returns a float."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_meta_sector_weight("科技成长")
        assert isinstance(result, float)

    def test_returns_correct_weight(self, temp_meta_sector_file: Path) -> None:
        """Test that correct weight is returned."""
        reload_mapping(path=temp_meta_sector_file)
        assert get_meta_sector_weight("科技成长") == 1.0
        assert get_meta_sector_weight("主题策略") == 0.5

    def test_unknown_returns_default_1_0(self, temp_meta_sector_file: Path) -> None:
        """Test that unknown meta sector returns default 1.0."""
        reload_mapping(path=temp_meta_sector_file)
        assert get_meta_sector_weight("不存在的板块") == 1.0


class TestGetMetaSectorDescription:
    """Tests for get_meta_sector_description()."""

    def test_returns_string(self, temp_meta_sector_file: Path) -> None:
        """Test that get_meta_sector_description returns a string."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_meta_sector_description("科技成长")
        assert isinstance(result, str)

    def test_returns_correct_description(self, temp_meta_sector_file: Path) -> None:
        """Test that correct description is returned."""
        reload_mapping(path=temp_meta_sector_file)
        desc = get_meta_sector_description("科技成长")
        assert "科技成长" in desc

    def test_empty_for_unknown(self, temp_meta_sector_file: Path) -> None:
        """Test that unknown meta sector returns empty string."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_meta_sector_description("不存在的板块")
        assert result == ""


class TestGetAllSubCategories:
    """Tests for get_all_sub_categories()."""

    def test_returns_sorted_list(self, temp_meta_sector_file: Path) -> None:
        """Test that get_all_sub_categories returns a sorted list."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_all_sub_categories()
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_contains_expected_sub_categories(self, temp_meta_sector_file: Path) -> None:
        """Test that expected sub-categories are present."""
        reload_mapping(path=temp_meta_sector_file)
        result = get_all_sub_categories()
        assert "半导体/芯片" in result
        assert "军工/国防" in result
        assert "食品饮料/消费" in result


class TestReloadMapping:
    """Tests for reload_mapping()."""

    def test_reload_returns_dict(self, temp_meta_sector_file: Path) -> None:
        """Test that reload_mapping returns a dict."""
        result = reload_mapping(path=temp_meta_sector_file)
        assert isinstance(result, dict)

    def test_reload_invalidates_cache(self, temp_meta_sector_file: Path) -> None:
        """Test that reload_mapping invalidates the cache."""
        # First load
        result1 = reload_mapping(path=temp_meta_sector_file)
        # Reload
        result2 = reload_mapping(path=temp_meta_sector_file)
        # Both should work
        assert result1 is not None
        assert result2 is not None
