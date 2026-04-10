"""Tests for src/agent/features.py (AgentFeatureBuilder)"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.agent.features import AgentFeatureBuilder


@pytest.fixture
def mock_sentiment_df() -> pl.DataFrame:
    """Create a mock sentiment DataFrame for testing."""
    data = {
        "date": ["2024-09-25", "2024-09-26", "2024-09-27", "2024-09-28", "2024-09-29"] * 4,
        "sub_category": [
            "半导体/芯片",
            "半导体/芯片",
            "半导体/芯片",
            "半导体/芯片",
            "半导体/芯片",
            "软件/信创",
            "软件/信创",
            "软件/信创",
            "软件/信创",
            "软件/信创",
            "军工/国防",
            "军工/国防",
            "军工/国防",
            "军工/国防",
            "军工/国防",
            "食品饮料/消费",
            "食品饮料/消费",
            "食品饮料/消费",
            "食品饮料/消费",
            "食品饮料/消费",
        ],
        "sentiment_mean": [
            0.3,
            0.35,
            0.4,
            0.38,
            0.42,
            0.1,
            0.15,
            0.12,
            0.18,
            0.2,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.08,
            0.2,
            0.25,
            0.22,
            0.28,
            0.3,
        ],
        "sentiment_weighted": [
            0.3,
            0.35,
            0.4,
            0.38,
            0.42,
            0.1,
            0.15,
            0.12,
            0.18,
            0.2,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.08,
            0.2,
            0.25,
            0.22,
            0.28,
            0.3,
        ],
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_meta_sector_map() -> dict[str, Any]:
    """Create a mock meta sector mapping."""
    return {
        "meta_sectors": {
            "科技成长": {
                "sub_categories": ["半导体/芯片", "软件/信创", "TMT"],
                "market_cap_weight": 1.0,
                "description": "科技成长",
            },
            "高端制造": {
                "sub_categories": ["军工/国防", "新能源/光伏", "新能源车/锂电"],
                "market_cap_weight": 1.0,
                "description": "高端制造",
            },
            "消费文娱": {
                "sub_categories": ["食品饮料/消费", "传媒/游戏/文娱"],
                "market_cap_weight": 1.0,
                "description": "消费文娱",
            },
            "医药健康": {
                "sub_categories": ["生物医药/创新药"],
                "market_cap_weight": 1.0,
                "description": "医药健康",
            },
            "资源材料": {
                "sub_categories": ["化工/新材料"],
                "market_cap_weight": 1.0,
                "description": "资源材料",
            },
            "金融地产": {
                "sub_categories": ["金融/银行/证券"],
                "market_cap_weight": 1.0,
                "description": "金融地产",
            },
            "基础设施/公共": {
                "sub_categories": ["交通运输/物流"],
                "market_cap_weight": 1.0,
                "description": "基础设施/公共",
            },
            "主题策略": {
                "sub_categories": ["央企/国企/国资改革"],
                "market_cap_weight": 0.5,
                "description": "主题策略",
            },
        },
        "global_leader_map": {
            "科技成长": ["半导体/芯片", "软件/信创"],
            "高端制造": ["军工/国防"],
            "消费文娱": ["食品饮料/消费"],
            "医药健康": ["生物医药/创新药"],
            "资源材料": ["化工/新材料"],
            "金融地产": ["金融/银行/证券"],
            "基础设施/公共": ["交通运输/物流"],
            "主题策略": ["央企/国企/国资改革"],
        },
    }


@pytest.fixture
def mock_price_df() -> pl.DataFrame:
    """Create a mock price DataFrame for testing."""
    data = {
        "date": ["2024-09-25", "2024-09-26", "2024-09-27", "2024-09-28", "2024-09-29"] * 2,
        "etf_code": [
            "512000",
            "512000",
            "512000",
            "512000",
            "512000",
            "512660",
            "512660",
            "512660",
            "512660",
            "512660",
        ],
        "close": [1.0, 1.02, 1.05, 1.03, 1.08, 2.0, 2.05, 2.1, 2.08, 2.15],
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock config object."""
    config = MagicMock()
    config.data.output_sentiment = Path("/tmp/sentiment.parquet")
    config.data.etf_prices = Path("/tmp/prices.parquet")
    config.data.meta_sector_mapping = Path("/tmp/meta_sector_mapping.json")
    config.data.input_news_raw = Path("/tmp/news.parquet")
    return config


class TestBuildTCNSequence:
    """Tests for build_tcn_sequence()."""

    def test_returns_dict(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
        mock_config: MagicMock,
    ) -> None:
        """Test that build_tcn_sequence returns a dict."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=pl.DataFrame(),
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_tcn_sequence(date="2024-09-30", lookback=5)

        assert isinstance(result, dict)

    def test_keys_are_8_meta_sectors(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result keys are the 8 meta sectors."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_tcn_sequence(date="2024-09-30", lookback=5)

        expected_meta_sectors = [
            "科技成长",
            "高端制造",
            "消费文娱",
            "医药健康",
            "资源材料",
            "金融地产",
            "基础设施/公共",
            "主题策略",
        ]
        for ms in expected_meta_sectors:
            assert ms in result, f"{ms} not found in tcn_sequence keys"

    def test_values_are_lists(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that values in result are lists of floats."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_tcn_sequence(date="2024-09-30", lookback=5)

        for sector, values in result.items():
            assert isinstance(values, list), f"{sector} value is not a list"
            for v in values:
                assert isinstance(v, (int, float)), f"{sector} value contains non-numeric: {v}"

    def test_list_length_matches_lookback(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that each meta sector has a list of length equal to lookback."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        builder._meta_sector_map = mock_meta_sector_map

        lookback = 5
        result = builder.build_tcn_sequence(date="2024-09-30", lookback=lookback)

        for sector, values in result.items():
            assert len(values) == lookback, f"{sector} has {len(values)} values, expected {lookback}"

    def test_accepts_date_typed_sentiment_dates(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Polars Date columns should not break string date filters."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_tcn_sequence(date="2024-09-30", lookback=5)

        assert "科技成长" in result
        assert len(result["科技成长"]) == 5


class TestBuildNewsSummary:
    """Tests for build_news_summary()."""

    def test_returns_dict(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that build_news_summary returns a dict."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        builder._meta_sector_map = mock_meta_sector_map

        # Mock the news data
        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_news_summary(date="2024-09-30", top_k=1)

        assert isinstance(result, dict)

    def test_keys_are_meta_sectors(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result keys are meta sectors."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")
            result = builder.build_news_summary(date="2024-09-30", top_k=1)

        for key in result.keys():
            assert key in mock_meta_sector_map["meta_sectors"], f"{key} is not a valid meta sector"

    def test_values_are_lists(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that values in result are lists (news titles)."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")
            result = builder.build_news_summary(date="2024-09-30", top_k=1)

        for sector, news_list in result.items():
            assert isinstance(news_list, list), f"{sector} value is not a list"

    def test_falls_back_to_keyword_match_when_raw_news_has_no_sub_category(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        raw_news_path = tmp_path / "raw_news.parquet"
        pl.DataFrame(
            {
                "datetime": [
                    "2024-09-24 09:00:00",
                    "2024-09-25 10:00:00",
                    "2024-09-26 11:00:00",
                ],
                "content": [
                    "半导体设备国产化推进，芯片景气改善",
                    "软件信创订单增长",
                    "军工订单释放",
                ],
                "title": [
                    "半导体设备国产化推进",
                    "软件信创订单增长",
                    "军工订单释放",
                ],
                "source": ["s1", "s2", "s3"],
            }
        ).write_parquet(raw_news_path)

        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = raw_news_path
            result = builder.build_news_summary(date="2024-09-30", top_k=1)

        assert isinstance(result, dict)
        assert result["科技成长"]
        assert "半导体" in result["科技成长"][0] or "软件" in result["科技成长"][0]


class TestBuildMarketState:
    """Tests for build_market_state()."""

    def test_returns_dict(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that build_market_state returns a dict."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_market_state(date="2024-09-30")

        assert isinstance(result, dict)

    def test_contains_expected_keys(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result contains expected market state keys."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_market_state(date="2024-09-30")

        expected_keys = [
            "market_return_1w",
            "market_return_2w",
            "market_volatility",
            "volume_ratio",
            "market_state",
        ]
        for key in expected_keys:
            assert key in result, f"Expected key '{key}' not found in market_state"

    def test_market_state_is_string(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that market_state value is a string."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_market_state(date="2024-09-30")

        assert isinstance(result["market_state"], str)
        assert result["market_state"] in ["bullish", "bearish", "neutral"]


class TestBuildPositionState:
    """Tests for build_position_state()."""

    def test_returns_dict(self) -> None:
        """Test that build_position_state returns a dict."""
        builder = AgentFeatureBuilder()

        result = builder.build_position_state(
            current_holdings={"科技成长": 0.3, "高端制造": 0.2},
            weekly_returns={"科技成长": 0.05, "高端制造": -0.02},
            agent_perf_1w=0.03,
            agent_perf_4w=0.12,
        )

        assert isinstance(result, dict)

    def test_contains_expected_keys(self) -> None:
        """Test that result contains expected position state keys."""
        builder = AgentFeatureBuilder()

        result = builder.build_position_state(
            current_holdings={"科技成长": 0.3, "高端制造": 0.2},
            weekly_returns={"科技成长": 0.05, "高端制造": -0.02},
            agent_perf_1w=0.03,
            agent_perf_4w=0.12,
        )

        expected_keys = [
            "total_weight",
            "invested_weight",
            "num_positions",
            "top_holdings",
            "portfolio_return_1w",
            "agent_perf_1w",
            "agent_perf_4w",
        ]
        for key in expected_keys:
            assert key in result, f"Expected key '{key}' not found in position_state"

    def test_total_weight_calculation(self) -> None:
        """Test that total_weight is calculated correctly."""
        builder = AgentFeatureBuilder()

        holdings = {"科技成长": 0.3, "高端制造": 0.2}
        result = builder.build_position_state(
            current_holdings=holdings,
            weekly_returns={},
            agent_perf_1w=0.0,
            agent_perf_4w=0.0,
        )

        assert result["total_weight"] == 0.5

    def test_invested_weight_excludes_zero_holdings(self) -> None:
        """Test that invested_weight excludes holdings below threshold."""
        builder = AgentFeatureBuilder()

        holdings = {"科技成长": 0.3, "高端制造": 0.2, "极小仓位": 0.005}
        result = builder.build_position_state(
            current_holdings=holdings,
            weekly_returns={},
            agent_perf_1w=0.0,
            agent_perf_4w=0.0,
        )

        # Only holdings >= 0.01 are counted as invested
        assert result["invested_weight"] == 0.5

    def test_num_positions_count(self) -> None:
        """Test that num_positions counts only invested positions."""
        builder = AgentFeatureBuilder()

        holdings = {"科技成长": 0.3, "高端制造": 0.2, "极小仓位": 0.005}
        result = builder.build_position_state(
            current_holdings=holdings,
            weekly_returns={},
            agent_perf_1w=0.0,
            agent_perf_4w=0.0,
        )

        assert result["num_positions"] == 2


class TestBuildSentPDivergence:
    """Tests for build_sent_p_divergence()."""

    def test_returns_dict(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that build_sent_p_divergence returns a dict."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_sent_p_divergence(date="2024-09-30")

        assert isinstance(result, dict)

    def test_keys_are_meta_sectors(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result keys are meta sectors."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_sent_p_divergence(date="2024-09-30")

        for key in result.keys():
            assert key in mock_meta_sector_map["meta_sectors"], f"{key} is not a valid meta sector"

    def test_values_are_floats(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that divergence values are floats."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        result = builder.build_sent_p_divergence(date="2024-09-30")

        for sector, divergence in result.items():
            assert isinstance(divergence, (int, float)), f"{sector} divergence is not numeric"


class TestBuildAgentFeatures:
    """Tests for build_agent_features() - main entry point."""

    def test_returns_dict(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that build_agent_features returns a dict."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_agent_features(
                date="2024-09-30",
                current_holdings={"科技成长": 0.3, "高端制造": 0.2},
                current_time="2024-09-30 08:30:00",
            )

        assert isinstance(result, dict)

    def test_contains_tcn_sequence(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result contains tcn_sequence (Feature A)."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_agent_features(
                date="2024-09-30",
                current_holdings={},
                current_time="2024-09-30 08:30:00",
            )

        assert "tcn_sequence" in result
        assert isinstance(result["tcn_sequence"], dict)

    def test_contains_news_summary(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result contains news_summary (Feature B)."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_agent_features(
                date="2024-09-30",
                current_holdings={},
                current_time="2024-09-30 08:30:00",
            )

        assert "news_summary" in result
        assert isinstance(result["news_summary"], dict)

    def test_contains_market_state(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result contains market_state (Feature C)."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_agent_features(
                date="2024-09-30",
                current_holdings={},
                current_time="2024-09-30 08:30:00",
            )

        assert "market_state" in result
        assert isinstance(result["market_state"], dict)

    def test_contains_position_state(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result contains position_state (Feature D)."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_agent_features(
                date="2024-09-30",
                current_holdings={"科技成长": 0.3},
                current_time="2024-09-30 08:30:00",
            )

        assert "position_state" in result
        assert isinstance(result["position_state"], dict)

    def test_contains_sent_p_divergence(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that result contains sent_p_divergence (Feature E)."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_agent_features(
                date="2024-09-30",
                current_holdings={},
                current_time="2024-09-30 08:30:00",
            )

        assert "sent_p_divergence" in result
        assert isinstance(result["sent_p_divergence"], dict)

    def test_all_5_features_present(
        self,
        mock_sentiment_df: pl.DataFrame,
        mock_price_df: pl.DataFrame,
        mock_meta_sector_map: dict[str, Any],
    ) -> None:
        """Test that all 5 features A/B/C/D/E are present."""
        builder = AgentFeatureBuilder(
            sentiment_df=mock_sentiment_df,
            price_df=mock_price_df,
        )
        builder._meta_sector_map = mock_meta_sector_map

        with patch.object(builder, "config") as mock_cfg:
            mock_cfg.data.input_news_raw = Path("/tmp/news.parquet")

            result = builder.build_agent_features(
                date="2024-09-30",
                current_holdings={},
                current_time="2024-09-30 08:30:00",
            )

        expected_keys = [
            "tcn_sequence",  # Feature A
            "news_summary",  # Feature B
            "market_state",  # Feature C
            "position_state",  # Feature D
            "sent_p_divergence",  # Feature E
        ]
        for key in expected_keys:
            assert key in result, f"Feature '{key}' not found in build_agent_features result"


class TestAgentFeatureBuilderInitialization:
    """Tests for AgentFeatureBuilder initialization."""

    def test_init_with_no_args(self) -> None:
        """Test that builder can be initialized with no arguments."""
        builder = AgentFeatureBuilder()
        assert builder._sentiment_df is None
        assert builder._price_df is None

    def test_init_with_sentiment_df(self, mock_sentiment_df: pl.DataFrame) -> None:
        """Test initialization with a pre-loaded sentiment DataFrame."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        assert builder._sentiment_df is not None

    def test_init_with_price_df(self, mock_price_df: pl.DataFrame) -> None:
        """Test initialization with a pre-loaded price DataFrame."""
        builder = AgentFeatureBuilder(price_df=mock_price_df)
        assert builder._price_df is not None

    def test_sentiment_df_property(self, mock_sentiment_df: pl.DataFrame) -> None:
        """Test sentiment_df property returns DataFrame."""
        builder = AgentFeatureBuilder(sentiment_df=mock_sentiment_df)
        result = builder.sentiment_df
        assert isinstance(result, pl.DataFrame)

    def test_price_df_property(self, mock_price_df: pl.DataFrame) -> None:
        """Test price_df property returns DataFrame."""
        builder = AgentFeatureBuilder(price_df=mock_price_df)
        result = builder.price_df
        assert isinstance(result, pl.DataFrame)
