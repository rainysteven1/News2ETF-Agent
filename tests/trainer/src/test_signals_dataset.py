"""Tests for signals dataset helpers under ``trainer/src/datasets``."""

from datetime import date

import numpy as np
import polars as pl
import pytest
import torch

from trainer.src.datasets.signals import (
    build_sub_category_sequences,
    compute_global_leader_sentiment,
    compute_market_beta,
    export_phase2_dataset,
)
from trainer.src.models.signals import TCNFanIn

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_mock_sentiment_df(n_dates=30, n_subs=47):
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, n_dates), "1d", eager=True)
    sub_categories = [f"sub_{i}" for i in range(n_subs)]
    rows = []
    np.random.seed(42)
    for d in dates:
        for s in sub_categories:
            rows.append(
                {
                    "date": d,
                    "sub_category": s,
                    "sentiment_mean": np.random.randn(),
                    "news_count": np.random.randint(0, 20),
                }
            )
    return pl.DataFrame(rows)


def make_mock_meta_sector_map():
    return {
        "meta_sectors": {
            f"meta_{i}": {
                "sub_categories": [f"sub_{j}" for j in range(i * 5, i * 5 + 7)],
                "market_cap_weight": 1.0,
            }
            for i in range(8)
        },
        "global_leader_map": {
            "meta_0": ["sub_5"],
            "meta_1": ["sub_0"],
        },
    }


@pytest.fixture
def sentiment_df():
    return make_mock_sentiment_df()


@pytest.fixture
def meta_sector_map():
    return make_mock_meta_sector_map()


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestBuildSubCategorySequences:
    def test_output_shapes(self, sentiment_df, meta_sector_map):
        X, y, dates, subs = build_sub_category_sequences(sentiment_df, meta_sector_map, lookback_days=5)
        n_sub = len(subs)
        assert X.shape[1:] == (5, n_sub, 6), f"X shape wrong: {X.shape}"
        assert y.shape[1] == 8, f"y should have 8 meta sectors, got {y.shape}"
        assert len(dates) == len(X)

    def test_dates_match(self, sentiment_df, meta_sector_map):
        X, y, dates, subs = build_sub_category_sequences(sentiment_df, meta_sector_map, lookback_days=5)
        assert len(dates) == X.shape[0]
        # Dates should be sorted
        assert dates == sorted(dates)

    def test_sub_industries_populated(self, sentiment_df, meta_sector_map):
        _, _, _, subs = build_sub_category_sequences(sentiment_df, meta_sector_map)
        assert len(subs) == 47

    def test_y_in_valid_range(self, sentiment_df, meta_sector_map):
        _, y, _, _ = build_sub_category_sequences(sentiment_df, meta_sector_map)
        assert (y >= -1).all() and (y <= 1).all(), "y should be in [-1, 1] after tanh"


class TestComputeGlobalLeaderSentiment:
    def test_returns_dataframe(self, sentiment_df, meta_sector_map):
        result = compute_global_leader_sentiment(sentiment_df, meta_sector_map)
        assert isinstance(result, pl.DataFrame)
        assert "date" in result.columns

    def test_has_leader_columns(self, sentiment_df, meta_sector_map):
        result = compute_global_leader_sentiment(sentiment_df, meta_sector_map)
        # Should have columns for sectors in global_leader_map
        assert "global_leader_meta_0" in result.columns
        assert "global_leader_meta_1" in result.columns


class TestComputeMarketBeta:
    def test_returns_dataframe(self, meta_sector_map):
        # Minimal price df
        price_df = pl.DataFrame(
            {
                "date": pl.date_range(date(2024, 1, 1), date(2024, 2, 1), "1d", eager=True),
            }
        )
        index_df = price_df.clone()
        result = compute_market_beta(price_df, index_df, meta_sector_map)
        assert isinstance(result, pl.DataFrame)
        assert "date" in result.columns
        assert "beta_meta_0" in result.columns


class TestExportPhase2Dataset:
    @pytest.mark.skip(
        reason=(
            "Integration test: requires full price_df with sector columns; "
            "build_sub_category_sequences still needs realistic price fixtures."
        )
    )
    def test_creates_parquet(self, sentiment_df, meta_sector_map, tmp_path):
        # Train a tiny model for export test
        model = TCNFanIn(n_sub=47, n_meta=8, input_size=6, hidden_size=16, num_layers=1)
        model.eval()

        price_df = pl.DataFrame({"date": [date(2024, 1, 1)]})
        index_df = price_df.clone()
        output_path = tmp_path / "agent_features.parquet"

        export_phase2_dataset(
            sentiment_df,
            price_df,
            index_df,
            meta_sector_map,
            tcn_model=model,
            lgbm_models={},
            iforest_model=None,
            device=torch.device("cpu"),
            output_path=output_path,
        )
        assert output_path.exists()
        # Verify it's a valid parquet
        df = pl.read_parquet(output_path)
        assert "date" in df.columns
        assert len(df) > 0
