from __future__ import annotations

import polars as pl

from src.signals import signal_helpers, signals_inference


def test_signals_inference_uses_runtime_signal_helpers() -> None:
    assert signals_inference.build_sub_category_sequences is signal_helpers.build_sub_category_sequences
    assert signals_inference.compute_global_leader_sentiment is signal_helpers.compute_global_leader_sentiment
    assert signals_inference.get_market_cap_weight is signal_helpers.get_market_cap_weight


def test_runtime_signal_helpers_build_sequences_without_trainer_imports() -> None:
    meta_sector_map = {
        "meta_sectors": {
            "科技成长": {"sub_categories": ["半导体/芯片"]},
            "医药健康": {"sub_categories": ["生物医药/创新药"]},
        },
        "notes": {
            "核心驱动（×1.5）": ["半导体/芯片"],
            "重要辅助（×1.0）": ["生物医药/创新药"],
            "边缘平滑（×0.5）": [],
        },
    }
    sentiment_df = pl.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
                "2024-01-05",
                "2024-01-05",
                "2024-01-06",
                "2024-01-06",
            ],
            "sub_category": [
                "半导体/芯片",
                "生物医药/创新药",
                "半导体/芯片",
                "生物医药/创新药",
                "半导体/芯片",
                "生物医药/创新药",
                "半导体/芯片",
                "生物医药/创新药",
                "半导体/芯片",
                "生物医药/创新药",
                "半导体/芯片",
                "生物医药/创新药",
            ],
            "sentiment_mean": [0.1, 0.2, 0.15, 0.18, 0.2, 0.16, 0.25, 0.14, 0.3, 0.12, 0.35, 0.1],
            "news_count": [3, 2, 4, 2, 5, 1, 6, 1, 5, 2, 4, 2],
        }
    )

    X, y, sample_dates, sub_industries = signal_helpers.build_sub_category_sequences(
        sentiment_df=sentiment_df,
        meta_sector_map=meta_sector_map,
        lookback_days=3,
        forecast_days=1,
        label_stats_path="missing_label_stats.json",
    )

    assert X.shape == (3, 3, 2, 6)
    assert y.shape == (3, 2)
    assert sample_dates == ["2024-01-03", "2024-01-04", "2024-01-05"]
    assert sub_industries == ["半导体/芯片", "生物医药/创新药"]
