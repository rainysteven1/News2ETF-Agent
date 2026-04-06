"""SHAP Explainability Analysis for LightGBM models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

LIGHTGBM_16_FEATURE_NAMES = [
    "delta_sentiment_1w",
    "delta_sentiment_2w",
    "news_count",
    "news_heat",
    "tcn_reg",
    "tcn_reg_delta",
    "tcn_prediction_stability",
    "news_count_std_5d",
    "sentiment_volatility_5d",
    "tcn_heat_interaction",
    "volume_ratio",
    "intraday_vol",
    "avg_price",
    "global_leader_sentiment",
    "market_beta",
    "sentiment_entropy",
]


class SHAPAnalyzer:
    """SHAP analysis for LightGBM stacking model."""

    def __init__(self, lgbm_model: Any, X_test: np.ndarray):
        import shap

        self.lgbm_model = lgbm_model
        self.X_test = X_test
        self.explainer = shap.TreeExplainer(lgbm_model)
        self._shap_values = None

    def compute_shap_values(self) -> np.ndarray:
        self._shap_values = self.explainer.shap_values(self.X_test)
        return self._shap_values

    def generate_summary_plot(self, output_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        shap.summary_plot(
            self._shap_values,
            self.X_test,
            feature_names=LIGHTGBM_16_FEATURE_NAMES,
            show=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def generate_force_plot(self, date: str, output_dir: Path) -> None:
        import shap

        output_dir.mkdir(parents=True, exist_ok=True)
        shap.force_plot(
            self.explainer.expected_value,
            self._shap_values[0],
            self.X_test[0],
            feature_names=LIGHTGBM_16_FEATURE_NAMES,
            matplotlib=False,
        )
        # Save as HTML
        shap.save_html(
            str(output_dir / f"force_plot_{date}.html"),
            shap.force_plot(
                self.explainer.expected_value,
                self._shap_values[0],
                self.X_test[0],
                feature_names=LIGHTGBM_16_FEATURE_NAMES,
            ),
        )

    def export_shap_values(self, dates: list, output_dir: Path) -> None:
        import pandas as pd

        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._shap_values, columns=LIGHTGBM_16_FEATURE_NAMES)
        df["date"] = dates
        df.to_csv(output_dir / "shap_values.csv", index=False)
        return df

    def check_tcn_dominance(self, threshold: float = 0.7) -> dict[str, Any]:
        """Check if TCN features dominate (>threshold combined SHAP weight)."""
        if self._shap_values is None:
            self.compute_shap_values()

        mean_abs_shap = np.mean(np.abs(self._shap_values), axis=0)
        tcn_features = ["tcn_reg", "tcn_reg_delta", "tcn_prediction_stability"]
        tcn_indices = [LIGHTGBM_16_FEATURE_NAMES.index(f) for f in tcn_features if f in LIGHTGBM_16_FEATURE_NAMES]
        tcn_weight = sum(mean_abs_shap[i] for i in tcn_indices)
        total_weight = sum(mean_abs_shap)
        ratio = tcn_weight / (total_weight + 1e-9)

        result = {
            "tcn_combined_weight": float(ratio),
            "dominant": ratio > threshold,
            "feature_importance": {name: float(mean_abs_shap[i]) for i, name in enumerate(LIGHTGBM_16_FEATURE_NAMES)},
        }

        if result["dominant"]:
            logger.warning(f"  [SHAP] TCN features dominate: {ratio:.1%} > {threshold:.0%}")
            logger.warning("  [SHAP] Consider further regularization or dimensionality reduction")

        return result
