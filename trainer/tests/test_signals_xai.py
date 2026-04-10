"""Tests for SHAPAnalyzer in ``trainer/src/utils/signals_xai.py``."""

from pathlib import Path

import numpy as np
import pytest

from trainer.src.utils.signals_xai import LIGHTGBM_16_FEATURE_NAMES, SHAPAnalyzer


def _make_lgbm_model():
    """Create a simple LightGBM model for testing."""
    import lightgbm as lgb

    np.random.seed(42)
    X = np.random.randn(200, 16)
    y = X[:, 0] * 0.5 + X[:, 4] * 0.3 + np.random.randn(200) * 0.1
    model = lgb.LGBMRegressor(n_estimators=10, num_leaves=4, verbose=-1)
    model.fit(X, y)
    return model, X


class TestFeatureNames:
    def test_feature_count(self):
        assert len(LIGHTGBM_16_FEATURE_NAMES) == 16

    def test_tcn_features_present(self):
        assert "tcn_reg" in LIGHTGBM_16_FEATURE_NAMES
        assert "tcn_prediction_stability" in LIGHTGBM_16_FEATURE_NAMES


class TestSHAPAnalyzer:
    @pytest.fixture
    def analyzer(self):
        model, X = _make_lgbm_model()
        return SHAPAnalyzer(model, X)

    def test_compute_shap_values(self, analyzer):
        sv = analyzer.compute_shap_values()
        assert sv.shape[1] == 16

    def test_export_shap_values(self, analyzer, tmp_path):
        analyzer.compute_shap_values()
        dates = [f"2024-01-{i:02d}" for i in range(1, len(analyzer.X_test) + 1)]
        analyzer.export_shap_values(dates, tmp_path)
        csv_path = tmp_path / "shap_values.csv"
        assert csv_path.exists()

    def test_generate_summary_plot(self, analyzer, tmp_path):
        analyzer.compute_shap_values()
        out = tmp_path / "shap_summary.png"
        analyzer.generate_summary_plot(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_check_tcn_dominance(self, analyzer):
        analyzer.compute_shap_values()
        result = analyzer.check_tcn_dominance()
        assert "dominant" in result
        assert "tcn_combined_weight" in result
        assert isinstance(result["dominant"], (bool, np.bool_))
