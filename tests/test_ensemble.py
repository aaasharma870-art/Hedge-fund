"""Tests for hedge_fund.ensemble - stacked ensemble model."""

import numpy as np
import pytest

from hedge_fund.ensemble import EnsembleModel


def _make_data(n=500, n_features=10, seed=42):
    """Create synthetic regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Simple linear relationship + noise
    coefs = rng.randn(n_features)
    y = X @ coefs + rng.randn(n) * 0.5
    return X, y


class TestEnsembleModel:
    def test_fit_and_predict(self):
        X, y = _make_data()
        model = EnsembleModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype == np.float64 or preds.dtype == np.float32

    def test_predict_before_fit_raises(self):
        model = EnsembleModel()
        X, _ = _make_data(n=10)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_score_returns_float(self):
        X, y = _make_data()
        model = EnsembleModel()
        model.fit(X, y)
        r2 = model.score(X, y)
        assert isinstance(r2, float)
        # Should fit training data reasonably well
        assert r2 > 0.5

    def test_feature_importances(self):
        X, y = _make_data(n_features=5)
        model = EnsembleModel()
        model.fit(X, y)
        imp = model.feature_importances_
        assert len(imp) == 5
        assert all(i >= 0 for i in imp)

    def test_small_dataset_fallback(self):
        """Very small datasets should use direct fit (no cross-val)."""
        X, y = _make_data(n=50, n_features=3)
        model = EnsembleModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 50

    def test_predictions_better_than_random(self):
        """Ensemble predictions should correlate with targets on held-out split."""
        X, y = _make_data(n=500, seed=1)
        X_train, y_train = X[:400], y[:400]
        X_test, y_test = X[400:], y[400:]
        model = EnsembleModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        correlation = np.corrcoef(preds, y_test)[0, 1]
        assert correlation > 0.3  # Should have some predictive power on same DGP

    def test_custom_xgb_params(self):
        X, y = _make_data(n=200)
        model = EnsembleModel(xgb_params={
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "verbosity": 0,
        })
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 200
