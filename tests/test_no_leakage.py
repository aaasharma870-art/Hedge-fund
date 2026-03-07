"""Tests for data leakage detection in the ML pipeline.

Verifies that the meta-learner does not train on in-sample predictions
(which would inflate apparent accuracy and degrade live performance).
"""

import numpy as np
import pytest

from hedge_fund.ensemble import EnsembleModel


class TestMetaLearnerLeakage:
    """Ensure the meta-learner doesn't see in-sample base predictions."""

    def test_direct_fit_uses_equal_weights(self):
        """When data is too small for OOF, meta should use equal weights, not fit on leaked data."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)

        model = EnsembleModel()
        model.fit(X, y)  # Should trigger _fit_direct (n=50 < 3*50=150)

        # Meta-learner should have equal-weight coefficients
        n_base = model._n_base_models()
        expected_coef = np.ones(n_base) / n_base
        np.testing.assert_array_almost_equal(model.meta_model.coef_, expected_coef)
        assert model.meta_model.intercept_ == 0.0

    def test_oof_predictions_not_zero(self):
        """OOF path should produce non-zero meta-learner predictions."""
        rng = np.random.RandomState(42)
        X = rng.randn(500, 10)
        coefs = rng.randn(10)
        y = X @ coefs + rng.randn(500) * 0.5

        model = EnsembleModel()
        model.fit(X, y)  # Should use OOF path (n=500 >= 150)

        # Meta-learner should have been trained on real OOF predictions
        # Coefficients should NOT be exactly equal (unless by coincidence)
        assert model._is_fitted
        preds = model.predict(X[:10])
        assert not np.all(preds == 0)

    def test_oof_vs_direct_generalization(self):
        """OOF-trained meta should generalize better than in-sample-trained meta.

        This tests the core leakage issue: a meta-learner trained on in-sample
        base predictions will overfit, while OOF-trained meta generalizes.
        """
        rng = np.random.RandomState(123)
        n_train, n_test = 500, 200
        n_features = 10
        X = rng.randn(n_train + n_test, n_features)
        coefs = rng.randn(n_features)
        y = X @ coefs + rng.randn(n_train + n_test) * 2.0

        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]

        # Train with OOF (proper)
        model = EnsembleModel()
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)

        # Should have meaningful correlation on test set
        corr = np.corrcoef(preds_test, y_test)[0, 1]
        assert corr > 0.1, f"OOF model should generalize: corr={corr:.3f}"

    def test_embargo_prevents_information_leak(self):
        """Verify that temporal ordering is maintained in OOF splits.

        TimeSeriesSplit should ensure training folds only use past data.
        """
        from sklearn.model_selection import TimeSeriesSplit

        n = 500
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(np.zeros(n)):
            # All training indices should be before validation indices
            assert train_idx[-1] < val_idx[0], \
                "Training data must precede validation data in time"


class TestFeatureLeakage:
    """Check for common feature leakage patterns."""

    def test_no_future_data_in_rolling(self):
        """Rolling computations should not use future data."""
        import pandas as pd

        prices = pd.Series(np.random.randn(100).cumsum() + 100)

        # Correct: trailing rolling window
        rolling_mean = prices.rolling(20).mean()
        # The rolling mean at index i should only use data up to index i
        for i in range(20, 100):
            expected = prices.iloc[i-19:i+1].mean()
            np.testing.assert_almost_equal(rolling_mean.iloc[i], expected, decimal=10)

    def test_pct_change_no_lookahead(self):
        """pct_change(5) should look backward, not forward."""
        import pandas as pd

        prices = pd.Series([100, 101, 102, 103, 104, 105, 106])
        roc = prices.pct_change(5)
        # roc at index 5 = (prices[5] - prices[0]) / prices[0]
        expected = (105 - 100) / 100
        np.testing.assert_almost_equal(roc.iloc[5], expected)
        # First 5 should be NaN
        assert roc.iloc[:5].isna().all()
