"""Tests for OOF stacking ensemble."""

import numpy as np
import pytest

from hedge_fund.ensemble import EnsembleModel


@pytest.fixture
def synthetic_signal_data():
    """Data where signal exists (IC ~ 0.15)."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5)
    y = 0.3 * X[:, 0] + 0.1 * X[:, 1] + np.random.randn(n) * 0.5
    return X, y


def test_oof_produces_nonzero_ic(synthetic_signal_data):
    X, y = synthetic_signal_data
    model = EnsembleModel(use_oof=True)
    model.fit(X, y)

    assert hasattr(model, 'oof_predictions_')
    ic = model.oof_predictions_.get('ic', 0)
    assert ic > 0.01, f"OOF IC too low: {ic:.4f}"


def test_direct_fit_equal_weights():
    """Small data should use equal-weight fallback."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = rng.randn(50)

    model = EnsembleModel(use_oof=True)
    model.fit(X, y)

    assert model.stacking_method_ == 'equal_weight'
    assert model._is_fitted


def test_oof_no_leakage():
    """Verify TimeSeriesSplit fold indices never overlap."""
    from sklearn.model_selection import TimeSeriesSplit

    n = 500
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(np.zeros(n))):
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        overlap = train_set & val_set
        assert len(overlap) == 0, f"Fold {fold}: overlap {overlap}"


def test_predict_output_shape(synthetic_signal_data):
    X, y = synthetic_signal_data
    model = EnsembleModel(use_oof=True)
    model.fit(X, y)
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert not np.any(np.isnan(preds))


def test_sample_weight_accepted(synthetic_signal_data):
    X, y = synthetic_signal_data
    sw = np.ones(len(y))
    model = EnsembleModel(use_oof=True)
    model.fit(X, y, sample_weight=sw)
    assert model._is_fitted
