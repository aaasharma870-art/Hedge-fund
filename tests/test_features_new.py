"""Tests for new V7 feature functions."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """Generate realistic-looking 200-bar OHLCV data."""
    np.random.seed(42)
    n = 200
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.005)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.003)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.003)
    open_ = close * (1 + np.random.randn(n) * 0.002)
    volume = np.random.lognormal(15, 0.5, n)
    dates = pd.date_range('2024-01-02 09:30', periods=n, freq='1h')
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)


def test_ofi_range(sample_ohlcv):
    from hedge_fund.features import compute_ofi
    ofi = compute_ofi(sample_ohlcv)
    assert ofi.notna().sum() > 100, "OFI has too many NaNs"
    assert ofi.abs().max() <= 4.0, "OFI exceeds clip range"
    assert ofi.std() > 0.01, "OFI is degenerate"


def test_rv_ratio_positive(sample_ohlcv):
    from hedge_fund.features import compute_rv_ratio
    rv = compute_rv_ratio(sample_ohlcv)
    valid = rv.dropna()
    assert (valid > 0).all(), "RV_Ratio must be positive"
    assert valid.max() <= 5.0, "RV_Ratio exceeds clip range"


def test_momentum_decomp_returns_two_series(sample_ohlcv):
    from hedge_fund.features import compute_momentum_decomp
    og, im = compute_momentum_decomp(sample_ohlcv)
    assert og.name == 'Overnight_Gap'
    assert im.name == 'Intraday_Mom'
    assert len(og) == len(sample_ohlcv)
    assert og.abs().max() <= 0.10
    assert im.abs().max() <= 0.10


def test_efficiency_ratio_bounded(sample_ohlcv):
    from hedge_fund.features import compute_efficiency_ratio
    er = compute_efficiency_ratio(sample_ohlcv)
    valid = er.dropna()
    assert (valid >= 0).all(), "ER must be >= 0"
    assert (valid <= 1).all(), "ER must be <= 1"


def test_bar_patterns_sum(sample_ohlcv):
    from hedge_fund.features import compute_bar_patterns
    uw, lw, br = compute_bar_patterns(sample_ohlcv, smooth=1)
    total = uw + lw + br
    valid_mask = total.notna()
    assert np.allclose(total[valid_mask], 1.0, atol=0.05)


def test_vpt_acceleration_range(sample_ohlcv):
    from hedge_fund.features import compute_vpt_acceleration
    vpt = compute_vpt_acceleration(sample_ohlcv)
    valid = vpt.dropna()
    assert len(valid) > 50
    assert valid.abs().max() <= 4.0


def test_atr_channel_pos_range(sample_ohlcv):
    from hedge_fund.features import compute_atr_channel_pos
    pos = compute_atr_channel_pos(sample_ohlcv)
    valid = pos.dropna()
    assert (valid >= -0.5).all()
    assert (valid <= 0.5).all()


def test_expected_features_constant():
    from hedge_fund.features import EXPECTED_FEATURES
    assert isinstance(EXPECTED_FEATURES, list)
    assert len(EXPECTED_FEATURES) >= 15
    assert 'OFI' in EXPECTED_FEATURES
    assert 'VPIN' in EXPECTED_FEATURES
    assert 'Efficiency_Ratio' in EXPECTED_FEATURES


def test_ofi_no_nan_after_warmup(sample_ohlcv):
    from hedge_fund.features import compute_ofi
    ofi = compute_ofi(sample_ohlcv)
    assert ofi.iloc[60:].isna().sum() == 0
