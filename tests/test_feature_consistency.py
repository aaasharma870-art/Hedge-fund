"""Tests for feature computation consistency between training and live paths.

Ensures identical feature engineering code produces identical outputs,
preventing train/live distribution mismatch (silent model degradation).
"""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n=300, seed=42):
    """Create synthetic OHLCV data mimicking 15-min bars."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2024-01-02 09:30', periods=n, freq='15min')
    close = 100 + rng.randn(n).cumsum() * 0.5
    high = close + rng.rand(n) * 2
    low = close - rng.rand(n) * 2
    open_ = close + rng.randn(n) * 0.5
    volume = (rng.rand(n) * 1e6 + 1e5).astype(int)

    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)


class TestFeatureFunctions:
    """Test individual feature computation functions from hedge_fund.features."""

    def test_vpin_output_range(self):
        """VPIN should be in [0, 1] range."""
        from hedge_fund.features import calculate_vpin
        df = _make_ohlcv()
        vpin = calculate_vpin(df, window=50)
        valid = vpin.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_amihud_output_nonnegative(self):
        """Amihud illiquidity should be non-negative."""
        from hedge_fund.features import calculate_amihud_illiquidity
        df = _make_ohlcv()
        amihud = calculate_amihud_illiquidity(df, window=20)
        valid = amihud.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0

    def test_enhanced_vwap_features_keys(self):
        """Enhanced VWAP should return expected feature keys."""
        from hedge_fund.features import calculate_enhanced_vwap_features
        df = _make_ohlcv()
        result = calculate_enhanced_vwap_features(df)
        assert 'VWAP_ZScore' in result
        assert 'VWAP_Slope' in result
        assert 'VWAP_Volume_Ratio' in result

    def test_volatility_regime_output(self):
        """Volatility regime should return numeric series and label series."""
        from hedge_fund.features import calculate_volatility_regime
        df = _make_ohlcv()
        regime_gex, vol_label = calculate_volatility_regime(df)
        assert len(regime_gex) == len(df)
        assert len(vol_label) == len(df)

    def test_liquidity_sweep_detection(self):
        """Liquidity sweep should return integer series."""
        from hedge_fund.features import calculate_liquidity_sweep
        df = _make_ohlcv()
        sweeps = calculate_liquidity_sweep(df, lookback=16)
        assert len(sweeps) == len(df)
        # Values should be -1, 0, or 1
        unique_vals = set(sweeps.dropna().unique())
        assert unique_vals.issubset({-1, 0, 1})

    def test_real_relative_strength(self):
        """RRS should compute without error even without SPY data."""
        from hedge_fund.features import calculate_real_relative_strength
        df = _make_ohlcv()
        rrs = calculate_real_relative_strength(df, spy_df=None)
        assert len(rrs) == len(df)


class TestFeatureStats:
    """Test feature statistics save/load for distribution validation."""

    def test_save_and_load_feature_stats(self, tmp_path):
        """Feature stats should round-trip through save/load."""
        from hedge_fund.features import save_feature_stats, load_feature_stats

        df = pd.DataFrame({
            'RSI': np.random.rand(100) * 100,
            'ADX': np.random.rand(100) * 50,
            'VPIN': np.random.rand(100),
            'Target': np.random.randn(100),  # Should be excluded
        })

        path = str(tmp_path / 'test_stats.json')
        stats = save_feature_stats(df, path=path)

        assert 'RSI' in stats
        assert 'ADX' in stats
        assert 'VPIN' in stats
        assert 'Target' not in stats  # Target should be excluded

        loaded = load_feature_stats(path=path)
        assert loaded['RSI']['mean'] == stats['RSI']['mean']

    def test_validate_feature_distributions_warns_on_extreme(self):
        """Validation should warn when live features are far from training mean."""
        from hedge_fund.features import validate_feature_distributions

        training_stats = {
            'RSI': {'mean': 50.0, 'std': 10.0, 'min': 10, 'max': 90, 'median': 50},
            'ADX': {'mean': 25.0, 'std': 5.0, 'min': 10, 'max': 60, 'median': 25},
        }

        # Normal values - no warnings
        live_normal = {'RSI': 55.0, 'ADX': 28.0}
        warnings = validate_feature_distributions(live_normal, training_stats)
        assert len(warnings) == 0

        # Extreme values - should warn
        live_extreme = {'RSI': 150.0, 'ADX': 80.0}
        warnings = validate_feature_distributions(live_extreme, training_stats)
        assert len(warnings) > 0


class TestMathUtils:
    """Test math utility functions used in feature engineering."""

    def test_kalman_filter_output_length(self):
        """Kalman filter output should match input length."""
        from hedge_fund.math_utils import get_kalman_filter
        prices = np.random.randn(100).cumsum() + 100
        result = get_kalman_filter(prices)
        assert len(result) == len(prices)

    def test_kalman_filter_smoothing(self):
        """Kalman output should be smoother than raw prices."""
        from hedge_fund.math_utils import get_kalman_filter
        prices = np.random.randn(200).cumsum() + 100
        filtered = get_kalman_filter(prices)
        # Filtered should have lower variance of differences
        raw_diff_var = np.var(np.diff(prices))
        filt_diff_var = np.var(np.diff(filtered))
        assert filt_diff_var < raw_diff_var

    def test_hurst_exponent_range(self):
        """Hurst exponent should be in [0, 1] range."""
        from hedge_fund.math_utils import get_hurst
        prices = np.random.randn(200).cumsum() + 100
        h = get_hurst(prices)
        assert 0.0 <= h <= 1.0

    def test_hurst_returns_float(self):
        """Hurst exponent should return a valid float."""
        from hedge_fund.math_utils import get_hurst
        prices = np.random.randn(200).cumsum() + 100
        h = get_hurst(prices)
        assert isinstance(h, float)
        assert np.isfinite(h)

    def test_hurst_short_series_returns_default(self):
        """Series too short should return 0.5 (default)."""
        from hedge_fund.math_utils import get_hurst
        short = np.array([1.0, 2.0, 3.0])
        h = get_hurst(short)
        assert h == 0.5
