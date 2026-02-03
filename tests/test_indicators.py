"""Tests for hedge_fund.indicators - ManualTA technical indicators."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.indicators import ManualTA


def _make_price_series(n=100, start=100.0, seed=42):
    """Generate a synthetic price series for testing."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(0, 0.02, n)
    prices = start * np.exp(np.cumsum(returns))
    return pd.Series(prices, name="Close")


def _make_ohlcv(n=100, start=100.0, seed=42):
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    close = _make_price_series(n, start, seed)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    volume = pd.Series(rng.randint(100_000, 1_000_000, n), dtype=float)
    return high, low, close, volume


class TestRSI:
    def test_rsi_returns_series(self):
        close = _make_price_series()
        result = ManualTA.rsi(close, length=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_rsi_bounded_0_100(self):
        close = _make_price_series(200)
        result = ManualTA.rsi(close, length=14)
        valid = result.dropna()
        assert (valid >= 0).all(), "RSI should not go below 0"
        assert (valid <= 100).all(), "RSI should not exceed 100"

    def test_rsi_overbought_on_strong_uptrend(self):
        # Monotonically increasing prices should yield RSI near 100
        close = pd.Series(np.linspace(100, 200, 100))
        result = ManualTA.rsi(close, length=14)
        assert result.iloc[-1] > 90, "Strong uptrend should have RSI > 90"

    def test_rsi_oversold_on_strong_downtrend(self):
        close = pd.Series(np.linspace(200, 100, 100))
        result = ManualTA.rsi(close, length=14)
        assert result.iloc[-1] < 10, "Strong downtrend should have RSI < 10"

    def test_rsi_around_50_on_flat(self):
        # Alternating up/down should keep RSI near 50
        vals = [100 + (0.5 if i % 2 == 0 else -0.5) for i in range(200)]
        close = pd.Series(vals)
        result = ManualTA.rsi(close, length=14)
        assert 40 < result.iloc[-1] < 60, "Flat series should have RSI near 50"

    def test_rsi_different_lengths(self):
        close = _make_price_series()
        r7 = ManualTA.rsi(close, length=7)
        r14 = ManualTA.rsi(close, length=14)
        # Shorter period should be more volatile (higher std)
        assert r7.std() > r14.std(), "Shorter RSI should be more volatile"


class TestATR:
    def test_atr_returns_series(self):
        high, low, close, _ = _make_ohlcv()
        result = ManualTA.atr(high, low, close, length=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_atr_always_positive(self):
        high, low, close, _ = _make_ohlcv(200)
        result = ManualTA.atr(high, low, close, length=14)
        valid = result.dropna()
        assert (valid > 0).all(), "ATR should always be positive"

    def test_atr_higher_for_volatile_series(self):
        # Low vol series
        rng = np.random.RandomState(42)
        close_calm = pd.Series(100 + rng.normal(0, 0.1, 200).cumsum())
        high_calm = close_calm + 0.2
        low_calm = close_calm - 0.2

        # High vol series
        close_wild = pd.Series(100 + rng.normal(0, 2.0, 200).cumsum())
        high_wild = close_wild + 3.0
        low_wild = close_wild - 3.0

        atr_calm = ManualTA.atr(high_calm, low_calm, close_calm).iloc[-1]
        atr_wild = ManualTA.atr(high_wild, low_wild, close_wild).iloc[-1]
        assert atr_wild > atr_calm, "Volatile series should have higher ATR"

    def test_atr_length_parameter(self):
        high, low, close, _ = _make_ohlcv(200)
        atr_7 = ManualTA.atr(high, low, close, length=7)
        atr_14 = ManualTA.atr(high, low, close, length=14)
        # Both should produce valid output
        assert atr_7.dropna().shape[0] > 0
        assert atr_14.dropna().shape[0] > 0


class TestBBands:
    def test_bbands_returns_dataframe(self):
        close = _make_price_series()
        result = ManualTA.bbands(close, length=20, std=2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3

    def test_bbands_column_names(self):
        close = _make_price_series()
        result = ManualTA.bbands(close, length=20, std=2)
        assert "BBL_20_2.0" in result.columns
        assert "BBM_20_2.0" in result.columns
        assert "BBU_20_2.0" in result.columns

    def test_bbands_ordering(self):
        """Lower band < Middle < Upper band."""
        close = _make_price_series(200)
        result = ManualTA.bbands(close, length=20, std=2).dropna()
        assert (result["BBL_20_2.0"] <= result["BBM_20_2.0"]).all()
        assert (result["BBM_20_2.0"] <= result["BBU_20_2.0"]).all()

    def test_bbands_middle_is_sma(self):
        close = _make_price_series(200)
        result = ManualTA.bbands(close, length=20, std=2)
        sma = close.rolling(20).mean()
        pd.testing.assert_series_equal(
            result["BBM_20_2.0"], sma, check_names=False
        )

    def test_bbands_width_scales_with_std(self):
        close = _make_price_series(200)
        bb1 = ManualTA.bbands(close, length=20, std=1).dropna()
        bb2 = ManualTA.bbands(close, length=20, std=2).dropna()
        width1 = (bb1.iloc[:, 2] - bb1.iloc[:, 0]).mean()
        width2 = (bb2.iloc[:, 2] - bb2.iloc[:, 0]).mean()
        assert width2 > width1, "2-std bands should be wider than 1-std"


class TestADX:
    def test_adx_returns_dataframe(self):
        high, low, close, _ = _make_ohlcv()
        result = ManualTA.adx(high, low, close, length=14)
        assert isinstance(result, pd.DataFrame)
        assert f"ADX_14" in result.columns

    def test_adx_non_negative(self):
        high, low, close, _ = _make_ohlcv(200)
        result = ManualTA.adx(high, low, close, length=14)
        valid = result["ADX_14"].dropna()
        assert (valid >= 0).all(), "ADX should be non-negative"

    def test_adx_high_for_trending_market(self):
        # Strong uptrend
        n = 200
        close = pd.Series(np.linspace(100, 200, n))
        high = close + 1.0
        low = close - 0.5
        result = ManualTA.adx(high, low, close, length=14)
        assert result["ADX_14"].iloc[-1] > 30, "Strong trend should have ADX > 30"

    def test_adx_low_for_choppy_market(self):
        # Choppy: oscillating prices
        n = 200
        vals = [100 + 2 * np.sin(i * 0.5) for i in range(n)]
        close = pd.Series(vals)
        high = close + 0.5
        low = close - 0.5
        result = ManualTA.adx(high, low, close, length=14)
        assert result["ADX_14"].iloc[-1] < 30, "Choppy market should have lower ADX"
