"""Tests for hedge_fund.simulation - bracket trade simulation and labeling."""

import numpy as np
import pandas as pd
import pytest

from hedge_fund.simulation import simulate_exit, compute_bracket_labels


class TestSimulateExit:
    def test_long_hits_take_profit(self):
        # Price goes straight up
        highs = np.array([105.0, 110.0, 115.0])
        lows = np.array([99.0, 104.0, 109.0])
        outcome, price = simulate_exit(highs, lows, sl=95.0, tp=110.0, side="LONG")
        assert outcome == "win"
        assert price == 110.0

    def test_long_hits_stop_loss(self):
        # Price goes straight down
        highs = np.array([99.0, 96.0, 93.0])
        lows = np.array([97.0, 94.0, 91.0])
        outcome, price = simulate_exit(highs, lows, sl=95.0, tp=120.0, side="LONG")
        assert outcome == "loss"
        assert price == 95.0

    def test_short_hits_take_profit(self):
        # Price goes down
        highs = np.array([99.0, 96.0, 93.0])
        lows = np.array([97.0, 94.0, 89.0])
        outcome, price = simulate_exit(highs, lows, sl=105.0, tp=90.0, side="SHORT")
        assert outcome == "win"
        assert price == 90.0

    def test_short_hits_stop_loss(self):
        # Price goes up past stop
        highs = np.array([102.0, 106.0, 110.0])
        lows = np.array([99.0, 103.0, 107.0])
        outcome, price = simulate_exit(highs, lows, sl=105.0, tp=80.0, side="SHORT")
        assert outcome == "loss"
        assert price == 105.0

    def test_timeout_when_no_barrier_hit(self):
        # Price stays in range
        highs = np.array([101.0, 101.5, 102.0])
        lows = np.array([99.0, 99.5, 98.5])
        outcome, price = simulate_exit(highs, lows, sl=90.0, tp=120.0, side="LONG")
        assert outcome == "timeout"
        assert price is None

    def test_stop_loss_checked_before_take_profit(self):
        # On the same bar, both SL and TP are hit.
        # SL should be checked first (conservative).
        highs = np.array([120.0])  # TP at 110, hits
        lows = np.array([90.0])   # SL at 95, also hits
        outcome, price = simulate_exit(highs, lows, sl=95.0, tp=110.0, side="LONG")
        assert outcome == "loss", "SL should be checked before TP"

    def test_trailing_stop_long(self):
        # Price goes up, then comes back down
        highs = np.array([105.0, 110.0, 108.0, 106.0])
        lows = np.array([102.0, 107.0, 105.0, 103.0])
        # Trail dist = 3.0, so after high=110, SL moves to 107
        # Then low=103 doesn't hit 107, low=103 < 107? Actually 103 < 107 yes
        # Wait: bar 2: low=105 > 107? No, 105 < 107, so trail stop hits
        outcome, price = simulate_exit(
            highs, lows, sl=95.0, tp=120.0, side="LONG", trail_dist=3.0
        )
        assert outcome == "trail_stop"

    def test_trailing_stop_short(self):
        # Price goes down, then comes back up
        highs = np.array([98.0, 94.0, 96.0, 99.0])
        lows = np.array([95.0, 91.0, 93.0, 96.0])
        # Trail dist = 3.0, after low=91, SL moves to 94
        # Then high=96 < 94? No, 96 > 94, so trail stop hits
        outcome, price = simulate_exit(
            highs, lows, sl=105.0, tp=80.0, side="SHORT", trail_dist=3.0
        )
        assert outcome == "trail_stop"

    def test_empty_bars_returns_timeout(self):
        outcome, price = simulate_exit(
            np.array([]), np.array([]), sl=95.0, tp=110.0, side="LONG"
        )
        assert outcome == "timeout"
        assert price is None

    def test_trailing_stop_never_moves_backward_long(self):
        # For LONG, SL should never decrease
        # bar0: high=108, trail SL = max(95, 108-3) = 105. low=103 > 95 (orig) but < 105 -> trail_stop at 105
        # Actually bar0 checks SL=95 first (l=103 > 95), then TP (h=108 < 120), then trail: SL=105
        # bar1: SL=105, l=100 <= 105 -> trail_stop at 105
        highs = np.array([108.0, 105.0, 110.0, 103.0])
        lows = np.array([103.0, 100.0, 106.0, 99.0])
        outcome, price = simulate_exit(
            highs, lows, sl=95.0, tp=120.0, side="LONG", trail_dist=3.0
        )
        assert outcome == "trail_stop"
        assert price == 105.0  # bar0 moves SL to 105, bar1 low=100 triggers it


class TestComputeBracketLabels:
    def _make_df(self, n=200):
        """Create a simple OHLCV DataFrame with ATR for testing."""
        rng = np.random.RandomState(42)
        close = 100 + rng.normal(0, 1, n).cumsum()
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)
        atr = pd.Series(np.abs(rng.normal(1.5, 0.3, n)), name="ATR")
        df = pd.DataFrame({
            "Close": close, "High": high, "Low": low, "ATR": atr
        })
        return df

    def test_returns_series(self):
        df = self._make_df()
        result = compute_bracket_labels(df, max_bars=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_labels_are_bounded(self):
        df = self._make_df(500)
        result = compute_bracket_labels(df, sl_mult=1.5, tp_mult=3.0, max_bars=10)
        rr = 3.0 / 1.5  # = 2.0
        assert (result.abs() <= rr + 0.01).all(), "Labels should be bounded by R:R ratio"

    def test_positive_labels_mean_long(self):
        """Positive labels indicate LONG is preferred."""
        df = self._make_df()
        result = compute_bracket_labels(df, max_bars=10)
        # At least some positive and some negative labels in random data
        assert (result > 0).any(), "Should have some positive (long) labels"
        assert (result < 0).any(), "Should have some negative (short) labels"

    def test_last_bars_are_zero(self):
        """Last max_bars should be 0 (not enough future data)."""
        df = self._make_df(100)
        max_bars = 10
        result = compute_bracket_labels(df, max_bars=max_bars)
        assert (result.iloc[-max_bars:] == 0).all()

    def test_zero_atr_bars_skipped(self):
        df = self._make_df()
        df.loc[50, "ATR"] = 0.0
        df.loc[51, "ATR"] = np.nan
        result = compute_bracket_labels(df, max_bars=10)
        assert result.iloc[50] == 0.0
        assert result.iloc[51] == 0.0

    def test_different_params_give_different_labels(self):
        df = self._make_df(300)
        labels_tight = compute_bracket_labels(df, sl_mult=0.5, tp_mult=1.0, max_bars=5)
        labels_wide = compute_bracket_labels(df, sl_mult=3.0, tp_mult=6.0, max_bars=20)
        # Different params should produce different distributions
        assert not labels_tight.equals(labels_wide)
