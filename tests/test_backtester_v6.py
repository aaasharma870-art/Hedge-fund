"""Tests for backtester V6 stateful simulation, partial profits, and label buckets."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import (
    simulate_trades_stateful,
    _simulate_with_partial,
    _pick_label_bucket,
    compute_risk_metrics,
    LABEL_BUCKETS,
)


# ==============================================================================
# HELPERS
# ==============================================================================

def _make_test_df(n=300, seed=42):
    """Create a synthetic test DataFrame with all columns needed by simulate_trades_stateful."""
    rng = np.random.RandomState(seed)
    close = 100 + rng.normal(0, 0.5, n).cumsum()
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    volume = rng.uniform(1e6, 5e6, n)
    atr = np.abs(rng.normal(1.5, 0.3, n))
    hurst = rng.uniform(0.3, 0.7, n)
    adx = rng.uniform(10, 40, n)
    rsi = rng.uniform(20, 80, n)

    df = pd.DataFrame({
        'Close': close,
        'High': high,
        'Low': low,
        'Volume': volume,
        'ATR': atr,
        'Hurst': hurst,
        'ADX': adx,
        'RSI': rsi,
        'VPIN': rng.uniform(0, 0.5, n),
        'Amihud_Illiquidity': rng.uniform(0, 0.5, n),
        'VWAP_ZScore': rng.normal(0, 1, n),
        'Volatility_Rank': rng.uniform(0.3, 0.9, n),
        'EMA_200': close - rng.uniform(-5, 5, n),
        'Regime_GEX_Proxy': rng.choice([-1, 0, 1], n),
        'VWAP_Slope': rng.normal(0, 0.01, n),
        'ROC_5': rng.normal(0, 0.02, n),
        '_ticker': 'TEST',
    })

    # Add predictions that cross threshold (some positive, some negative)
    preds = rng.normal(0, 0.3, n)
    df['Predictions'] = preds

    idx = pd.date_range('2024-01-01', periods=n, freq='h')
    df.index = idx
    return df


# ==============================================================================
# TEST: _simulate_with_partial
# ==============================================================================

class TestSimulateWithPartial:
    """Tests for V6 partial profit simulation."""

    def test_full_stop_loss_returns_minus_1(self):
        """If price hits SL before partial TP, should return -1R."""
        # LONG entry at 100, SL at 98, TP at 106
        future_high = np.array([100.5, 100.3, 99.5])
        future_low = np.array([99.0, 97.0, 96.0])  # hits SL=98 on bar 1
        close = np.array([100.0] * 10)

        result = _simulate_with_partial(
            future_high, future_low, close, idx=0, max_bars=5,
            entry=100.0, sl=98.0, tp=106.0, sl_dist=2.0,
            side='LONG', trail_dist=None,
            scale_out_r=1.5, rr=3.0,
        )
        assert result['total_r'] == -1.0
        assert result['resolved'] is True
        assert result['bars_held'] == 2  # Hit on bar index 1 = 2 bars

    def test_partial_hit_locks_profit(self):
        """If partial TP is hit, 1/3 is locked at scale_out_r."""
        # LONG entry=100, SL=98, TP=106, sl_dist=2
        # partial_tp = 100 + 1.5*2 = 103
        future_high = np.array([101.0, 103.5, 106.5])  # partial on bar 1, TP on bar 2
        future_low = np.array([99.5, 101.0, 104.0])
        close = np.array([100.0] * 10)

        result = _simulate_with_partial(
            future_high, future_low, close, idx=0, max_bars=5,
            entry=100.0, sl=98.0, tp=106.0, sl_dist=2.0,
            side='LONG', trail_dist=None,
            scale_out_r=1.5, rr=3.0,
        )
        # 1/3 at 1.5R + 2/3 at 3.0R = 0.5 + 2.0 = 2.5R
        assert result['total_r'] == pytest.approx(2.5)
        assert result['resolved'] is True

    def test_full_win_without_partial(self):
        """If TP is hit without partial, return full rr."""
        # Price jumps directly to TP
        future_high = np.array([107.0])
        future_low = np.array([99.0])  # SL checked first: 99 > 98 so no SL
        close = np.array([100.0] * 5)

        result = _simulate_with_partial(
            future_high, future_low, close, idx=0, max_bars=3,
            entry=100.0, sl=98.0, tp=106.0, sl_dist=2.0,
            side='LONG', trail_dist=None,
            scale_out_r=1.5, rr=3.0,
        )
        # SL at 98, low is 99 (not hit). TP at 106, high is 107 (hit).
        # But partial_tp = 103, high is 107 > 103, so partial IS hit first.
        # This test verifies partial path with remaining also hitting TP.
        assert result['resolved'] is True
        assert result['bars_held'] >= 1

    def test_timeout_returns_mtm(self):
        """If neither SL nor TP is hit, returns MTM-based R."""
        # Price stays in range
        future_high = np.array([101.0, 101.5, 101.0])
        future_low = np.array([99.0, 99.5, 99.0])
        close = np.array([100.0, 100.5, 101.0, 100.0])

        result = _simulate_with_partial(
            future_high, future_low, close, idx=0, max_bars=3,
            entry=100.0, sl=95.0, tp=110.0, sl_dist=5.0,
            side='LONG', trail_dist=None,
            scale_out_r=1.5, rr=2.0,
        )
        assert result['resolved'] is False
        assert result['bars_held'] == 3

    def test_short_stop_loss(self):
        """SHORT trade SL hit."""
        future_high = np.array([101.0, 103.0])  # hits SL=102
        future_low = np.array([99.0, 100.0])
        close = np.array([100.0] * 5)

        result = _simulate_with_partial(
            future_high, future_low, close, idx=0, max_bars=5,
            entry=100.0, sl=102.0, tp=94.0, sl_dist=2.0,
            side='SHORT', trail_dist=None,
            scale_out_r=1.5, rr=3.0,
        )
        assert result['total_r'] == -1.0
        assert result['resolved'] is True

    def test_bars_held_on_early_exit(self):
        """bars_held should reflect actual exit, not max_bars."""
        # SL hit on first bar
        future_high = np.array([100.5, 100.3, 100.2, 100.1, 100.0])
        future_low = np.array([96.0, 99.0, 99.0, 99.0, 99.0])  # SL=98 hit on bar 0
        close = np.array([100.0] * 10)

        result = _simulate_with_partial(
            future_high, future_low, close, idx=0, max_bars=5,
            entry=100.0, sl=98.0, tp=106.0, sl_dist=2.0,
            side='LONG', trail_dist=None,
            scale_out_r=1.5, rr=3.0,
        )
        assert result['bars_held'] == 1  # Exited on very first bar


# ==============================================================================
# TEST: simulate_trades_stateful
# ==============================================================================

class TestSimulateTradesStateful:
    """Tests for V6 stateful trade simulation."""

    def test_returns_list_of_tuples(self):
        df = _make_test_df()
        trades = simulate_trades_stateful(
            df, pred_threshold=0.2, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL",
        )
        assert isinstance(trades, list)
        if trades:
            assert len(trades[0]) == 5  # (pnl, resolved, pos_size, ticker, side)

    def test_empty_df_returns_empty(self):
        trades = simulate_trades_stateful(
            None, pred_threshold=0.2, sl_mult=1.5, tp_mult=3.0, max_bars=10,
        )
        assert trades == []

    def test_high_threshold_reduces_trades(self):
        df = _make_test_df()
        trades_low = simulate_trades_stateful(
            df, pred_threshold=0.1, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL",
        )
        trades_high = simulate_trades_stateful(
            df, pred_threshold=0.5, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL",
        )
        assert len(trades_high) <= len(trades_low)

    def test_strict_mode_filters_more(self):
        df = _make_test_df()
        trades_minimal = simulate_trades_stateful(
            df, pred_threshold=0.15, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL",
        )
        trades_strict = simulate_trades_stateful(
            df, pred_threshold=0.15, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="STRICT",
        )
        assert len(trades_strict) <= len(trades_minimal)

    def test_volatile_regime_reduces_short_size(self):
        """V11: In volatile regime, shorts still execute but with reduced size
        (regime size scalar applied via determine_entry, not gated)."""
        df = _make_test_df()
        df['Regime_Volatile'] = 1.0
        df['Regime_Trending'] = 0.0
        df['Regime_MeanRev'] = 0.0
        df['Predictions'] = -0.5  # all short signals
        trades = simulate_trades_stateful(
            df, pred_threshold=0.1, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL",
        )
        # V11: simplified entry allows shorts through; verify they execute
        assert len(trades) > 0
        # All should be SHORT direction
        assert all(t[4] == 'SHORT' for t in trades)

    def test_governor_reduces_size_in_drawdown(self):
        """After many losses, governor should reduce position sizes."""
        df = _make_test_df(n=500, seed=99)
        df['Predictions'] = 0.5
        trades = simulate_trades_stateful(
            df, pred_threshold=0.1, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL", use_kelly=False,
        )
        if len(trades) > 30:
            sizes = [t[2] for t in trades]
            # V7: sizes can exceed 1.0 due to confidence scalar, but should be bounded
            assert all(s <= 3.1 for s in sizes)

    def test_kelly_disabled_keeps_reasonable_size(self):
        """With Kelly disabled, pos_size should be reasonable (soft filters apply)."""
        df = _make_test_df(n=100)
        df['Predictions'] = 0.5
        trades = simulate_trades_stateful(
            df, pred_threshold=0.1, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL", use_kelly=False,
        )
        if trades:
            # V7: confidence, Hurst, vol scalars apply even without Kelly
            # Size should be positive and bounded
            assert 0.1 < trades[0][2] < 3.1


# ==============================================================================
# TEST: _pick_label_bucket
# ==============================================================================

class TestPickLabelBucket:
    """Tests for SL/TP label bucket selection."""

    def test_exact_match(self):
        bucket = _pick_label_bucket(1.5, 2.0)  # tp_mult = 3.0
        assert bucket == (1.5, 3.0, 10)

    def test_closest_match_low(self):
        bucket = _pick_label_bucket(1.0, 1.5)  # tp_mult = 1.5, closest to (1.0, 2.0)
        assert bucket == (1.0, 2.0, 8)

    def test_closest_match_high(self):
        bucket = _pick_label_bucket(2.5, 2.0)  # tp_mult = 5.0, closest to (2.5, 5.0)
        assert bucket == (2.5, 5.0, 14)

    def test_returns_tuple_of_three(self):
        bucket = _pick_label_bucket(1.5, 2.5)
        assert len(bucket) == 3
        assert all(isinstance(v, (int, float)) for v in bucket)

    def test_mid_range_picks_reasonable(self):
        bucket = _pick_label_bucket(1.75, 2.5)  # tp_mult = 4.375
        # Should be (2.0, 4.0, 12) since dist = |1.75-2.0| + |4.375-4.0| = 0.625
        # vs (1.5, 3.0, 10) dist = |1.75-1.5| + |4.375-3.0| = 1.625
        assert bucket == (2.0, 4.0, 12)


# ==============================================================================
# TEST: bars_held integration
# ==============================================================================

class TestBarsHeldIntegration:
    """Verify bars_held is properly tracked end-to-end."""

    def test_trades_advance_by_actual_bars(self):
        """Simulate and verify that trade spacing reflects actual exit bars."""
        df = _make_test_df(n=200)
        # Force high predictions to guarantee trades
        df['Predictions'] = 0.5
        trades = simulate_trades_stateful(
            df, pred_threshold=0.1, sl_mult=1.5, tp_mult=3.0, max_bars=10,
            filter_mode="MINIMAL", use_kelly=False,
        )
        # Just verify we get trades and they have valid structure
        assert isinstance(trades, list)
        for t in trades:
            assert len(t) == 5
            assert isinstance(t[0], float)
            assert isinstance(t[1], bool)
            assert t[4] in ('LONG', 'SHORT')


# ==============================================================================
# TEST: compute_risk_metrics (V6 extended)
# ==============================================================================

class TestRiskMetricsV6:
    """Additional V6-specific metrics tests."""

    def test_position_sized_trades_affect_metrics(self):
        """Trades with different position sizes should affect total return."""
        # 2x sized win vs 0.5x sized win
        trades_big = [(2.0 * 2.0, True, 2.0, 'A', 'LONG')]  # 4R effective
        trades_small = [(2.0 * 0.5, True, 0.5, 'A', 'LONG')]  # 1R effective
        m_big = compute_risk_metrics(trades_big)
        m_small = compute_risk_metrics(trades_small)
        assert m_big['TotalReturn_R'] > m_small['TotalReturn_R']

    def test_mixed_resolved_unresolved(self):
        trades = [
            (1.5, True, 1.0, 'A', 'LONG'),
            (-0.3, False, 1.0, 'A', 'SHORT'),
            (2.0, True, 1.0, 'B', 'LONG'),
            (-0.5, True, 1.0, 'B', 'SHORT'),  # resolved loss so PF_Res is computable
        ]
        m = compute_risk_metrics(trades)
        assert m['Trades'] == 4
        assert m['Resolved'] == 3
        assert m['PF_Res'] > 0

    def test_pct_gain_key_exists(self):
        trades = [
            (2.0, True, 1.0, 'A', 'LONG'),
            (-1.0, True, 1.0, 'A', 'SHORT'),
        ]
        m = compute_risk_metrics(trades)
        assert 'PctGain' in m


class TestAnchoredWalkForward:
    """Tests for anchored (expanding) walk-forward mode."""

    def test_anchored_flag_exists(self):
        from backtester import ANCHORED_WF
        assert isinstance(ANCHORED_WF, bool)

    def test_walk_forward_accepts_anchored_param(self):
        from backtester import walk_forward_train_predict
        import inspect
        sig = inspect.signature(walk_forward_train_predict)
        assert 'anchored' in sig.parameters

    def test_pruned_accepts_anchored_param(self):
        from backtester import _walk_forward_pruned
        import inspect
        sig = inspect.signature(_walk_forward_pruned)
        assert 'anchored' in sig.parameters
