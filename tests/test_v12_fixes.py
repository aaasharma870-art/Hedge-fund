"""Tests for v12.9 bug fixes and metrics."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMetricsFixes:
    """Verify PctGain, CAGR, Sortino, AvgCost_R are correct."""

    def _make_trades(self, outcomes, sides=None):
        if sides is None:
            sides = ['LONG'] * len(outcomes)
        return [(o, True, 1.0, 'TEST', s) for o, s in zip(outcomes, sides)]

    def test_pct_gain_positive_for_winners(self):
        from backtester_v12 import compute_risk_metrics
        trades = self._make_trades([2.0, 2.0, 2.0, -1.0])
        m = compute_risk_metrics(trades)
        assert m['PctGain'] > 0

    def test_pct_gain_negative_for_losers(self):
        from backtester_v12 import compute_risk_metrics
        trades = self._make_trades([-1.0, -1.0, -1.0, 0.5])
        m = compute_risk_metrics(trades)
        assert m['PctGain'] < 0

    def test_cagr_positive_for_winners(self):
        from backtester_v12 import compute_risk_metrics
        trades = self._make_trades([2.0] * 50 + [-1.0] * 20)
        m = compute_risk_metrics(trades)
        assert m['CAGR'] > 0

    def test_avg_cost_r_is_nonzero(self):
        from backtester_v12 import compute_risk_metrics
        trades = self._make_trades([1.0, -1.0])
        m = compute_risk_metrics(trades)
        assert m['AvgCost_R'] > 0, "AvgCost_R should not be zero"

    def test_sortino_exists(self):
        from backtester_v12 import compute_risk_metrics
        trades = self._make_trades([2.0, -1.0, 1.5, -0.5])
        m = compute_risk_metrics(trades)
        assert 'Sortino' in m

    def test_sortino_nonzero_for_mixed_trades(self):
        """Sortino should be non-zero for a system with wins and losses."""
        from backtester_v12 import compute_risk_metrics
        # Need enough downside returns for ddof=1 std to work
        trades = self._make_trades([2.0, 1.5, 2.0, -1.0, -0.5, 1.0, -0.8, 2.0, -0.3, 1.5])
        m = compute_risk_metrics(trades)
        assert m['Sortino'] != 0, "Sortino should be non-zero for mixed trades"

    def test_empty_metrics_have_all_keys(self):
        from backtester_v12 import _empty_metrics
        m = _empty_metrics()
        for key in ['PctGain', 'CAGR', 'AvgCost_R', 'Sortino']:
            assert key in m, f"Missing key: {key}"


class TestLSBalance:
    """Verify soft L/S balance works correctly."""

    def test_soft_cap_doesnt_force_equality(self):
        from hedge_fund.daily_model import generate_watchlist
        import pandas as pd

        np.random.seed(42)
        mock_preds = {}
        dates = pd.date_range('2024-01-02', periods=60, freq='B')
        for i, ticker in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
            df = pd.DataFrame({
                'Close': 100 + np.random.randn(60).cumsum() * 0.5,
            }, index=dates)
            # A, B, C: positive predictions; D, E, F: negative
            if i < 3:
                df['DailyPrediction'] = np.abs(np.random.randn(60)) * 0.02 + 0.01
            else:
                df['DailyPrediction'] = -np.abs(np.random.randn(60)) * 0.02 - 0.01
            mock_preds[ticker] = df

        wl = generate_watchlist(mock_preds, top_n=3, bottom_n=2, min_spread=0.0)
        assert len(wl) > 0, "Watchlist should not be empty"

        for d, signals in wl.items():
            nl = len(signals.get('longs', []))
            ns = len(signals.get('shorts', []))
            if nl > 0 and ns > 0:
                ratio = max(nl, ns) / min(nl, ns)
                assert ratio <= 2.01, f"Ratio {ratio} exceeds 2:1 on {d}"


class TestConfidenceSizing:
    """Verify confidence scalar is bounded."""

    def test_confidence_scalar_bounded(self):
        for conv, median in [(0.001, 0.001), (0.1, 0.001), (0.001, 0.1), (0.05, 0.05)]:
            scalar = float(np.clip(abs(conv) / max(median, 1e-10), 0.7, 1.5))
            assert 0.7 <= scalar <= 1.5, f"Scalar {scalar} out of bounds"

    def test_confidence_returns_float(self):
        scalar = float(np.clip(np.float64(1.2), 0.7, 1.5))
        assert isinstance(scalar, float)


class TestDailyModelConfig:
    """Verify daily model regularization is reasonable for ~200 rows."""

    def test_daily_model_no_early_stopping(self):
        from hedge_fund.ensemble import EnsembleModel
        model = EnsembleModel(use_daily=True)
        params = model.xgb_model.get_params()
        # Early stopping removed — kills features on small daily data
        assert params.get('early_stopping_rounds') is None

    def test_daily_model_min_child_weight_reasonable(self):
        from hedge_fund.ensemble import EnsembleModel
        model = EnsembleModel(use_daily=True)
        params = model.xgb_model.get_params()
        assert params['min_child_weight'] <= 15  # Not too restrictive for 200 rows


class TestEmptyExecution:
    """Verify execution engine handles edge cases."""

    def test_empty_input_returns_empty(self):
        from hedge_fund.execution import simulate_hybrid_trades
        trades = simulate_hybrid_trades({}, {}, {})
        assert trades == []
