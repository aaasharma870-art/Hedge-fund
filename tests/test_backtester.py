"""Tests for backtester V5 functions: risk metrics, Monte Carlo, per-ticker breakdown."""

import numpy as np
import pytest
from unittest.mock import patch

# Import backtester functions directly
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtester import compute_risk_metrics, per_ticker_breakdown, monte_carlo_test


class TestComputeRiskMetrics:
    """Tests for the advanced risk metrics calculator."""

    def _make_trades(self, outcomes, resolved=None, tickers=None):
        """Helper to create trade tuples: (outcome, is_resolved, size, ticker)."""
        if resolved is None:
            resolved = [True] * len(outcomes)
        if tickers is None:
            tickers = ['TEST'] * len(outcomes)
        return [(o, r, 1.0, t) for o, r, t in zip(outcomes, resolved, tickers)]

    def test_basic_profit_factor(self):
        trades = self._make_trades([2.0, 2.0, -1.0, -1.0])
        m = compute_risk_metrics(trades)
        assert m['PF_Raw'] == pytest.approx(2.0)
        assert m['WR_Raw'] == pytest.approx(0.5)

    def test_perfect_win_rate(self):
        trades = self._make_trades([1.0, 1.5, 2.0])
        m = compute_risk_metrics(trades)
        assert m['WR_Raw'] == pytest.approx(1.0)
        assert m['Trades'] == 3

    def test_all_losses(self):
        trades = self._make_trades([-1.0, -1.0, -1.0])
        m = compute_risk_metrics(trades)
        assert m['WR_Raw'] == pytest.approx(0.0)
        assert m['PF_Raw'] == 0
        assert m['LongestLoss'] == 3

    def test_empty_trades(self):
        m = compute_risk_metrics([])
        assert m['Trades'] == 0
        assert m['PF_Res'] == 0

    def test_max_drawdown_calculated(self):
        # Win, win, lose, lose, lose -> drawdown should be negative
        trades = self._make_trades([2.0, 2.0, -1.0, -1.0, -1.0])
        m = compute_risk_metrics(trades)
        assert m['MaxDD_R'] < 0
        # Max drawdown = from peak of 4.0 to 1.0 = -3.0
        assert m['MaxDD_R'] == pytest.approx(-3.0)

    def test_no_drawdown_on_monotonic_wins(self):
        trades = self._make_trades([1.0, 1.0, 1.0, 1.0])
        m = compute_risk_metrics(trades)
        assert m['MaxDD_R'] == pytest.approx(0.0)

    def test_longest_losing_streak(self):
        trades = self._make_trades([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
        m = compute_risk_metrics(trades)
        assert m['LongestLoss'] == 3

    def test_sharpe_positive_for_winning_system(self):
        trades = self._make_trades([1.5, 1.5, 1.5, -0.5, 1.5])
        m = compute_risk_metrics(trades)
        assert m['Sharpe'] > 0

    def test_calmar_ratio(self):
        trades = self._make_trades([2.0, -1.0, 2.0])
        m = compute_risk_metrics(trades)
        # Total return = 3.0, max DD = -1.0, Calmar = 3.0
        assert m['Calmar'] == pytest.approx(3.0)

    def test_payoff_ratio(self):
        # 2 wins of 3.0R, 2 losses of 1.0R -> payoff = 3.0/1.0 = 3.0
        trades = self._make_trades([3.0, 3.0, -1.0, -1.0])
        m = compute_risk_metrics(trades)
        assert m['PayoffRatio'] == pytest.approx(3.0)

    def test_resolved_vs_raw_metrics(self):
        # Mix of resolved and unresolved trades
        trades = [
            (2.0, True, 1.0, 'A'),   # resolved win
            (-1.0, True, 1.0, 'A'),  # resolved loss
            (0.5, False, 1.0, 'A'),  # timeout (not resolved)
        ]
        m = compute_risk_metrics(trades)
        assert m['Trades'] == 3
        assert m['Resolved'] == 2

    def test_total_return(self):
        trades = self._make_trades([1.0, -0.5, 2.0, -0.5])
        m = compute_risk_metrics(trades)
        assert m['TotalReturn_R'] == pytest.approx(2.0)


class TestPerTickerBreakdown:
    """Tests for per-ticker results decomposition."""

    def test_separates_by_ticker(self):
        trades = [
            (2.0, True, 1.0, 'NVDA'),
            (-1.0, True, 1.0, 'NVDA'),
            (1.5, True, 1.0, 'TSLA'),
        ]
        breakdown = per_ticker_breakdown(trades)
        assert 'NVDA' in breakdown
        assert 'TSLA' in breakdown
        assert breakdown['NVDA']['Trades'] == 2
        assert breakdown['TSLA']['Trades'] == 1

    def test_single_ticker(self):
        trades = [
            (1.0, True, 1.0, 'AMD'),
            (1.0, True, 1.0, 'AMD'),
        ]
        breakdown = per_ticker_breakdown(trades)
        assert len(breakdown) == 1
        assert breakdown['AMD']['WR_Raw'] == pytest.approx(1.0)

    def test_empty_trades(self):
        breakdown = per_ticker_breakdown([])
        assert len(breakdown) == 0

    def test_metrics_per_ticker_independent(self):
        trades = [
            (3.0, True, 1.0, 'WINNER'),
            (3.0, True, 1.0, 'WINNER'),
            (-1.0, True, 1.0, 'LOSER'),
            (-1.0, True, 1.0, 'LOSER'),
        ]
        breakdown = per_ticker_breakdown(trades)
        assert breakdown['WINNER']['WR_Raw'] == pytest.approx(1.0)
        assert breakdown['LOSER']['WR_Raw'] == pytest.approx(0.0)


class TestMonteCarloTest:
    """Tests for Monte Carlo significance testing."""

    def test_strong_signal_low_pvalue(self):
        # System with very high PF should have low p-value
        # Uses magnitudes that are symmetric so sign-randomization creates
        # meaningful variation in PF. 50 wins@2R, 20 losses@2R -> PF=5.0
        trades = [(2.0, True, 1.0, 'A')] * 50 + [(-2.0, True, 1.0, 'A')] * 20
        observed_pf = sum(x for x, _, _, _ in trades if x > 0) / abs(sum(x for x, _, _, _ in trades if x < 0))
        p = monte_carlo_test(trades, observed_pf=observed_pf, n_simulations=500)
        assert p < 0.10

    def test_random_signal_high_pvalue(self):
        # System with PF near 1.0 should have high p-value
        trades = [(1.0, True, 1.0, 'A')] * 30 + [(-1.0, True, 1.0, 'A')] * 30
        p = monte_carlo_test(trades, observed_pf=1.0, n_simulations=200)
        assert p > 0.20

    def test_too_few_trades_returns_1(self):
        trades = [(1.0, True, 1.0, 'A')] * 5
        p = monte_carlo_test(trades, observed_pf=2.0)
        assert p == 1.0

    def test_returns_float_between_0_and_1(self):
        trades = [(1.5, True, 1.0, 'A')] * 30 + [(-1.0, True, 1.0, 'A')] * 20
        p = monte_carlo_test(trades, observed_pf=1.5, n_simulations=100)
        assert 0.0 <= p <= 1.0
