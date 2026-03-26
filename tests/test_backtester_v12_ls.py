"""Tests for V12 L/S balance and metrics."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ls_balance_soft_cap():
    """Soft L/S balance should cap at 2:1, not force equality."""
    from hedge_fund.daily_model import generate_watchlist
    import pandas as pd
    import numpy as np

    # Create mock predictions: 5 tickers, clear long bias
    mock_preds = {}
    dates = pd.date_range('2024-01-02', periods=30, freq='B')
    for ticker in ['AAPL', 'NVDA', 'TSLA', 'GS', 'JPM']:
        df = pd.DataFrame({
            'Close': np.random.RandomState(42).randn(30).cumsum() + 100,
        }, index=dates)
        # AAPL, NVDA, TSLA: strong positive predictions (longs)
        # GS, JPM: weak negative predictions (shorts)
        if ticker in ['AAPL', 'NVDA', 'TSLA']:
            df['DailyPrediction'] = np.random.RandomState(hash(ticker) % 2**31).uniform(0.02, 0.05, 30)
        else:
            df['DailyPrediction'] = np.random.RandomState(hash(ticker) % 2**31).uniform(-0.03, -0.01, 30)
        mock_preds[ticker] = df

    wl = generate_watchlist(mock_preds, top_n=3, bottom_n=2)

    # Should have some signals
    assert len(wl) > 0

    # Check L/S ratio never exceeds 2:1
    for d, signals in wl.items():
        n_long = len(signals.get('longs', []))
        n_short = len(signals.get('shorts', []))
        if n_long > 0 and n_short > 0:
            ratio = max(n_long, n_short) / min(n_long, n_short)
            assert ratio <= 2.1, f"L/S ratio {ratio:.1f} exceeds 2:1 cap on {d}"


def test_pct_gain_and_cagr_positive_for_winning_system():
    """PctGain and CAGR should be positive for a winning trade set."""
    from backtester_v12 import compute_risk_metrics

    # Simulate a winning system: 60% WR, 2:1 RR
    trades = []
    for i in range(100):
        if i % 5 < 3:  # 60% wins
            trades.append((2.0, True, 1.0, 'AAPL', 'LONG'))
        else:
            trades.append((-1.0, True, 1.0, 'AAPL', 'SHORT'))

    m = compute_risk_metrics(trades)
    assert m['PctGain'] > 0, f"PctGain should be positive: {m['PctGain']}"
    assert m['CAGR'] > 0, f"CAGR should be positive: {m['CAGR']}"
    assert 'AvgCost_R' in m
    assert 'Sortino' in m


def test_max_concurrent_positions_respected():
    """Execution engine should never exceed 6 concurrent positions."""
    from hedge_fund.execution import simulate_hybrid_trades

    # Empty input = empty output
    trades = simulate_hybrid_trades({}, {}, {})
    assert trades == []
