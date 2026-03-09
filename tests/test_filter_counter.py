"""Tests for TradeFilterCounter."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_filter_counter_basics():
    from backtester import TradeFilterCounter
    counter = TradeFilterCounter()
    counter.increment('raw_bars_evaluated', 1000)
    counter.increment('passed_pred_threshold', 300)
    counter.increment('killed_by_adx', 50)
    counter.increment('submitted_to_simulation', 220)
    counter.increment('executed_tp', 100)
    counter.increment('executed_sl', 80)
    counter.increment('executed_timeout', 40)

    assert counter.counts['raw_bars_evaluated'] == 1000
    assert counter.counts['passed_pred_threshold'] == 300
    # Should not crash
    counter.report(trial_num=99)


def test_filter_counter_zero_trades():
    from backtester import TradeFilterCounter
    counter = TradeFilterCounter()
    counter.increment('raw_bars_evaluated', 1000)
    # No executed trades
    counter.report()  # Should print warning but not crash
