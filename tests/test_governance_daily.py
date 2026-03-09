"""Tests for DailyRiskManager."""

from hedge_fund.governance import DailyRiskManager
import datetime


def test_daily_risk_normal():
    mgr = DailyRiskManager(daily_loss_limit=0.02, daily_profit_lock_pct=0.03)
    mgr.reset_if_new_day(datetime.date.today(), 100000)
    assert mgr.get_daily_size_scalar() == 1.0
    assert not mgr.is_halted


def test_daily_risk_loss_halt():
    mgr = DailyRiskManager(daily_loss_limit=0.02)
    mgr.reset_if_new_day(datetime.date.today(), 100000)
    mgr.record_pnl(-2100)  # > 2% loss
    assert mgr.get_daily_size_scalar() == 0.0
    assert mgr.is_halted


def test_daily_risk_profit_lock():
    mgr = DailyRiskManager(daily_profit_lock_pct=0.03)
    mgr.reset_if_new_day(datetime.date.today(), 100000)
    mgr.record_pnl(3500)  # > 3% gain
    assert mgr.get_daily_size_scalar() == 0.5


def test_daily_risk_resets_on_new_day():
    mgr = DailyRiskManager(daily_loss_limit=0.02)
    mgr.reset_if_new_day(datetime.date(2024, 1, 1), 100000)
    mgr.record_pnl(-2500)
    assert mgr.is_halted
    # New day
    mgr.reset_if_new_day(datetime.date(2024, 1, 2), 97500)
    assert not mgr.is_halted
    assert mgr.get_daily_size_scalar() == 1.0
