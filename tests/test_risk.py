"""Tests for hedge_fund.risk - position sizing and Kelly criterion."""

import pytest
from hedge_fund.risk import calculate_position_size, kelly_criterion


class TestPositionSize:
    def test_basic_sizing(self):
        # $100k equity, $100 entry, $95 stop = $5 risk distance
        # 1.5% of 100k = $1500 risk budget, qty = 1500 / 5 = 300
        # But max 20% of equity / price = $20k / $100 = 200 cap
        qty = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=95.0
        )
        assert qty == 200  # Capped by max_pct_of_equity (20%)

    def test_zero_stop_distance_returns_zero(self):
        qty = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=100.0
        )
        assert qty == 0

    def test_negative_equity_returns_zero(self):
        qty = calculate_position_size(
            equity=-10_000, entry_price=100.0, stop_price=95.0
        )
        assert qty == 0

    def test_zero_entry_returns_zero(self):
        qty = calculate_position_size(
            equity=100_000, entry_price=0.0, stop_price=95.0
        )
        assert qty == 0

    def test_max_position_cap(self):
        # Very tight stop would yield huge qty, should be capped at 20% of equity
        # $100k equity, $10 entry, $9.99 stop = $0.01 risk distance
        # Risk budget = $1500, qty = 1500/0.01 = 150000
        # But max 20% = $20k / $10 = 2000 shares
        qty = calculate_position_size(
            equity=100_000, entry_price=10.0, stop_price=9.99
        )
        assert qty <= 2000

    def test_vix_multiplier_reduces_size(self):
        qty_normal = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=95.0, vix_mult=1.0
        )
        qty_high_vix = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=95.0, vix_mult=0.5
        )
        assert qty_high_vix < qty_normal

    def test_slippage_haircut(self):
        qty_no_slip = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=95.0,
            use_market_orders=False
        )
        qty_with_slip = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=95.0,
            use_market_orders=True, slippage_haircut=0.10
        )
        assert qty_with_slip < qty_no_slip
        # 10% haircut should reduce by ~10%
        assert qty_with_slip == int(qty_no_slip * 0.9)

    def test_custom_risk_pct(self):
        # Use a wider stop so max_pct_of_equity doesn't cap both at 200
        qty_low = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=90.0, risk_pct=0.01
        )
        qty_high = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=90.0, risk_pct=0.03
        )
        assert qty_high > qty_low

    def test_short_position_sizing(self):
        # Stop above entry for shorts
        qty = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=105.0
        )
        assert qty > 0  # Should still work (abs of distance)

    def test_returns_integer(self):
        qty = calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=95.0
        )
        assert isinstance(qty, int)


class TestKellyCriterion:
    def test_basic_kelly(self):
        # 60% win rate, 2:1 avg win/loss
        result = kelly_criterion(
            win_rate=0.60, avg_win_r=2.0, avg_loss_r=1.0
        )
        assert result > 0
        assert result <= 0.03  # capped at max_risk

    def test_negative_edge_returns_minimum(self):
        # 30% win rate with 1:1 = negative edge
        result = kelly_criterion(
            win_rate=0.30, avg_win_r=1.0, avg_loss_r=1.0
        )
        assert result == 0.003  # min_risk

    def test_zero_win_rate_returns_minimum(self):
        result = kelly_criterion(
            win_rate=0.0, avg_win_r=2.0, avg_loss_r=1.0
        )
        assert result == 0.003

    def test_high_edge_capped_at_max(self):
        # Extreme edge should be capped
        result = kelly_criterion(
            win_rate=0.90, avg_win_r=5.0, avg_loss_r=0.5,
            max_risk=0.03
        )
        assert result <= 0.03

    def test_shrinkage_reduces_output(self):
        result_full = kelly_criterion(
            win_rate=0.60, avg_win_r=2.0, avg_loss_r=1.0, shrinkage=1.0
        )
        result_shrunk = kelly_criterion(
            win_rate=0.60, avg_win_r=2.0, avg_loss_r=1.0, shrinkage=0.35
        )
        assert result_shrunk <= result_full

    def test_timeout_rate_affects_result(self):
        result_no_timeout = kelly_criterion(
            win_rate=0.60, avg_win_r=2.0, avg_loss_r=1.0,
            timeout_rate=0.0
        )
        # Adding timeout with negative R reduces edge
        result_with_timeout = kelly_criterion(
            win_rate=0.50, avg_win_r=2.0, avg_loss_r=1.0,
            timeout_rate=0.10, avg_timeout_r=-0.2
        )
        # Timeouts with negative R should reduce or maintain sizing
        assert result_with_timeout <= result_no_timeout or result_with_timeout == 0.003

    def test_returns_float(self):
        result = kelly_criterion(
            win_rate=0.60, avg_win_r=2.0, avg_loss_r=1.0
        )
        assert isinstance(result, float)
