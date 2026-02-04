"""
Risk management utilities for position sizing and trade risk.

Provides risk-based position sizing, Kelly criterion calculations,
overnight gap risk management, slippage modeling, and portfolio heat management.
"""

import datetime
import logging
from zoneinfo import ZoneInfo

import numpy as np

_ET = ZoneInfo("America/New_York")


def calculate_position_size(equity, entry_price, stop_price, risk_pct=0.015,
                            max_pct_of_equity=0.20, vix_mult=1.0,
                            use_market_orders=False, slippage_haircut=0.001):
    """
    Risk-based position sizing: qty = (equity * risk_pct) / stop_distance.

    Args:
        equity: Current account equity in dollars.
        entry_price: Expected entry price.
        stop_price: Stop-loss price.
        risk_pct: Fraction of equity to risk per trade (default 1.5%).
        max_pct_of_equity: Maximum position size as fraction of equity.
        vix_mult: VIX-based risk multiplier (lower in high-VIX environments).
        use_market_orders: If True, apply slippage haircut.
        slippage_haircut: Fraction to reduce quantity by for market orders.

    Returns:
        Integer quantity of shares to trade. Returns 0 if inputs are invalid.
    """
    if equity <= 0 or entry_price <= 0:
        return 0

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0

    risk_per_trade = equity * risk_pct
    qty = int((risk_per_trade * vix_mult) / stop_distance)
    max_qty = int(equity * max_pct_of_equity / entry_price)
    qty = min(max(qty, 0), max_qty)

    if use_market_orders:
        qty = int(qty * (1 - slippage_haircut))

    return qty


def kelly_criterion(win_rate, avg_win_r, avg_loss_r, timeout_rate=0.0,
                    avg_timeout_r=0.0, shrinkage=0.35, confidence_boost=0.5,
                    min_risk=0.003, max_risk=0.03):
    """
    3-outcome Kelly criterion for optimal bet sizing.

    Computes the Kelly fraction adjusted for three possible outcomes:
    win, loss, and timeout (neither barrier hit).

    Args:
        win_rate: Probability of winning (0-1).
        avg_win_r: Average R-multiple on wins (positive).
        avg_loss_r: Average R-multiple on losses (positive, magnitude).
        timeout_rate: Probability of timeout (0-1).
        avg_timeout_r: Average R-multiple on timeouts (can be negative).
        shrinkage: Kelly shrinkage factor (0-1). Lower = more conservative.
        confidence_boost: Multiplier for confidence-adjusted Kelly.
        min_risk: Minimum risk fraction floor.
        max_risk: Maximum risk fraction ceiling.

    Returns:
        Float representing the optimal risk fraction of equity per trade.
    """
    if win_rate <= 0 or avg_win_r <= 0 or avg_loss_r <= 0:
        return min_risk

    loss_rate = 1.0 - win_rate - timeout_rate
    if loss_rate <= 0:
        loss_rate = 0.01

    # Edge = E[R] = win_rate * avg_win - loss_rate * avg_loss + timeout_rate * avg_timeout
    edge = (win_rate * avg_win_r) - (loss_rate * avg_loss_r) + (timeout_rate * avg_timeout_r)

    if edge <= 0:
        return min_risk

    # Kelly fraction = edge / avg_win_r (simplified)
    kelly_f = edge / avg_win_r

    # Apply shrinkage and confidence adjustment
    adjusted = kelly_f * shrinkage * confidence_boost

    return float(np.clip(adjusted, min_risk, max_risk))


class OvernightGapModel:
    """
    Overnight gap risk management.

    Before market close, assesses gap risk for each position and either:
    - Exits positions with high gap risk (earnings, high VIX, low PnL)
    - Tightens stops on remaining positions to reduce overnight exposure
    """

    PRE_CLOSE_MINUTES = 15  # Start managing 15 min before close (15:45 ET)

    def __init__(self, earnings_guard=None):
        """
        Args:
            earnings_guard: Optional object with a ``check_safe(ticker)`` method
                that returns True if the ticker has no upcoming earnings risk.
        """
        self.earnings = earnings_guard
        self._gap_stats = {}  # ticker -> (avg_gap_pct, gap_std)

    def is_pre_close(self):
        """True if within PRE_CLOSE_MINUTES of market close (16:00 ET)."""
        try:
            now = datetime.datetime.now(_ET)
        except Exception:
            now = datetime.datetime.now()
        h, m = now.hour, now.minute
        if now.weekday() >= 5:
            return False
        return (h == 15 and m >= (60 - self.PRE_CLOSE_MINUTES)) or h >= 16

    def gap_risk_score(self, ticker, vix, pnl_r, has_earnings_tomorrow=False):
        """
        Compute overnight gap risk score (0-1).

        Args:
            ticker: Stock ticker symbol.
            vix: Current VIX level.
            pnl_r: Current position PnL in R-multiples.
            has_earnings_tomorrow: Whether earnings are expected next day.

        Returns:
            Float 0-1. Higher = more risk. >= 0.55 suggests exit.
        """
        score = 0.0

        # VIX component
        if vix > 30:
            score += 0.35
        elif vix > 25:
            score += 0.20
        elif vix > 20:
            score += 0.10

        # Earnings component
        if has_earnings_tomorrow:
            score += 0.40

        # PnL component: low/negative PnL = worse risk/reward overnight
        if pnl_r < 0.3:
            score += 0.20
        elif pnl_r < 0.0:
            score += 0.30

        return min(1.0, score)

    def should_exit_pre_close(self, ticker, vix, pnl_r):
        """Returns True if position should be closed before market close."""
        has_earnings = False
        if self.earnings:
            try:
                has_earnings = not self.earnings.check_safe(ticker)
            except Exception:
                pass
        risk = self.gap_risk_score(ticker, vix, pnl_r, has_earnings)
        return risk >= 0.55

    def pre_close_stop_tightening(self, side, entry, current_sl, atr, pnl_r):
        """
        Tighten stop for positions held overnight.

        Args:
            side: 'LONG' or 'SHORT'.
            entry: Entry price.
            current_sl: Current stop-loss price.
            atr: Current ATR value.
            pnl_r: Current PnL in R-multiples.

        Returns:
            New stop-loss price, or None if no change needed.
        """
        if pnl_r < 0.5:
            return None  # Should be exited, not tightened

        # Move stop to lock in at least 0.25R profit overnight
        lock_level = 0.25
        if side == 'LONG':
            min_sl = entry + lock_level * abs(entry - current_sl)
            return max(current_sl, min_sl)
        else:
            min_sl = entry - lock_level * abs(entry - current_sl)
            return min(current_sl, min_sl)


class SlippageCalculator:
    """
    Realistic execution cost model for trade simulation.

    Models three components of slippage:
    - Bid-ask spread cost (fixed per side)
    - Market impact (proportional to order size relative to ADV)
    - Commission (per-share, typically zero for retail)
    """

    def __init__(self, spread_pct=0.03, impact_pct=0.02, commission_per_share=0.0):
        """
        Args:
            spread_pct: Bid-ask spread cost per side as percentage (default 0.03%).
            impact_pct: Market impact estimate per side as percentage (default 0.02%).
            commission_per_share: Per-share commission (default 0.0).
        """
        self.spread_pct = spread_pct / 100.0
        self.impact_pct = impact_pct / 100.0
        self.commission_per_share = commission_per_share

    def one_way_cost(self, price, qty=1):
        """
        Compute one-way (entry or exit) execution cost in dollars.

        Args:
            price: Execution price per share.
            qty: Number of shares.

        Returns:
            Total one-way cost in dollars.
        """
        spread_cost = price * self.spread_pct * qty
        impact_cost = price * self.impact_pct * qty
        commission = self.commission_per_share * qty
        return spread_cost + impact_cost + commission

    def round_trip_cost(self, price, qty=1):
        """
        Compute round-trip (entry + exit) execution cost in dollars.

        Args:
            price: Execution price per share.
            qty: Number of shares.

        Returns:
            Total round-trip cost in dollars.
        """
        return 2 * self.one_way_cost(price, qty)

    def round_trip_pct(self):
        """Return round-trip cost as a decimal fraction of price."""
        return 2 * (self.spread_pct + self.impact_pct)

    def cost_in_r(self, entry_price, sl_distance):
        """
        Express round-trip cost as R-multiples for bracket trade sizing.

        Args:
            entry_price: Trade entry price.
            sl_distance: Stop-loss distance in price units.

        Returns:
            Round-trip cost expressed in R-multiples.
        """
        if sl_distance <= 0:
            return 0.0
        cost_per_share = entry_price * self.round_trip_pct()
        return cost_per_share / sl_distance
