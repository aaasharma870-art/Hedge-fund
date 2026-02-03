"""
Risk management utilities for position sizing and trade risk.

Provides risk-based position sizing, Kelly criterion calculations,
and portfolio heat management.
"""

import numpy as np


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
