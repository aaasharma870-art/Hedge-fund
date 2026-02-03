"""
hedge_fund - Quantitative algorithmic trading system.

Provides shared modules for indicators, math utilities, simulation,
feature engineering, risk management, and data helpers used by both
the backtester and the live trading bot.
"""

from hedge_fund.indicators import ManualTA
from hedge_fund.math_utils import get_kalman_filter, get_hurst
from hedge_fund.simulation import simulate_exit, compute_bracket_labels
from hedge_fund.features import (
    calculate_vpin,
    calculate_enhanced_vwap_features,
    calculate_volatility_regime,
    calculate_amihud_illiquidity,
    calculate_real_relative_strength,
    calculate_liquidity_sweep,
)
from hedge_fund.risk import calculate_position_size, kelly_criterion
from hedge_fund.data import RateLimiter
from hedge_fund.objectives import profit_factor_objective, asymmetric_loss_objective

__all__ = [
    "ManualTA",
    "get_kalman_filter",
    "get_hurst",
    "simulate_exit",
    "compute_bracket_labels",
    "calculate_vpin",
    "calculate_enhanced_vwap_features",
    "calculate_volatility_regime",
    "calculate_amihud_illiquidity",
    "calculate_real_relative_strength",
    "calculate_liquidity_sweep",
    "calculate_position_size",
    "kelly_criterion",
    "RateLimiter",
    "profit_factor_objective",
    "asymmetric_loss_objective",
]
