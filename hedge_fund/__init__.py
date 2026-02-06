"""
hedge_fund - Quantitative algorithmic trading system.

Provides shared modules for indicators, math utilities, simulation,
feature engineering, risk management, portfolio optimization,
governance, analysis, and data helpers used by both the backtester
and the live trading bot.
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
    CrossSectionalRanker,
)
from hedge_fund.risk import (
    calculate_position_size,
    kelly_criterion,
    OvernightGapModel,
    SlippageCalculator,
)
from hedge_fund.data import RateLimiter
from hedge_fund.objectives import profit_factor_objective, asymmetric_loss_objective
from hedge_fund.optimization import PortfolioOptimizer
from hedge_fund.governance import MonteCarloGovernor
from hedge_fund.analysis import run_attribution_analysis
from hedge_fund.config import save_optimal_params, load_optimal_params, apply_to_settings
from hedge_fund.dashboard import Dashboard
from hedge_fund.scanner import CandidateScanner
from hedge_fund.ensemble import EnsembleModel

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
    "CrossSectionalRanker",
    "calculate_position_size",
    "kelly_criterion",
    "OvernightGapModel",
    "SlippageCalculator",
    "RateLimiter",
    "profit_factor_objective",
    "asymmetric_loss_objective",
    "PortfolioOptimizer",
    "MonteCarloGovernor",
    "run_attribution_analysis",
    "save_optimal_params",
    "load_optimal_params",
    "apply_to_settings",
    "Dashboard",
    "CandidateScanner",
    "EnsembleModel",
]
