"""
Portfolio optimization using Mean-Variance Optimization with beta constraints.

Provides institutional-grade portfolio weight allocation that maximizes
Sharpe Ratio while constraining market beta exposure.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Institutional-Grade Mean-Variance Optimizer.

    Maximizes: UTILITY = Expected_Return - (Risk_Aversion * Portfolio_Variance)

    Aggressive but safe:
    - Maximizes Sharpe Ratio (Return / Risk).
    - Penalizes correlation risk (Variance).
    - Allows high volatility if the expected return justifies it.
    """

    def __init__(self, risk_free_rate=0.04, target_vol=0.25):
        self.risk_free_rate = risk_free_rate
        # Target Volatility (Annualized). 0.25 = 25% (Aggressive).
        # SPY is usually ~15%. 25% allows for hedge-fund level risk.
        self.target_vol = target_vol

    def get_optimal_weights(self, candidates, lookback_prices_df, market_ticker='SPY'):
        """
        Compute optimal portfolio weights via constrained Sharpe maximization.

        Args:
            candidates: List of dicts [{'symbol': 'NVDA', 'ev': 2.5}, ...]
                Each dict must have 'symbol' and 'ev' (expected value) keys.
            lookback_prices_df: DataFrame of historical CLOSE prices
                (index=datetime, columns=symbols). Should include market_ticker.
            market_ticker: Ticker for beta constraint calculation (default 'SPY').

        Returns:
            Dict mapping symbol -> weight (floats summing to 1.0).
            Returns equal weights on failure or insufficient data.
        """
        if not candidates or lookback_prices_df.empty:
            return {}

        symbols = [c['symbol'] for c in candidates]

        # 1. Expected Returns (Alpha Vector)
        alpha_vector = np.array([c.get('ev', 0.0) for c in candidates])

        # 2. Daily Returns & Covariance
        returns_df = lookback_prices_df.pct_change().dropna()

        if len(returns_df) < 20:
            logging.warning("Not enough history. Using equal weights.")
            return {s: 1.0 / len(symbols) for s in symbols}

        # Extract subset for candidates
        try:
            cand_returns = returns_df[symbols]
            # Ledoit-Wolf shrinkage for stable covariance with limited data
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(cand_returns.dropna())
                cov_matrix = pd.DataFrame(
                    lw.covariance_ * 252, index=symbols, columns=symbols
                )
                logging.debug(f"LW shrinkage coef: {lw.shrinkage_:.3f}")
            except Exception:
                cov_matrix = cand_returns.cov() * 252
        except KeyError as e:
            logging.error(f"Missing data for symbols: {e}")
            return {s: 1.0 / len(symbols) for s in symbols}

        # 3. Calculate Betas (Sensitivity to Market)
        betas = np.ones(len(symbols))
        if market_ticker in returns_df.columns:
            market_ret = returns_df[market_ticker]
            market_var = market_ret.var()
            if market_var > 1e-6:
                for i, sym in enumerate(symbols):
                    cov_sm = cand_returns[sym].cov(market_ret)
                    betas[i] = cov_sm / market_var

        # 4. Optimization
        num_assets = len(symbols)
        initial_weights = np.ones(num_assets) / num_assets
        bounds = tuple((0.0, 0.40) for _ in range(num_assets))

        # Constraints
        # A. Fully Invested
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

        # B. Beta Constraint: Portfolio Beta <= 0.90
        def beta_constraint(x):
            port_beta = np.sum(x * betas)
            return 0.90 - port_beta

        cons.append({'type': 'ineq', 'fun': beta_constraint})

        # Objective: Maximize Sharpe
        def negative_sharpe(weights):
            port_return = np.sum(weights * alpha_vector)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = port_return / (port_vol + 1e-6)
            return -sharpe

        try:
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=cons
            )

            if not result.success:
                # Fallback to unconstrained Sharpe optimization
                logging.debug("Beta constraint infeasible, retrying without beta cap...")
                result = minimize(
                    negative_sharpe,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[cons[0]]  # Only sum=1
                )

            optimal_weights = result.x
            optimal_weights[optimal_weights < 0.01] = 0.0
            if np.sum(optimal_weights) > 0:
                optimal_weights /= np.sum(optimal_weights)

            final_beta = np.sum(optimal_weights * betas)
            logging.info(f"Quant Portfolio Optimized: Est. Beta = {final_beta:.2f}")

            return {sym: w for sym, w in zip(symbols, optimal_weights)}

        except Exception as e:
            logging.error(f"Optimizer breakdown: {e}")
            return {s: 1.0 / len(symbols) for s in symbols}

    def calculate_allocation(self, total_equity, optimal_weights, max_leverage=1.5):
        """
        Convert weights to dollar allocations with optional leverage.

        Args:
            total_equity: Total account equity in dollars.
            optimal_weights: Dict of symbol -> weight from get_optimal_weights.
            max_leverage: Maximum leverage multiplier (1.5 = 150% long).

        Returns:
            Dict mapping symbol -> dollar allocation.
        """
        allocations = {}
        for sym, w in optimal_weights.items():
            if w > 0:
                allocations[sym] = total_equity * w * max_leverage
        return allocations
