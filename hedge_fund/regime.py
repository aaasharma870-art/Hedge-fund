"""
hedge_fund/regime.py

Real-time regime state machine for live trading.
Classifies market regime per bar and adjusts signal weights, position sizing,
and stop parameters accordingly.
"""

import numpy as np

REGIME_TRENDING = 0
REGIME_MEAN_REVERTING = 1
REGIME_VOLATILE = 2


class RegimeManager:
    """
    Tracks current market regime and returns regime-conditional parameters.
    Instantiated once per simulation call or live session.
    """

    def __init__(self):
        self.current_regime = REGIME_MEAN_REVERTING
        self.regime_counts = {0: 0, 1: 0, 2: 0}

    def update(self, regime_trending: float, regime_meanrev: float,
               regime_volatile: float) -> int:
        if regime_volatile > 0.5:
            self.current_regime = REGIME_VOLATILE
        elif regime_trending > 0.5:
            self.current_regime = REGIME_TRENDING
        else:
            self.current_regime = REGIME_MEAN_REVERTING
        self.regime_counts[self.current_regime] += 1
        return self.current_regime

    def get_signal_weights(self) -> dict:
        """
        Signal combination weights by regime.
        In trending regimes: weight momentum and order flow.
        In MR regimes: weight the mean reversion score.
        In volatile regimes: weight model prediction most (most stable signal).
        """
        if self.current_regime == REGIME_TRENDING:
            return {'model': 0.45, 'cofi': 0.30, 'cs_rank': 0.20, 'mr': 0.05}
        elif self.current_regime == REGIME_MEAN_REVERTING:
            return {'model': 0.40, 'cofi': 0.20, 'cs_rank': 0.05, 'mr': 0.35}
        else:  # VOLATILE
            return {'model': 0.65, 'cofi': 0.25, 'cs_rank': 0.05, 'mr': 0.05}

    def get_size_scalar(self) -> float:
        if self.current_regime == REGIME_VOLATILE:
            return 0.40
        elif self.current_regime == REGIME_TRENDING:
            return 1.00
        else:
            return 0.80

    def allow_short(self) -> bool:
        return self.current_regime != REGIME_VOLATILE

    def get_stop_scalar(self, direction: str) -> float:
        if self.current_regime == REGIME_VOLATILE:
            return 1.30 if direction == 'long' else 0.65
        elif self.current_regime == REGIME_TRENDING:
            return 1.05 if direction == 'long' else 0.85
        else:
            return 0.90
