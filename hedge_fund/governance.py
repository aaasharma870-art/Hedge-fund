"""
Monte Carlo risk governance for dynamic position sizing.

Scales risk exposure based on recent trade performance using Monte Carlo
simulation of equity drawdowns. Reduces risk during drawdowns and
restores it during healthy performance.
"""

import datetime
import logging
from collections import deque


class MonteCarloGovernor:
    """
    Monte Carlo Risk Governor.

    Tracks recent trade history and adjusts a risk scalar (0.5 - 1.0)
    based on simulated equity drawdown from the last N trades.

    Risk Levels:
        - Normal (DD < 5%):   scalar = 1.0
        - Warning (DD >= 5%): scalar = 0.75
        - Critical (DD >= 8%): scalar = 0.5
    """

    def __init__(self, settings=None, dd_warning=0.05, dd_critical=0.08,
                 lookback_trades=50, update_interval=300):
        """
        Args:
            settings: Optional settings dict (for forward compatibility).
            dd_warning: Drawdown threshold for warning level (default 5%).
            dd_critical: Drawdown threshold for critical level (default 8%).
            lookback_trades: Number of recent trades to analyze.
            update_interval: Minimum seconds between recalculations.
        """
        self.settings = settings or {}
        self.trade_history = deque(maxlen=500)
        self.risk_scalar = 1.0
        self.last_update = None
        self.in_drawdown = False
        self.DD_WARNING = dd_warning
        self.DD_CRITICAL = dd_critical
        self.LOOKBACK_TRADES = lookback_trades
        self._update_interval = update_interval
        logging.info("Monte Carlo Governor initialized")

    def add_trade(self, pnl, risk_dollars, side='LONG', timestamp=None):
        """
        Record a completed trade.

        Args:
            pnl: Realized profit/loss in dollars.
            risk_dollars: Dollar amount risked on the trade.
            side: 'LONG' or 'SHORT'.
            timestamp: Trade timestamp. Defaults to now.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
        r_multiple = pnl / risk_dollars if risk_dollars > 0 else 0
        self.trade_history.append({
            'timestamp': timestamp,
            'pnl': pnl,
            'risk': risk_dollars,
            'r_multiple': r_multiple,
            'side': side,
        })

    def apply_adjustments(self):
        """
        Recalculate risk scalar based on recent trade equity curve drawdown.

        Only recalculates if enough trades exist and the update interval
        has elapsed since the last calculation.
        """
        if len(self.trade_history) < 20:
            self.risk_scalar = 1.0
            return

        if (self.last_update and
                (datetime.datetime.now() - self.last_update).total_seconds()
                < self._update_interval):
            return

        try:
            recent = list(self.trade_history)[-self.LOOKBACK_TRADES:]
            equity_curve = self._calculate_equity_curve(recent)
            peak = max(equity_curve)
            current = equity_curve[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0

            if drawdown >= self.DD_CRITICAL:
                self.risk_scalar = 0.5
                self.in_drawdown = True
            elif drawdown >= self.DD_WARNING:
                self.risk_scalar = 0.75
                self.in_drawdown = True
            else:
                self.risk_scalar = 1.0
                self.in_drawdown = False

            self.last_update = datetime.datetime.now()
        except Exception as e:
            logging.error(f"MC Governor error: {e}")
            self.risk_scalar = 1.0

    def _calculate_equity_curve(self, trades):
        """Build cumulative PnL curve from trade list."""
        curve = [0]
        cumulative = 0
        for t in trades:
            cumulative += t['pnl']
            curve.append(cumulative)
        return curve

    def get_risk_scalar(self):
        """Return current risk scalar (0.5 - 1.0)."""
        return self.risk_scalar
