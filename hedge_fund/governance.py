"""
Monte Carlo risk governance for dynamic position sizing.

Scales risk exposure based on recent trade performance using Monte Carlo
simulation of equity drawdowns. Reduces risk during drawdowns and
restores it during healthy performance.
"""

import datetime
import json
import logging
import os
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
                 lookback_trades=50, update_interval=300, state_dir='data'):
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
        self._state_file = os.path.join(state_dir, 'governor_state.json')
        self._load_state()
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
        self._save_state()

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

    def _save_state(self):
        """Persist trade history and risk scalar to disk for crash recovery."""
        try:
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            state = {
                'recent_trades': [
                    {'pnl': t['pnl'], 'risk': t['risk'], 'r_multiple': t['r_multiple'],
                     'side': t.get('side', 'LONG'),
                     'timestamp': t['timestamp'].isoformat() if hasattr(t['timestamp'], 'isoformat') else str(t['timestamp'])}
                    for t in list(self.trade_history)[-self.LOOKBACK_TRADES:]
                ],
                'risk_scalar': self.risk_scalar,
                'in_drawdown': self.in_drawdown,
            }
            with open(self._state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logging.warning(f"Failed to save Governor state: {e}")

    def _load_state(self):
        """Restore trade history from disk on startup."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file) as f:
                state = json.load(f)
            for t in state.get('recent_trades', []):
                self.trade_history.append({
                    'timestamp': datetime.datetime.fromisoformat(t['timestamp']) if isinstance(t.get('timestamp'), str) else datetime.datetime.now(),
                    'pnl': t.get('pnl', 0),
                    'risk': t.get('risk', 1),
                    'r_multiple': t.get('r_multiple', 0),
                    'side': t.get('side', 'LONG'),
                })
            self.risk_scalar = state.get('risk_scalar', 1.0)
            self.in_drawdown = state.get('in_drawdown', False)
            logging.info(f"Governor state restored: {len(self.trade_history)} trades, scalar={self.risk_scalar}")
        except Exception as e:
            logging.warning(f"Failed to load Governor state: {e}")

    def get_risk_scalar(self):
        """Return current risk scalar (0.5 - 1.0)."""
        return self.risk_scalar


class DailyRiskManager:
    """Daily P&L tracking with loss halt and profit lock-in."""

    def __init__(self, daily_loss_limit=0.02, daily_profit_lock_pct=0.03):
        self.daily_loss_limit = daily_loss_limit
        self.daily_profit_lock_pct = daily_profit_lock_pct
        self._daily_pnl = 0.0
        self._daily_start_equity = 0.0
        self._current_date = None

    def reset_if_new_day(self, current_date, equity):
        """Reset daily P&L tracker at start of new trading day."""
        if current_date != self._current_date:
            self._current_date = current_date
            self._daily_pnl = 0.0
            self._daily_start_equity = equity

    def record_pnl(self, pnl):
        """Record a trade's P&L."""
        self._daily_pnl += pnl

    def get_daily_size_scalar(self):
        """Returns size scalar based on current day's P&L."""
        if self._daily_start_equity <= 0:
            return 1.0

        daily_pnl_pct = self._daily_pnl / self._daily_start_equity

        if daily_pnl_pct <= -self.daily_loss_limit:
            return 0.0  # full stop

        if daily_pnl_pct >= self.daily_profit_lock_pct:
            return 0.5  # reduce to protect gains

        return 1.0

    @property
    def is_halted(self):
        if self._daily_start_equity <= 0:
            return False
        return (self._daily_pnl / self._daily_start_equity) <= -self.daily_loss_limit
