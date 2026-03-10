"""
hedge_fund/portfolio.py

Portfolio-level book management for long/short equity system.
Controls gross exposure, net exposure, and ticker-regime parameter adjustment.
"""


TICKER_REGIME_CLASS = {
    'RKLB': 'momentum', 'ASTS': 'momentum', 'AMD': 'momentum',
    'NVDA': 'momentum', 'PLTR': 'momentum', 'COIN': 'momentum',
    'GS': 'value', 'GE': 'value', 'COST': 'value',
    'JPM': 'value', 'UNH': 'value', 'CAT': 'value',
    'XOM': 'value', 'JNJ': 'value', 'PG': 'value',
}

TICKER_BETA = {
    'RKLB': 1.8, 'ASTS': 2.1, 'AMD': 1.6,
    'NVDA': 1.7, 'PLTR': 1.9, 'COIN': 2.3,
    'GS': 1.3, 'GE': 1.1, 'COST': 0.7,
    'JPM': 1.2, 'UNH': 0.8, 'CAT': 1.1,
    'XOM': 0.9, 'JNJ': 0.6, 'PG': 0.5,
}


class PortfolioManager:
    """
    Manages the long/short book at portfolio level.
    Instantiated once per simulation call or live session.
    """

    def __init__(self, max_gross: float = 1.20, max_net: float = 0.40,
                 max_single: float = 0.25):
        self.max_gross = max_gross
        self.max_net = max_net
        self.max_single = max_single
        self.positions = {}  # {ticker: {'direction': str, 'size': float}}

    def add(self, ticker: str, direction: str, size: float):
        self.positions[ticker] = {'direction': direction, 'size': size}

    def remove(self, ticker: str):
        self.positions.pop(ticker, None)

    def has_position(self, ticker: str) -> bool:
        return ticker in self.positions

    def get_exposure(self) -> dict:
        long_e = sum(p['size'] for p in self.positions.values() if p['direction'] == 'long')
        short_e = sum(p['size'] for p in self.positions.values() if p['direction'] == 'short')
        return {'gross': long_e + short_e, 'net': long_e - short_e,
                'long': long_e, 'short': short_e}

    def allowable_size(self, ticker: str, direction: str, requested: float) -> float:
        """
        Returns the actual allowed position size after all portfolio constraints.
        Returns 0.0 if the trade must be blocked.
        """
        # Block flips (same ticker, opposite direction already open)
        if ticker in self.positions and self.positions[ticker]['direction'] != direction:
            return 0.0

        exp = self.get_exposure()
        size = min(requested, self.max_single)

        # Gross limit
        if exp['gross'] + size > self.max_gross:
            size = max(0.0, self.max_gross - exp['gross'])

        # Net limit
        if direction == 'long':
            if exp['net'] + size > self.max_net:
                size = max(0.0, self.max_net - exp['net'])
        else:
            if exp['net'] - size < -self.max_net:
                size = max(0.0, exp['net'] + self.max_net)

        return size

    def regime_params(self, ticker: str, base_sl: float, base_tp_rr: float,
                      direction: str, market_regime: int) -> tuple:
        """
        Returns (sl_mult, tp_rr, max_bars_scalar) adjusted for ticker class and regime.

        Momentum tickers (RKLB, ASTS, AMD): wider stops, higher targets, more patience
        Value tickers (GS, GE, COST): tighter stops, lower targets, quicker exits
        Short trades universally: tighter stops
        Volatile market regime: reduce everything
        """
        sl, tp, mb = base_sl, base_tp_rr, 1.0
        cls = TICKER_REGIME_CLASS.get(ticker, 'neutral')

        if cls == 'momentum':
            sl *= 1.25
            tp *= 1.15
            mb = 1.20
        elif cls == 'value':
            sl *= 0.85
            tp *= 0.90
            mb = 0.85

        if direction == 'short':
            sl *= 0.80   # tighter stop on shorts always
            tp *= 0.85
            mb *= 0.75

        if market_regime == 2:  # volatile
            sl *= 0.75
            mb *= 0.70

        return sl, tp, mb
