"""
Technical indicator calculations with no external TA library dependencies.

Provides ManualTA with RSI, ATR, Bollinger Bands, and ADX implementations
that match the pandas_ta interface used by the bot and backtester.
"""

import numpy as np
import pandas as pd


class ManualTA:
    """Manual technical analysis indicators (no pandas_ta dependency required)."""

    @staticmethod
    def rsi(close, length=14):
        """
        Relative Strength Index.

        Args:
            close: Series of closing prices.
            length: RSI lookback period.

        Returns:
            Series of RSI values (0-100).
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / length, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / length, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high, low, close, length=14):
        """
        Average True Range.

        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of closing prices.
            length: ATR smoothing period.

        Returns:
            Series of ATR values.
        """
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / length, adjust=False).mean()

    @staticmethod
    def bbands(close, length=20, std=2):
        """
        Bollinger Bands.

        Args:
            close: Series of closing prices.
            length: Moving average period.
            std: Number of standard deviations.

        Returns:
            DataFrame with columns BBL_{length}_{std}, BBM_{length}_{std}, BBU_{length}_{std}.
        """
        ma = close.rolling(length).mean()
        std_dev = close.rolling(length).std()
        upper = ma + std * std_dev
        lower = ma - std * std_dev
        return pd.DataFrame(
            {
                f"BBL_{length}_{float(std)}": lower,
                f"BBM_{length}_{float(std)}": ma,
                f"BBU_{length}_{float(std)}": upper,
            },
            index=close.index,
        )

    @staticmethod
    def adx(high, low, close, length=14):
        """
        Average Directional Index.

        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of closing prices.
            length: ADX smoothing period.

        Returns:
            DataFrame with column ADX_{length}.
        """
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up = high - high.shift()
        down = low.shift() - low
        pos_dm = ((up > down) & (up > 0)) * up
        neg_dm = ((down > up) & (down > 0)) * down

        tr_s = tr.ewm(alpha=1 / length, adjust=False).mean()
        pos_dm_s = pos_dm.ewm(alpha=1 / length, adjust=False).mean()
        neg_dm_s = neg_dm.ewm(alpha=1 / length, adjust=False).mean()

        plus_di = 100 * (pos_dm_s / tr_s)
        minus_di = 100 * (neg_dm_s / tr_s)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_val = dx.ewm(alpha=1 / length, adjust=False).mean()

        return pd.DataFrame({f"ADX_{length}": adx_val}, index=close.index)
