"""
Bracket trade simulation engine.

Provides bar-by-bar simulation of bracket orders (stop-loss / take-profit)
with optional trailing stops, and label generation for ML training.
"""

import numpy as np
import pandas as pd


def simulate_exit(highs, lows, sl, tp, side, trail_dist=None):
    """
    Simulate a bracket exit bar-by-bar with optional trailing stop.

    Checks stop-loss before take-profit on each bar (conservative assumption).

    Args:
        highs: Array-like of high prices for future bars.
        lows: Array-like of low prices for future bars.
        sl: Initial stop-loss price.
        tp: Take-profit price.
        side: 'LONG' or 'SHORT'.
        trail_dist: If set, trailing stop distance (e.g. 1.0 * ATR).
            For LONG, SL trails below the highest high by trail_dist.
            For SHORT, SL trails above the lowest low by trail_dist.

    Returns:
        Tuple of (outcome, exit_price):
            outcome: 'win', 'loss', 'trail_stop', or 'timeout'
            exit_price: Price at which the exit occurred, or None for timeout.
    """
    current_sl = sl

    for i in range(len(highs)):
        h = highs[i]
        l = lows[i]

        if side == "LONG":
            if l <= current_sl:
                status = "loss" if current_sl == sl else "trail_stop"
                return status, current_sl
            if h >= tp:
                return "win", tp
            if trail_dist:
                new_sl = h - trail_dist
                if new_sl > current_sl:
                    current_sl = new_sl

        else:  # SHORT
            if h >= current_sl:
                status = "loss" if current_sl == sl else "trail_stop"
                return status, current_sl
            if l <= tp:
                return "win", tp
            if trail_dist:
                new_sl = l + trail_dist
                if new_sl < current_sl:
                    current_sl = new_sl

    return "timeout", None


def compute_bracket_labels(df, sl_mult=1.5, tp_mult=3.0, max_bars=20, atr_col="ATR",
                           mode="regression"):
    """
    Generate ML training labels by simulating bracket trades at each bar.

    For each bar, simulates both a LONG and SHORT bracket trade and assigns:
      +R => LONG expected to profit with R-multiple R
      -R => SHORT expected to profit with R-multiple R
       0 => Neither direction has positive expected value (HOLD)

    Timeouts use mark-to-market at the horizon bar with a small decay penalty
    for opportunity cost.

    Args:
        df: DataFrame with columns 'Close', 'High', 'Low', and atr_col.
        sl_mult: Stop-loss distance as multiple of ATR.
        tp_mult: Take-profit distance as multiple of ATR.
        max_bars: Maximum bars before timeout (vertical barrier).
        atr_col: Name of the ATR column in df.
        mode: 'regression' returns continuous R-values (default).

    Returns:
        If mode='regression', returns Series of signed R-values.
        Also adds 'Target' column to df and returns the modified DataFrame
        for backward compatibility with the backtester.
    """
    n = len(df)
    labels = np.zeros(n, dtype=float)

    atr = df[atr_col].values
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    rr = tp_mult / sl_mult

    for i in range(n - max_bars - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        entry = close[i]
        risk = sl_mult * a
        if risk <= 0:
            continue

        # LONG bracket
        long_sl = entry - risk
        long_tp = entry + tp_mult * a
        long_out, _ = simulate_exit(
            high[i + 1 : i + max_bars + 1],
            low[i + 1 : i + max_bars + 1],
            long_sl,
            long_tp,
            "LONG",
        )
        if long_out == "win":
            long_r = rr
        elif long_out == "loss":
            long_r = -1.0
        else:
            mtm = (close[min(i + max_bars, n - 1)] - entry) / risk
            long_r = float(np.clip(mtm - 0.05, -1.0, rr))

        # SHORT bracket
        short_sl = entry + risk
        short_tp = entry - tp_mult * a
        short_out, _ = simulate_exit(
            high[i + 1 : i + max_bars + 1],
            low[i + 1 : i + max_bars + 1],
            short_sl,
            short_tp,
            "SHORT",
        )
        if short_out == "win":
            short_r = rr
        elif short_out == "loss":
            short_r = -1.0
        else:
            mtm = (entry - close[min(i + max_bars, n - 1)]) / risk
            short_r = float(np.clip(mtm - 0.05, -1.0, rr))

        # Choose best positive EV; otherwise HOLD = 0
        best = max(long_r, short_r)
        if best <= 0.0:
            labels[i] = 0.0
        elif long_r >= short_r:
            labels[i] = float(best)       # positive => go long
        else:
            labels[i] = float(-best)      # negative => go short

    if mode == "regression":
        return pd.Series(labels, index=df.index, name="Target")

    df["Target"] = labels
    return df
