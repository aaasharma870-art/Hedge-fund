"""
Mathematical utilities for signal processing and market regime detection.

Provides Kalman filter for trend estimation and Hurst exponent for
distinguishing trending vs mean-reverting market regimes.
"""

import numpy as np


def get_kalman_filter(series, q_base=0.01, r_base=0.1, vol_span=20):
    """
    Volatility-adaptive Kalman filter for price trend estimation.

    Dynamically adjusts process noise (Q) and measurement noise (R)
    based on a trimmed-mean baseline of exponentially-smoothed absolute returns.

    Args:
        series: 1-D numpy array of prices.
        q_base: Base process noise. Higher values make the filter more responsive.
        r_base: Base measurement noise. Higher values make the filter smoother.
        vol_span: EMA span for volatility estimation.

    Returns:
        numpy array of filtered (estimated) prices, same length as input.
    """
    n = len(series)
    if n == 0:
        return np.array([])

    abs_returns = np.abs(np.diff(series, prepend=series[0]))
    alpha_ema = 2.0 / (vol_span + 1)
    vol_ema = np.empty(n)
    vol_ema[0] = abs_returns[0] if abs_returns[0] > 0 else 1e-8
    for i in range(1, n):
        vol_ema[i] = alpha_ema * abs_returns[i] + (1 - alpha_ema) * vol_ema[i - 1]

    sorted_vol = np.sort(vol_ema)
    trim_lo = max(1, int(n * 0.10))
    trim_hi = max(trim_lo + 1, int(n * 0.90))
    vol_baseline = float(np.mean(sorted_vol[trim_lo:trim_hi]))
    if vol_baseline <= 0:
        vol_baseline = 1e-8

    x = series[0]
    p = 1.0
    estimates = np.empty(n)

    for i in range(n):
        ratio = max(0.1, min(10.0, vol_ema[i] / vol_baseline))
        q = max(0.001, min(0.1, q_base * ratio))
        r = max(0.01, min(1.0, r_base / ratio))
        z = series[i]
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        estimates[i] = x

    return estimates


def get_hurst(series):
    """
    Estimate the Hurst exponent using rescaled range analysis.

    The Hurst exponent characterizes the long-term memory of a time series:
      H < 0.5: Mean-reverting (anti-persistent)
      H = 0.5: Random walk
      H > 0.5: Trending (persistent)

    Args:
        series: 1-D array-like of prices (minimum 100 observations).

    Returns:
        Float in [0, 1]. Returns 0.5 (random walk) on insufficient data or errors.
    """
    try:
        if len(series) < 100:
            return 0.5
        lags = range(2, 20)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        if any(t <= 0 for t in tau):
            return 0.5
        slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5
