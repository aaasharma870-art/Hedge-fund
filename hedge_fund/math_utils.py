"""
Mathematical utilities for signal processing and market regime detection.

Provides Kalman filter for trend estimation and Hurst exponent for
distinguishing trending vs mean-reverting market regimes.
"""

import numpy as np


def get_kalman_filter(series, q_base=0.01, r_base=0.1, vol_span=20, return_velocity=False):
    """
    Volatility-adaptive Kalman filter for price trend estimation.

    Uses a constant-velocity state model: [price_level, price_velocity].

    Args:
        series: 1-D numpy array of prices.
        q_base: Base process noise.
        r_base: Base measurement noise.
        vol_span: EMA span for volatility estimation.
        return_velocity: If True, returns (estimates, velocities) tuple.

    Returns:
        numpy array of filtered prices (or tuple of (prices, velocities) if return_velocity=True).
    """
    n = len(series)
    if n == 0:
        if return_velocity:
            return np.array([]), np.array([])
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

    # State: [price_level, price_velocity]
    x = np.array([series[0], 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])  # constant velocity transition
    H = np.array([[1.0, 0.0]])  # observe price level only

    estimates = np.empty(n)
    velocities = np.empty(n)

    for i in range(n):
        ratio = max(0.1, min(10.0, vol_ema[i] / vol_baseline))
        q = max(0.001, min(0.1, q_base * ratio))
        r = max(0.01, min(1.0, r_base / ratio))

        Q = np.array([[q, q * 0.5], [q * 0.5, q]])
        R_mat = np.array([[r]])

        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        z = series[i]
        y_innov = z - H @ x
        S = H @ P @ H.T + R_mat
        K = P @ H.T / S[0, 0]
        x = x + K.flatten() * y_innov[0]
        P = (np.eye(2) - K @ H) @ P

        estimates[i] = x[0]
        velocities[i] = x[1]

    if return_velocity:
        return estimates, velocities
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
