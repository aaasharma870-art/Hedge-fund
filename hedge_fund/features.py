"""
Feature engineering for institutional-grade trading signals.

Provides microstructure features (VPIN, VWAP, Amihud), regime detection
(GEX proxy, volatility regime), alpha signals (RRS, liquidity sweeps),
and enhanced triple-barrier labeling.
"""

import logging
import threading
import time

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_vpin(df, volume_bucket_size=None, window=50):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    Measures order flow toxicity. High VPIN indicates informed traders
    are present and adverse selection risk is elevated.

    Uses Bulk Volume Classification (BVC) to estimate buy/sell volume
    from price changes, then computes rolling imbalance across
    volume-bucketed bars.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns.
        volume_bucket_size: Volume per bucket. If None, auto-determined
            as median(20-day avg volume) / 50.
        window: Number of volume buckets for the rolling VPIN average.

    Returns:
        Series of VPIN values clipped to [0, 1]. Higher = more toxic.
    """
    if df is None or len(df) < window:
        return pd.Series(0.0, index=df.index if df is not None else [])

    df = df.copy()

    if volume_bucket_size is None:
        avg_daily_volume = df["Volume"].rolling(20).mean().median()
        volume_bucket_size = max(1, int(avg_daily_volume / 50))

    price_change = df["Close"].diff()
    price_std = price_change.rolling(20).std()
    z_score = price_change / (price_std + 1e-9)
    buy_prob = norm.cdf(z_score)

    df["Buy_Volume"] = df["Volume"] * buy_prob
    df["Sell_Volume"] = df["Volume"] * (1 - buy_prob)

    cumulative_volume = df["Volume"].cumsum()
    bucket_id = (cumulative_volume / volume_bucket_size).astype(int)

    bucket_imbalance = df.groupby(bucket_id).apply(
        lambda x: abs(x["Buy_Volume"].sum() - x["Sell_Volume"].sum())
    )
    bucket_volume = df.groupby(bucket_id)["Volume"].sum()

    vpin_buckets = bucket_imbalance / (bucket_volume + 1)
    vpin_rolling = vpin_buckets.rolling(window, min_periods=10).mean()

    bucket_to_vpin = vpin_rolling.to_dict()
    result = df.groupby(bucket_id).ngroup().map(bucket_to_vpin).fillna(0).clip(0, 1)
    result.index = df.index
    return result


def calculate_enhanced_vwap_features(df):
    """
    Enhanced VWAP features for institutional mean-reversion signals.

    Returns:
        Dict with keys:
            VWAP_ZScore: Standardized distance from 20-period VWAP.
            VWAP_Slope: Rate of VWAP change (trending vs ranging).
            VWAP_Volume_Ratio: Current volume vs VWAP-period average volume.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (
        (typical_price * df["Volume"]).rolling(20).sum()
        / df["Volume"].rolling(20).sum()
    )

    vwap_dist = df["Close"] - vwap
    vwap_std = vwap_dist.rolling(20).std()
    vwap_zscore = vwap_dist / (vwap_std + 1e-9)

    vwap_roc = vwap.pct_change(5)
    vwap_slope = vwap_roc.rolling(10).mean()

    avg_vwap_volume = df["Volume"].rolling(20).mean()
    vwap_volume_ratio = df["Volume"] / (avg_vwap_volume + 1)

    return {
        "VWAP_ZScore": vwap_zscore,
        "VWAP_Slope": vwap_slope,
        "VWAP_Volume_Ratio": vwap_volume_ratio,
    }


def calculate_volatility_regime(df, atr_func=None):
    """
    GEX Proxy via volatility regime detection.

    Since real GEX requires options data, this proxies it using realized
    volatility patterns:
      - Low RV + contracting ATR = Positive GEX (mean reversion)
      - High RV + expanding ATR = Negative GEX (trending/breakouts)

    Args:
        df: DataFrame with 'Close', 'High', 'Low' columns.
        atr_func: Optional callable(high, low, close, length) -> Series.
            If None, computes ATR internally.

    Returns:
        Tuple of (regime_gex, vol_regime_label):
            regime_gex: Series of -1 (negative GEX), 0 (neutral), +1 (positive GEX).
            vol_regime_label: Series of 'LOW', 'MEDIUM', 'HIGH'.
    """
    returns = df["Close"].pct_change()
    realized_vol = returns.rolling(20).std() * np.sqrt(252)

    if atr_func is not None:
        atr_20 = atr_func(df["High"], df["Low"], df["Close"], length=20)
    else:
        from hedge_fund.indicators import ManualTA
        atr_20 = ManualTA.atr(df["High"], df["Low"], df["Close"], length=20)

    atr_change = atr_20.pct_change(5)

    vol_percentile = realized_vol.rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )

    regime = pd.Series(0, index=df.index)
    regime[(vol_percentile < 0.3) & (atr_change < 0)] = 1
    regime[(vol_percentile > 0.7) & (atr_change > 0)] = -1

    vol_regime = pd.Series("MEDIUM", index=df.index)
    vol_regime[vol_percentile < 0.3] = "LOW"
    vol_regime[vol_percentile > 0.7] = "HIGH"

    return regime, vol_regime


def calculate_amihud_illiquidity(df, window=20):
    """
    Amihud Illiquidity Ratio: price impact per dollar traded.

    High values indicate low liquidity and high slippage risk.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns.
        window: Rolling window for the illiquidity average.

    Returns:
        Series of illiquidity percentile ranks (0-1).
    """
    returns = df["Close"].pct_change().abs()
    dollar_volume = df["Volume"] * df["Close"]

    illiquidity = returns / (dollar_volume + 1e-9)
    illiquidity_ratio = illiquidity.rolling(window).mean()

    illiquidity_rank = illiquidity_ratio.rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )
    return illiquidity_rank


def calculate_real_relative_strength(stock_df, spy_df=None):
    """
    Real Relative Strength (RRS) vs the market.

    Measures how much a stock outperforms its beta-adjusted expected return.
    Positive RRS means institutional accumulation (stronger than expected).

    Args:
        stock_df: DataFrame with 'Close' for the stock.
        spy_df: DataFrame with 'Close' for SPY. If None, falls back to
            5-period momentum as a proxy.

    Returns:
        Series of cumulative 5-period RRS values.
    """
    if spy_df is None or len(spy_df) < 50:
        return stock_df["Close"].pct_change(5)

    stock_returns = stock_df["Close"].pct_change()
    spy_returns = spy_df["Close"].pct_change()

    spy_returns_aligned = spy_returns.reindex(stock_df.index, method="ffill")

    covariance = stock_returns.rolling(50).cov(spy_returns_aligned)
    spy_variance = spy_returns_aligned.rolling(50).var()
    beta = covariance / (spy_variance + 1e-9)
    beta = beta.fillna(1.0).clip(-3, 3)

    expected_return = beta * spy_returns_aligned
    rrs = stock_returns - expected_return

    rrs_cumulative = rrs.rolling(5).sum()
    return rrs_cumulative.fillna(0.0)


def calculate_liquidity_sweep(df, lookback=16):
    """
    Institutional liquidity sweep detection.

    Detects "fakeouts" where price breaks a key level to trigger retail
    stops, then reverses. High win-rate mean reversion signal.

    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns.
        lookback: Periods to look back for high/low levels.

    Returns:
        Series of signals: +1 (bullish sweep), -1 (bearish sweep), 0 (none).
    """
    signals = pd.Series(0, index=df.index)

    rolling_high = df["High"].rolling(lookback).max()
    rolling_low = df["Low"].rolling(lookback).min()

    avg_volume = df["Volume"].rolling(20).mean()
    volume_surge = df["Volume"] > (avg_volume * 1.5)

    for i in range(lookback + 1, len(df)):
        current_bar = df.iloc[i]
        prev_high = rolling_high.iloc[i - 1]
        prev_low = rolling_low.iloc[i - 1]

        if (
            current_bar["High"] > prev_high
            and current_bar["Close"] < prev_high
            and volume_surge.iloc[i]
        ):
            signals.iloc[i] = -1  # Bearish sweep
        elif (
            current_bar["Low"] < prev_low
            and current_bar["Close"] > prev_low
            and volume_surge.iloc[i]
        ):
            signals.iloc[i] = 1  # Bullish sweep

    return signals


class CrossSectionalRanker:
    """
    Cross-sectional factor ranking: rank stocks vs peers by momentum, value, quality.

    Produces percentile scores (0-1) per ticker for use as model features
    and score boosters. Recomputed once per scan cycle from snapshot data.
    """

    def __init__(self):
        self._ranks = {}  # ticker -> {'momentum_rank': 0.75, ...}
        self._last_update = 0
        self._lock = threading.Lock()

    def update(self, snap_data, fundamental_getter, universe):
        """
        Rank universe cross-sectionally.

        Args:
            snap_data: Dict from market snapshot
                {ticker: {'price': float, 'dayVol': int, 'dayOpen': float, ...}}
            fundamental_getter: Callable(ticker) -> dict with at least 'pe_ratio'.
                Can be None to skip value factor.
            universe: List of ticker strings to rank.
        """
        if not snap_data or len(snap_data) < 5:
            return

        tickers = [t for t in universe if t in snap_data]
        if len(tickers) < 5:
            return

        # Momentum factor: intraday return
        mom = {}
        for t in tickers:
            s = snap_data[t]
            day_open = s.get('dayOpen', 0)
            price = s.get('price', 0)
            if day_open > 0 and price > 0:
                mom[t] = (price - day_open) / day_open
            else:
                mom[t] = 0.0

        # Volume factor: relative dollar volume (liquidity + attention)
        vol = {}
        for t in tickers:
            s = snap_data[t]
            vol[t] = s.get('dayVol', 0) * s.get('price', 1)

        # Value factor: inverse P/E (high = cheap = good value)
        value = {}
        for t in tickers:
            try:
                if fundamental_getter is not None:
                    feats = fundamental_getter(t)
                    pe = feats.get('pe_ratio', 20)
                    if pe and pe > 0:
                        value[t] = 1.0 / pe
                    else:
                        value[t] = 0.05
                else:
                    value[t] = 0.05
            except Exception:
                value[t] = 0.05

        # Compute percentile ranks
        def _pct_rank(data_dict):
            if not data_dict:
                return {}
            items = sorted(data_dict.items(), key=lambda x: x[1])
            n = len(items)
            return {t: (i / max(1, n - 1)) for i, (t, _) in enumerate(items)}

        mom_ranks = _pct_rank(mom)
        vol_ranks = _pct_rank(vol)
        val_ranks = _pct_rank(value)

        # Composite score: 50% momentum + 25% volume + 25% value
        ranks = {}
        for t in tickers:
            mr = mom_ranks.get(t, 0.5)
            vr = vol_ranks.get(t, 0.5)
            vlr = val_ranks.get(t, 0.5)
            composite = 0.50 * mr + 0.25 * vr + 0.25 * vlr
            ranks[t] = {
                'momentum_rank': round(mr, 3),
                'volume_rank': round(vr, 3),
                'value_rank': round(vlr, 3),
                'composite_rank': round(composite, 3),
            }

        with self._lock:
            self._ranks = ranks
            self._last_update = time.time()

        logging.info(f"Cross-sectional ranking: {len(ranks)} tickers ranked")

    def get_ranks(self, ticker):
        """Return factor ranks for a ticker, or neutral defaults."""
        with self._lock:
            return self._ranks.get(ticker, {
                'momentum_rank': 0.5, 'volume_rank': 0.5,
                'value_rank': 0.5, 'composite_rank': 0.5,
            })

    def get_score_boost(self, ticker):
        """
        Score multiplier based on composite rank.

        Top-ranked stocks get up to 1.3x score boost, bottom get 0.7x.
        """
        r = self.get_ranks(ticker)
        comp = r['composite_rank']
        return float(np.clip(0.7 + 0.6 * comp, 0.7, 1.3))
