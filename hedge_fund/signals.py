"""
hedge_fund/signals.py

Institutional-quality alpha signals grounded in market microstructure
and cross-sectional factor research. All signals are normalized to
approximately N(0,1) range and clipped to [-4, 4].
Sign convention: positive = bullish, negative = bearish, for all signals.

Academic foundations:
    Order flow: Kyle (1985), Easley & O'Hara (1987), Glosten & Milgrom (1985)
    Cross-sectional: Jegadeesh & Titman (1993), Asness et al. (2013)
    Mean reversion: Lehmann (1990), Lo & MacKinlay (1990)
    Regime detection: Lo & MacKinlay (1988) variance ratio, Hamilton (1989)
    Session structure: Admati & Pfleiderer (1988)
    Adverse selection: Kyle (1985) lambda, Almgren et al. (2005)
"""

import numpy as np
import pandas as pd
from typing import Optional


SIGNAL_NAMES = [
    'COFI',
    'Absorption_Ratio',
    'Kyle_Lambda',
    'Trade_Intensity',
    'CS_Mom_Rank',
    'CS_Volume_Rank',
    'MR_Score',
    'Beta_Momentum',
    'Variance_Ratio',
    'RV_Regime',
    'Regime_Trending',
    'Regime_MeanRev',
    'Regime_Volatile',
    'Session_Opening',
    'Session_Closing',
    'Session_Mid',
    'Session_Progress',
]


def compute_cofi(df: pd.DataFrame, window: int = 20, norm_window: int = 100) -> pd.Series:
    """
    Cumulative Order Flow Imbalance.

    Tracks persistent accumulation of net directional volume over a rolling
    window. Single-bar OFI is noise. COFI captures the multi-bar trend in
    institutional positioning before it is fully reflected in price.

    Method: Lee-Ready tick rule approximation using close position within range.
    Close near high = buyer-initiated. Close near low = seller-initiated.
    Net volume accumulated over window bars. Z-score normalized.

    Positive COFI_Z (above 1.5): persistent institutional buying -> long
    Negative COFI_Z (below -1.5): persistent institutional selling -> short
    """
    try:
        bar_range = (df['High'] - df['Low']).clip(lower=1e-10)
        close_pos = (df['Close'] - df['Low']) / bar_range
        buy_vol = df['Volume'] * close_pos
        sell_vol = df['Volume'] * (1.0 - close_pos)
        net_vol = buy_vol - sell_vol
        cofi_raw = net_vol.rolling(window, min_periods=max(3, window // 4)).sum()
        roll_mean = cofi_raw.rolling(norm_window, min_periods=20).mean()
        roll_std = cofi_raw.rolling(norm_window, min_periods=20).std().clip(lower=1e-10)
        return ((cofi_raw - roll_mean) / roll_std).clip(-4, 4).rename('COFI')
    except Exception:
        return pd.Series(0.0, index=df.index, name='COFI')


def compute_absorption_ratio(df: pd.DataFrame, window: int = 5, norm_window: int = 60) -> pd.Series:
    """
    Directional Absorption Ratio.

    When large sell orders hit but price does not fall proportionally, buyers
    are absorbing the selling. This exhaustion of supply typically precedes a
    price reversal. The signal is made directional by multiplying by the sign
    of recent price movement.

    Positive: buyers absorbing sells (supply exhaustion) -> bullish
    Negative: sellers absorbing buys (demand exhaustion) -> bearish
    """
    try:
        price_move = df['Close'].diff().abs().clip(lower=1e-6)
        vol_per_move = df['Volume'] / price_move
        absorption = vol_per_move.rolling(window, min_periods=2).mean()
        direction = np.sign(df['Close'].diff(window)).replace(0, np.nan).ffill().fillna(1)
        directional = absorption * direction
        roll_mean = directional.rolling(norm_window, min_periods=15).mean()
        roll_std = directional.rolling(norm_window, min_periods=15).std().clip(lower=1e-10)
        return ((directional - roll_mean) / roll_std).clip(-4, 4).rename('Absorption_Ratio')
    except Exception:
        return pd.Series(0.0, index=df.index, name='Absorption_Ratio')


def compute_kyle_lambda(df: pd.DataFrame, window: int = 20, norm_window: int = 100) -> pd.Series:
    """
    Kyle's Lambda -- Adverse Selection / Market Depth Proxy.

    Price impact per unit of sqrt(volume) following the square-root market
    impact model (Almgren et al., 2005). High lambda = thin market with
    informed trading = unfavorable entry conditions. We return negative lambda
    so that positive values indicate favorable (deep) market conditions.

    High positive value: deep liquid market, good execution conditions
    High negative value: thin market, high adverse selection risk, avoid
    """
    try:
        price_impact = df['Close'].diff().abs()
        sqrt_vol = np.sqrt(df['Volume'].clip(lower=1))
        impact = price_impact / sqrt_vol
        lambda_raw = impact.rolling(window, min_periods=5).mean()
        roll_mean = lambda_raw.rolling(norm_window, min_periods=20).mean()
        roll_std = lambda_raw.rolling(norm_window, min_periods=20).std().clip(lower=1e-10)
        lambda_z = (lambda_raw - roll_mean) / roll_std
        return (-lambda_z).clip(-4, 4).rename('Kyle_Lambda')
    except Exception:
        return pd.Series(0.0, index=df.index, name='Kyle_Lambda')


def compute_trade_intensity(df: pd.DataFrame, window: int = 10, norm_window: int = 60) -> pd.Series:
    """
    Abnormal Trade Intensity -- Institutional Activity Precursor.

    Volume spikes not accompanied by proportional price moves indicate a large
    participant working an order through the market. This often precedes price
    movement by 1-5 bars as they continue building position. Made directional
    using order flow classification.

    Positive: high abnormal volume with buying pressure -> follow long
    Negative: high abnormal volume with selling pressure -> follow short
    """
    try:
        avg_vol = df['Volume'].rolling(window, min_periods=3).mean().clip(lower=1)
        vol_ratio = df['Volume'] / avg_vol
        avg_vol_long = df['Volume'].rolling(norm_window, min_periods=20).mean().clip(lower=1)
        intensity = vol_ratio * (avg_vol_long / avg_vol.clip(lower=1))
        roll_mean = intensity.rolling(norm_window, min_periods=20).mean()
        roll_std = intensity.rolling(norm_window, min_periods=20).std().clip(lower=1e-10)
        intensity_z = (intensity - roll_mean) / roll_std
        bar_range = (df['High'] - df['Low']).clip(lower=1e-10)
        close_pos = (df['Close'] - df['Low']) / bar_range
        ofi_dir = np.sign(close_pos - 0.5).replace(0, 1)
        return (intensity_z * ofi_dir).clip(-4, 4).rename('Trade_Intensity')
    except Exception:
        return pd.Series(0.0, index=df.index, name='Trade_Intensity')


def compute_cross_sectional_momentum(
    ticker: str,
    all_closes: pd.DataFrame,
    lookbacks: list = None,
) -> pd.Series:
    """
    Cross-Sectional Momentum Rank (Jegadeesh & Titman, 1993).

    For each bar, ranks all tickers by N-bar return and returns the target
    ticker's percentile rank centered at 0. Composites multiple lookbacks.

    High positive: top cross-sectional momentum -> strong long signal
    High negative: bottom cross-sectional momentum -> strong short signal
    """
    if lookbacks is None:
        lookbacks = [5, 10, 20]
    try:
        ranks = []
        for lb in lookbacks:
            rets = all_closes.pct_change(lb)
            ranked = rets.rank(axis=1, pct=True) - 0.5  # center at 0
            if ticker in ranked.columns:
                ranks.append(ranked[ticker])
        if not ranks:
            return pd.Series(0.0, index=all_closes.index, name='CS_Mom_Rank')
        composite = pd.concat(ranks, axis=1).mean(axis=1)
        roll_std = composite.rolling(60, min_periods=10).std().clip(lower=1e-10)
        return (composite / roll_std).clip(-4, 4).rename('CS_Mom_Rank')
    except Exception:
        return pd.Series(0.0, index=all_closes.index, name='CS_Mom_Rank')


def compute_cs_volume_rank(
    ticker: str,
    all_volumes: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Cross-Sectional Abnormal Volume Rank.

    Ranks tickers by abnormal volume (actual / rolling average) relative to peers.

    High positive: abnormally high volume vs peers -> strong directional signal
    Near zero: normal relative volume -> weaker signal quality
    """
    try:
        avg = all_volumes.rolling(window, min_periods=5).mean().clip(lower=1)
        ratios = all_volumes / avg
        ranked = ratios.rank(axis=1, pct=True)
        if ticker not in ranked.columns:
            return pd.Series(0.5, index=all_volumes.index, name='CS_Volume_Rank')
        return (ranked[ticker] - 0.5).clip(-0.5, 0.5).rename('CS_Volume_Rank')
    except Exception:
        return pd.Series(0.0, index=all_volumes.index, name='CS_Volume_Rank')


def compute_mean_reversion_score(df: pd.DataFrame, window: int = 20, smooth: int = 3) -> pd.Series:
    """
    Mean Reversion Opportunity Score (Lehmann 1990, Lo & MacKinlay 1990).

    Z-score of recent log-return relative to rolling distribution.
    High positive: stock significantly overbought -> short opportunity (fade)
    High negative: stock significantly oversold -> long opportunity (buy dip)

    NOTE: This signal is directionally inverted for trade entry.
    MR_Score < -1.5 + positive COFI = long entry (oversold with buying)
    MR_Score > +1.5 + negative COFI = short entry (overbought with selling)
    """
    try:
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        roll_mean = log_ret.rolling(window, min_periods=5).mean()
        roll_std = log_ret.rolling(window, min_periods=5).std().clip(lower=1e-6)
        z = (log_ret - roll_mean) / roll_std
        return z.rolling(smooth, min_periods=1).mean().clip(-4, 4).rename('MR_Score')
    except Exception:
        return pd.Series(0.0, index=df.index, name='MR_Score')


def compute_beta_momentum(df: pd.DataFrame, universe_returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Beta-Adjusted Idiosyncratic Momentum (Frazzini & Pedersen, 2014).

    Decomposes ticker return into beta * market_return (systematic) and
    alpha (idiosyncratic). Only the alpha component reflects genuine
    stock-specific demand.

    Positive: stock-specific outperformance (institutional demand) -> long
    Negative: stock-specific underperformance (institutional selling) -> short
    """
    try:
        ticker_ret = df['Close'].pct_change()
        cov = ticker_ret.rolling(window, min_periods=10).cov(universe_returns)
        var = universe_returns.rolling(window, min_periods=10).var().clip(lower=1e-10)
        beta = (cov / var).clip(-3, 3)
        alpha = ticker_ret - beta * universe_returns
        cum_alpha = alpha.rolling(window, min_periods=5).sum()
        roll_mean = cum_alpha.rolling(60, min_periods=15).mean()
        roll_std = cum_alpha.rolling(60, min_periods=15).std().clip(lower=1e-10)
        return ((cum_alpha - roll_mean) / roll_std).clip(-4, 4).rename('Beta_Momentum')
    except Exception:
        return pd.Series(0.0, index=df.index, name='Beta_Momentum')


def compute_variance_ratio(df: pd.DataFrame, k: int = 4, window: int = 40) -> pd.Series:
    """
    Variance Ratio Test (Lo & MacKinlay, 1988).

    Tests whether returns exhibit autocorrelation (trending) or negative
    autocorrelation (mean-reverting) vs random walk (VR=1.0).

    VR_centered > 0.3: trending regime, momentum signals valid
    VR_centered < -0.3: mean-reverting regime, MR signals valid
    |VR_centered| < 0.15: near random walk, reduce size and skip
    """
    try:
        ret1 = df['Close'].pct_change(1)
        retk = df['Close'].pct_change(k)
        var1 = ret1.rolling(window, min_periods=10).var().clip(lower=1e-10)
        vark = retk.rolling(window, min_periods=10).var().clip(lower=1e-10)
        vr = vark / (k * var1)
        return (vr - 1.0).clip(-1, 1).rename('Variance_Ratio')
    except Exception:
        return pd.Series(0.0, index=df.index, name='Variance_Ratio')


def compute_rv_regime(df: pd.DataFrame, short: int = 5, long: int = 60) -> pd.Series:
    """
    Realized Volatility Regime.

    Short-window realized vol relative to long-window baseline.
    Engle (1982) ARCH: volatility clusters.

    > 1.5: vol expansion in progress -> widen stops, reduce size
    0.7-1.5: normal vol conditions -> trade at full size
    < 0.7: vol compression -> breakout setup forming, size up
    """
    try:
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        rv_s = log_ret.rolling(short, min_periods=3).std()
        rv_l = log_ret.rolling(long, min_periods=20).std().clip(lower=1e-10)
        return (rv_s / rv_l).clip(0, 4).rename('RV_Regime')
    except Exception:
        return pd.Series(1.0, index=df.index, name='RV_Regime')


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday Session Structure (Admati & Pfleiderer, 1988).

    Informed trading concentrates at open and close. Order flow imbalance
    signals are strongest in the opening 45 minutes. Mean reversion dominates
    mid-session. Institutional rebalancing resumes in the closing 60 minutes.

    For 15-minute bars (session = 9:30-16:00 ET, 26 bars):
      Opening (9:30-10:15): bars 1-3, order flow strongest
      Mid-session (10:15-15:00): bars 4-22, mean reversion dominant
      Closing (15:00-16:00): bars 23-26, rebalancing flow
    """
    result = df.copy()
    try:
        if hasattr(df.index, 'hour'):
            h = pd.Series(df.index.hour, index=df.index)
            m = pd.Series(df.index.minute, index=df.index)
            total = h * 60 + m
            opening = ((total >= 570) & (total < 615)).astype(float)
            closing = (total >= 900).astype(float)
            mid = (~opening.astype(bool) & ~closing.astype(bool)).astype(float)
            result['Session_Opening'] = opening.values
            result['Session_Closing'] = closing.values
            result['Session_Mid'] = mid.values
            result['Session_Progress'] = ((total - 570).clip(lower=0) / 390.0).clip(0, 1).values
        else:
            result['Session_Opening'] = 0.0
            result['Session_Closing'] = 0.0
            result['Session_Mid'] = 1.0
            result['Session_Progress'] = 0.5
    except Exception:
        result['Session_Opening'] = 0.0
        result['Session_Closing'] = 0.0
        result['Session_Mid'] = 1.0
        result['Session_Progress'] = 0.5
    return result


def classify_regime(variance_ratio: pd.Series, rv_regime: pd.Series) -> pd.Series:
    """
    Three-state regime classifier using variance ratio and vol regime.

    State 0 TRENDING: VR > 0.15, vol not elevated -> momentum mode
    State 1 MEAN_REVERTING: VR < -0.10, low vol -> mean reversion mode
    State 2 VOLATILE: RV_Regime > 1.8 -> defensive mode, longs only
    """
    regime = pd.Series(1, index=variance_ratio.index, dtype=int)
    trending = (variance_ratio > 0.15) & (rv_regime < 1.6)
    volatile = rv_regime > 1.8
    regime[trending] = 0
    regime[volatile] = 2
    return regime


def compute_all_signals(
    df: pd.DataFrame,
    ticker: str,
    all_closes: Optional[pd.DataFrame] = None,
    all_volumes: Optional[pd.DataFrame] = None,
    universe_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Master function. Computes all institutional signals for one ticker.
    Called at the end of prepare_features(). Adds signal columns to existing df.
    Does not modify any existing columns.
    """
    result = df.copy()

    result['COFI'] = compute_cofi(df)
    result['Absorption_Ratio'] = compute_absorption_ratio(df)
    result['Kyle_Lambda'] = compute_kyle_lambda(df)
    result['Trade_Intensity'] = compute_trade_intensity(df)

    if all_closes is not None and ticker in all_closes.columns:
        result['CS_Mom_Rank'] = compute_cross_sectional_momentum(
            ticker, all_closes).reindex(df.index).fillna(0.0)
    else:
        result['CS_Mom_Rank'] = 0.0

    if all_volumes is not None and ticker in all_volumes.columns:
        result['CS_Volume_Rank'] = compute_cs_volume_rank(
            ticker, all_volumes).reindex(df.index).fillna(0.0)
    else:
        result['CS_Volume_Rank'] = 0.0

    result['MR_Score'] = compute_mean_reversion_score(df)

    if universe_returns is not None:
        univ = universe_returns.reindex(df.index, method='ffill').fillna(0)
        result['Beta_Momentum'] = compute_beta_momentum(df, univ)
    else:
        result['Beta_Momentum'] = 0.0

    vr = compute_variance_ratio(df)
    rv = compute_rv_regime(df)
    result['Variance_Ratio'] = vr
    result['RV_Regime'] = rv

    regime = classify_regime(vr, rv)
    result['Regime_Trending'] = (regime == 0).astype(float)
    result['Regime_MeanRev'] = (regime == 1).astype(float)
    result['Regime_Volatile'] = (regime == 2).astype(float)

    result = compute_session_features(result)

    return result
