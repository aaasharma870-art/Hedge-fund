"""Daily-frequency features for alpha signal generation."""

import numpy as np
import pandas as pd

DAILY_FEATURES = [
    'Mom_5d', 'Mom_21d', 'Mom_63d', 'Mom_126d_skip21',
    'MR_ZScore_20d', 'RSI_14d',
    'Vol_Ratio', 'ATR_Pct_20d',
    'Amihud_20d', 'Volume_ZScore',
    'CS_Mom_Rank', 'CS_Vol_Rank', 'CS_MR_Rank',
    'Reversal_1d', 'Volume_Price_Div', 'Vol_Surprise',
]


def compute_daily_features(daily_df, ticker=None, universe_daily=None):
    """
    Compute daily features for one ticker.

    Args:
        daily_df: Daily OHLCV DataFrame
        ticker: Ticker symbol (for cross-sectional ranking)
        universe_daily: Dict {ticker: daily_df} for cross-sectional features

    Returns:
        DataFrame with DAILY_FEATURES columns added. Also adds Daily_ATR
        (not a feature, used for bracket sizing).
    """
    df = daily_df.copy()
    c, v, h, l = df['Close'], df['Volume'], df['High'], df['Low']
    returns = c.pct_change()

    # Momentum
    df['Mom_5d'] = c.pct_change(5)
    df['Mom_21d'] = c.pct_change(21)
    df['Mom_63d'] = c.pct_change(63)
    df['Mom_126d_skip21'] = c.shift(21).pct_change(105)

    # Mean reversion
    ma20 = c.rolling(20).mean()
    df['MR_ZScore_20d'] = (c - ma20) / c.rolling(20).std().clip(lower=1e-10)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().clip(lower=1e-10)
    df['RSI_14d'] = (100 - 100 / (1 + gain / loss)) / 100 - 0.5

    # Volatility
    vol10 = returns.rolling(10).std()
    vol60 = returns.rolling(60).std().clip(lower=1e-10)
    df['Vol_Ratio'] = vol10 / vol60
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df['ATR_Pct_20d'] = tr.rolling(20).mean() / c
    df['Daily_ATR'] = tr.rolling(20).mean()  # For bracket sizing, not a feature

    # Liquidity
    dollar_vol = (c * v).clip(lower=1.0)
    raw_amihud = (returns.abs() / dollar_vol * 1e9).rolling(20).mean()
    am_mean = raw_amihud.rolling(120).mean()
    am_std = raw_amihud.rolling(120).std().clip(lower=1e-10)
    df['Amihud_20d'] = (raw_amihud - am_mean) / am_std
    vol_ma60 = v.rolling(60).mean().clip(lower=1.0)
    df['Volume_ZScore'] = (v - vol_ma60) / v.rolling(60).std().clip(lower=1.0)

    # Short-term reversal (Lehmann 1990)
    df['Reversal_1d'] = -c.pct_change(1)

    # Volume-price divergence (accumulation/distribution)
    abs_ret = returns.abs().clip(lower=1e-10)
    vol_z = (v - v.rolling(20).mean()) / v.rolling(20).std().clip(lower=1e-10)
    ret_z = abs_ret / abs_ret.rolling(20).mean().clip(lower=1e-10)
    df['Volume_Price_Div'] = vol_z - ret_z

    # Volatility surprise
    expected_move = tr.rolling(20).mean().clip(lower=1e-10)
    actual_move = (c - c.shift(1)).abs()
    df['Vol_Surprise'] = actual_move / expected_move

    # Cross-sectional (fill 0.5 if no universe)
    df['CS_Mom_Rank'] = 0.5
    df['CS_Vol_Rank'] = 0.5
    df['CS_MR_Rank'] = 0.5

    if universe_daily and len(universe_daily) >= 3 and ticker:
        try:
            all_mom21 = pd.DataFrame({t: udf['Close'].pct_change(21)
                                       for t, udf in universe_daily.items()})
            all_vol = pd.DataFrame({t: udf['Close'].pct_change().rolling(20).std()
                                     for t, udf in universe_daily.items()})
            all_mr = pd.DataFrame({
                t: (udf['Close'] - udf['Close'].rolling(20).mean()) /
                    udf['Close'].rolling(20).std().clip(lower=1e-10)
                for t, udf in universe_daily.items()})
            mom_rank = all_mom21.rank(axis=1, pct=True)
            vol_rank = all_vol.rank(axis=1, pct=True)
            mr_rank = all_mr.rank(axis=1, pct=True)
            if ticker in mom_rank.columns:
                df['CS_Mom_Rank'] = mom_rank[ticker].reindex(df.index).fillna(0.5)
                df['CS_Vol_Rank'] = vol_rank[ticker].reindex(df.index).fillna(0.5)
                df['CS_MR_Rank'] = mr_rank[ticker].reindex(df.index).fillna(0.5)
        except Exception:
            pass

    # Clip features
    for col in DAILY_FEATURES:
        if col in df.columns:
            df[col] = df[col].clip(-4, 4)

    df.dropna(subset=[f for f in DAILY_FEATURES if f in df.columns], inplace=True)
    return df
