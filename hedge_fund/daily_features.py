"""
Daily-frequency feature engineering for alpha signal generation.

All features are normalized to approximately N(0,1) and computed on daily OHLCV bars.
Based on documented factor research:
  - Momentum: Jegadeesh & Titman (1993), Carhart (1997)
  - Mean reversion: DeBondt & Thaler (1985), Lo & MacKinlay (1990)
  - Volatility: Ang et al. (2006)
  - Liquidity: Amihud (2002)
  - Quality/Value: Asness et al. (2019)
"""

import numpy as np
import pandas as pd


DAILY_FEATURES = [
    # Momentum factors (multiple horizons)
    'Mom_5d',          # 5-day return (short-term momentum)
    'Mom_10d',         # 10-day return
    'Mom_21d',         # 1-month return
    'Mom_63d',         # 3-month return (strongest documented factor)
    'Mom_126d_skip21', # 6-month momentum skipping recent month (Jegadeesh & Titman)

    # Mean reversion
    'MR_ZScore_20d',   # 20-day mean reversion Z-score
    'MR_ZScore_60d',   # 60-day mean reversion Z-score
    'RSI_14d',         # 14-day RSI (overbought/oversold)

    # Volatility
    'Vol_Ratio',       # 10d vol / 60d vol (vol expansion/contraction)
    'ATR_Pct_20d',     # 20-day ATR as % of price
    'Realized_Vol_20d', # 20-day realized volatility

    # Liquidity
    'Amihud_20d',      # 20-day Amihud illiquidity ratio
    'Volume_ZScore',   # Volume relative to 60-day average

    # Cross-sectional (computed across the universe)
    'CS_Mom_Rank',     # Cross-sectional momentum rank (0 to 1)
    'CS_MR_Rank',      # Cross-sectional mean-reversion rank
    'CS_Vol_Rank',     # Cross-sectional volatility rank

    # Calendar/Structural
    'Day_of_Week',     # Monday=0, Friday=4 (documented day-of-week effects)
    'Month_End',       # Binary: within 3 days of month end (window dressing)
]


def compute_daily_features(daily_df, universe_daily=None, ticker=None):
    """
    Compute all daily features for a single ticker.

    Args:
        daily_df: DataFrame with daily OHLCV (columns: Open, High, Low, Close, Volume)
        universe_daily: Dict of {ticker: daily_df} for cross-sectional features
        ticker: Name of this ticker (for cross-sectional lookups)

    Returns:
        DataFrame with all DAILY_FEATURES columns added. NaN rows dropped.
    """
    df = daily_df.copy()

    c = df['Close']
    v = df['Volume']
    h = df['High']
    l = df['Low']

    # ── Momentum factors ──
    df['Mom_5d'] = c.pct_change(5)
    df['Mom_10d'] = c.pct_change(10)
    df['Mom_21d'] = c.pct_change(21)
    df['Mom_63d'] = c.pct_change(63)
    # 6-month momentum skipping most recent month (classic Jegadeesh-Titman)
    df['Mom_126d_skip21'] = c.shift(21).pct_change(105)

    # ── Mean reversion ──
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std().clip(lower=1e-10)
    df['MR_ZScore_20d'] = (c - ma20) / std20

    ma60 = c.rolling(60).mean()
    std60 = c.rolling(60).std().clip(lower=1e-10)
    df['MR_ZScore_60d'] = (c - ma60) / std60

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().clip(lower=1e-10)
    rs = gain / loss
    df['RSI_14d'] = (100 - 100 / (1 + rs)) / 100.0 - 0.5  # Centered at 0

    # ── Volatility ──
    returns = c.pct_change()
    vol_10 = returns.rolling(10).std()
    vol_60 = returns.rolling(60).std().clip(lower=1e-10)
    df['Vol_Ratio'] = vol_10 / vol_60

    df['Realized_Vol_20d'] = returns.rolling(20).std() * np.sqrt(252)

    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR_Pct_20d'] = tr.rolling(20).mean() / c

    # Also store raw ATR for position sizing (not a feature, but needed for exits)
    df['Daily_ATR'] = tr.rolling(20).mean()

    # ── Liquidity ──
    abs_ret = returns.abs()
    dollar_vol = (c * v).clip(lower=1.0)
    df['Amihud_20d'] = (abs_ret / dollar_vol * 1e9).rolling(20).mean()
    # Normalize to Z-score
    amihud_mean = df['Amihud_20d'].rolling(120).mean()
    amihud_std = df['Amihud_20d'].rolling(120).std().clip(lower=1e-10)
    df['Amihud_20d'] = (df['Amihud_20d'] - amihud_mean) / amihud_std

    vol_ma60 = v.rolling(60).mean().clip(lower=1.0)
    df['Volume_ZScore'] = (v - vol_ma60) / v.rolling(60).std().clip(lower=1.0)

    # ── Cross-sectional features ──
    df['CS_Mom_Rank'] = 0.5
    df['CS_MR_Rank'] = 0.5
    df['CS_Vol_Rank'] = 0.5

    if universe_daily and len(universe_daily) >= 3 and ticker:
        try:
            all_mom21 = pd.DataFrame({
                t: udf['Close'].pct_change(21)
                for t, udf in universe_daily.items()
            })
            all_mr = pd.DataFrame({
                t: (udf['Close'] - udf['Close'].rolling(20).mean()) /
                    udf['Close'].rolling(20).std().clip(lower=1e-10)
                for t, udf in universe_daily.items()
            })
            all_vol = pd.DataFrame({
                t: udf['Close'].pct_change().rolling(20).std()
                for t, udf in universe_daily.items()
            })

            mom_rank = all_mom21.rank(axis=1, pct=True)
            mr_rank = all_mr.rank(axis=1, pct=True)
            vol_rank = all_vol.rank(axis=1, pct=True)

            if ticker in mom_rank.columns:
                df['CS_Mom_Rank'] = mom_rank[ticker].reindex(df.index).fillna(0.5)
                df['CS_MR_Rank'] = mr_rank[ticker].reindex(df.index).fillna(0.5)
                df['CS_Vol_Rank'] = vol_rank[ticker].reindex(df.index).fillna(0.5)
        except Exception:
            pass

    # ── Calendar ──
    if hasattr(df.index, 'dayofweek'):
        df['Day_of_Week'] = df.index.dayofweek / 4.0 - 0.5
    else:
        df['Day_of_Week'] = 0.0

    if hasattr(df.index, 'day'):
        df['Month_End'] = ((df.index.days_in_month - df.index.day) <= 3).astype(float)
    else:
        df['Month_End'] = 0.0

    # Clip all features to [-4, 4]
    for col in DAILY_FEATURES:
        if col in df.columns:
            df[col] = df[col].clip(-4, 4)

    df.dropna(subset=[f for f in DAILY_FEATURES if f in df.columns], inplace=True)
    return df
