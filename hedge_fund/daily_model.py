"""
Daily Alpha Signal Generator (Tier 1).

Predicts 5-day forward cross-sectional returns using walk-forward validated
ML ensemble on daily bars. Outputs a ranked watchlist.
"""

import numpy as np
import pandas as pd
from hedge_fund.ensemble import EnsembleModel
from hedge_fund.daily_features import DAILY_FEATURES, compute_daily_features


# Label: 5-day forward return (continuous, for regression)
FORWARD_DAYS = 5


def compute_daily_labels(df, forward_days=FORWARD_DAYS):
    """
    Compute forward N-day returns as regression labels.

    Simple and direct: if you buy at today's close, what's your
    return after N trading days? No bracket simulation needed at
    the daily alpha level — brackets are for the execution layer.
    """
    fwd_ret = df['Close'].pct_change(forward_days).shift(-forward_days)
    return fwd_ret


def walk_forward_daily(df, features, train_days=250, test_days=60,
                       step_days=60, forward_days=FORWARD_DAYS):
    """
    Walk-forward training on daily bars with expanding window.

    Train on 250 days (~1 year), predict next 60 days (~3 months).
    Expanding window: training always starts at day 0.

    Returns DataFrame of test predictions with 'DailyPrediction' column,
    or None if insufficient data.
    """
    labels = compute_daily_labels(df, forward_days)
    df = df.copy()
    df['DailyTarget'] = labels

    avail = [f for f in features if f in df.columns]
    if len(avail) < 5:
        return None

    n = len(df)
    all_test_dfs = []
    embargo = forward_days  # Prevent label leakage

    start = 0
    while start + train_days + embargo + test_days <= n:
        train_end = start + train_days
        test_start = train_end + embargo
        test_end = min(test_start + test_days, n)

        # Expanding window: always train from bar 0
        train_df = df.iloc[0:train_end]
        test_df = df.iloc[test_start:test_end].copy()

        # Drop rows with NaN labels
        train_clean = train_df.dropna(subset=['DailyTarget'])
        test_clean = test_df.dropna(subset=['DailyTarget'])

        if len(train_clean) < 100 or len(test_clean) < 10:
            start += step_days
            continue

        model = EnsembleModel(use_daily=True)
        model.fit(train_clean[avail], train_clean['DailyTarget'])

        preds = model.predict(test_clean[avail])
        test_clean = test_clean.copy()
        test_clean['DailyPrediction'] = preds
        all_test_dfs.append(test_clean)

        start += step_days

    if not all_test_dfs:
        return None

    return pd.concat(all_test_dfs)


def generate_daily_watchlist(daily_predictions_by_ticker, top_n=4, bottom_n=4,
                             min_conviction=0.0):
    """
    Generate daily long/short watchlist from cross-sectional ranking.

    For each trading day:
    1. Collect all tickers' predictions for that day
    2. Rank them cross-sectionally
    3. Top N = long watchlist, Bottom N = short watchlist
    4. Conviction = prediction magnitude

    Returns dict: {date: {'longs': [(ticker, conviction), ...],
                          'shorts': [(ticker, conviction), ...]}}
    """
    pred_panel = {}
    for ticker, df in daily_predictions_by_ticker.items():
        if 'DailyPrediction' not in df.columns:
            continue
        for date_idx, row in df.iterrows():
            d = date_idx.date() if hasattr(date_idx, 'date') else date_idx
            if d not in pred_panel:
                pred_panel[d] = {}
            pred_panel[d][ticker] = row['DailyPrediction']

    watchlist = {}
    for d in sorted(pred_panel.keys()):
        day_preds = pred_panel[d]
        if len(day_preds) < top_n + bottom_n:
            continue

        sorted_tickers = sorted(day_preds.items(), key=lambda x: x[1], reverse=True)

        longs = [(t, score) for t, score in sorted_tickers[:top_n]
                 if score > min_conviction]
        shorts = [(t, abs(score)) for t, score in sorted_tickers[-bottom_n:]
                  if score < -min_conviction]

        if longs or shorts:
            watchlist[d] = {'longs': longs, 'shorts': shorts}

    return watchlist
