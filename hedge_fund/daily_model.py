"""
Daily Alpha Signal Generator.

Walk-forward ML on daily bars -> cross-sectional ranked watchlist.
"""

import numpy as np
import pandas as pd
from hedge_fund.ensemble import EnsembleModel
from hedge_fund.daily_features import DAILY_FEATURES, compute_daily_features

FORWARD_DAYS = 5  # Predict 5-day return


def compute_daily_labels(df, forward_days=FORWARD_DAYS):
    """5-day forward return as regression label."""
    return df['Close'].pct_change(forward_days).shift(-forward_days)


def walk_forward_daily(df, features, train_days=250, test_days=60,
                       step_days=60):
    """
    Walk-forward on daily bars. Expanding (anchored) window.
    Returns DataFrame with 'DailyPrediction' column or None.
    """
    labels = compute_daily_labels(df)
    df = df.copy()
    df['DailyTarget'] = labels

    avail = [f for f in features if f in df.columns]
    if len(avail) < 5:
        return None

    n = len(df)
    all_test = []
    embargo = FORWARD_DAYS + 5  # Extra gap to avoid label leakage from overlapping forward windows

    start = 0
    while start + train_days + embargo + test_days <= n:
        train_end = start + train_days
        test_start = train_end + embargo
        test_end = min(test_start + test_days, n)

        train_df = df.iloc[0:train_end].dropna(subset=['DailyTarget'])
        test_df = df.iloc[test_start:test_end].copy()
        test_clean = test_df.dropna(subset=['DailyTarget'])

        if len(train_df) < 200 or len(test_clean) < 10:
            start += step_days
            continue

        model = EnsembleModel(use_daily=True)
        model.fit(train_df[avail], train_df['DailyTarget'])

        preds = model.predict(test_clean[avail])
        test_clean = test_clean.copy()
        test_clean['DailyPrediction'] = preds
        all_test.append(test_clean)

        start += step_days

    if not all_test:
        return None
    return pd.concat(all_test)


def generate_watchlist(predictions_by_ticker, top_n=3, bottom_n=3,
                       min_spread=0.0):
    """
    Cross-sectional ranking -> daily long/short watchlist.

    IMPORTANT: Predictions are shifted forward by 1 day to eliminate
    same-day entry bias. The model computes features from day T's bar,
    but the signal is acted on day T+1 (next trading day).

    Returns: {date: {'longs': [(ticker, score), ...], 'shorts': [...]}}
    """
    # Build date -> {ticker: prediction} panel
    panel = {}
    all_prediction_dates = set()

    for ticker, df in predictions_by_ticker.items():
        if 'DailyPrediction' not in df.columns:
            continue
        for idx, row in df.iterrows():
            d = idx.date() if hasattr(idx, 'date') else idx
            all_prediction_dates.add(d)
            if d not in panel:
                panel[d] = {}
            panel[d][ticker] = row['DailyPrediction']

    # Build sorted list of all dates for shifting
    sorted_dates = sorted(all_prediction_dates)
    next_day_map = {}
    for i in range(len(sorted_dates) - 1):
        next_day_map[sorted_dates[i]] = sorted_dates[i + 1]

    watchlist = {}
    for d in sorted_dates:
        preds = panel.get(d, {})
        if len(preds) < top_n + bottom_n:
            continue

        # Shift signal to NEXT trading day (eliminates same-day bias)
        trade_date = next_day_map.get(d)
        if trade_date is None:
            continue  # Last day has no next day

        ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)

        # Conviction filter: skip days with low spread between best and worst
        if min_spread > 0:
            spread = ranked[0][1] - ranked[-1][1]
            if spread < min_spread:
                continue

        # Cross-sectional: long the best, short the worst — absolute sign irrelevant
        longs = [(t, s) for t, s in ranked[:top_n]]
        shorts = [(t, abs(s)) for t, s in ranked[-bottom_n:]]

        # Force equal L/S balance to prevent directional bias
        if len(longs) > len(shorts):
            longs = longs[:len(shorts)]
        elif len(shorts) > len(longs):
            shorts = shorts[:len(longs)]

        if longs or shorts:
            watchlist[trade_date] = {'longs': longs, 'shorts': shorts}

    return watchlist
