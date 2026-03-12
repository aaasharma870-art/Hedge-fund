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
    embargo = FORWARD_DAYS

    start = 0
    while start + train_days + embargo + test_days <= n:
        train_end = start + train_days
        test_start = train_end + embargo
        test_end = min(test_start + test_days, n)

        train_df = df.iloc[0:train_end].dropna(subset=['DailyTarget'])
        test_df = df.iloc[test_start:test_end].copy()
        test_clean = test_df.dropna(subset=['DailyTarget'])

        if len(train_df) < 100 or len(test_clean) < 10:
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


def generate_watchlist(predictions_by_ticker, top_n=3, bottom_n=3):
    """
    Cross-sectional ranking -> daily long/short watchlist.

    Returns: {date: {'longs': [(ticker, score), ...], 'shorts': [...]}}
    """
    # Build date -> {ticker: prediction} panel
    panel = {}
    for ticker, df in predictions_by_ticker.items():
        if 'DailyPrediction' not in df.columns:
            continue
        for idx, row in df.iterrows():
            d = idx.date() if hasattr(idx, 'date') else idx
            if d not in panel:
                panel[d] = {}
            panel[d][ticker] = row['DailyPrediction']

    watchlist = {}
    for d in sorted(panel.keys()):
        preds = panel[d]
        if len(preds) < top_n + bottom_n:
            continue

        ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        longs = [(t, s) for t, s in ranked[:top_n] if s > 0]
        shorts = [(t, abs(s)) for t, s in ranked[-bottom_n:] if s < 0]

        if longs or shorts:
            watchlist[d] = {'longs': longs, 'shorts': shorts}

    return watchlist
