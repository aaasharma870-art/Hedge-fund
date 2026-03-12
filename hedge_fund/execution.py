"""
Intraday Execution Engine.

Uses 15-minute VPIN, VWAP, OFI for entry timing.
Uses DAILY ATR for bracket sizing.
Holding period: 3-15 days.
"""

import numpy as np
import pandas as pd


def score_entry_bar(row, direction):
    """Score a 15-min bar for entry quality. Returns 0-1."""
    vpin = float(row.get('VPIN', 0.5))
    vwap_z = float(row.get('VWAP_ZScore', 0.0))
    ofi = float(row.get('OFI', 0.0))

    if direction == 'LONG':
        vpin_score = max(0, 1.0 - vpin)
        vwap_score = max(0, min(1, 0.5 - vwap_z * 0.3))
        ofi_score = max(0, min(1, 0.5 + ofi * 0.2))
    else:
        vpin_score = max(0, 1.0 - vpin)
        vwap_score = max(0, min(1, 0.5 + vwap_z * 0.3))
        ofi_score = max(0, min(1, 0.5 - ofi * 0.2))

    return 0.4 * vpin_score + 0.35 * vwap_score + 0.25 * ofi_score


def is_entry_window(ts):
    """First 90 min (9:30-11:00) or last 60 min (15:00-16:00)."""
    if not hasattr(ts, 'hour'):
        return True
    t = ts.hour * 60 + ts.minute
    return (570 <= t <= 660) or (900 <= t <= 960)


def find_intraday_entry(intra_df, trade_date, direction, threshold=0.4):
    """Find best entry point on 15-min bars for trade_date."""
    try:
        if hasattr(intra_df.index, 'date'):
            day_bars = intra_df[intra_df.index.date == trade_date]
        else:
            return None

        if len(day_bars) == 0:
            return None

        best = None
        best_score = 0

        for idx, row in day_bars.iterrows():
            if not is_entry_window(idx):
                continue
            score = score_entry_bar(row, direction)
            if score > threshold and score > best_score:
                best_score = score
                best = {'price': float(row['Close']), 'score': score, 'time': idx}

        return best
    except Exception:
        return None


def simulate_hybrid_trades(watchlist, intraday_data, daily_data,
                           sl_atr_mult=1.5, tp_atr_mult=3.0,
                           max_hold_days=10, entry_threshold=0.4,
                           partial_exit_atr=1.5, cost_pct=0.001):
    """
    Full hybrid backtest.

    For each day: check exits on open positions, then enter new from watchlist.
    Exits use daily close prices + daily ATR brackets.

    Returns list of trade tuples: (pnl_r, resolved, size, ticker, direction)
    compatible with compute_risk_metrics() format.
    """
    trades = []
    positions = {}
    sorted_dates = sorted(watchlist.keys())

    for day_idx, today in enumerate(sorted_dates):

        # -- CHECK EXITS --
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in daily_data:
                to_close.append((ticker, 0.0, True))
                continue

            daily_df = daily_data[ticker]
            current_price = _get_price(daily_df, today)
            if current_price is None:
                continue

            # Count days held
            days_held = sum(1 for d in sorted_dates[max(0, day_idx - pos.get('max_idx', 50)):day_idx]
                          if d > pos['entry_date'])

            # Check daily high/low for SL/TP intraday hits
            daily_high = _get_field(daily_df, today, 'High')
            daily_low = _get_field(daily_df, today, 'Low')

            if pos['direction'] == 'LONG':
                # SL check
                if daily_low and daily_low <= pos['sl_price']:
                    pnl_r = -1.0 - pos['cost_r'] + pos.get('partial_pnl', 0)
                    to_close.append((ticker, pnl_r, True))
                    continue
                # TP check
                if daily_high and daily_high >= pos['tp_price']:
                    rr = tp_atr_mult / sl_atr_mult
                    pnl_r = rr - pos['cost_r'] + pos.get('partial_pnl', 0)
                    to_close.append((ticker, pnl_r, True))
                    continue
                # Partial profit
                partial_level = pos['entry_price'] + partial_exit_atr * pos['daily_atr']
                if daily_high and daily_high >= partial_level and not pos.get('partial_taken'):
                    pos['partial_taken'] = True
                    pos['partial_pnl'] = (1 / 3) * partial_exit_atr / sl_atr_mult
                    pos['sl_price'] = pos['entry_price']  # Move to breakeven
            else:
                if daily_high and daily_high >= pos['sl_price']:
                    pnl_r = -1.0 - pos['cost_r'] + pos.get('partial_pnl', 0)
                    to_close.append((ticker, pnl_r, True))
                    continue
                if daily_low and daily_low <= pos['tp_price']:
                    rr = tp_atr_mult / sl_atr_mult
                    pnl_r = rr - pos['cost_r'] + pos.get('partial_pnl', 0)
                    to_close.append((ticker, pnl_r, True))
                    continue
                partial_level = pos['entry_price'] - partial_exit_atr * pos['daily_atr']
                if daily_low and daily_low <= partial_level and not pos.get('partial_taken'):
                    pos['partial_taken'] = True
                    pos['partial_pnl'] = (1 / 3) * partial_exit_atr / sl_atr_mult
                    pos['sl_price'] = pos['entry_price']

            # Timeout
            if days_held >= max_hold_days:
                r = _calc_r(pos, current_price)
                pnl_r = r - pos['cost_r'] + pos.get('partial_pnl', 0)
                to_close.append((ticker, pnl_r, False))

        for ticker, pnl_r, resolved in to_close:
            pos = positions.pop(ticker, None)
            if pos:
                trades.append((pnl_r, resolved, 1.0, ticker, pos['direction']))

        # -- ENTER NEW POSITIONS --
        signals = watchlist.get(today, {})

        for direction_key in ['longs', 'shorts']:
            direction = 'LONG' if direction_key == 'longs' else 'SHORT'

            for ticker, conviction in signals.get(direction_key, []):
                if ticker in positions:
                    continue
                if ticker not in daily_data:
                    continue

                daily_df = daily_data[ticker]
                daily_atr = _get_field(daily_df, today, 'Daily_ATR')
                if daily_atr is None or daily_atr <= 0:
                    continue

                # Find intraday entry
                entry_info = None
                if ticker in intraday_data:
                    entry_info = find_intraday_entry(
                        intraday_data[ticker], today, direction, entry_threshold)

                if entry_info:
                    entry_price = entry_info['price']
                else:
                    entry_price = _get_price(daily_df, today)
                    if entry_price is None:
                        continue

                sl_dist = sl_atr_mult * daily_atr
                tp_dist = tp_atr_mult * daily_atr

                if direction == 'LONG':
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                cost_r = (cost_pct * entry_price * 2) / sl_dist if sl_dist > 0 else 0

                positions[ticker] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_date': today,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'sl_dist': sl_dist,
                    'daily_atr': daily_atr,
                    'conviction': conviction,
                    'cost_r': cost_r,
                    'partial_taken': False,
                    'partial_pnl': 0.0,
                    'max_idx': max_hold_days + 5,
                }

    # Close remaining at last price
    if sorted_dates:
        last = sorted_dates[-1]
        for ticker, pos in positions.items():
            if ticker in daily_data:
                p = _get_price(daily_data[ticker], last)
                if p:
                    r = _calc_r(pos, p)
                    trades.append((r - pos['cost_r'] + pos.get('partial_pnl', 0),
                                   False, 1.0, ticker, pos['direction']))

    return trades


def _get_price(df, date):
    try:
        if hasattr(df.index, 'date'):
            mask = df.index.date == date
            if mask.any():
                return float(df.loc[mask, 'Close'].iloc[-1])
        if date in df.index:
            return float(df.loc[date, 'Close'])
    except Exception:
        pass
    return None


def _get_field(df, date, field):
    try:
        if hasattr(df.index, 'date'):
            mask = df.index.date == date
            if mask.any() and field in df.columns:
                return float(df.loc[mask, field].iloc[-1])
    except Exception:
        pass
    return None


def _calc_r(pos, price):
    if pos['direction'] == 'LONG':
        return (price - pos['entry_price']) / pos['sl_dist']
    else:
        return (pos['entry_price'] - price) / pos['sl_dist']
