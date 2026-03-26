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

    Iterates over ALL trading days (not just watchlist dates) so SL/TP
    is checked every day. New entries only happen on watchlist dates.

    Returns list of trade tuples: (pnl_r, resolved, size, ticker, direction)
    compatible with compute_risk_metrics() format.
    """
    trades = []
    positions = {}
    recent_outcomes = {'LONG': [], 'SHORT': []}  # Track last N outcomes per direction
    REGIME_LOOKBACK = 10   # Check last 10 trades per direction
    REGIME_MIN_WR = 0.40   # If last 10 trades have <40% WR, reduce exposure
    MAX_CONCURRENT_POSITIONS = 6  # Limit concurrent positions

    # Build complete list of ALL trading days from daily data
    all_dates = set()
    for ticker, df in daily_data.items():
        if hasattr(df.index, 'date'):
            all_dates.update(df.index.date)
        else:
            all_dates.update(df.index)

    # Only keep dates within the watchlist date range
    wl_dates = sorted(watchlist.keys())
    if not wl_dates:
        return trades
    min_date, max_date = wl_dates[0], wl_dates[-1]
    all_trading_days = sorted(d for d in all_dates if min_date <= d <= max_date)

    for today in all_trading_days:

        # -- CHECK EXITS (runs EVERY trading day) --
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in daily_data:
                to_close.append((ticker, 0.0, True))
                continue

            daily_df = daily_data[ticker]
            current_price = _get_price(daily_df, today)
            if current_price is None:
                continue

            # Count ACTUAL trading days held
            days_held = sum(1 for d in all_trading_days
                           if pos['entry_date'] < d <= today)

            # Check daily high/low for SL/TP intraday hits
            daily_high = _get_field(daily_df, today, 'High')
            daily_low = _get_field(daily_df, today, 'Low')

            # Track best price for trailing stop
            if current_price:
                if pos['direction'] == 'LONG':
                    if current_price > pos.get('best_price', 0):
                        pos['best_price'] = current_price
                else:
                    if current_price < pos.get('best_price', float('inf')):
                        pos['best_price'] = current_price

            if pos['direction'] == 'LONG':
                # SL check
                if daily_low and daily_low <= pos['sl_price']:
                    # Compute ACTUAL R-multiple from current SL position
                    actual_r = (pos['sl_price'] - pos['entry_price']) / pos['sl_dist']
                    pnl_r = actual_r - pos['cost_r'] + pos.get('partial_pnl', 0)
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
                # Trailing stop: after partial taken, trail behind best price
                if pos.get('partial_taken') and current_price:
                    if current_price > pos.get('best_price', 0):
                        pos['best_price'] = current_price
                    trail_level = pos['best_price'] - 0.8 * pos['daily_atr']
                    if trail_level > pos['sl_price']:
                        pos['sl_price'] = trail_level
            else:
                if daily_high and daily_high >= pos['sl_price']:
                    # Compute ACTUAL R-multiple from current SL position
                    actual_r = (pos['entry_price'] - pos['sl_price']) / pos['sl_dist']
                    pnl_r = actual_r - pos['cost_r'] + pos.get('partial_pnl', 0)
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
                # Trailing stop for shorts
                if pos.get('partial_taken') and current_price:
                    if current_price < pos.get('best_price', float('inf')):
                        pos['best_price'] = current_price
                    trail_level = pos['best_price'] + 0.8 * pos['daily_atr']
                    if trail_level < pos['sl_price']:
                        pos['sl_price'] = trail_level

            # Timeout
            if days_held >= max_hold_days:
                r = _calc_r(pos, current_price)
                pnl_r = r - pos['cost_r'] + pos.get('partial_pnl', 0)
                to_close.append((ticker, pnl_r, False))

        for ticker, pnl_r, resolved in to_close:
            pos = positions.pop(ticker, None)
            if pos:
                sized_pnl = pnl_r * pos.get('size', 1.0)
                trades.append((sized_pnl, resolved, pos.get('size', 1.0), ticker, pos['direction']))
                # Track recent outcomes for regime detection
                recent_outcomes[pos['direction']].append(1 if pnl_r > 0 else 0)
                if len(recent_outcomes[pos['direction']]) > REGIME_LOOKBACK:
                    recent_outcomes[pos['direction']].pop(0)

        # -- ENTER NEW POSITIONS (only on watchlist dates) --
        signals = watchlist.get(today, {})
        if not signals:
            continue

        # Compute median conviction for confidence sizing
        all_convictions = []
        for dk in ['longs', 'shorts']:
            for _, conv in signals.get(dk, []):
                all_convictions.append(abs(conv))
        median_conviction = float(np.median(all_convictions)) if all_convictions else 1.0
        median_conviction = max(median_conviction, 1e-10)  # avoid division by zero

        for direction_key in ['longs', 'shorts']:
            direction = 'LONG' if direction_key == 'longs' else 'SHORT'

            for ticker, conviction in signals.get(direction_key, []):
                if len(positions) >= MAX_CONCURRENT_POSITIONS:
                    break
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

                # Regime gate: reduce size (don't skip) if recent signals failing
                regime_size = 1.0
                recent = recent_outcomes.get(direction, [])
                if len(recent) >= REGIME_LOOKBACK:
                    recent_wr = sum(recent) / len(recent)
                    if recent_wr < REGIME_MIN_WR:
                        regime_size = 0.5  # Half size, don't skip entirely

                # Confidence sizing: scale by conviction relative to median
                confidence_scalar = min(1.5, abs(conviction) / median_conviction)
                size = 1.0 * regime_size * confidence_scalar

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
                    'best_price': entry_price,
                    'size': size,
                }

    # Close remaining at last price
    if all_trading_days:
        last = all_trading_days[-1]
        for ticker, pos in positions.items():
            if ticker in daily_data:
                p = _get_price(daily_data[ticker], last)
                if p:
                    r = _calc_r(pos, p)
                    pnl_r = r - pos['cost_r'] + pos.get('partial_pnl', 0)
                    sized_pnl = pnl_r * pos.get('size', 1.0)
                    trades.append((sized_pnl, False, pos.get('size', 1.0),
                                   ticker, pos['direction']))

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
