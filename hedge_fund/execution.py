"""
Intraday Execution Engine (Tier 2).

Given a daily watchlist (direction + conviction), finds optimal entry points
on 15-minute bars using microstructure signals.

Entry logic:
- Only trade during entry windows (first 90 min, last 60 min of session)
- Score each 15-min bar using VPIN, VWAP deviation, order flow
- Enter when execution score exceeds threshold
- If no good entry found by end of day, enter at close (TWAP proxy)

Exit logic:
- Stop-loss: N × Daily ATR from entry
- Take-profit: M × Daily ATR from entry
- Timeout: max_hold_days trading days
- Partial exit: 1/3 at partial_exit_atr × Daily ATR profit, move stop to breakeven
"""

import numpy as np
import pandas as pd


def score_entry_bar(row, direction):
    """
    Score a 15-minute bar for entry quality.

    For LONG entries, we want:
    - Low VPIN (no informed selling)
    - VWAP_ZScore < 0 (price below VWAP = buying below average)
    - Positive OFI (buy pressure building)

    For SHORT entries, the opposite.

    Returns score in [0, 1]. Higher = better entry.
    """
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


def is_entry_window(timestamp):
    """
    Check if current time is within an entry window.
    First 90 minutes: 9:30 - 11:00 ET
    Last 60 minutes: 15:00 - 16:00 ET
    """
    if not hasattr(timestamp, 'hour'):
        return True

    h, m = timestamp.hour, timestamp.minute
    t = h * 60 + m

    morning_open = 9 * 60 + 30
    morning_close = 11 * 60
    afternoon_open = 15 * 60
    afternoon_close = 16 * 60

    return (morning_open <= t <= morning_close) or (afternoon_open <= t <= afternoon_close)


def simulate_hybrid_trades(watchlist, intraday_data, daily_data,
                           sl_atr_mult=1.5, tp_atr_mult=3.0,
                           max_hold_days=10, entry_threshold=0.4,
                           partial_exit_atr=1.5, cost_pct=0.0005):
    """
    Full hybrid backtest: daily signals + intraday execution.

    Args:
        watchlist: Output from generate_daily_watchlist()
        intraday_data: Dict of {ticker: 15min_df} with VPIN, OFI, VWAP_ZScore columns
        daily_data: Dict of {ticker: daily_df} with Daily_ATR column
        sl_atr_mult: Stop-loss as multiple of daily ATR
        tp_atr_mult: Take-profit as multiple of daily ATR
        max_hold_days: Maximum holding period in trading days
        entry_threshold: Minimum execution score to enter (0-1)
        partial_exit_atr: ATR multiple for partial profit take
        cost_pct: Round-trip transaction cost as fraction of price

    Returns:
        List of trade dicts with full P&L details
    """
    all_trades = []
    open_positions = {}

    sorted_dates = sorted(watchlist.keys())

    for day_idx, trade_date in enumerate(sorted_dates):
        signals = watchlist[trade_date]

        # ── Process exits for open positions ──
        tickers_to_close = []
        for ticker, pos in open_positions.items():
            if ticker not in daily_data:
                continue

            daily_df = daily_data[ticker]
            entry_date = pos['entry_date']
            days_held = sum(1 for d in sorted_dates
                           if entry_date < d <= trade_date)

            current_price = _get_close_price(daily_df, trade_date)
            if current_price is None:
                continue

            r_current = _calc_r_multiple(pos, current_price)

            if days_held >= max_hold_days:
                pnl_r = r_current - pos['cost_r'] + pos.get('partial_pnl', 0)
                all_trades.append(_make_trade(
                    pos, current_price, trade_date, pnl_r, 'timeout', days_held))
                tickers_to_close.append(ticker)
                continue

            if r_current <= -1.0:
                pnl_r = -1.0 - pos['cost_r']
                all_trades.append(_make_trade(
                    pos, pos['sl_price'], trade_date, pnl_r, 'stop_loss', days_held))
                tickers_to_close.append(ticker)

            elif r_current >= tp_atr_mult / sl_atr_mult:
                rr = tp_atr_mult / sl_atr_mult
                pnl_r = rr - pos['cost_r'] + pos.get('partial_pnl', 0)
                all_trades.append(_make_trade(
                    pos, pos['tp_price'], trade_date, pnl_r, 'take_profit', days_held))
                tickers_to_close.append(ticker)

            elif (r_current >= partial_exit_atr / sl_atr_mult
                  and not pos.get('partial_taken')):
                pos['partial_taken'] = True
                pos['sl_price'] = pos['entry_price']
                pos['partial_pnl'] = (1 / 3) * partial_exit_atr / sl_atr_mult

        for t in tickers_to_close:
            del open_positions[t]

        # ── Process new entries ──
        for direction_key in ['longs', 'shorts']:
            direction = 'LONG' if direction_key == 'longs' else 'SHORT'

            for ticker, conviction in signals.get(direction_key, []):
                if ticker in open_positions:
                    continue

                if ticker not in intraday_data or ticker not in daily_data:
                    continue

                intra_df = intraday_data[ticker]
                daily_df = daily_data[ticker]

                daily_atr = _get_daily_atr(daily_df, trade_date)
                if daily_atr is None or daily_atr <= 0:
                    continue

                entry = _find_intraday_entry(
                    intra_df, trade_date, direction, entry_threshold)

                if entry is None:
                    entry_price = _get_close_price(daily_df, trade_date)
                    if entry_price is None:
                        continue
                    entry_quality = 0.3
                else:
                    entry_price = entry['price']
                    entry_quality = entry['score']

                sl_dist = sl_atr_mult * daily_atr
                tp_dist = tp_atr_mult * daily_atr

                if direction == 'LONG':
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                cost_r = (cost_pct * entry_price) / sl_dist if sl_dist > 0 else 0

                open_positions[ticker] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_date': trade_date,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'sl_dist': sl_dist,
                    'daily_atr': daily_atr,
                    'conviction': conviction,
                    'entry_quality': entry_quality,
                    'cost_r': cost_r,
                    'partial_taken': False,
                    'partial_pnl': 0.0,
                    'ticker': ticker,
                }

    # Close remaining open positions
    for ticker, pos in open_positions.items():
        if ticker in daily_data and sorted_dates:
            last_date = sorted_dates[-1]
            current_price = _get_close_price(daily_data[ticker], last_date)
            if current_price:
                pnl_r = _calc_r_multiple(pos, current_price)
                pnl_r = pnl_r - pos['cost_r'] + pos.get('partial_pnl', 0)
                all_trades.append(_make_trade(
                    pos, current_price, last_date, pnl_r, 'end_of_backtest', 0))

    return all_trades


# ── Helper functions ──

def _make_trade(pos, exit_price, exit_date, pnl_r, exit_type, days_held):
    """Create a standardized trade dict."""
    return {
        'ticker': pos['ticker'],
        'direction': pos['direction'],
        'entry_price': pos['entry_price'],
        'exit_price': exit_price,
        'entry_date': pos['entry_date'],
        'exit_date': exit_date,
        'pnl_r': pnl_r,
        'exit_type': exit_type,
        'conviction': pos['conviction'],
        'days_held': days_held,
        'entry_quality': pos.get('entry_quality', 0),
    }


def _get_close_price(daily_df, date):
    """Get close price for a date, with fuzzy matching."""
    try:
        if hasattr(daily_df.index, 'date'):
            mask = daily_df.index.date == date
            if mask.any():
                return float(daily_df.loc[mask, 'Close'].iloc[-1])
        if date in daily_df.index:
            return float(daily_df.loc[date, 'Close'])
    except Exception:
        pass
    return None


def _get_daily_atr(daily_df, date):
    """Get daily ATR for a date."""
    try:
        if 'Daily_ATR' in daily_df.columns:
            if hasattr(daily_df.index, 'date'):
                mask = daily_df.index.date == date
                if mask.any():
                    return float(daily_df.loc[mask, 'Daily_ATR'].iloc[-1])
            if date in daily_df.index:
                return float(daily_df.loc[date, 'Daily_ATR'])
    except Exception:
        pass
    return None


def _find_intraday_entry(intra_df, trade_date, direction, threshold):
    """
    Scan 15-minute bars on trade_date for best entry point.

    Returns {'price': float, 'score': float, 'time': timestamp} or None.
    """
    try:
        if hasattr(intra_df.index, 'date'):
            day_bars = intra_df[intra_df.index.date == trade_date]
        else:
            return None

        if len(day_bars) == 0:
            return None

        best_entry = None
        best_score = 0

        for idx, row in day_bars.iterrows():
            if not is_entry_window(idx):
                continue

            score = score_entry_bar(row, direction)

            if score > threshold and score > best_score:
                best_score = score
                best_entry = {
                    'price': float(row['Close']),
                    'score': score,
                    'time': idx,
                }

        return best_entry

    except Exception:
        return None


def _calc_pnl(pos, exit_price, cost_pct):
    """Calculate PnL in R-multiples."""
    if pos['direction'] == 'LONG':
        raw = (exit_price - pos['entry_price']) / pos['sl_dist']
    else:
        raw = (pos['entry_price'] - exit_price) / pos['sl_dist']
    return raw - pos['cost_r'] + pos.get('partial_pnl', 0)


def _calc_r_multiple(pos, current_price):
    """Calculate current R-multiple (unrealized)."""
    if pos['direction'] == 'LONG':
        return (current_price - pos['entry_price']) / pos['sl_dist']
    else:
        return (pos['entry_price'] - current_price) / pos['sl_dist']
