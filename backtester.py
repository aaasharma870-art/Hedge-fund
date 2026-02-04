# ==============================================================================
# GOD MODE BACKTESTER V5: STATISTICALLY ROBUST GRID SEARCH
# ==============================================================================
# V5 improvements over V4:
#   1. 3-year backtest window (vs 1 year) for multi-regime coverage
#   2. Walk-forward validation (rolling train/test, no single split)
#   3. Monte Carlo significance test (shuffle labels 1000x)
#   4. Per-ticker results breakdown (detect single-stock dependency)
#   5. Advanced risk metrics (drawdown, Sharpe, Calmar, losing streaks)
#   6. Feature importance pruning (auto-drop bottom 50%)
#   7. Realistic execution costs (spread + commission + impact)
#   8. Diversified universe (tech + healthcare + energy + financials)
# ==============================================================================

import os
import sys
import subprocess
import time
import warnings
import random
import numpy as np
import pandas as pd
import json
import requests
import threading
import concurrent.futures
import datetime

try:
    import xgboost as xgb
    from rich.console import Console
    from rich.table import Table
    from scipy.stats import norm
except ImportError:
    print("Missing deps, running pip install...", flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'xgboost', 'rich', 'scipy'], check=True)
    import xgboost as xgb
    from rich.console import Console
    from rich.table import Table
    from scipy.stats import norm

from hedge_fund.indicators import ManualTA
from hedge_fund.math_utils import get_kalman_filter, get_hurst
from hedge_fund.simulation import simulate_exit as _simulate_exit
from hedge_fund.features import (
    calculate_vpin,
    calculate_enhanced_vwap_features,
    calculate_volatility_regime,
    calculate_amihud_illiquidity,
    calculate_liquidity_sweep,
)
from hedge_fund.objectives import profit_factor_objective
from hedge_fund.data import RateLimiter

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

KEYS = {
    "FMP": os.environ.get("FMP_API_KEY", ""),
    "POLY": os.environ.get("POLYGON_API_KEY", ""),
}

IO_WORKERS = 16

# --- IMPROVEMENT #1: 3 years of data ---
LOOKBACK_DAYS = 1095

# --- IMPROVEMENT #2: Walk-forward settings ---
WF_TRAIN_BARS = 1500   # ~9 months of hourly bars for training
WF_TEST_BARS = 500     # ~3 months of hourly bars for testing
WF_STEP_BARS = 500     # step forward by 3 months each window

# --- IMPROVEMENT #7: Execution cost model ---
SPREAD_COST_PCT = 0.03      # bid-ask spread per side (0.03%)
COMMISSION_PER_SHARE = 0.0  # most brokers are zero-commission
MARKET_IMPACT_PCT = 0.02    # market impact estimate per side
ROUND_TRIP_COST_PCT = 2 * (SPREAD_COST_PCT + MARKET_IMPACT_PCT) / 100  # total as decimal

# --- IMPROVEMENT #8: Diversified universe ---
TICKERS = [
    # Tech / Growth
    'NVDA', 'PLTR', 'TSLA', 'AMD', 'MSFT', 'META',
    # Small-cap momentum (original)
    'RKLB', 'ASTS',
    # Financials
    'JPM', 'GS',
    # Healthcare
    'UNH', 'LLY',
    # Energy
    'XOM', 'CVX',
    # Industrials
    'CAT', 'GE',
    # Consumer
    'AMZN', 'COST',
]

# --- IMPROVEMENT #3: Monte Carlo settings ---
MONTE_CARLO_RUNS = 1000

# --- IMPROVEMENT #6: Feature pruning ---
PRUNE_FEATURES = True
PRUNE_KEEP_RATIO = 0.5  # keep top 50% by importance

GRID_SETTINGS = {
    "PRED_THRESHOLD": [0.20, 0.30],
    "SL_MULT": [1.5],
    "RISK_REWARD": ["1:1.5", "1:2"],
    "MAX_BARS": [8, 12],
    "FILTER_MODE": ["STRICT", "MINIMAL"],
    "TRAIL_MULT": [1.0, 1.5],
    "HURST_LIMIT": [0.45, 0.55],
    "ADX_MIN": [20, 25],
    "DYNAMIC_SIZING": [True],
}

ta = ManualTA
console = Console()


# ==============================================================================
# INFRASTRUCTURE
# ==============================================================================

class Polygon_Helper:
    def __init__(self):
        self.sess = requests.Session()
        self.base = "https://api.polygon.io"
        self.last_429 = 0
        self._lock = threading.Lock()
        self._mem_cache = {}
        self._rate_limiter = RateLimiter(rate_per_sec=12.0, burst=20)

    def _throttle(self):
        self._rate_limiter.acquire()

    def fetch_data(self, t, days=365, mult=1, timespan='hour'):
        with self._lock:
            if time.time() - self.last_429 < 60:
                time.sleep(60 - (time.time() - self.last_429))

        end = datetime.datetime.now(datetime.timezone.utc)
        start = end - datetime.timedelta(days=days)

        print(f"   {t}: Fetching {days}d of {mult}-{timespan} data from Polygon...", flush=True)

        url = (
            f"{self.base}/v2/aggs/ticker/{t}/range/{mult}/{timespan}/"
            f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
            f"?adjusted=true&limit=50000&sort=asc&apiKey={KEYS['POLY']}"
        )

        all_rows = []
        max_retries = 5

        while url:
            retries = 0
            while retries < max_retries:
                try:
                    self._throttle()
                    r = self.sess.get(url, timeout=15)
                    if r.status_code == 200:
                        js = r.json()
                        all_rows.extend(js.get("results", []) or [])
                        url = js.get("next_url")
                        if url and "apiKey=" not in url:
                            url = url + ("&" if "?" in url else "?") + f"apiKey={KEYS['POLY']}"
                        break
                    elif r.status_code == 429:
                        retries += 1
                        with self._lock:
                            self.last_429 = time.time()
                        time.sleep(5 + (retries * 5))
                    else:
                        print(f"   Polygon {t} status {r.status_code}")
                        url = None
                        break
                except Exception as e:
                    retries += 1
                    time.sleep(3)

            if retries >= max_retries:
                url = None

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).rename(columns={
            't': 'Datetime', 'c': 'Close', 'o': 'Open',
            'h': 'High', 'l': 'Low', 'v': 'Volume'
        })
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms', utc=True).dt.tz_convert("America/New_York")
        df = df.set_index('Datetime').sort_index()
        df = df[~df.index.duplicated()]
        return df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

ALL_FEATURES = [
    'RSI', 'ADX', 'ATR_Pct', 'Vol_Rel', 'Kalman_Dist', 'Hurst',
    'BB_Width', 'BB_Position', 'VWAP_Dist', 'HL_Range',
    'ROC_5', 'ROC_20',
    'Vol_Surge', 'Money_Flow',
    'Volatility_Rank', 'Trend_Consistency',
    'Hour', 'Day_of_Week',
    'VPIN', 'VWAP_ZScore', 'VWAP_Slope', 'VWAP_Volume_Ratio',
    'Regime_GEX_Proxy', 'Amihud_Illiquidity', 'Volatility_Regime_Score',
    'RRS_Cumulative', 'Liquidity_Sweep',
]


def prepare_features(df):
    """Compute all 27 features on a DataFrame of OHLCV bars."""
    df = df.copy()

    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    df['Vol_Rel'] = df['Volume'] / df['Volume'].rolling(20).mean()

    df['Kalman'] = get_kalman_filter(df['Close'].values)
    df['Kalman_Dist'] = (df['Close'] - df['Kalman']) / df['Close']

    df['Hurst'] = np.nan
    close_vals = df['Close'].values
    for i in range(50, len(df), 10):
        window = close_vals[i - 50:i]
        df.iloc[i, df.columns.get_loc('Hurst')] = get_hurst(window)
    df['Hurst'] = df['Hurst'].ffill().fillna(0.5)

    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / df['Close']
    df['BB_Position'] = (df['Close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])

    df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['Close']

    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Money_Flow'] = (df['Close'] * df['Volume']).rolling(10).sum()
    df['Money_Flow'] = df['Money_Flow'] / df['Money_Flow'].rolling(50).mean()

    df['ROC_5'] = df['Close'].pct_change(5)
    df['ROC_20'] = df['Close'].pct_change(20)

    try:
        df['VPIN'] = calculate_vpin(df)
    except Exception:
        df['VPIN'] = 0.5

    df['Vol_Surge'] = df['Volume'] / df['Volume'].rolling(5).mean()

    df['Volatility_Rank'] = df['ATR_Pct'].rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5, raw=False)

    _ret = df['Close'].pct_change()
    df['Trend_Consistency'] = _ret.rolling(20).apply(lambda s: (s > 0).mean(), raw=False)

    df['Hour'] = df.index.hour
    df['Day_of_Week'] = df.index.dayofweek

    try:
        vwap_feats = calculate_enhanced_vwap_features(df)
        df['VWAP_ZScore'] = vwap_feats['VWAP_ZScore'].fillna(0.0)
        df['VWAP_Slope'] = vwap_feats['VWAP_Slope'].fillna(0.0)
        df['VWAP_Volume_Ratio'] = vwap_feats['VWAP_Volume_Ratio'].fillna(1.0)
    except Exception:
        df['VWAP_ZScore'] = 0.0
        df['VWAP_Slope'] = 0.0
        df['VWAP_Volume_Ratio'] = 1.0

    try:
        regime_gex, vol_regime_label = calculate_volatility_regime(df)
        df['Regime_GEX_Proxy'] = regime_gex.fillna(0)
        regime_score_map = {'LOW': -1, 'MEDIUM': 0, 'HIGH': 1}
        df['Volatility_Regime_Score'] = vol_regime_label.map(regime_score_map).fillna(0)
    except Exception:
        df['Regime_GEX_Proxy'] = 0
        df['Volatility_Regime_Score'] = 0

    try:
        df['Amihud_Illiquidity'] = calculate_amihud_illiquidity(df, window=20)
    except Exception:
        df['Amihud_Illiquidity'] = 0.5

    df['RRS_Cumulative'] = df['Close'].pct_change(5).rolling(5).sum().fillna(0.0)

    try:
        df['Liquidity_Sweep'] = calculate_liquidity_sweep(df, lookback=16)
    except Exception:
        df['Liquidity_Sweep'] = 0

    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()

    df.dropna(inplace=True)
    return df


# ==============================================================================
# LABEL GENERATION
# ==============================================================================

def compute_bracket_labels(df, sl_mult=1.5, tp_mult=3.0, max_bars=20, atr_col='ATR'):
    """EV-style continuous labels: +R = LONG, -R = SHORT, 0 = HOLD."""
    n = len(df)
    labels = np.zeros(n, dtype=float)
    atr = df[atr_col].values
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    rr = tp_mult / sl_mult

    for i in range(n - max_bars - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = close[i]
        risk = sl_mult * a
        if risk <= 0:
            continue

        long_sl = entry - risk
        long_tp = entry + tp_mult * a
        long_out, _ = _simulate_exit(high[i + 1:i + max_bars + 1], low[i + 1:i + max_bars + 1],
                                     long_sl, long_tp, 'LONG')
        if long_out == 'win':
            long_r = rr
        elif long_out == 'loss':
            long_r = -1.0
        else:
            mtm = (close[min(i + max_bars, n - 1)] - entry) / risk
            long_r = float(np.clip(mtm - 0.05, -1.0, rr))

        short_sl = entry + risk
        short_tp = entry - tp_mult * a
        short_out, _ = _simulate_exit(high[i + 1:i + max_bars + 1], low[i + 1:i + max_bars + 1],
                                      short_sl, short_tp, 'SHORT')
        if short_out == 'win':
            short_r = rr
        elif short_out == 'loss':
            short_r = -1.0
        else:
            mtm = (entry - close[min(i + max_bars, n - 1)]) / risk
            short_r = float(np.clip(mtm - 0.05, -1.0, rr))

        best = max(long_r, short_r)
        if best <= 0.0:
            labels[i] = 0.0
        elif long_r >= short_r:
            labels[i] = float(best)
        else:
            labels[i] = float(-best)

    return labels


# ==============================================================================
# IMPROVEMENT #2: WALK-FORWARD TRAIN + PREDICT
# ==============================================================================

def walk_forward_train_predict(df, features, sl_mult, tp_mult, max_bars,
                               train_bars=WF_TRAIN_BARS, test_bars=WF_TEST_BARS,
                               step_bars=WF_STEP_BARS, prune=PRUNE_FEATURES):
    """
    Walk-forward validation: train on rolling window, predict on next window.
    Returns concatenated test predictions across all windows and the pruned
    feature list (if pruning enabled).
    """
    labels = compute_bracket_labels(df, sl_mult=sl_mult, tp_mult=tp_mult, max_bars=max_bars)
    df = df.copy()
    df['Target'] = labels

    n = len(df)
    all_test_dfs = []
    importance_accum = np.zeros(len(features))
    window_count = 0
    active_features = list(features)

    start = 0
    while start + train_bars + test_bars <= n:
        train_end = start + train_bars
        test_end = min(train_end + test_bars, n)

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end].copy()

        if len(train_df) < 200 or len(test_df) < 30:
            start += step_bars
            continue

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.50,
            colsample_bytree=0.50,
            min_child_weight=10,
            reg_alpha=5.0,
            reg_lambda=10.0,
            gamma=0.5,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(train_df[active_features], train_df['Target'])

        # Accumulate feature importance for pruning
        importance_accum[:len(active_features)] += model.feature_importances_
        window_count += 1

        preds = model.predict(test_df[active_features])
        test_df['Predictions'] = preds

        # Overfitting dampener
        train_r2 = model.score(train_df[active_features], train_df['Target'])
        test_r2 = model.score(test_df[active_features], test_df['Target'])
        if train_r2 - test_r2 > 0.15:
            test_df['Predictions'] = test_df['Predictions'] * 0.90

        all_test_dfs.append(test_df)
        start += step_bars

    if not all_test_dfs:
        return None, features

    # --- IMPROVEMENT #6: Feature pruning ---
    pruned_features = list(active_features)
    if prune and window_count > 0:
        avg_importance = importance_accum[:len(active_features)] / window_count
        feat_imp = sorted(zip(active_features, avg_importance), key=lambda x: x[1], reverse=True)
        keep_n = max(5, int(len(active_features) * PRUNE_KEEP_RATIO))
        pruned_features = [f for f, _ in feat_imp[:keep_n]]
        dropped = [f for f, _ in feat_imp[keep_n:]]
        if dropped:
            print(f"      [Pruned] Dropped {len(dropped)} weak features: {', '.join(dropped[:5])}...")
            # Re-run walk-forward with pruned features for final predictions
            return _walk_forward_pruned(df, pruned_features, sl_mult, tp_mult, max_bars,
                                        train_bars, test_bars, step_bars), pruned_features

    combined = pd.concat(all_test_dfs)
    return combined, pruned_features


def _walk_forward_pruned(df, features, sl_mult, tp_mult, max_bars,
                         train_bars, test_bars, step_bars):
    """Re-run walk-forward with pruned feature set."""
    n = len(df)
    all_test_dfs = []
    start = 0

    while start + train_bars + test_bars <= n:
        train_end = start + train_bars
        test_end = min(train_end + test_bars, n)
        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end].copy()

        if len(train_df) < 200 or len(test_df) < 30:
            start += step_bars
            continue

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.50, colsample_bytree=0.50, min_child_weight=10,
            reg_alpha=5.0, reg_lambda=10.0, gamma=0.5,
            n_jobs=-1, verbosity=0,
        )
        model.fit(train_df[features], train_df['Target'])
        preds = model.predict(test_df[features])
        test_df['Predictions'] = preds

        train_r2 = model.score(train_df[features], train_df['Target'])
        test_r2 = model.score(test_df[features], test_df['Target'])
        if train_r2 - test_r2 > 0.15:
            test_df['Predictions'] = test_df['Predictions'] * 0.90

        all_test_dfs.append(test_df)
        start += step_bars

    if not all_test_dfs:
        return None
    return pd.concat(all_test_dfs)


# ==============================================================================
# TRADE SIMULATION (with execution costs)
# ==============================================================================

def simulate_trades(test_df, pred_threshold, sl_mult, tp_mult, max_bars,
                    trail_mult=None, filter_mode="STRICT",
                    hurst_limit=0.5, adx_min=0, dynamic_sizing=False,
                    execution_cost_pct=ROUND_TRIP_COST_PCT):
    """
    Simulate trades with institutional entry filters and execution costs.
    Returns list of (outcome_r, is_resolved, position_size, ticker).
    """
    if test_df is None or len(test_df) == 0:
        return []

    rr = tp_mult / sl_mult
    trades = []

    close = test_df['Close'].values
    high = test_df['High'].values
    low = test_df['Low'].values
    atr_vals = test_df['ATR'].values
    hurst_vals = test_df['Hurst'].values
    adx_vals = test_df['ADX'].values
    preds = test_df['Predictions'].values
    ticker = test_df.get('_ticker', pd.Series('UNK', index=test_df.index)).values

    i = 0
    while i < len(test_df):
        row = test_df.iloc[i]
        pred_r = row['Predictions']

        if abs(pred_r) < pred_threshold:
            i += 1
            continue

        if row.get('VPIN', 0.0) > 0.85:
            i += 1
            continue

        if filter_mode in ["STRICT", "MODERATE"] and row.get('Amihud_Illiquidity', 0.5) > 0.90:
            i += 1
            continue

        side = 'LONG' if pred_r > 0 else 'SHORT'

        if filter_mode in ["STRICT", "MODERATE"]:
            vwap_z = row.get('VWAP_ZScore', 0.0)
            if abs(vwap_z) > 2.5:
                if vwap_z > 2.5 and side == 'LONG':
                    i += 1
                    continue
                if vwap_z < -2.5 and side == 'SHORT':
                    i += 1
                    continue

            if filter_mode == "STRICT":
                last_close = row['Close']
                ema_200 = row.get('EMA_200', 0)
                if ema_200 > 0:
                    if last_close < ema_200 and side == 'LONG':
                        if row.get('RSI', 50) > 25:
                            i += 1
                            continue
                    if last_close > ema_200 and side == 'SHORT':
                        if row.get('RSI', 50) < 75:
                            i += 1
                            continue

            if filter_mode == "STRICT":
                gex_proxy = row.get('Regime_GEX_Proxy', 0)
                if gex_proxy == 1:
                    if abs(row.get('VWAP_ZScore', 0.0)) < 1.0:
                        i += 1
                        continue
                elif gex_proxy == -1:
                    if abs(row.get('VWAP_Slope', 0.0)) < 0.001:
                        i += 1
                        continue

            if row.get('Volatility_Rank', 0) < 0.5:
                i += 1
                continue

        if hurst_vals[i] > hurst_limit:
            i += 1
            continue

        if adx_vals[i] < adx_min:
            i += 1
            continue

        # --- Trade execution ---
        idx = i
        entry = close[idx]
        a = atr_vals[idx]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue

        sl_dist = sl_mult * a
        tp_dist = tp_mult * a
        sl = entry - sl_dist if side == 'LONG' else entry + sl_dist
        tp = entry + tp_dist if side == 'LONG' else entry - tp_dist

        trail_dist = (trail_mult * a) if trail_mult else None

        end_idx = min(idx + max_bars + 1, len(test_df))
        future_high = high[idx + 1:end_idx]
        future_low = low[idx + 1:end_idx]

        if len(future_high) == 0:
            i += 1
            continue

        outcome_str, exit_price = _simulate_exit(future_high, future_low, sl, tp, side, trail_dist)

        is_resolved = False
        outcome = 0.0

        if outcome_str == 'win':
            outcome = rr
            bars_held = max_bars
            is_resolved = True
        elif outcome_str == 'loss':
            outcome = -1.0
            bars_held = max_bars
            is_resolved = True
        elif outcome_str == 'trail_stop':
            if side == 'LONG':
                raw_pnl = exit_price - entry
            else:
                raw_pnl = entry - exit_price
            outcome = raw_pnl / sl_dist if sl_dist > 0 else 0.0
            bars_held = max_bars
            is_resolved = True
        else:
            exit_idx = min(idx + max_bars, len(close) - 1)
            exit_price_to = close[exit_idx]
            raw = (exit_price_to - entry) if side == 'LONG' else (entry - exit_price_to)
            outcome = raw / sl_dist if sl_dist > 0 else 0.0
            bars_held = max_bars
            is_resolved = False

        # --- IMPROVEMENT #7: Deduct execution costs ---
        cost_in_r = (execution_cost_pct * entry) / sl_dist if sl_dist > 0 else 0
        outcome -= cost_in_r

        # Dynamic sizing
        pos_size = 1.0
        if dynamic_sizing:
            pred_val = preds[idx]
            diff = abs(pred_val) - pred_threshold
            if diff > 0:
                pos_size += (diff * 2.0)
                pos_size = min(pos_size, 2.0)

        final_pnl = outcome * pos_size
        tick = ticker[idx] if idx < len(ticker) else 'UNK'
        trades.append((final_pnl, is_resolved, pos_size, tick))

        i += max(bars_held, 2)

    return trades


# ==============================================================================
# IMPROVEMENT #5: ADVANCED RISK METRICS
# ==============================================================================

def compute_risk_metrics(trades):
    """
    Compute comprehensive risk metrics from a list of trade outcomes.
    Returns dict with PF, WR, Sharpe, MaxDD, Calmar, longest losing streak, etc.
    """
    if not trades:
        return _empty_metrics()

    outcomes = [t[0] for t in trades]
    resolved = [t[0] for t in trades if t[1]]
    n_total = len(outcomes)

    # --- Basic metrics ---
    wins = sum(1 for x in outcomes if x > 0)
    gross_win = sum(x for x in outcomes if x > 0)
    gross_loss = abs(sum(x for x in outcomes if x < 0))
    pf_raw = gross_win / gross_loss if gross_loss > 0 else 0
    wr_raw = wins / n_total if n_total > 0 else 0

    # Resolved metrics
    if resolved:
        res_wins = sum(1 for x in resolved if x > 0)
        gross_win_res = sum(x for x in resolved if x > 0)
        gross_loss_res = abs(sum(x for x in resolved if x < 0))
        pf_res = gross_win_res / gross_loss_res if gross_loss_res > 0 else 0
        wr_res = res_wins / len(resolved)
    else:
        pf_res = 0
        wr_res = 0

    # --- Equity curve ---
    equity = np.cumsum(outcomes)

    # --- Max Drawdown ---
    peak = np.maximum.accumulate(equity)
    drawdowns = equity - peak
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # --- Longest losing streak ---
    longest_loss = 0
    current_loss = 0
    for x in outcomes:
        if x < 0:
            current_loss += 1
            longest_loss = max(longest_loss, current_loss)
        else:
            current_loss = 0

    # --- Sharpe Ratio (annualized, assuming ~6.5 trades/day is wrong; use per-trade) ---
    if len(outcomes) > 1:
        mean_r = np.mean(outcomes)
        std_r = np.std(outcomes, ddof=1)
        sharpe_per_trade = mean_r / std_r if std_r > 0 else 0
        # Annualize: assume ~250 trading days, ~2 trades/day avg
        trades_per_year = min(len(outcomes) * (252 * 6.5 / max(len(outcomes), 1)), 2000)
        sharpe_annual = sharpe_per_trade * np.sqrt(trades_per_year)
    else:
        sharpe_annual = 0

    # --- Calmar Ratio (total return / max drawdown) ---
    total_return = equity[-1] if len(equity) > 0 else 0
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

    # --- Average win / average loss ---
    avg_win = gross_win / wins if wins > 0 else 0
    avg_loss = gross_loss / (n_total - wins) if (n_total - wins) > 0 else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        'PF_Res': pf_res, 'WR_Res': wr_res,
        'PF_Raw': pf_raw, 'WR_Raw': wr_raw,
        'Trades': n_total,
        'Resolved': len(resolved),
        'MaxDD_R': round(max_dd, 2),
        'Sharpe': round(sharpe_annual, 2),
        'Calmar': round(calmar, 2),
        'LongestLoss': longest_loss,
        'TotalReturn_R': round(total_return, 2),
        'AvgWin_R': round(avg_win, 2),
        'AvgLoss_R': round(avg_loss, 2),
        'PayoffRatio': round(payoff_ratio, 2),
    }


def _empty_metrics():
    return {
        'PF_Res': 0, 'WR_Res': 0, 'PF_Raw': 0, 'WR_Raw': 0,
        'Trades': 0, 'Resolved': 0, 'MaxDD_R': 0, 'Sharpe': 0,
        'Calmar': 0, 'LongestLoss': 0, 'TotalReturn_R': 0,
        'AvgWin_R': 0, 'AvgLoss_R': 0, 'PayoffRatio': 0,
    }


# ==============================================================================
# IMPROVEMENT #4: PER-TICKER BREAKDOWN
# ==============================================================================

def per_ticker_breakdown(trades):
    """Group trades by ticker and compute metrics for each."""
    by_ticker = {}
    for t in trades:
        tick = t[3]
        if tick not in by_ticker:
            by_ticker[tick] = []
        by_ticker[tick].append(t)

    breakdown = {}
    for tick, tick_trades in sorted(by_ticker.items()):
        breakdown[tick] = compute_risk_metrics(tick_trades)
    return breakdown


# ==============================================================================
# IMPROVEMENT #3: MONTE CARLO SIGNIFICANCE TEST
# ==============================================================================

def monte_carlo_test(trades, observed_pf, n_simulations=MONTE_CARLO_RUNS):
    """
    Shuffle trade outcomes and re-compute PF to test significance.
    Returns p-value (fraction of shuffled runs that beat observed PF).
    """
    if len(trades) < 10:
        return 1.0  # not enough data

    outcomes = [t[0] for t in trades]
    beat_count = 0

    for _ in range(n_simulations):
        shuffled = list(outcomes)
        random.shuffle(shuffled)
        # Randomly flip signs to destroy any directional signal
        shuffled = [x * random.choice([1, -1]) for x in shuffled]

        gross_win = sum(x for x in shuffled if x > 0)
        gross_loss = abs(sum(x for x in shuffled if x < 0))
        pf_shuffled = gross_win / gross_loss if gross_loss > 0 else 0

        if pf_shuffled >= observed_pf:
            beat_count += 1

    return beat_count / n_simulations


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    console.print("[bold green]GOD MODE BACKTESTER V5 (STATISTICALLY ROBUST)[/bold green]")
    console.print(f"[dim]Lookback: {LOOKBACK_DAYS}d | Walk-Forward: {WF_TRAIN_BARS}/{WF_TEST_BARS}/{WF_STEP_BARS} bars[/dim]")
    console.print(f"[dim]Universe: {len(TICKERS)} tickers | Execution cost: {ROUND_TRIP_COST_PCT*100:.2f}% round-trip[/dim]")
    console.print(f"[dim]Feature pruning: {'ON (keep top '+str(int(PRUNE_KEEP_RATIO*100))+'%)' if PRUNE_FEATURES else 'OFF'}[/dim]")
    console.print(f"[dim]Monte Carlo: {MONTE_CARLO_RUNS} shuffles[/dim]\n")

    # ── 1. Download & Prep ──
    poly = Polygon_Helper()
    data_cache = {}
    print("Downloading data (3-year lookback)...")

    for t in TICKERS:
        try:
            raw = poly.fetch_data(t, days=LOOKBACK_DAYS, mult=1, timespan='hour')
            if len(raw) > 500:
                df_proc = prepare_features(raw)
                df_proc['_ticker'] = t
                data_cache[t] = df_proc
                print(f"   {t}: {len(df_proc)} bars ({df_proc.index[0].date()} to {df_proc.index[-1].date()})")
            else:
                print(f"   {t}: Insufficient data ({len(raw)} bars)")
        except Exception as e:
            print(f"   {t}: Failed ({e})")

    if not data_cache:
        print("No data available. Set POLYGON_API_KEY env var.")
        return

    # ── 2. Grid Search with Walk-Forward ──
    results = []
    n_combos = (len(GRID_SETTINGS["SL_MULT"]) * len(GRID_SETTINGS["RISK_REWARD"]) *
                len(GRID_SETTINGS["MAX_BARS"]) * len(GRID_SETTINGS["PRED_THRESHOLD"]) *
                len(GRID_SETTINGS["FILTER_MODE"]) * len(GRID_SETTINGS["TRAIL_MULT"]) *
                len(GRID_SETTINGS["HURST_LIMIT"]) * len(GRID_SETTINGS["ADX_MIN"]))
    print(f"\nGrid Search: {n_combos} parameter combinations\n")
    ctr = 0

    for sl_m in GRID_SETTINGS["SL_MULT"]:
        for rr_str in GRID_SETTINGS["RISK_REWARD"]:
            reward_ratio = float(rr_str.split(":")[1])
            tp_m = sl_m * reward_ratio

            for mb in GRID_SETTINGS["MAX_BARS"]:
                print(f"   Walk-Forward Training: SL={sl_m} TP={tp_m:.1f} ({rr_str}) MB={mb}...",
                      flush=True)

                # Walk-forward per ticker
                predictions_cache = {}
                pruned_features_cache = {}
                for t, df in data_cache.items():
                    wf_result, pruned_feats = walk_forward_train_predict(
                        df, ALL_FEATURES, sl_m, tp_m, mb)
                    if wf_result is not None:
                        predictions_cache[t] = wf_result
                        pruned_features_cache[t] = pruned_feats

                if not predictions_cache:
                    continue

                # Sweep filters
                for thresh in GRID_SETTINGS["PRED_THRESHOLD"]:
                    for fmode in GRID_SETTINGS["FILTER_MODE"]:
                        for trail in GRID_SETTINGS["TRAIL_MULT"]:
                            for h_lim in GRID_SETTINGS["HURST_LIMIT"]:
                                for adx_m in GRID_SETTINGS["ADX_MIN"]:
                                    ctr += 1
                                    all_trades = []

                                    for t, test_df in predictions_cache.items():
                                        t_results = simulate_trades(
                                            test_df, thresh, sl_m, tp_m, mb,
                                            trail_mult=trail, filter_mode=fmode,
                                            hurst_limit=h_lim, adx_min=adx_m,
                                            dynamic_sizing=True)
                                        all_trades.extend(t_results)

                                    if len(all_trades) < 10:
                                        continue

                                    metrics = compute_risk_metrics(all_trades)

                                    if ctr % 50 == 0 or (metrics['PF_Res'] > 1.3 and metrics['WR_Res'] > 0.45):
                                        print(f"   [{ctr}/{n_combos}] {rr_str} T={thresh} "
                                              f"H<{h_lim} ADX>{adx_m} {fmode} -> "
                                              f"PF={metrics['PF_Res']:.2f} WR={metrics['WR_Res']:.1%} "
                                              f"DD={metrics['MaxDD_R']:.1f}R "
                                              f"N={metrics['Trades']}", flush=True)

                                    result = {
                                        "SL": sl_m, "R:R": rr_str, "MB": mb,
                                        "Thresh": thresh, "Mode": fmode,
                                        "Trail": str(trail), "Hurst": str(h_lim),
                                        "ADX": str(adx_m),
                                        **metrics,
                                        "_trades": all_trades,
                                    }
                                    results.append(result)

    if not results:
        console.print("[red]No results. Check data and parameters.[/red]")
        return

    # ── 3. Sort and display top results ──
    results.sort(key=lambda x: x['PF_Res'], reverse=True)

    print("\n" + "=" * 100)
    print("FINAL RESULTS (SORTED BY RESOLVED PF) - WITH RISK METRICS")
    print("=" * 100)

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"Top 20 Configs ({len(TICKERS)} tickers, {LOOKBACK_DAYS}d, walk-forward)")
    table.add_column("SL")
    table.add_column("R:R")
    table.add_column("MB")
    table.add_column("T")
    table.add_column("Mode")
    table.add_column("Trail")
    table.add_column("H")
    table.add_column("ADX")
    table.add_column("PF", justify="right", style="bold green")
    table.add_column("WR", justify="right")
    table.add_column("N", justify="right")
    table.add_column("MaxDD", justify="right", style="red")
    table.add_column("Sharpe", justify="right", style="cyan")
    table.add_column("Calmar", justify="right")
    table.add_column("LStrk", justify="right")
    table.add_column("Ret", justify="right")
    table.add_column("Stat")

    best_config = None

    for r in results[:20]:
        status = ""
        if r['Trades'] < 30:
            status = "Low N"
        elif r['PF_Res'] > 1.5 and r['WR_Res'] > 0.50 and r['Sharpe'] > 1.0:
            status = "STRONG"
            if not best_config:
                best_config = r
        elif r['PF_Res'] > 1.3 and r['WR_Res'] > 0.48:
            status = "OK"
            if not best_config:
                best_config = r

        table.add_row(
            str(r['SL']), r['R:R'], str(r['MB']),
            str(r['Thresh']), r['Mode'], r['Trail'],
            str(r['Hurst']), str(r['ADX']),
            f"{r['PF_Res']:.2f}", f"{r['WR_Res']:.1%}",
            str(r['Trades']),
            f"{r['MaxDD_R']:.1f}R",
            f"{r['Sharpe']:.1f}",
            f"{r['Calmar']:.1f}",
            str(r['LongestLoss']),
            f"{r['TotalReturn_R']:.1f}R",
            status,
        )

    console.print(table)

    # ── 4. Best config deep-dive ──
    if best_config:
        console.print(f"\n[bold green]BEST CONFIG:[/bold green]")
        console.print(f"   SL_MULT={best_config['SL']}, R:R={best_config['R:R']}, MAX_BARS={best_config['MB']}")
        console.print(f"   THRESHOLD={best_config['Thresh']}, MODE={best_config['Mode']}, TRAIL={best_config['Trail']}")
        console.print(f"   HURST<{best_config['Hurst']}, ADX>{best_config['ADX']}")
        console.print(f"   PF={best_config['PF_Res']:.2f} | WR={best_config['WR_Res']:.1%} | "
                       f"Sharpe={best_config['Sharpe']:.2f} | MaxDD={best_config['MaxDD_R']:.1f}R | "
                       f"Calmar={best_config['Calmar']:.1f}")
        console.print(f"   Trades: {best_config['Trades']} | "
                       f"Avg Win: {best_config['AvgWin_R']:.2f}R | "
                       f"Avg Loss: {best_config['AvgLoss_R']:.2f}R | "
                       f"Payoff: {best_config['PayoffRatio']:.2f}")

        # ── IMPROVEMENT #4: Per-ticker breakdown ──
        console.print(f"\n[bold cyan]PER-TICKER BREAKDOWN:[/bold cyan]")
        ticker_table = Table(show_header=True, header_style="bold cyan")
        ticker_table.add_column("Ticker")
        ticker_table.add_column("PF", justify="right")
        ticker_table.add_column("WR", justify="right")
        ticker_table.add_column("Trades", justify="right")
        ticker_table.add_column("Return", justify="right")
        ticker_table.add_column("MaxDD", justify="right")
        ticker_table.add_column("LStrk", justify="right")
        ticker_table.add_column("Verdict", justify="right")

        breakdown = per_ticker_breakdown(best_config['_trades'])
        profitable_tickers = 0
        for tick, m in breakdown.items():
            verdict = ""
            if m['Trades'] < 5:
                verdict = "Too few"
            elif m['PF_Res'] > 1.3 and m['TotalReturn_R'] > 0:
                verdict = "Profitable"
                profitable_tickers += 1
            elif m['TotalReturn_R'] > 0:
                verdict = "Marginal"
                profitable_tickers += 1
            else:
                verdict = "Losing"

            ticker_table.add_row(
                tick,
                f"{m['PF_Res']:.2f}",
                f"{m['WR_Res']:.1%}",
                str(m['Trades']),
                f"{m['TotalReturn_R']:.1f}R",
                f"{m['MaxDD_R']:.1f}R",
                str(m['LongestLoss']),
                verdict,
            )

        console.print(ticker_table)
        console.print(f"   Profitable tickers: {profitable_tickers}/{len(breakdown)}")

        if profitable_tickers < len(breakdown) * 0.5:
            console.print("[yellow]   WARNING: Edge concentrated in <50% of tickers. "
                          "Strategy may be ticker-dependent, not systematic.[/yellow]")

        # ── IMPROVEMENT #3: Monte Carlo significance ──
        console.print(f"\n[bold yellow]MONTE CARLO SIGNIFICANCE TEST ({MONTE_CARLO_RUNS} shuffles):[/bold yellow]")
        p_value = monte_carlo_test(best_config['_trades'], best_config['PF_Res'])

        if p_value < 0.01:
            sig_label = "[bold green]HIGHLY SIGNIFICANT (p < 0.01)[/bold green]"
        elif p_value < 0.05:
            sig_label = "[green]SIGNIFICANT (p < 0.05)[/green]"
        elif p_value < 0.10:
            sig_label = "[yellow]MARGINAL (p < 0.10)[/yellow]"
        else:
            sig_label = "[red]NOT SIGNIFICANT (p >= 0.10) - LIKELY NOISE[/red]"

        console.print(f"   p-value: {p_value:.4f}")
        console.print(f"   Verdict: {sig_label}")
        console.print(f"   (Probability that random trading achieves PF >= {best_config['PF_Res']:.2f})")

    else:
        console.print("\n[yellow]No config passed minimum thresholds. "
                      "Consider loosening filters or extending data.[/yellow]")

    # ── 5. Summary warnings ──
    console.print(f"\n[dim]{'='*60}[/dim]")
    console.print("[dim]NOTES:[/dim]")
    console.print(f"[dim]  - Execution cost deducted: {ROUND_TRIP_COST_PCT*100:.2f}% per round trip[/dim]")
    console.print(f"[dim]  - Walk-forward windows: train={WF_TRAIN_BARS} test={WF_TEST_BARS} step={WF_STEP_BARS}[/dim]")
    if PRUNE_FEATURES:
        console.print(f"[dim]  - Features auto-pruned to top {int(PRUNE_KEEP_RATIO*100)}% by importance[/dim]")
    console.print(f"[dim]  - Universe: {', '.join(TICKERS)}[/dim]")
    console.print(f"[dim]  - All metrics include unrealized timeout trades[/dim]")


if __name__ == "__main__":
    main()
