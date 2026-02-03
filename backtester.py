# ==============================================================================
# GOD MODE BACKTESTER: GRID SEARCH OPTIMIZATION (COLAB EDITION V4 - FINAL)
# ==============================================================================
# Purpose: Scientifically find the "Golden Parameters" for >1.5 PF and >50% WR.
#
# CHANGES IN V4 (THE "GOD MODE" UPGRADE):
# - 1:1 PARITY with Main Bot Logic via `WalkForwardAI` port.
# - ADDED: Kalman Filter & Hurst Exponent (Trend vs Mean Reversion).
# - ADDED: Custom "Profit Factor" Objective Function for XGBoost.
# - ADDED: Institutional Features (VPIN, Liquidity Sweeps, RRS).
# - SWITCHED: From Classifier (Up/Down) to Regressor (Predicting R-Value).
#
# This script is the "Truth Teller". If it says a config is bad, it is bad.
# If it says a config is good, it is statistically robust.
# ==============================================================================

import os
import sys
import subprocess
import time
import warnings
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

# Import shared modules from hedge_fund package
from hedge_fund.indicators import ManualTA
from hedge_fund.math_utils import get_kalman_filter, get_hurst
from hedge_fund.simulation import simulate_exit as _simulate_exit, compute_bracket_labels
from hedge_fund.features import (
    calculate_vpin,
    calculate_enhanced_vwap_features,
    calculate_volatility_regime,
    calculate_amihud_illiquidity,
    calculate_liquidity_sweep,
)
from hedge_fund.objectives import profit_factor_objective
from hedge_fund.data import RateLimiter

# Configuration
KEYS = {
    "FMP": os.environ.get("FMP_API_KEY", ""),
    "POLY": os.environ.get("POLYGON_API_KEY", ""),
}

IO_WORKERS = 16

ta = ManualTA
console = Console()

# === 1b. INFRASTRUCTURE (POLYGON HELPER) ===

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
        """
        Fetch Polygon aggregate bars with pagination.
        Defaulting to 1-hour bars for backtester compatibility.
        """
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
                        print(f"   ⚠️ Polygon {t} status {r.status_code}")
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
        # Filter duplicates
        df = df[~df.index.duplicated()]

        return df

# === 2. Math and simulation functions now imported from hedge_fund package ===

# === 3. FEATURE KITCHEN (1:1 Parity with Main Bot) ===

def prepare_god_mode_features(df):
    # 1. Basic Technicals
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']
    df['Vol_Rel'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # 2. Kalman Filter (Trend)
    df['Kalman'] = get_kalman_filter(df['Close'].values)
    df['Kalman_Dist'] = (df['Close'] - df['Kalman']) / df['Close']

    # 3. Hurst Exponent (Market Regiment: Trend vs Chop)
    # Optimized: Compute every 10 bars
    df['Hurst'] = np.nan
    close_vals = df['Close'].values
    for i in range(50, len(df), 10):
        window = close_vals[i-50:i]
        df.iloc[i, df.columns.get_loc('Hurst')] = get_hurst(window)
    df['Hurst'] = df['Hurst'].ffill().fillna(0.5)

    # 4. Bollinger Bands (20, 2)
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / df['Close']
    df['BB_Position'] = (df['Close'] - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])

    # 5. VWAP (Institutional Gravity)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    df['VWAP_Dist'] = (df['Close'] - df['VWAP']) / df['Close']

    # 6. Microstructure
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Money_Flow'] = (df['Close'] * df['Volume']).rolling(10).sum()
    df['Money_Flow'] = df['Money_Flow'] / df['Money_Flow'].rolling(50).mean()

    # 7. Momentum
    df['ROC_5'] = df['Close'].pct_change(5)
    df['ROC_20'] = df['Close'].pct_change(20)

    # 8. VPIN (Toxic Flow)
    try:
        df['VPIN'] = calculate_vpin(df)
    except:
        df['VPIN'] = 0.5

    # 9. Vol_Surge (Volume spike detection - from bot)
    df['Vol_Surge'] = df['Volume'] / df['Volume'].rolling(5).mean()

    # 10. Volatility_Rank (ATR percentile rank - from bot)
    df['Volatility_Rank'] = df['ATR_Pct'].rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5, raw=False)

    # 11. Trend_Consistency (fraction of positive returns - from bot)
    _ret = df['Close'].pct_change()
    df['Trend_Consistency'] = _ret.rolling(20).apply(lambda s: (s > 0).mean(), raw=False)

    # 12. Time features (intraday patterns - from bot)
    df['Hour'] = df.index.hour
    df['Day_of_Week'] = df.index.dayofweek

    # 13. Enhanced VWAP features (institutional mean reversion - from hedge_fund package)
    try:
        vwap_feats = calculate_enhanced_vwap_features(df)
        df['VWAP_ZScore'] = vwap_feats['VWAP_ZScore'].fillna(0.0)
        df['VWAP_Slope'] = vwap_feats['VWAP_Slope'].fillna(0.0)
        df['VWAP_Volume_Ratio'] = vwap_feats['VWAP_Volume_Ratio'].fillna(1.0)
    except:
        df['VWAP_ZScore'] = 0.0
        df['VWAP_Slope'] = 0.0
        df['VWAP_Volume_Ratio'] = 1.0

    # 14. GEX Proxy / Volatility Regime (from bot)
    try:
        regime_gex, vol_regime_label = calculate_volatility_regime(df)
        df['Regime_GEX_Proxy'] = regime_gex.fillna(0)
        regime_score_map = {'LOW': -1, 'MEDIUM': 0, 'HIGH': 1}
        df['Volatility_Regime_Score'] = vol_regime_label.map(regime_score_map).fillna(0)
    except:
        df['Regime_GEX_Proxy'] = 0
        df['Volatility_Regime_Score'] = 0

    # 15. Amihud Illiquidity (from bot)
    try:
        df['Amihud_Illiquidity'] = calculate_amihud_illiquidity(df, window=20)
    except:
        df['Amihud_Illiquidity'] = 0.5

    # 16. RRS Cumulative (momentum proxy without SPY data - from bot)
    df['RRS_Cumulative'] = df['Close'].pct_change(5).rolling(5).sum().fillna(0.0)

    # 17. Liquidity Sweep (institutional reversal signal - from bot)
    try:
        df['Liquidity_Sweep'] = calculate_liquidity_sweep(df, lookback=16)
    except:
        df['Liquidity_Sweep'] = 0

    # Smart Filters (kept for simulation entry logic)
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()

    df.dropna(inplace=True)

    # Feature list matches bot's active_features (minus API-only features)
    features = [
        # Core technicals
        'RSI', 'ADX', 'ATR_Pct', 'Vol_Rel', 'Kalman_Dist', 'Hurst',
        # Price action
        'BB_Width', 'BB_Position', 'VWAP_Dist', 'HL_Range',
        # Momentum
        'ROC_5', 'ROC_20',
        # Volume
        'Vol_Surge', 'Money_Flow',
        # Regime
        'Volatility_Rank', 'Trend_Consistency',
        # Time
        'Hour', 'Day_of_Week',
        # Institutional microstructure
        'VPIN', 'VWAP_ZScore', 'VWAP_Slope', 'VWAP_Volume_Ratio',
        'Regime_GEX_Proxy', 'Amihud_Illiquidity', 'Volatility_Regime_Score',
        # Alpha features
        'RRS_Cumulative', 'Liquidity_Sweep',
    ]
    return df, features

# === 4. SIMULATION ENGINE ===

def compute_bracket_labels(df, sl_mult=1.5, tp_mult=3.0, max_bars=20, atr_col='ATR'):
    """
    EV-style continuous labels with direction encoded in sign.
    PORTED FROM MAIN BOT (compute_bracket_labels) for 1:1 parity.
      +x  => LONG with expected R ~ x
      -x  => SHORT with expected R ~ |x|
       0  => HOLD / no trade (both directions negative EV)
    Timeouts use mark-to-market at horizon (continuous gradient).
    """
    n = len(df)
    labels = np.zeros(n, dtype=float)

    atr = df[atr_col].values
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values

    rr = tp_mult / sl_mult  # e.g. 3.0/1.5 = 2.0R reward

    for i in range(n - max_bars - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        entry = close[i]
        risk = sl_mult * a
        if risk <= 0:
            continue

        # --- LONG bracket ---
        long_sl = entry - risk
        long_tp = entry + tp_mult * a
        long_out, _ = _simulate_exit(high[i+1:i+max_bars+1], low[i+1:i+max_bars+1],
                                  long_sl, long_tp, 'LONG')
        if long_out == 'win':
            long_r = rr
        elif long_out == 'loss':
            long_r = -1.0
        else:
            # Mark-to-market at horizon with small decay for dead money
            mtm = (close[min(i + max_bars, n - 1)] - entry) / risk
            long_r = float(np.clip(mtm - 0.05, -1.0, rr))

        # --- SHORT bracket ---
        short_sl = entry + risk
        short_tp = entry - tp_mult * a
        short_out, _ = _simulate_exit(high[i+1:i+max_bars+1], low[i+1:i+max_bars+1],
                                   short_sl, short_tp, 'SHORT')
        if short_out == 'win':
            short_r = rr
        elif short_out == 'loss':
            short_r = -1.0
        else:
            mtm = (entry - close[min(i + max_bars, n - 1)]) / risk
            short_r = float(np.clip(mtm - 0.05, -1.0, rr))

        # --- Choose best POSITIVE EV only; otherwise HOLD = 0 ---
        best = max(long_r, short_r)
        if best <= 0.0:
            labels[i] = 0.0                    # Both directions negative EV
        elif long_r >= short_r:
            labels[i] = float(best)            # + => go long
        else:
            labels[i] = float(-best)           # - => go short

    df['Target'] = labels
    return df

def train_and_predict(df_orig, features, sl_mult, tp_mult, max_bars):
    """
    Train XGBoost model for given SL/TP/max_bars and return test_df with predictions.
    Uses compute_bracket_labels (ported from bot) for 1:1 target parity.
    Separated from simulation so we train once per param combo.
    """
    df = df_orig.copy()

    df['Target'] = compute_bracket_labels(df, sl_mult=sl_mult, tp_mult=tp_mult, max_bars=max_bars)

    # Train/Test Split (Time Series)
    train_size = int(len(df) * 0.75)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:].copy()

    if len(train_df) < 100 or len(test_df) < 30:
        return None

    # Train Regressor (Highly Regularized to fix Overfitting)
    model = xgb.XGBRegressor(
        n_estimators=100, # Reduced from 300
        max_depth=2,      # Stump-like trees to force generalization
        learning_rate=0.05,
        subsample=0.50,   # Aggressive bagging
        colsample_bytree=0.50, # Aggressive feature sampling
        min_child_weight=10, # Require more data per leaf
        reg_alpha=5.0,    # High L1
        reg_lambda=10.0,  # High L2
        gamma=0.5,        # Min loss reduction
        n_jobs=-1,
        verbosity=0
    )

    model.fit(train_df[features], train_df['Target'])

    # Log Feature Importance (to see what's driving the logic)
    if random.random() < 0.05: # Print only occasionally
        imps = list(zip(features, model.feature_importances_))
        imps.sort(key=lambda x: x[1], reverse=True)
        top_5 = [f"{f}:{score:.2f}" for f, score in imps[:5]]
        print(f"      [Model DNA] Top Features: {', '.join(top_5)}")

    preds = model.predict(test_df[features])
    test_df['Predictions'] = preds

    # Overfitting detection (from bot: R^2 gap check)
    train_r2 = model.score(train_df[features], train_df['Target'])
    test_r2 = model.score(test_df[features], test_df['Target'])
    dampener = 0.90 if (train_r2 - test_r2 > 0.15) else 1.0
    if dampener < 1.0:
        test_df['Predictions'] = test_df['Predictions'] * dampener
        print(f"      (overfit dampener applied: train_R2={train_r2:.3f} test_R2={test_r2:.3f})")

    return test_df


def simulate_trades(test_df, pred_threshold, sl_mult, tp_mult, max_bars, trail_mult=None, filter_mode="STRICT",
                    hurst_limit=0.5, adx_min=0, dynamic_sizing=False):
    """
    Simulate bidirectional trades with bot's institutional entry filters.
    Returns list of tuples: (outcome_r, is_resolved_bool, position_size)
    """
    if test_df is None:
        return []

    rr = tp_mult / sl_mult
    trades = []  # List of (outcome, is_resolved, size)

    close = test_df['Close'].values
    high = test_df['High'].values
    low = test_df['Low'].values
    atr = test_df['ATR'].values
    hurst = test_df['Hurst'].values
    adx = test_df['ADX'].values
    preds = test_df['Predictions'].values

    i = 0
    while i < len(test_df):
        row = test_df.iloc[i]
        pred_r = row['Predictions']

        # ============================================================
        # INSTITUTIONAL ENTRY FILTERS (Ported from main bot)
        # ============================================================

        # 1. Base R-value threshold
        if abs(pred_r) < pred_threshold:
            i += 1
            continue

        # 2. VPIN Filter: Block toxic order flow
        # APPLIES TO: STRICT, MODERATE, MINIMAL (Safety Check)
        if row.get('VPIN', 0.0) > 0.85:
            i += 1
            continue

        # 3. Amihud Filter: Block illiquid conditions
        # APPLIES TO: STRICT, MODERATE
        if filter_mode in ["STRICT", "MODERATE"] and row.get('Amihud_Illiquidity', 0.5) > 0.90:
            i += 1
            continue

        # 4. Hurst Filter (used as a scalar later, not an entry block)
        hurst_val = row.get('Hurst', 0.5)

        # 5. Determine side from prediction sign
        side = 'LONG' if pred_r > 0 else 'SHORT'

        if filter_mode in ["STRICT", "MODERATE"]:
            # 6. VWAP Z-Score extreme filter (bot: block against mean reversion)
            vwap_z = row.get('VWAP_ZScore', 0.0)
            if abs(vwap_z) > 2.5:
                if vwap_z > 2.5 and side == 'LONG':
                    i += 1
                    continue
                if vwap_z < -2.5 and side == 'SHORT':
                    i += 1
                    continue

            # 7. EMA 200 Trend Filter ("Golden Gate" from bot)
            # APPLIES TO: STRICT only
            if filter_mode == "STRICT":
                last_close = row['Close']
                ema_200 = row.get('EMA_200', 0)
                if ema_200 > 0:
                    if last_close < ema_200 and side == 'LONG':
                        if row.get('RSI', 50) > 25:  # Exception: deep oversold bounce
                            i += 1
                            continue
                    if last_close > ema_200 and side == 'SHORT':
                        if row.get('RSI', 50) < 75:  # Exception: deep overbought dump
                            i += 1
                            continue

            # 8. GEX Regime filter (bot: match strategy to regime)
            # APPLIES TO: STRICT only
            if filter_mode == "STRICT":
                gex_proxy = row.get('Regime_GEX_Proxy', 0)
                if gex_proxy == 1:  # Positive GEX = mean reversion
                    if abs(vwap_z) < 1.0:
                        i += 1
                        continue
                elif gex_proxy == -1:  # Negative GEX = trending
                    vwap_slope = row.get('VWAP_Slope', 0.0)
                    if abs(vwap_slope) < 0.001:
                        i += 1
                        continue

            # Filter: Volatility Rank > 0.5 (avoid sleepy markets)
            if row.get('Volatility_Rank', 0) < 0.5:
                i += 1
                continue

        # --- NEW REGIME GATES (Round 6) ---
        # 1. Hurst Exponent Check (Avoid Chop)
        # Hurst < 0.5 = Mean Reverting, Hurst > 0.5 = Trending
        # But commonly Hurst close to 0.5 is Random Walk.
        # We want to AVOID high Hurst if we are Mean Reverting?
        # Actually standard interpretation:
        # H < 0.5: Mean Reverting (Anti-persistent) -> Good for range trading
        # H > 0.5: Trending (Persistent) -> Good for trend following
        # H ~ 0.5: Random Walk -> Dangerous
        # Bot logic often wants H < 0.4 for mean reversion or H > 0.6 for trend.
        # Here we let Grid decide.
        if hurst[i] > hurst_limit:
            i += 1
            continue

        # 2. ADX Check (Trend Strength)
        if adx[i] < adx_min:
            i += 1
            continue

        # ============================================================
        # TRADE EXECUTION (using bot's _simulate_exit)
        # ============================================================
        idx = i
        entry = close[idx]
        a = atr[idx]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue

        sl_dist = sl_mult * a
        tp_dist = tp_mult * a
        sl = entry - sl_dist if side == 'LONG' else entry + sl_dist
        tp = entry + tp_dist if side == 'LONG' else entry - tp_dist

        # Trailing Distance (e.g. 1.5 * ATR)
        trail_dist = (trail_mult * a) if trail_mult else None

        # Simulate bracket exit (SL checked first per bar - matches bot)
        end_idx = min(idx + max_bars + 1, len(test_df))
        future_high = high[idx+1:end_idx]
        future_low = low[idx+1:end_idx]

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
            # Calculate Real PnL based on exit price
            if side == 'LONG':
                raw_pnl = exit_price - entry
            else:
                raw_pnl = entry - exit_price

            outcome = raw_pnl / sl_dist if sl_dist > 0 else 0.0
            bars_held = max_bars
            is_resolved = True
        else:
            # Timeout: mark-to-market (from bot)
            exit_idx = min(idx + max_bars, len(close) - 1)
            exit_price_to = close[exit_idx]
            raw = (exit_price_to - entry) if side == 'LONG' else (entry - exit_price_to)
            outcome = raw / sl_dist if sl_dist > 0 else 0.0
            bars_held = max_bars
            is_resolved = False


        # Note: Bot uses Hurst < 0.38 to halve POSITION SIZE, not R-multiple.
        # For PF/WR measurement, we record raw R-multiples.
        # Hurst filter above already skips truly choppy conditions via feature model.
        # Apply Hurst scalar to outcome (from bot's position sizing logic)
        # Apply Hurst scalar to outcome (from bot's position sizing logic)
        # OLD: hurst_scalar = 0.5 if hurst_val < 0.38 else 1.0

        # NEW: Confidence-Based Sizing (Kelly-lite)
        # If dynamic_sizing=True: Size = 1.0 + (Prob - Threshold) * Scale
        # e.g. Prob 0.8, Thresh 0.6 -> Size = 1.0 + 0.2*2 = 1.4
        pos_size = 1.0
        if dynamic_sizing:
             pred_val = preds[idx]
             diff = abs(pred_val) - pred_threshold
             if diff > 0:
                 pos_size += (diff * 2.0) # Bonus size for high conviction
                 pos_size = min(pos_size, 2.0) # Cap at 2x

        # Calculate final R-multiple PnL
        final_pnl = outcome * pos_size

        # Append tuple (outcome, is_resolved, size)
        trades.append((final_pnl, is_resolved, pos_size))

        i += max(bars_held, 2)  # min 2-bar cooldown

    return trades

# === 5. MAIN EXECUTION ===

GRID_SETTINGS = {
    # Raw prediction threshold
    "PRED_THRESHOLD": [0.20, 0.30],
    # Stop loss ATR multiplier
    "SL_MULT": [1.5],
    # Take profit as ratio of SL
    "RISK_REWARD": ["1:1.5", "1:2"],
    # Max bars to hold (favorites from last round)
    "MAX_BARS": [8, 12],
    # New Filter Modes
    "FILTER_MODE": ["STRICT", "MINIMAL"],
    # Trailing Stop (ATR multiplier)
    "TRAIL_MULT": [1.0, 1.5],
    # Regime Filters (Round 6)
    "HURST_LIMIT": [0.45, 0.55], # Test stricter vs looser chop filter
    "ADX_MIN": [20, 25],
    "DYNAMIC_SIZING": [True]
}

TICKERS = ['NVDA', 'PLTR', 'TSLA', 'AMD', 'MSFT', 'META', 'RKLB', 'ASTS']

def main():
    console.print("[bold green]GOD MODE BACKTESTER V4 (FULL BRAIN)[/bold green]")

    # 1. Download & Prep
    poly = Polygon_Helper()
    data_cache = {}
    print("Downloading high-res data...")
    for t in TICKERS:
        try:
            # Fetch 1 year of hourly data to match original logic
            # Polygon allows direct hourly fetch
            raw = poly.fetch_data(t, days=365, mult=1, timespan='hour')

            if len(raw) > 200:
                df_proc, feats = prepare_god_mode_features(raw)
                data_cache[t] = (df_proc, feats)

                # Show Training Logic
                train_sz = int(len(df_proc) * 0.75)
                train_end = df_proc.index[train_sz]
                test_start = df_proc.index[train_sz]
                print(f"   {t}: Ready ({len(df_proc)} bars)")
                print(f"      [LEARNING]: {df_proc.index[0].date()} -> {train_end.date()}")
                print(f"      [TESTING ]: {test_start.date()} -> {df_proc.index[-1].date()}")
            else:
                print(f"   {t}: Insufficient data ({len(raw)} bars)")
        except Exception as e:
            print(f"   {t}: Failed ({e})")

    if not data_cache:
        print("No data available.")
        return

    # 2. Grid Search
    # Outer loop: (SL_MULT x R:R x MAX_BARS) = model training combos
    # Inner loop: PRED_THRESHOLD + FILTER_MODE + TRAIL_MULT
    results = []
    n_sl = len(GRID_SETTINGS["SL_MULT"])
    n_rr = len(GRID_SETTINGS["RISK_REWARD"])
    n_mb = len(GRID_SETTINGS["MAX_BARS"])
    n_thresh = len(GRID_SETTINGS["PRED_THRESHOLD"])
    n_modes = len(GRID_SETTINGS["FILTER_MODE"])
    n_trail = len(GRID_SETTINGS["TRAIL_MULT"])

    n_train_combos = n_sl * n_rr * n_mb
    total = n_train_combos * n_thresh * n_modes * n_trail
    ctr = 0

    print(f"\nGrid Search: {total} combos ({n_train_combos} model trains x {n_thresh} thresh x {n_modes} features x {n_trail} trails)")
    print(f"   SL_MULT: {GRID_SETTINGS['SL_MULT']}")
    print(f"   R:R: {GRID_SETTINGS['RISK_REWARD']}")
    print(f"   MAX_BARS: {GRID_SETTINGS['MAX_BARS']}")
    print(f"   THRESHOLDS: {GRID_SETTINGS['PRED_THRESHOLD']}")
    print(f"   MODES: {GRID_SETTINGS['FILTER_MODE']}")
    print(f"   TRAILS: {GRID_SETTINGS['TRAIL_MULT']}")

    for sl_m in GRID_SETTINGS["SL_MULT"]:
        for rr_str in GRID_SETTINGS["RISK_REWARD"]:
            reward_ratio = float(rr_str.split(":")[1])
            tp_m = sl_m * reward_ratio

            for mb in GRID_SETTINGS["MAX_BARS"]:
                # Train models once per (SL, R:R, MAX_BARS) combo
                print(f"\n   Training: SL={sl_m} TP={tp_m:.1f} ({rr_str}) MB={mb}...", flush=True)
                predictions_cache = {}
                for t, (df, feats) in data_cache.items():
                    test_df = train_and_predict(df, feats, sl_m, tp_m, mb)
                    if test_df is not None:
                        predictions_cache[t] = test_df

                # Sweep thresholds & modes & trails & regime
                for thresh in GRID_SETTINGS["PRED_THRESHOLD"]:
                    for fmode in GRID_SETTINGS["FILTER_MODE"]:
                        for trail in GRID_SETTINGS["TRAIL_MULT"]:
                            for h_lim in GRID_SETTINGS["HURST_LIMIT"]:
                                for adx_m in GRID_SETTINGS["ADX_MIN"]:

                                    ctr += 1
                                    all_trades = [] # List of (outcome, is_resolved, size)

                                    for t, test_df in predictions_cache.items():
                                        t_results = simulate_trades(test_df, thresh, sl_m, tp_m, mb, trail_mult=trail, filter_mode=fmode,
                                                                  hurst_limit=h_lim, adx_min=adx_m, dynamic_sizing=True)
                                        all_trades.extend(t_results)

                                    if not all_trades:
                                        continue

                                    # RAW METRICS (Diluted by timeouts)
                                    raw_outcomes = [x[0] for x in all_trades]
                                    resol_flags = [x[1] for x in all_trades]

                                    raw_wins = sum(1 for x in raw_outcomes if x > 0)
                                    gross_win_raw = sum(x for x in raw_outcomes if x > 0)
                                    gross_loss_raw = abs(sum(x for x in raw_outcomes if x < 0))
                                    pf_raw = gross_win_raw / gross_loss_raw if gross_loss_raw > 0 else 0
                                    wr_raw = raw_wins / len(raw_outcomes)

                                    # RESOLVED METRICS
                                    resolved_outcomes = [x[0] for x in all_trades if x[1]]
                                    if resolved_outcomes:
                                        res_wins = sum(1 for x in resolved_outcomes if x > 0)
                                        gross_win_res = sum(x for x in resolved_outcomes if x > 0)
                                        gross_loss_res = abs(sum(x for x in resolved_outcomes if x < 0))
                                        pf_res = gross_win_res / gross_loss_res if gross_loss_res > 0 else 0
                                        wr_res = res_wins / len(resolved_outcomes)
                                    else:
                                        pf_res = 0
                                        wr_res = 0

                                    # Print status periodically
                                    if ctr % 100 == 0 or (pf_res > 1.5 and wr_res > 0.45):
                                         t_str = f"Tr={trail}" if trail else "Tr=OFF"
                                         print(f"   [{ctr}/{total}] {rr_str} T={thresh} H<{h_lim} ADX>{adx_m} -> PF={pf_res:.2f} WR={wr_res:.1%} (N={len(all_trades)})", flush=True)

                                    results.append({
                                        "SL": sl_m, "R:R": rr_str, "MB": mb,
                                        "Thresh": thresh, "Mode": fmode, "Trail": str(trail),
                                        "Hurst": str(h_lim), "ADX": str(adx_m),
                                        "PF_Res": pf_res, "WR_Res": wr_res,
                                        "PF_Raw": pf_raw, "WR_Raw": wr_raw,
                                        "Trades": len(all_trades)
                                    })

    # 3. Report
    print("\n" + "="*80)
    print("FINAL RESULTS (SORTED BY RESOLVED PF)")
    print("="*80)

    results.sort(key=lambda x: x['PF_Res'], reverse=True)

    table = Table(show_header=True, header_style="bold magenta", title="Top 40 Configs")
    table.add_column("SL")
    table.add_column("R:R")
    table.add_column("MB")
    table.add_column("Thresh")
    table.add_column("Mode")
    table.add_column("Trail")
    table.add_column("Hurst")
    table.add_column("ADX")
    table.add_column("PF (Res)", justify="right", style="bold green")
    table.add_column("WR (Res)", justify="right")
    table.add_column("PF (Raw)", justify="right", style="dim")
    table.add_column("Trades", justify="right")
    table.add_column("Status", justify="right")

    best_config = None

    for r in results[:40]:  # Top 40
        status = ""
        style = "white"

        if r['Trades'] < 30:
            status = "Low N"
            style = "dim"
        elif r['PF_Res'] > 1.5 and r['WR_Res'] > 0.50:
            status = "HOLY GRAIL"
            if not best_config: best_config = r
        elif r['PF_Res'] > 1.4:
            status = "Strong"
            if not best_config: best_config = r

        table.add_row(
            str(r['SL']),
            r['R:R'],
            str(r['MB']),
            str(r['Thresh']),
            r['Mode'],
            r['Trail'],
            str(r['Hurst']),
            str(r['ADX']),
            f"{r['PF_Res']:.2f}",
            f"{r['WR_Res']:.1%}",
            f"{r['PF_Raw']:.2f}",
            str(r['Trades']),
            status
        )

    console.print(table)

    if best_config:
        console.print(f"\n[bold green]BEST CONFIG:[/bold green]")
        console.print(f"   SL_MULT={best_config['SL']}, R:R={best_config['R:R']}")
        console.print(f"   MAX_BARS={best_config['MB']}, THRESHOLD={best_config['Thresh']}")
        console.print(f"   FILTER_MODE={best_config['Mode']}, TRAIL={best_config['Trail']}")
        console.print(f"   HURST<{best_config['Hurst']}, ADX>{best_config['ADX']}")
        console.print(f"   PF (Res): {best_config['PF_Res']:.2f} | WR (Res): {best_config['WR_Res']:.1%}")
    else:
        console.print("\n[yellow]No Holy Grail found yet, but check top results.[/yellow]")

import random
if __name__ == "__main__":
    main()
