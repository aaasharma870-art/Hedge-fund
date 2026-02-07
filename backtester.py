# ==============================================================================
# GOD MODE BACKTESTER V6.3: OPTUNA BAYESIAN + ENSEMBLE + PARETO
# ==============================================================================
# V6.3 additions:
#   1. Anchored (expanding) walk-forward: training window grows from bar 0
#      instead of sliding. Uses all available history for each fold.
# V6.2 additions:
#   1. Ensemble model stacking (XGBoost + LightGBM + Ridge -> meta-learner)
#   2. Multi-objective Optuna: Pareto frontier for PF vs MaxDD
#   3. Config sync: saves optimal params to JSON for bot auto-loading
# V6.1 fixes over V6:
#   1. Label-parameter alignment: SL/TP label buckets match Optuna trials
#   2. Holdout validation: final N bars reserved, Optuna never sees them
#   3. Actual bars_held tracking (was always max_bars, now tracks real exit)
#   4. Parallel data fetching via ThreadPoolExecutor
#   5. Optuna SQLite persistence for cross-run study tracking
# V6 upgrades over V5:
#   1. Optuna Bayesian optimization (replaces grid search)
#   2. Stateful simulation with MonteCarloGovernor equity tracking
#   3. Partial profit simulation (scale out 1/3 at 1.5R)
#   4. Regime-specific logic (Hurst-based Trend vs MeanReversion)
#   5. Dynamic Kelly criterion sizing from walk-forward stats
#   6. SlippageCalculator for realistic execution cost model
# Preserved from V5:
#   - 3-year backtest window, walk-forward validation
#   - Monte Carlo significance test, per-ticker breakdown
#   - Feature importance pruning, diversified universe
# ==============================================================================

import sys
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import subprocess
import time
import warnings
import random
import numpy as np
import pandas as pd
import requests
import threading
import datetime
import concurrent.futures

try:
    import importlib.metadata  # FIX: Required for pandas_ta-openbb
    import pandas_ta as ta
except ImportError:
    try:
        import pandas_ta_openbb as ta
    except ImportError:
        ta = None  # Will be replaced by ManualTA below

from collections import defaultdict, deque
import joblib

try:
    import xgboost as xgb
    from rich.console import Console
    from rich.table import Table
    import optuna
except ImportError:
    print("Missing deps, running pip install...", flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'xgboost', 'rich', 'scipy', 'optuna'], check=True)
    import xgboost as xgb
    from rich.console import Console
    from rich.table import Table
    import optuna

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
from hedge_fund.data import RateLimiter
from hedge_fund.governance import MonteCarloGovernor
from hedge_fund.risk import kelly_criterion, SlippageCalculator
from hedge_fund.config import save_optimal_params
from hedge_fund.ensemble import EnsembleModel

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

KEYS = {
    "FMP": os.environ.get("FMP_API_KEY", ""),
    "POLY": os.environ.get("POLYGON_API_KEY", ""),
}

IO_WORKERS = 16

# --- Data window ---
LOOKBACK_DAYS = 1095  # 3 years

# --- Walk-forward settings ---
WF_TRAIN_BARS = 1500   # ~9 months of hourly bars for training
WF_TEST_BARS = 500     # ~3 months of hourly bars for testing
WF_STEP_BARS = 500     # step forward by 3 months each window

# --- Execution cost model (via SlippageCalculator) ---
SLIPPAGE = SlippageCalculator(spread_pct=0.03, impact_pct=0.02)
ROUND_TRIP_COST_PCT = SLIPPAGE.round_trip_pct()

# --- Diversified universe ---
TICKERS = [
    # Tech / Growth
    'NVDA', 'PLTR', 'TSLA', 'AMD', 'MSFT', 'META',
    # Small-cap momentum
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

# --- Monte Carlo settings ---
MONTE_CARLO_RUNS = 1000

# --- Feature pruning ---
PRUNE_FEATURES = True
PRUNE_KEEP_RATIO = 0.5

# --- Optuna settings ---
OPTUNA_N_TRIALS = 60     # Bayesian optimization trials
OPTUNA_TIMEOUT = None    # No time limit (set to seconds to cap)
OPTUNA_STORAGE = "sqlite:///optuna_backtester.db"  # Persistent study storage

# --- SL/TP label buckets for Optuna ---
# Pre-compute labels for a small set of SL/TP combos so Optuna trials
# use labels that match their simulation parameters.
LABEL_BUCKETS = [
    (1.0, 2.0, 8),
    (1.5, 3.0, 10),
    (2.0, 4.0, 12),
    (2.5, 5.0, 14),
]

# --- Holdout settings ---
HOLDOUT_BARS = 500  # Reserve last ~3 months as final validation

# --- Ensemble settings ---
USE_ENSEMBLE = True  # Use XGBoost + LightGBM + Ridge stacked ensemble

# --- Walk-forward mode ---
ANCHORED_WF = True  # True = expanding (anchored) window; False = rolling fixed-width

# --- Multi-objective settings ---
MULTI_OBJECTIVE = True  # Pareto frontier: maximize PF, minimize MaxDD

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
                except Exception:
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
# WALK-FORWARD TRAIN + PREDICT
# ==============================================================================

def walk_forward_train_predict(df, features, sl_mult, tp_mult, max_bars,
                               train_bars=WF_TRAIN_BARS, test_bars=WF_TEST_BARS,
                               step_bars=WF_STEP_BARS, prune=PRUNE_FEATURES,
                               anchored=ANCHORED_WF):
    """
    Walk-forward validation: train on window, predict on next window.
    anchored=True  -> expanding (anchored) window: train always starts at 0
    anchored=False -> rolling fixed-width window: train slides forward
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

        train_start = 0 if anchored else start
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end].copy()

        if len(train_df) < 200 or len(test_df) < 30:
            start += step_bars
            continue

        if USE_ENSEMBLE:
            model = EnsembleModel()
        else:
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

    # Feature pruning
    pruned_features = list(active_features)
    if prune and window_count > 0:
        avg_importance = importance_accum[:len(active_features)] / window_count
        feat_imp = sorted(zip(active_features, avg_importance), key=lambda x: x[1], reverse=True)
        keep_n = max(5, int(len(active_features) * PRUNE_KEEP_RATIO))
        pruned_features = [f for f, _ in feat_imp[:keep_n]]
        dropped = [f for f, _ in feat_imp[keep_n:]]
        if dropped:
            print(f"      [Pruned] Dropped {len(dropped)} weak features: {', '.join(dropped[:5])}...")
            return _walk_forward_pruned(df, pruned_features, sl_mult, tp_mult, max_bars,
                                        train_bars, test_bars, step_bars, anchored), pruned_features

    combined = pd.concat(all_test_dfs)
    return combined, pruned_features


def _walk_forward_pruned(df, features, sl_mult, tp_mult, max_bars,
                         train_bars, test_bars, step_bars, anchored=ANCHORED_WF):
    """Re-run walk-forward with pruned feature set."""
    n = len(df)
    all_test_dfs = []
    start = 0

    while start + train_bars + test_bars <= n:
        train_end = start + train_bars
        test_end = min(train_end + test_bars, n)
        train_start = 0 if anchored else start
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end].copy()

        if len(train_df) < 200 or len(test_df) < 30:
            start += step_bars
            continue

        if USE_ENSEMBLE:
            model = EnsembleModel()
        else:
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
# V6: STATEFUL TRADE SIMULATION
# ==============================================================================

def simulate_trades_stateful(test_df, pred_threshold, sl_mult, tp_mult, max_bars,
                             trail_mult=None, filter_mode="STRICT",
                             hurst_limit=0.5, adx_min=0,
                             scale_out_r=1.5, use_kelly=True,
                             regime_hurst_filter=True):
    """
    V6 stateful trade simulation with:
    - MonteCarloGovernor for equity-aware risk scaling
    - Partial profit taking (scale out 1/3 at scale_out_r * R)
    - Regime-specific logic (Hurst-based Trend vs MeanReversion)
    - Dynamic Kelly criterion sizing
    - SlippageCalculator execution costs

    Returns list of (outcome_r, is_resolved, position_size, ticker).
    """
    if test_df is None or len(test_df) == 0:
        return []

    rr = tp_mult / sl_mult
    trades = []

    # Stateful governor tracks simulated equity
    governor = MonteCarloGovernor(dd_warning=0.05, dd_critical=0.08,
                                  lookback_trades=50, update_interval=0)

    # Running stats for Kelly
    win_count = 0
    loss_count = 0
    total_win_r = 0.0
    total_loss_r = 0.0

    close = test_df['Close'].values
    high = test_df['High'].values
    low = test_df['Low'].values
    atr_vals = test_df['ATR'].values
    hurst_vals = test_df['Hurst'].values
    adx_vals = test_df['ADX'].values
    ticker = test_df.get('_ticker', pd.Series('UNK', index=test_df.index)).values

    i = 0
    while i < len(test_df):
        row = test_df.iloc[i]
        pred_r = row['Predictions']

        if abs(pred_r) < pred_threshold:
            i += 1
            continue

        # VPIN toxicity filter
        if row.get('VPIN', 0.0) > 0.85:
            i += 1
            continue

        # Amihud illiquidity filter
        if filter_mode in ("STRICT", "MODERATE") and row.get('Amihud_Illiquidity', 0.5) > 0.90:
            i += 1
            continue

        side = 'LONG' if pred_r > 0 else 'SHORT'
        hurst_val = hurst_vals[i]

        # V6: Regime-specific filtering
        if regime_hurst_filter:
            if hurst_val > 0.6:
                # Trending regime: only allow trend-following (momentum)
                if side == 'LONG' and row.get('ROC_5', 0) < 0:
                    i += 1
                    continue
                if side == 'SHORT' and row.get('ROC_5', 0) > 0:
                    i += 1
                    continue
            elif hurst_val < 0.4:
                # Mean-reverting regime: only allow mean-reversion
                if side == 'LONG' and row.get('RSI', 50) > 60:
                    i += 1
                    continue
                if side == 'SHORT' and row.get('RSI', 50) < 40:
                    i += 1
                    continue

        # Standard filters
        if filter_mode in ("STRICT", "MODERATE"):
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

        # V6: Partial profit simulation (scale out 1/3 at scale_out_r)
        partial_r = _simulate_with_partial(
            future_high, future_low, close, idx, max_bars,
            entry, sl, tp, sl_dist, side, trail_dist,
            scale_out_r=scale_out_r, rr=rr
        )

        outcome = partial_r['total_r']
        is_resolved = partial_r['resolved']
        bars_held = partial_r.get('bars_held', max_bars)

        # Deduct execution costs via SlippageCalculator
        cost_r = SLIPPAGE.cost_in_r(entry, sl_dist)
        outcome -= cost_r

        # V6: Dynamic sizing via Kelly + Governor
        pos_size = 1.0

        if use_kelly and win_count + loss_count >= 20:
            wr = win_count / (win_count + loss_count)
            avg_w = total_win_r / win_count if win_count > 0 else rr
            avg_l = total_loss_r / loss_count if loss_count > 0 else 1.0
            kelly_frac = kelly_criterion(wr, avg_w, avg_l, shrinkage=0.35)
            # Scale position: kelly_frac is 0.003-0.03, map to 0.5-2.0x sizing
            pos_size = float(np.clip(kelly_frac / 0.015, 0.5, 2.0))

        # Governor risk scaling
        governor.apply_adjustments()
        pos_size *= governor.get_risk_scalar()

        final_pnl = outcome * pos_size
        tick = ticker[idx] if idx < len(ticker) else 'UNK'
        trades.append((final_pnl, is_resolved, pos_size, tick))

        # Update running stats
        if outcome > 0:
            win_count += 1
            total_win_r += outcome
        else:
            loss_count += 1
            total_loss_r += abs(outcome)

        # Feed governor with simulated PnL
        governor.add_trade(pnl=final_pnl, risk_dollars=1.0, side=side)

        i += max(bars_held, 2)

    return trades


def _simulate_with_partial(future_high, future_low, close, idx, max_bars,
                           entry, sl, tp, sl_dist, side, trail_dist,
                           scale_out_r=1.5, rr=2.0):
    """
    Simulate bracket exit with partial profit taking.

    Scale out 1/3 of position at scale_out_r (e.g. 1.5R), let remaining
    2/3 ride to full TP or trail stop.
    """
    partial_tp_level = entry + (scale_out_r * sl_dist) if side == 'LONG' else entry - (scale_out_r * sl_dist)

    # Check if partial TP is hit first
    partial_hit = False
    for j in range(len(future_high)):
        h, l = future_high[j], future_low[j]
        if side == 'LONG':
            if l <= sl:
                # Full stop loss before any partial
                return {'total_r': -1.0, 'resolved': True, 'bars_held': j + 1}
            if h >= partial_tp_level and not partial_hit:
                partial_hit = True
                break
        else:
            if h >= sl:
                return {'total_r': -1.0, 'resolved': True, 'bars_held': j + 1}
            if l <= partial_tp_level and not partial_hit:
                partial_hit = True
                break

    if not partial_hit:
        # No partial hit, simulate normally
        outcome_str, exit_price = _simulate_exit(future_high, future_low, sl, tp, side, trail_dist)

        # Find actual exit bar
        exit_bar = len(future_high)  # default: timeout at end
        if outcome_str in ('win', 'loss', 'trail_stop'):
            for k in range(len(future_high)):
                fh, fl = future_high[k], future_low[k]
                if side == 'LONG':
                    if fl <= sl or fh >= tp:
                        exit_bar = k + 1
                        break
                else:
                    if fh >= sl or fl <= tp:
                        exit_bar = k + 1
                        break

        if outcome_str == 'win':
            return {'total_r': rr, 'resolved': True, 'bars_held': exit_bar}
        elif outcome_str == 'loss':
            return {'total_r': -1.0, 'resolved': True, 'bars_held': exit_bar}
        elif outcome_str == 'trail_stop':
            if side == 'LONG':
                raw_pnl = exit_price - entry
            else:
                raw_pnl = entry - exit_price
            r_val = raw_pnl / sl_dist if sl_dist > 0 else 0.0
            return {'total_r': r_val, 'resolved': True, 'bars_held': exit_bar}
        else:
            n = len(close)
            exit_idx = min(idx + max_bars, n - 1)
            exit_p = close[exit_idx]
            raw = (exit_p - entry) if side == 'LONG' else (entry - exit_p)
            r_val = raw / sl_dist if sl_dist > 0 else 0.0
            return {'total_r': r_val, 'resolved': False, 'bars_held': max_bars}

    # Partial hit: 1/3 locked at scale_out_r, 2/3 continues
    partial_pnl = (1.0 / 3.0) * scale_out_r

    # Move stop to breakeven for remaining 2/3
    new_sl = entry  # breakeven for both LONG and SHORT

    # Simulate remaining 2/3 from partial hit point onward
    remaining_high = future_high[j + 1:] if j + 1 < len(future_high) else np.array([])
    remaining_low = future_low[j + 1:] if j + 1 < len(future_low) else np.array([])

    if len(remaining_high) == 0:
        # Partial was hit on last bar
        return {'total_r': partial_pnl, 'resolved': True, 'bars_held': j + 1}

    remaining_out, remaining_exit = _simulate_exit(
        remaining_high, remaining_low, new_sl, tp, side, trail_dist
    )

    # Find exit bar in remaining segment
    remaining_exit_bar = len(remaining_high)
    if remaining_out in ('win', 'loss', 'trail_stop'):
        for k in range(len(remaining_high)):
            fh, fl = remaining_high[k], remaining_low[k]
            if side == 'LONG':
                if fl <= new_sl or fh >= tp:
                    remaining_exit_bar = k + 1
                    break
            else:
                if fh >= new_sl or fl <= tp:
                    remaining_exit_bar = k + 1
                    break

    total_bars = j + 1 + remaining_exit_bar

    if remaining_out == 'win':
        remaining_r = rr
    elif remaining_out == 'loss':
        # Stop at breakeven = 0R for remaining
        remaining_r = 0.0
    elif remaining_out == 'trail_stop':
        if side == 'LONG':
            raw = remaining_exit - entry
        else:
            raw = entry - remaining_exit
        remaining_r = raw / sl_dist if sl_dist > 0 else 0.0
    else:
        n = len(close)
        exit_idx = min(idx + max_bars, n - 1)
        exit_p = close[exit_idx]
        raw = (exit_p - entry) if side == 'LONG' else (entry - exit_p)
        remaining_r = raw / sl_dist if sl_dist > 0 else 0.0

    total_r = partial_pnl + (2.0 / 3.0) * remaining_r
    resolved = remaining_out in ('win', 'loss', 'trail_stop')
    return {'total_r': total_r, 'resolved': resolved, 'bars_held': total_bars}


# ==============================================================================
# RISK METRICS
# ==============================================================================

def compute_risk_metrics(trades):
    """Compute comprehensive risk metrics from trade outcomes."""
    if not trades:
        return _empty_metrics()

    outcomes = [t[0] for t in trades]
    resolved = [t[0] for t in trades if t[1]]
    n_total = len(outcomes)

    wins = sum(1 for x in outcomes if x > 0)
    gross_win = sum(x for x in outcomes if x > 0)
    gross_loss = abs(sum(x for x in outcomes if x < 0))
    pf_raw = gross_win / gross_loss if gross_loss > 0 else 0
    wr_raw = wins / n_total if n_total > 0 else 0

    if resolved:
        res_wins = sum(1 for x in resolved if x > 0)
        gross_win_res = sum(x for x in resolved if x > 0)
        gross_loss_res = abs(sum(x for x in resolved if x < 0))
        pf_res = gross_win_res / gross_loss_res if gross_loss_res > 0 else 0
        wr_res = res_wins / len(resolved)
    else:
        pf_res = 0
        wr_res = 0

    equity = np.cumsum(outcomes)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity - peak
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    longest_loss = 0
    current_loss = 0
    for x in outcomes:
        if x < 0:
            current_loss += 1
            longest_loss = max(longest_loss, current_loss)
        else:
            current_loss = 0

    if len(outcomes) > 1:
        mean_r = np.mean(outcomes)
        std_r = np.std(outcomes, ddof=1)
        sharpe_per_trade = mean_r / std_r if std_r > 0 else 0
        trades_per_year = min(len(outcomes) * (252 * 6.5 / max(len(outcomes), 1)), 2000)
        sharpe_annual = sharpe_per_trade * np.sqrt(trades_per_year)
    else:
        sharpe_annual = 0

    total_return = equity[-1] if len(equity) > 0 else 0
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0

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


def monte_carlo_test(trades, observed_pf, n_simulations=MONTE_CARLO_RUNS):
    """
    Shuffle trade outcomes and re-compute PF to test significance.
    Returns p-value (fraction of shuffled runs that beat observed PF).
    """
    if len(trades) < 10:
        return 1.0

    outcomes = [t[0] for t in trades]
    beat_count = 0

    for _ in range(n_simulations):
        shuffled = list(outcomes)
        random.shuffle(shuffled)
        shuffled = [x * random.choice([1, -1]) for x in shuffled]

        gross_win = sum(x for x in shuffled if x > 0)
        gross_loss = abs(sum(x for x in shuffled if x < 0))
        pf_shuffled = gross_win / gross_loss if gross_loss > 0 else 0

        if pf_shuffled >= observed_pf:
            beat_count += 1

    return beat_count / n_simulations


# ==============================================================================
# V6: OPTUNA OBJECTIVE
# ==============================================================================

def _pick_label_bucket(sl_mult, tp_rr):
    """Select the closest pre-computed label bucket for given SL/TP params."""
    tp_mult = sl_mult * tp_rr
    best = None
    best_dist = float('inf')
    for b_sl, b_tp, b_mb in LABEL_BUCKETS:
        dist = abs(sl_mult - b_sl) + abs(tp_mult - b_tp)
        if dist < best_dist:
            best_dist = dist
            best = (b_sl, b_tp, b_mb)
    return best


def create_optuna_objective(predictions_cache_by_bucket):
    """
    Create an Optuna objective function that evaluates parameter combinations.

    Uses label-bucket-aware predictions: each SL/TP bucket has its own
    walk-forward predictions so labels match simulation parameters.
    """
    def objective(trial):
        # Bayesian search space
        pred_threshold = trial.suggest_float("pred_threshold", 0.10, 0.40)
        sl_mult = trial.suggest_float("sl_mult", 1.0, 2.5, step=0.25)
        tp_rr = trial.suggest_float("tp_rr", 1.5, 3.5, step=0.5)
        tp_mult = sl_mult * tp_rr
        max_bars = trial.suggest_int("max_bars", 6, 16, step=2)
        trail_mult = trial.suggest_float("trail_mult", 0.5, 2.0, step=0.25)
        hurst_limit = trial.suggest_float("hurst_limit", 0.35, 0.65, step=0.05)
        adx_min = trial.suggest_int("adx_min", 15, 30, step=5)
        filter_mode = trial.suggest_categorical("filter_mode", ["STRICT", "MODERATE", "MINIMAL"])
        scale_out_r = trial.suggest_float("scale_out_r", 1.0, 2.0, step=0.25)
        regime_hurst = trial.suggest_categorical("regime_hurst_filter", [True, False])

        # Pick the label bucket whose SL/TP best matches this trial
        bucket_key = _pick_label_bucket(sl_mult, tp_rr)
        predictions_cache = predictions_cache_by_bucket.get(bucket_key, {})
        if not predictions_cache:
            return 0.0

        all_trades = []
        for t, test_df in predictions_cache.items():
            t_results = simulate_trades_stateful(
                test_df, pred_threshold, sl_mult, tp_mult, max_bars,
                trail_mult=trail_mult, filter_mode=filter_mode,
                hurst_limit=hurst_limit, adx_min=adx_min,
                scale_out_r=scale_out_r, use_kelly=True,
                regime_hurst_filter=regime_hurst,
            )
            all_trades.extend(t_results)

        if len(all_trades) < 15:
            if MULTI_OBJECTIVE:
                return 0.0, 99.0  # bad PF, bad DD
            return 0.0  # Not enough trades

        metrics = compute_risk_metrics(all_trades)

        pf = metrics['PF_Res']
        wr = metrics['WR_Res']
        n = metrics['Trades']
        dd = abs(metrics['MaxDD_R'])

        # Store metrics for later retrieval
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("trades", all_trades)
        trial.set_user_attr("tp_mult", tp_mult)
        trial.set_user_attr("bucket_key", list(bucket_key))

        if pf < 1.0 or wr < 0.40:
            if MULTI_OBJECTIVE:
                return 0.0, dd if dd > 0 else 99.0
            return 0.0

        if MULTI_OBJECTIVE:
            # Objective 1: maximize composite score (PF * WR * sqrt(N))
            # Objective 2: minimize max drawdown
            score = pf * wr * np.sqrt(n)
            return score, dd

        # Single objective: composite score
        score = pf * wr * np.sqrt(n) / (1.0 + dd)
        return score

    return objective


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    console.print("[bold green]GOD MODE BACKTESTER V6.2 (OPTUNA + ENSEMBLE + PARETO)[/bold green]")
    console.print(f"[dim]Lookback: {LOOKBACK_DAYS}d | Walk-Forward: {WF_TRAIN_BARS}/{WF_TEST_BARS}/{WF_STEP_BARS} bars[/dim]")
    console.print(f"[dim]Universe: {len(TICKERS)} tickers | Execution cost: {ROUND_TRIP_COST_PCT*100:.2f}% round-trip[/dim]")
    console.print(f"[dim]Feature pruning: {'ON (keep top '+str(int(PRUNE_KEEP_RATIO*100))+'%)' if PRUNE_FEATURES else 'OFF'}[/dim]")
    console.print(f"[dim]Optuna trials: {OPTUNA_N_TRIALS} | Monte Carlo: {MONTE_CARLO_RUNS} shuffles[/dim]")
    console.print(f"[dim]Label buckets: {len(LABEL_BUCKETS)} SL/TP combos | Holdout: {HOLDOUT_BARS} bars[/dim]")
    console.print(f"[dim]V6.1 fixes: Label-matched buckets, holdout validation, actual bars_held, parallel fetch[/dim]\n")

    # ── 1. Download & Prep (parallel) ──
    poly = Polygon_Helper()
    data_cache = {}
    print(f"Downloading data (3-year lookback, {IO_WORKERS} workers)...")

    def _fetch_and_prep(t):
        try:
            raw = poly.fetch_data(t, days=LOOKBACK_DAYS, mult=1, timespan='hour')
            if len(raw) > 500:
                df_proc = prepare_features(raw)
                df_proc['_ticker'] = t
                return t, df_proc
            else:
                print(f"   {t}: Insufficient data ({len(raw)} bars)")
                return t, None
        except Exception as e:
            print(f"   {t}: Failed ({e})")
            return t, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
        futures = {executor.submit(_fetch_and_prep, t): t for t in TICKERS}
        for future in concurrent.futures.as_completed(futures):
            t, df_proc = future.result()
            if df_proc is not None:
                data_cache[t] = df_proc
                print(f"   {t}: {len(df_proc)} bars ({df_proc.index[0].date()} to {df_proc.index[-1].date()})")

    if not data_cache:
        print("No data available. Set POLYGON_API_KEY env var.")
        return

    # ── 1b. Split holdout ──
    # Reserve the last HOLDOUT_BARS from each ticker for final validation.
    # Optuna only sees data before the holdout cutoff.
    train_data = {}
    holdout_data = {}
    for t, df in data_cache.items():
        if len(df) > HOLDOUT_BARS + WF_TRAIN_BARS:
            train_data[t] = df.iloc[:-HOLDOUT_BARS]
            holdout_data[t] = df.iloc[-HOLDOUT_BARS:]
            print(f"   {t}: Train {len(train_data[t])} bars | Holdout {len(holdout_data[t])} bars")
        else:
            train_data[t] = df
            holdout_data[t] = None
            print(f"   {t}: All {len(df)} bars used for training (insufficient for holdout)")

    # ── 2. Walk-Forward Training per label bucket ──
    # Train separate models for each SL/TP bucket so labels match Optuna's
    # simulation parameters.
    print(f"\nRunning walk-forward training for {len(LABEL_BUCKETS)} label buckets...")
    predictions_cache_by_bucket = {}

    for bucket_sl, bucket_tp, bucket_mb in LABEL_BUCKETS:
        bucket_key = (bucket_sl, bucket_tp, bucket_mb)
        bucket_cache = {}
        print(f"\n   Bucket SL={bucket_sl} TP={bucket_tp} MB={bucket_mb}:")

        for t, df in train_data.items():
            wf_result, pruned_feats = walk_forward_train_predict(
                df, ALL_FEATURES, bucket_sl, bucket_tp, bucket_mb)
            if wf_result is not None:
                bucket_cache[t] = wf_result
                print(f"      {t}: {len(wf_result)} test predictions")

        if bucket_cache:
            predictions_cache_by_bucket[bucket_key] = bucket_cache
            print(f"   Bucket ({bucket_sl},{bucket_tp},{bucket_mb}): "
                  f"{len(bucket_cache)} tickers ready")

    if not predictions_cache_by_bucket:
        print("Walk-forward produced no predictions. Check data quality.")
        return

    # ── 3. Optuna Bayesian Optimization (with persistence) ──
    mode_str = "multi-objective (PF vs MaxDD)" if MULTI_OBJECTIVE else "single-objective"
    print(f"\nStarting Optuna optimization ({OPTUNA_N_TRIALS} trials, {mode_str})...")

    study_kwargs = {
        "study_name": "backtester_v6",
        "sampler": optuna.samplers.TPESampler(seed=42),
    }

    if MULTI_OBJECTIVE:
        study_kwargs["directions"] = ["maximize", "minimize"]
    else:
        study_kwargs["direction"] = "maximize"

    try:
        study = optuna.create_study(
            storage=OPTUNA_STORAGE,
            load_if_exists=True,
            **study_kwargs,
        )
        print(f"   Optuna study persisted to {OPTUNA_STORAGE}")
    except Exception:
        study = optuna.create_study(**study_kwargs)
        print("   Optuna study in-memory (SQLite unavailable)")

    objective = create_optuna_objective(predictions_cache_by_bucket)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=True)

    # ── 4. Collect results from all trials ──
    results = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        # Handle both single and multi-objective trial values
        if MULTI_OBJECTIVE:
            if trial.values is None or len(trial.values) < 2:
                continue
            score_val = trial.values[0]  # composite
            dd_val = trial.values[1]     # drawdown
            if score_val <= 0:
                continue
            composite_score = score_val / (1.0 + dd_val)
        else:
            if trial.value is None or trial.value <= 0:
                continue
            composite_score = trial.value

        metrics = trial.user_attrs.get("metrics", _empty_metrics())
        all_trades = trial.user_attrs.get("trades", [])
        result = {
            "SL": trial.params.get("sl_mult", 1.5),
            "R:R": f"1:{trial.params.get('tp_rr', 2.0):.1f}",
            "MB": trial.params.get("max_bars", 10),
            "Thresh": round(trial.params.get("pred_threshold", 0.2), 2),
            "Mode": trial.params.get("filter_mode", "STRICT"),
            "Trail": str(trial.params.get("trail_mult", 1.0)),
            "Hurst": str(trial.params.get("hurst_limit", 0.5)),
            "ADX": str(trial.params.get("adx_min", 20)),
            "ScaleOut": str(trial.params.get("scale_out_r", 1.5)),
            "Regime": str(trial.params.get("regime_hurst_filter", True)),
            **metrics,
            "_trades": all_trades,
            "_score": composite_score,
        }
        results.append(result)

    if not results:
        console.print("[red]No results. Check data and parameters.[/red]")
        return

    results.sort(key=lambda x: x['_score'], reverse=True)

    # Show Pareto front if multi-objective
    if MULTI_OBJECTIVE:
        try:
            pareto_trials = optuna.importance.get_pareto_front_trials(study) \
                if hasattr(optuna.importance, 'get_pareto_front_trials') \
                else study.best_trials
            console.print(f"\n[bold cyan]PARETO FRONT: {len(pareto_trials)} non-dominated configs[/bold cyan]")
        except Exception:
            pass

    # ── 5. Display top results ──
    print("\n" + "=" * 110)
    print("FINAL RESULTS (SORTED BY COMPOSITE SCORE) - V6 STATEFUL SIMULATION")
    print("=" * 110)

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"Top 20 Configs ({len(TICKERS)} tickers, {LOOKBACK_DAYS}d, walk-forward, Optuna)")
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

    # ── 6. Best config deep-dive ──
    if best_config:
        console.print(f"\n[bold green]BEST CONFIG:[/bold green]")
        console.print(f"   SL_MULT={best_config['SL']}, R:R={best_config['R:R']}, MAX_BARS={best_config['MB']}")
        console.print(f"   THRESHOLD={best_config['Thresh']}, MODE={best_config['Mode']}, TRAIL={best_config['Trail']}")
        console.print(f"   HURST<{best_config['Hurst']}, ADX>{best_config['ADX']}")
        console.print(f"   ScaleOut={best_config['ScaleOut']}, Regime={best_config['Regime']}")
        console.print(f"   PF={best_config['PF_Res']:.2f} | WR={best_config['WR_Res']:.1%} | "
                       f"Sharpe={best_config['Sharpe']:.2f} | MaxDD={best_config['MaxDD_R']:.1f}R | "
                       f"Calmar={best_config['Calmar']:.1f}")
        console.print(f"   Trades: {best_config['Trades']} | "
                       f"Avg Win: {best_config['AvgWin_R']:.2f}R | "
                       f"Avg Loss: {best_config['AvgLoss_R']:.2f}R | "
                       f"Payoff: {best_config['PayoffRatio']:.2f}")

        # Per-ticker breakdown
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

        # Monte Carlo significance
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

        # ── 7. HOLDOUT VALIDATION ──
        # Evaluate the best config on data Optuna never saw
        holdout_tickers_with_data = {t: h for t, h in holdout_data.items() if h is not None and len(h) > 100}
        if holdout_tickers_with_data:
            console.print(f"\n[bold magenta]HOLDOUT VALIDATION ({HOLDOUT_BARS} bars, unseen by Optuna):[/bold magenta]")

            # Get best config params
            best_sl = best_config['SL']
            best_tp_rr = float(best_config['R:R'].split(':')[1])
            best_tp_mult = best_sl * best_tp_rr
            best_mb = best_config['MB']
            best_thresh = best_config['Thresh']
            best_trail = float(best_config['Trail'])
            best_hurst = float(best_config['Hurst'])
            best_adx = int(best_config['ADX'])
            best_mode = best_config['Mode']
            best_scale = float(best_config['ScaleOut'])
            best_regime = best_config['Regime'] == 'True'

            # Run walk-forward on holdout data to get predictions
            holdout_trades = []
            for t, h_df in holdout_tickers_with_data.items():
                # Use the training data to build a model, then predict on holdout
                full_df = data_cache[t]
                train_portion = full_df.iloc[:-HOLDOUT_BARS]
                if len(train_portion) < WF_TRAIN_BARS:
                    continue

                # Train on last WF_TRAIN_BARS of training data
                train_slice = train_portion.iloc[-WF_TRAIN_BARS:]
                labels = compute_bracket_labels(train_slice, sl_mult=best_sl,
                                                tp_mult=best_tp_mult, max_bars=best_mb)
                train_slice = train_slice.copy()
                train_slice['Target'] = labels

                avail_feats = [f for f in ALL_FEATURES if f in train_slice.columns and f in h_df.columns]
                if len(avail_feats) < 5:
                    continue

                model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=2, learning_rate=0.05,
                    subsample=0.50, colsample_bytree=0.50, min_child_weight=10,
                    reg_alpha=5.0, reg_lambda=10.0, gamma=0.5,
                    n_jobs=-1, verbosity=0,
                )
                model.fit(train_slice[avail_feats], train_slice['Target'])

                holdout_df = h_df.copy()
                holdout_df['Predictions'] = model.predict(holdout_df[avail_feats])

                h_trades = simulate_trades_stateful(
                    holdout_df, best_thresh, best_sl, best_tp_mult, best_mb,
                    trail_mult=best_trail, filter_mode=best_mode,
                    hurst_limit=best_hurst, adx_min=best_adx,
                    scale_out_r=best_scale, use_kelly=True,
                    regime_hurst_filter=best_regime,
                )
                holdout_trades.extend(h_trades)

            if holdout_trades and len(holdout_trades) >= 5:
                h_metrics = compute_risk_metrics(holdout_trades)
                console.print(f"   Trades: {h_metrics['Trades']} | "
                               f"PF: {h_metrics['PF_Res']:.2f} | "
                               f"WR: {h_metrics['WR_Res']:.1%} | "
                               f"Sharpe: {h_metrics['Sharpe']:.2f} | "
                               f"MaxDD: {h_metrics['MaxDD_R']:.1f}R | "
                               f"Return: {h_metrics['TotalReturn_R']:.1f}R")

                # Compare in-sample vs holdout
                is_pf = best_config['PF_Res']
                ho_pf = h_metrics['PF_Res']
                decay = (is_pf - ho_pf) / is_pf * 100 if is_pf > 0 else 0

                if ho_pf >= 1.3 and h_metrics['WR_Res'] >= 0.45:
                    console.print(f"   [bold green]HOLDOUT PASSED[/bold green] "
                                   f"(PF decay: {decay:.0f}%)")
                elif ho_pf >= 1.0:
                    console.print(f"   [yellow]HOLDOUT MARGINAL[/yellow] "
                                   f"(PF decay: {decay:.0f}% - edge weakened out-of-sample)")
                else:
                    console.print(f"   [red]HOLDOUT FAILED[/red] "
                                   f"(PF decay: {decay:.0f}% - likely overfit)")
            else:
                console.print("   [yellow]Not enough holdout trades for evaluation[/yellow]")
        else:
            console.print("\n[yellow]No holdout data available (tickers too short)[/yellow]")

        # ── 8. Save optimal parameters for bot ──
        console.print(f"\n[bold green]SAVING OPTIMAL PARAMETERS...[/bold green]")
        try:
            holdout_m = h_metrics if 'h_metrics' in dir() else None
            config_path = save_optimal_params(best_config, holdout_metrics=holdout_m)
            console.print(f"   Saved to {config_path}")
            console.print(f"   Bot will auto-load these on next startup")
        except Exception as e:
            console.print(f"   [red]Failed to save config: {e}[/red]")

    else:
        console.print("\n[yellow]No config passed minimum thresholds. "
                      "Consider loosening filters or extending data.[/yellow]")

    # ── 9. Summary ──
    console.print(f"\n[dim]{'='*60}[/dim]")
    console.print("[dim]NOTES:[/dim]")
    console.print(f"[dim]  - Execution cost deducted: {ROUND_TRIP_COST_PCT*100:.2f}% per round trip[/dim]")
    console.print(f"[dim]  - Walk-forward windows: train={WF_TRAIN_BARS} test={WF_TEST_BARS} step={WF_STEP_BARS}[/dim]")
    if PRUNE_FEATURES:
        console.print(f"[dim]  - Features auto-pruned to top {int(PRUNE_KEEP_RATIO*100)}% by importance[/dim]")
    console.print(f"[dim]  - Optuna trials: {OPTUNA_N_TRIALS} (TPE Bayesian sampler, persistent)[/dim]")
    console.print(f"[dim]  - Label buckets: {len(LABEL_BUCKETS)} SL/TP combos (labels match simulation)[/dim]")
    console.print(f"[dim]  - Holdout: {HOLDOUT_BARS} bars reserved for out-of-sample validation[/dim]")
    console.print(f"[dim]  - Ensemble: {'XGB+LGB+Ridge stacked' if USE_ENSEMBLE else 'XGBoost only'}[/dim]")
    console.print(f"[dim]  - Optuna mode: {'multi-objective Pareto (PF vs DD)' if MULTI_OBJECTIVE else 'single composite score'}[/dim]")
    console.print(f"[dim]  - V6.2: Partial profits, regime logic, Kelly sizing, MC Governor[/dim]")
    console.print(f"[dim]  - V6.2: Ensemble stacking, Pareto optimization, config sync[/dim]")
    console.print(f"[dim]  - Universe: {', '.join(TICKERS)}[/dim]")
    console.print(f"[dim]  - All metrics include unrealized timeout trades[/dim]")


if __name__ == "__main__":
    main()
