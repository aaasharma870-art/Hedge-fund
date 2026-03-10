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
    compute_ofi,
    compute_rv_ratio,
    compute_momentum_decomp,
    compute_efficiency_ratio,
    compute_vpt_acceleration,
    compute_bar_patterns,
    compute_beta_alpha,
    compute_atr_channel_pos,
    EXPECTED_FEATURES,
)
from hedge_fund.data import RateLimiter
from hedge_fund.governance import MonteCarloGovernor
from hedge_fund.risk import kelly_criterion, SlippageCalculator
from hedge_fund.config import save_optimal_params
from hedge_fund.ensemble import EnsembleModel
from scipy.stats import spearmanr

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
OPTUNA_N_TRIALS = 150    # Bayesian optimization trials
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
MULTI_OBJECTIVE = False  # V7: single composite objective

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

ALL_FEATURES = list(EXPECTED_FEATURES)


# Global market proxy returns - set during data loading
_UNIVERSE_RETURNS = None


def prepare_features(df, universe_returns=None):
    """Compute all features on a DataFrame of OHLCV bars."""
    df = df.copy()

    # Core indicators (kept for internal use but not in feature list)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_Pct'] = df['ATR'] / df['Close']

    # Kalman filter with velocity
    kalman_level, kalman_vel = get_kalman_filter(df['Close'].values, return_velocity=True)
    df['Kalman_Trend'] = (df['Close'] - kalman_level) / df['Close']
    df['Kalman_Velocity'] = kalman_vel

    # Hurst exponent (min 100 bars)
    df['Hurst_Exponent'] = np.nan
    close_vals = df['Close'].values
    for i in range(100, len(df), 10):
        window = close_vals[max(0, i - 100):i]
        df.iloc[i, df.columns.get_loc('Hurst_Exponent')] = get_hurst(window)
    df['Hurst_Exponent'] = df['Hurst_Exponent'].ffill().fillna(0.5)

    # VPIN
    try:
        df['VPIN'] = calculate_vpin(df)
    except Exception:
        df['VPIN'] = 0.5

    # VWAP features
    try:
        vwap_feats = calculate_enhanced_vwap_features(df)
        df['VWAP_ZScore'] = vwap_feats['VWAP_ZScore'].fillna(0.0)
    except Exception:
        df['VWAP_ZScore'] = 0.0

    # GEX Proxy
    try:
        regime_gex, _ = calculate_volatility_regime(df)
        df['GEX_Proxy'] = regime_gex.fillna(0)
    except Exception:
        df['GEX_Proxy'] = 0

    # Amihud
    try:
        df['Amihud_Illiquidity'] = calculate_amihud_illiquidity(df, window=20)
    except Exception:
        df['Amihud_Illiquidity'] = 0.5

    # Relative Return Strength
    df['Relative_Return_Strength'] = df['Close'].pct_change(5).rolling(5).sum().fillna(0.0)

    # NEW features
    df['OFI'] = compute_ofi(df)
    df['RV_Ratio'] = compute_rv_ratio(df)

    og, im = compute_momentum_decomp(df)
    df['Overnight_Gap'] = og
    df['Intraday_Mom'] = im

    df['Efficiency_Ratio'] = compute_efficiency_ratio(df)
    df['VPT_Acceleration'] = compute_vpt_acceleration(df)

    uw, lw, br = compute_bar_patterns(df)
    df['Upper_Wick'] = uw
    df['Lower_Wick'] = lw
    df['Body_Ratio'] = br

    df['ATR_Channel_Pos'] = compute_atr_channel_pos(df)

    # Beta Alpha (requires universe proxy returns)
    proxy = universe_returns if universe_returns is not None else _UNIVERSE_RETURNS
    if proxy is not None:
        proxy_aligned = proxy.reindex(df.index, method='ffill').fillna(0)
        df['Beta_Alpha'] = compute_beta_alpha(df, proxy_aligned)
    else:
        df['Beta_Alpha'] = 0.0

    # Keep ATR for label generation and EMA for reference (not in feature list)
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

    embargo = max_bars  # Prevent label look-ahead leak from train into test
    start = 0
    while start + train_bars + embargo + test_bars <= n:
        train_end = start + train_bars
        test_start = train_end + embargo  # Gap to avoid label leakage
        test_end = min(test_start + test_bars, n)

        train_start = 0 if anchored else start
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end].copy()

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
    embargo = max_bars  # Prevent label look-ahead leak from train into test
    start = 0

    while start + train_bars + embargo + test_bars <= n:
        train_end = start + train_bars
        test_start = train_end + embargo  # Gap to avoid label leakage
        test_end = min(test_start + test_bars, n)
        train_start = 0 if anchored else start
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end].copy()

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

class TradeFilterCounter:
    """Tracks how many candidate signals are killed at each filter stage."""

    def __init__(self):
        self.counts = {
            'raw_bars_evaluated': 0,
            'passed_pred_threshold': 0,
            'killed_by_vpin': 0,
            'killed_by_amihud': 0,
            'killed_by_adx': 0,
            'killed_by_hurst': 0,
            'killed_by_vwap_extreme': 0,
            'killed_by_insufficient_bars': 0,
            'submitted_to_simulation': 0,
            'executed_tp': 0,
            'executed_sl': 0,
            'executed_timeout': 0,
        }

    def increment(self, key, n=1):
        if key in self.counts:
            self.counts[key] += n

    def report(self, trial_num=None):
        label = f"Trial {trial_num}" if trial_num is not None else "Simulation"
        raw = max(self.counts['raw_bars_evaluated'], 1)
        print(f"\n{'='*60}")
        print(f"TRADE FILTER BREAKDOWN - {label}")
        print(f"{'='*60}")
        for k, v in self.counts.items():
            pct = v / raw * 100
            print(f"  {k:<40} {v:>6} ({pct:5.1f}%)")
        executed = (self.counts['executed_tp'] + self.counts['executed_sl'] +
                    self.counts['executed_timeout'])
        print(f"\n  SIGNAL SURVIVAL RATE: {executed}/{raw} = {executed/raw*100:.2f}%")
        if executed == 0:
            print("  WARNING: ZERO TRADES EXECUTED")
        print(f"{'='*60}\n")


def compute_vol_regime_scalar(ticker_returns, short_window=10, long_window=60, max_reduction=0.5):
    """Reduce position size when short-term vol >> long-term vol."""
    rv_short = ticker_returns.rolling(short_window).std()
    rv_long = ticker_returns.rolling(long_window).std().clip(lower=1e-10)
    vol_ratio = rv_short / rv_long
    excess = (vol_ratio - 1.0).clip(lower=0)
    scalar = 1.0 - (1.0 - max_reduction) * (1 - np.exp(-excess * 2))
    return scalar.clip(max_reduction, 1.0)


def compute_confidence_scalar(pred_score, threshold, max_scale=1.5, min_scale=0.75):
    """Scale position size by prediction confidence."""
    if threshold <= 1e-10:
        return 1.0
    confidence_ratio = abs(pred_score) / threshold
    scale = min_scale + (max_scale - min_scale) * np.log(max(confidence_ratio, 1.0)) / np.log(3.0)
    return float(np.clip(scale, min_scale, max_scale))


def simulate_trades_stateful(test_df, pred_threshold, sl_mult, tp_mult, max_bars,
                             trail_mult=None, filter_mode="MINIMAL",
                             hurst_limit=0.5, adx_min=0,
                             scale_out_r=1.5, use_kelly=True,
                             regime_filter_mode='soft',
                             filter_counter=None,
                             direction_bias=1.0):
    """
    V7 stateful trade simulation with:
    - Percentile-based pred_threshold (already computed as raw value)
    - Soft ADX and Hurst filters (reduce size, don't block)
    - Confidence-scaled position sizing
    - Vol regime scalar
    - MonteCarloGovernor for equity-aware risk scaling
    - Partial profit taking (scale out 1/3 at scale_out_r * R)
    - Dynamic Kelly criterion sizing
    - SlippageCalculator execution costs

    Returns list of (outcome_r, is_resolved, position_size, ticker).
    """
    if test_df is None or len(test_df) == 0:
        return []

    rr = tp_mult / sl_mult
    trades = []

    governor = MonteCarloGovernor(dd_warning=0.05, dd_critical=0.08,
                                  lookback_trades=50, update_interval=0)

    win_count = 0
    loss_count = 0
    total_win_r = 0.0
    total_loss_r = 0.0

    close = test_df['Close'].values
    high = test_df['High'].values
    low = test_df['Low'].values
    atr_vals = test_df['ATR'].values
    hurst_vals = test_df.get('Hurst_Exponent', test_df.get('Hurst', pd.Series(0.5, index=test_df.index))).values
    adx_vals = test_df['ADX'].values
    ticker = test_df.get('_ticker', pd.Series('UNK', index=test_df.index)).values

    # Compute vol regime scalars
    returns_series = pd.Series(close).pct_change()
    vol_scalars = compute_vol_regime_scalar(returns_series).values

    fc = filter_counter

    i = 0
    while i < len(test_df):
        if fc:
            fc.increment('raw_bars_evaluated')

        row = test_df.iloc[i]
        pred_r = row['Predictions']

        if abs(pred_r) < pred_threshold:
            i += 1
            continue

        if fc:
            fc.increment('passed_pred_threshold')

        # VPIN toxicity filter
        if row.get('VPIN', 0.0) > 0.85:
            if fc:
                fc.increment('killed_by_vpin')
            i += 1
            continue

        # Amihud illiquidity filter
        if row.get('Amihud_Illiquidity', 0.5) > 0.90:
            if fc:
                fc.increment('killed_by_amihud')
            i += 1
            continue

        side = 'LONG' if pred_r > 0 else 'SHORT'

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
            if fc:
                fc.increment('killed_by_insufficient_bars')
            i += 1
            continue

        if fc:
            fc.increment('submitted_to_simulation')

        # Simulate exit with partial profit
        partial_r = _simulate_with_partial(
            future_high, future_low, close, idx, max_bars,
            entry, sl, tp, sl_dist, side, trail_dist,
            scale_out_r=scale_out_r, rr=rr
        )

        outcome = partial_r['total_r']
        is_resolved = partial_r['resolved']
        bars_held = partial_r.get('bars_held', max_bars)

        # Track outcome type
        if fc:
            if outcome > 0:
                fc.increment('executed_tp')
            elif outcome < 0:
                fc.increment('executed_sl')
            else:
                fc.increment('executed_timeout')

        # Deduct execution costs
        cost_r = SLIPPAGE.cost_in_r(entry, sl_dist)
        outcome -= cost_r

        # --- Position sizing ---
        pos_size = 1.0

        # Kelly sizing
        if use_kelly and win_count + loss_count >= 20:
            wr = win_count / (win_count + loss_count)
            avg_w = total_win_r / win_count if win_count > 0 else rr
            avg_l = total_loss_r / loss_count if loss_count > 0 else 1.0
            kelly_frac = kelly_criterion(wr, avg_w, avg_l, shrinkage=0.35)
            pos_size = float(np.clip(kelly_frac / 0.015, 0.5, 2.0))

        # Direction bias: reduce short position sizes
        if side == 'SHORT' and direction_bias < 1.0:
            pos_size *= direction_bias

        # Confidence scalar
        conf_scalar = compute_confidence_scalar(abs(pred_r), pred_threshold)
        pos_size *= conf_scalar

        # ADX soft filter (reduce size when ADX < adx_min, don't block)
        if adx_min > 0 and adx_vals[idx] < adx_min:
            adx_scalar = 0.5 + 0.5 * (adx_vals[idx] / max(adx_min, 1e-6))
            pos_size *= adx_scalar

        # Hurst directional soft filter
        hurst_val = hurst_vals[idx]
        pred_direction = 1 if pred_r > 0 else -1
        if regime_filter_mode == 'soft':
            if pred_direction > 0:
                hurst_scalar = 0.5 + 0.5 * (hurst_val / max(hurst_limit, 0.01))
                hurst_scalar = min(1.5, hurst_scalar)
            else:
                hurst_scalar = 0.5 + 0.5 * ((1 - hurst_val) / max(1 - hurst_limit, 0.01))
                hurst_scalar = min(1.5, hurst_scalar)
            pos_size *= max(0.3, hurst_scalar)
        else:
            # Hard mode
            if hurst_val > hurst_limit:
                if fc:
                    fc.increment('killed_by_hurst')
                i += 1
                continue

        # Vol regime scalar
        if idx < len(vol_scalars):
            pos_size *= vol_scalars[idx] if np.isfinite(vol_scalars[idx]) else 1.0

        # Governor risk scaling
        governor.apply_adjustments()
        pos_size *= governor.get_risk_scalar()

        # Cap position size
        pos_size = min(pos_size, 3.0)

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
        # Normalize R-multiples to portfolio returns assuming 2% Kelly risk per trade
        normalized_returns = [o * 0.02 for o in outcomes]
        mean_r = np.mean(normalized_returns)
        std_r = np.std(normalized_returns, ddof=1)
        sharpe_per_trade = mean_r / std_r if std_r > 1e-10 else 0
        # Annualize: assume ~4 trades per day on average across universe
        sharpe_annual = sharpe_per_trade * np.sqrt(252 * 4)
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
    Sign-randomization test for profit factor significance.

    Under the null hypothesis that the strategy has no directional edge,
    each trade's sign (win vs loss) is equally likely. We randomly reassign
    signs while preserving trade magnitudes, then compute PF.

    FIX: The original test both shuffled order AND flipped signs independently.
    Order shuffling is unnecessary (PF is order-invariant). Sign-randomization
    alone is the correct null hypothesis test for directional edge.

    Returns p-value (fraction of randomized runs that beat observed PF).
    """
    if len(trades) < 10:
        return 1.0

    outcomes = [t[0] for t in trades]
    magnitudes = [abs(x) for x in outcomes]
    beat_count = 0

    for _ in range(n_simulations):
        # Randomly assign signs to magnitudes (null = no directional edge)
        randomized = [m * random.choice([1, -1]) for m in magnitudes]

        gross_win = sum(x for x in randomized if x > 0)
        gross_loss = abs(sum(x for x in randomized if x < 0))
        pf_randomized = gross_win / gross_loss if gross_loss > 0 else 0

        if pf_randomized >= observed_pf:
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
    Create an Optuna objective function with composite scoring.

    Uses percentile-based pred_threshold to guarantee trades.
    Single objective with composite score (PF + Sharpe + DD + trades + consistency).
    """
    def objective(trial):
        # V7 search space
        pred_threshold_pct = trial.suggest_float("pred_threshold_pct", 0.55, 0.85)
        sl_mult = trial.suggest_float("sl_mult", 0.8, 3.0)
        tp_rr = trial.suggest_float("tp_rr", 1.5, 5.0)
        tp_mult = sl_mult * tp_rr
        max_bars = trial.suggest_int("max_bars", 4, 24)
        trail_mult = trial.suggest_float("trail_mult", 0.5, 2.5)
        hurst_limit = trial.suggest_float("hurst_limit", 0.45, 0.75)
        adx_min = trial.suggest_float("adx_min", 0, 18)
        scale_out_r = trial.suggest_float("scale_out_r", 1.0, 3.0)
        regime_filter_mode = trial.suggest_categorical("regime_filter_mode", ["soft", "hard"])
        lambda_decay = trial.suggest_float("lambda_decay", 0.5, 4.0)
        direction_bias = trial.suggest_float("direction_bias", 0.3, 1.0)

        # Pick the label bucket whose SL/TP best matches this trial
        bucket_key = _pick_label_bucket(sl_mult, tp_rr)
        predictions_cache = predictions_cache_by_bucket.get(bucket_key, {})
        if not predictions_cache:
            return -5.0

        all_trades = []
        per_ticker_trades = {}
        tickers_with_trades = set()
        filter_counter = TradeFilterCounter()

        # Compute per-ticker prediction confidence for weighting
        all_pred_stds = []
        for t, test_df in predictions_cache.items():
            preds = test_df['Predictions'].dropna()
            if len(preds) > 10:
                all_pred_stds.append((t, float(np.std(preds))))
        universe_avg_confidence = np.mean([s for _, s in all_pred_stds]) if all_pred_stds else 1.0
        ticker_confidence_weights = {t: s / max(universe_avg_confidence, 1e-10) for t, s in all_pred_stds}

        for t, test_df in predictions_cache.items():
            # Compute percentile-based threshold from this ticker's predictions
            all_preds_abs = np.abs(test_df['Predictions'].values)
            valid_preds = all_preds_abs[np.isfinite(all_preds_abs)]
            if len(valid_preds) < 10:
                continue
            pred_threshold_raw = np.percentile(valid_preds, pred_threshold_pct * 100)

            t_results = simulate_trades_stateful(
                test_df, pred_threshold_raw, sl_mult, tp_mult, max_bars,
                trail_mult=trail_mult, filter_mode="MINIMAL",
                hurst_limit=hurst_limit, adx_min=adx_min,
                scale_out_r=scale_out_r, use_kelly=True,
                regime_filter_mode=regime_filter_mode,
                filter_counter=filter_counter,
                direction_bias=direction_bias,
            )
            all_trades.extend(t_results)
            if t_results:
                tickers_with_trades.add(t)
                per_ticker_trades[t] = t_results

        n_trades = len(all_trades)

        # Hard prune: too few trades (30 = ~1.7 per ticker minimum)
        if n_trades < 30:
            trial.report(-5.0, step=0)
            raise optuna.TrialPruned()

        metrics = compute_risk_metrics(all_trades)

        pf = metrics.get('PF_Raw', metrics['PF_Res'])
        wr = metrics['WR_Raw']
        dd = abs(metrics['MaxDD_R'])
        sharpe = metrics['Sharpe']
        avg_dd_per_trade = dd / max(n_trades, 1)

        if wr < 0.28:
            trial.report(-2.0, step=0)
            raise optuna.TrialPruned()

        # Composite score
        pf_score = np.clip(np.log(max(pf, 0.01)) / np.log(4.0), -1.0, 1.0)
        sharpe_score = np.clip(min(sharpe, 3.0) / 3.0, -1.0, 1.0)
        dd_score = np.clip(1.0 - (avg_dd_per_trade / 3.0), -1.0, 1.0)
        trade_score = np.clip(np.log(max(n_trades, 1) / 30) / np.log(500 / 30), 0.0, 1.0)
        ticker_score = np.clip(len(tickers_with_trades) / 5.0, 0.0, 1.0)

        # Per-ticker consistency: fraction of tickers with PF > 1.0
        ticker_pfs = {}
        for t, t_results in per_ticker_trades.items():
            if len(t_results) >= 5:
                t_metrics = compute_risk_metrics(t_results)
                ticker_pfs[t] = t_metrics.get('PF_Raw', 0)
        n_profitable_tickers = sum(1 for pf_val in ticker_pfs.values() if pf_val > 1.0)
        consistency_score = n_profitable_tickers / max(len(ticker_pfs), 1)

        composite = (
            0.28 * pf_score +
            0.22 * sharpe_score +
            0.18 * dd_score +
            0.15 * trade_score +
            0.12 * consistency_score +
            0.05 * ticker_score
        )

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("trades", all_trades)
        trial.set_user_attr("tp_mult", tp_mult)
        trial.set_user_attr("bucket_key", list(bucket_key))
        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("pf", round(pf, 4))
        trial.set_user_attr("sharpe", round(sharpe, 4))
        trial.set_user_attr("max_dd", round(dd, 4))
        trial.set_user_attr("win_rate", round(wr, 4))
        trial.set_user_attr("n_tickers_trading", len(tickers_with_trades))
        trial.set_user_attr("pred_threshold_pct", pred_threshold_pct)
        trial.set_user_attr("n_profitable_tickers", n_profitable_tickers)
        trial.set_user_attr("ticker_pfs", {k: round(v, 3) for k, v in ticker_pfs.items()})

        return composite

    return objective


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global _UNIVERSE_RETURNS

    console.print("[bold green]GOD MODE BACKTESTER V7.0 (COMPLETE OVERHAUL)[/bold green]")
    console.print(f"[dim]Lookback: {LOOKBACK_DAYS}d | Walk-Forward: {WF_TRAIN_BARS}/{WF_TEST_BARS}/{WF_STEP_BARS} bars[/dim]")
    console.print(f"[dim]Universe: {len(TICKERS)} tickers | Execution cost: {ROUND_TRIP_COST_PCT*100:.2f}% round-trip[/dim]")
    console.print(f"[dim]Features: {len(ALL_FEATURES)} | Pruning: top {int(PRUNE_KEEP_RATIO*100)}%[/dim]")
    console.print(f"[dim]Optuna trials: {OPTUNA_N_TRIALS} | Monte Carlo: {MONTE_CARLO_RUNS} shuffles[/dim]")
    console.print(f"[dim]V7: Percentile threshold, soft filters, OOF stacking, new features[/dim]\n")

    # ── 1. Download & Prep (parallel) ──
    poly = Polygon_Helper()
    raw_cache = {}
    print(f"Downloading data (3-year lookback, {IO_WORKERS} workers)...")

    def _fetch_raw(t):
        try:
            raw = poly.fetch_data(t, days=LOOKBACK_DAYS, mult=1, timespan='hour')
            if len(raw) > 500:
                return t, raw
            else:
                print(f"   {t}: Insufficient data ({len(raw)} bars)")
                return t, None
        except Exception as e:
            print(f"   {t}: Failed ({e})")
            return t, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
        futures = {executor.submit(_fetch_raw, t): t for t in TICKERS}
        for future in concurrent.futures.as_completed(futures):
            t, raw = future.result()
            if raw is not None:
                raw_cache[t] = raw
                print(f"   {t}: {len(raw)} raw bars")

    if not raw_cache:
        print("No data available. Set POLYGON_API_KEY env var.")
        return

    # ── 1b. Compute universe proxy returns ──
    print("\nComputing universe proxy returns...")
    all_returns = {}
    for t, raw in raw_cache.items():
        all_returns[t] = raw['Close'].pct_change()
    _UNIVERSE_RETURNS = pd.DataFrame(all_returns).mean(axis=1)

    # ── 1c. Prepare features ──
    data_cache = {}
    print("Computing features...")
    for t, raw in raw_cache.items():
        try:
            df_proc = prepare_features(raw, universe_returns=_UNIVERSE_RETURNS)
            df_proc['_ticker'] = t
            data_cache[t] = df_proc
            print(f"   {t}: {len(df_proc)} bars ({df_proc.index[0].date()} to {df_proc.index[-1].date()})")
        except Exception as e:
            print(f"   {t}: Feature computation failed ({e})")

    if not data_cache:
        print("No data after feature computation.")
        return

    # ── 1d. Split holdout ──
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
            print(f"   {t}: All {len(df)} bars for training (insufficient for holdout)")

    # ── 2. Walk-Forward Training per label bucket ──
    print(f"\nRunning walk-forward training for {len(LABEL_BUCKETS)} label buckets...")
    predictions_cache_by_bucket = {}

    for bucket_sl, bucket_tp, bucket_mb in LABEL_BUCKETS:
        bucket_key = (bucket_sl, bucket_tp, bucket_mb)
        bucket_cache = {}
        print(f"\n   Bucket SL={bucket_sl} TP={bucket_tp} MB={bucket_mb}:")

        for t, df in train_data.items():
            avail_feats = [f for f in ALL_FEATURES if f in df.columns]
            wf_result, pruned_feats = walk_forward_train_predict(
                df, avail_feats, bucket_sl, bucket_tp, bucket_mb)
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

    # ── 3. Optuna Optimization (single objective, composite score) ──
    print(f"\nStarting Optuna optimization ({OPTUNA_N_TRIALS} trials, single-objective composite)...")

    try:
        study = optuna.create_study(
            study_name="hedge_fund_v7",
            direction="maximize",
            storage=OPTUNA_STORAGE,
            load_if_exists=False,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=25,
                multivariate=True,
                seed=42,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=15,
                n_warmup_steps=0,
            ),
        )
        print(f"   Optuna study persisted to {OPTUNA_STORAGE}")
    except Exception:
        study = optuna.create_study(
            study_name="hedge_fund_v7",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=25, multivariate=True, seed=42),
        )
        print("   Optuna study in-memory (SQLite unavailable)")

    objective = create_optuna_objective(predictions_cache_by_bucket)
    study.optimize(
        objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: (
                print(
                    f"\n  Trial {trial.number}: "
                    f"composite={trial.value:.4f} | "
                    f"PF={trial.user_attrs.get('pf', 'pruned')} | "
                    f"Sharpe={trial.user_attrs.get('sharpe', 'pruned')} | "
                    f"Trades={trial.user_attrs.get('n_trades', 'pruned')} | "
                    f"Tickers={trial.user_attrs.get('n_tickers_trading', 'pruned')} | "
                    f"WR={trial.user_attrs.get('win_rate', 'pruned')}"
                )
                if trial.number % 5 == 0 and trial.value is not None else None
            )
        ],
    )

    # ── 4. Collect results ──
    results = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.value is None or trial.value <= -4.0:
            continue

        metrics = trial.user_attrs.get("metrics", _empty_metrics())
        all_trades = trial.user_attrs.get("trades", [])
        tp_rr_val = trial.params.get('tp_rr', 2.0)
        result = {
            "SL": trial.params.get("sl_mult", 1.5),
            "R:R": f"1:{tp_rr_val:.1f}",
            "MB": trial.params.get("max_bars", 10),
            "Thresh": round(trial.params.get("pred_threshold_pct", 0.70), 2),
            "Mode": "SOFT",
            "Trail": str(round(trial.params.get("trail_mult", 1.0), 2)),
            "Hurst": str(round(trial.params.get("hurst_limit", 0.5), 2)),
            "ADX": str(round(trial.params.get("adx_min", 0), 1)),
            "ScaleOut": str(round(trial.params.get("scale_out_r", 1.5), 2)),
            "Regime": trial.params.get("regime_filter_mode", "soft"),
            **metrics,
            "_trades": all_trades,
            "_score": trial.value,
            "_params": trial.params,
        }
        results.append(result)

    if not results:
        console.print("[red]No results. Check data and parameters.[/red]")
        return

    results.sort(key=lambda x: x['_score'], reverse=True)

    # ── 5. Display top results ──
    print("\n" + "=" * 110)
    print("TOP 10 OPTIMIZED CONFIGS - V7 COMPOSITE SCORE")
    print("=" * 110)

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"Top 10 Configs ({len(TICKERS)} tickers, {LOOKBACK_DAYS}d)")
    table.add_column("Rank")
    table.add_column("Score", justify="right")
    table.add_column("PF", justify="right", style="bold green")
    table.add_column("WR", justify="right")
    table.add_column("N", justify="right")
    table.add_column("MaxDD", justify="right", style="red")
    table.add_column("Sharpe", justify="right", style="cyan")
    table.add_column("SL")
    table.add_column("R:R")
    table.add_column("Thresh")

    best_config = None

    for idx_r, r in enumerate(results[:10]):
        if not best_config and r['Trades'] >= 15:
            best_config = r

        table.add_row(
            str(idx_r + 1),
            f"{r['_score']:.4f}",
            f"{r['PF_Res']:.2f}",
            f"{r['WR_Res']:.1%}",
            str(r['Trades']),
            f"{r['MaxDD_R']:.1f}R",
            f"{r['Sharpe']:.1f}",
            str(round(r['SL'], 2)),
            r['R:R'],
            str(r['Thresh']),
        )

    console.print(table)

    if not best_config and results:
        best_config = results[0]

    # ── 6. Best config deep-dive ──
    if best_config:
        console.print(f"\n[bold green]BEST CONFIG:[/bold green]")
        console.print(f"   PF={best_config['PF_Res']:.2f} | WR={best_config['WR_Res']:.1%} | "
                       f"Sharpe={best_config['Sharpe']:.2f} | MaxDD={best_config['MaxDD_R']:.1f}R")
        console.print(f"   Trades: {best_config['Trades']} | Score: {best_config['_score']:.4f}")

        # Per-ticker breakdown
        breakdown = per_ticker_breakdown(best_config['_trades'])
        if breakdown:
            console.print(f"\n[bold cyan]PER-TICKER BREAKDOWN:[/bold cyan]")
            ticker_table = Table(show_header=True, header_style="bold cyan")
            ticker_table.add_column("Ticker")
            ticker_table.add_column("PF", justify="right")
            ticker_table.add_column("WR", justify="right")
            ticker_table.add_column("Trades", justify="right")
            ticker_table.add_column("Return", justify="right")

            for tick, m in breakdown.items():
                ticker_table.add_row(tick, f"{m['PF_Res']:.2f}", f"{m['WR_Res']:.1%}",
                                    str(m['Trades']), f"{m['TotalReturn_R']:.1f}R")
            console.print(ticker_table)

        # Monte Carlo
        p_value = monte_carlo_test(best_config['_trades'], best_config['PF_Res'])
        console.print(f"\n   Monte Carlo p-value: {p_value:.4f}")

        # ── 7. Holdout ──
        holdout_tickers_with_data = {t: h for t, h in holdout_data.items() if h is not None and len(h) > 100}
        if holdout_tickers_with_data:
            console.print(f"\n[bold magenta]HOLDOUT VALIDATION ({HOLDOUT_BARS} bars):[/bold magenta]")
            params = best_config.get('_params', {})
            best_sl = params.get('sl_mult', best_config['SL'])
            best_tp_rr = params.get('tp_rr', 2.0)
            best_tp_mult = best_sl * best_tp_rr
            best_mb = params.get('max_bars', best_config['MB'])
            best_thresh_pct = params.get('pred_threshold_pct', 0.70)
            best_trail = params.get('trail_mult', 1.0)
            best_hurst = params.get('hurst_limit', 0.5)
            best_adx = params.get('adx_min', 0)
            best_scale = params.get('scale_out_r', 1.5)
            best_regime_mode = params.get('regime_filter_mode', 'soft')

            holdout_trades = []
            for t, h_df in holdout_tickers_with_data.items():
                full_df = data_cache[t]
                train_portion = full_df.iloc[:-HOLDOUT_BARS]
                if len(train_portion) < WF_TRAIN_BARS:
                    continue

                train_slice = train_portion.iloc[-WF_TRAIN_BARS:]
                labels = compute_bracket_labels(train_slice, sl_mult=best_sl,
                                                tp_mult=best_tp_mult, max_bars=best_mb)
                train_slice = train_slice.copy()
                train_slice['Target'] = labels

                avail_feats = [f for f in ALL_FEATURES if f in train_slice.columns and f in h_df.columns]
                if len(avail_feats) < 5:
                    continue

                model = EnsembleModel()
                model.fit(train_slice[avail_feats], train_slice['Target'])

                holdout_df = h_df.copy()
                holdout_df['Predictions'] = model.predict(holdout_df[avail_feats])

                # Percentile threshold on holdout predictions
                preds_abs = np.abs(holdout_df['Predictions'].values)
                valid_preds = preds_abs[np.isfinite(preds_abs)]
                if len(valid_preds) < 10:
                    continue
                thresh_raw = np.percentile(valid_preds, best_thresh_pct * 100)

                h_trades = simulate_trades_stateful(
                    holdout_df, thresh_raw, best_sl, best_tp_mult, best_mb,
                    trail_mult=best_trail, filter_mode="MINIMAL",
                    hurst_limit=best_hurst, adx_min=best_adx,
                    scale_out_r=best_scale, use_kelly=True,
                    regime_filter_mode=best_regime_mode,
                )
                holdout_trades.extend(h_trades)

            if holdout_trades and len(holdout_trades) >= 5:
                h_metrics = compute_risk_metrics(holdout_trades)
                console.print(f"   Trades: {h_metrics['Trades']} | PF: {h_metrics['PF_Res']:.2f} | "
                               f"WR: {h_metrics['WR_Res']:.1%} | Sharpe: {h_metrics['Sharpe']:.2f}")

        # ── 8. Save optimal parameters ──
        console.print(f"\n[bold green]SAVING OPTIMAL PARAMETERS...[/bold green]")
        try:
            holdout_m = h_metrics if 'h_metrics' in dir() else None
            config_path = save_optimal_params(best_config, holdout_metrics=holdout_m)
            console.print(f"   Saved to {config_path}")
        except Exception as e:
            console.print(f"   [red]Failed to save config: {e}[/red]")

        # Save ensemble models
        try:
            os.makedirs('data/models', exist_ok=True)
            console.print("   Models directory: data/models/")
        except Exception:
            pass

    else:
        console.print("\n[yellow]No config passed minimum thresholds.[/yellow]")

    console.print(f"\n[dim]Universe: {', '.join(TICKERS)}[/dim]")
    console.print(f"[dim]Features: {', '.join(ALL_FEATURES[:10])}...[/dim]")


if __name__ == "__main__":
    main()
