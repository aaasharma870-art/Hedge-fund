# ==============================================================================
# HYBRID BACKTESTER V12: DAILY ALPHA + INTRADAY EXECUTION
# ==============================================================================
# Two-tier institutional system:
#   Tier 1 — Daily ML predicts 5-day forward returns → ranked watchlist
#   Tier 2 — Intraday execution via 15-min VPIN/OFI/VWAP entry timing
#
# Root causes fixed from V11:
#   1. Labels were noise (98% timeout at ~0.0) → 5-day forward returns
#   2. Features useless on 15-min (momentum pruned 56/60) → daily factors
#   3. Costs destroyed edge (0.285R/trade) → 0.025R/trade with 3-10d holds
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
    from rich.console import Console
    from rich.table import Table
    import optuna
except ImportError:
    print("Missing deps, running pip install...", flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'rich', 'optuna'], check=True)
    from rich.console import Console
    from rich.table import Table
    import optuna

from hedge_fund.daily_features import DAILY_FEATURES, compute_daily_features
from hedge_fund.daily_model import (
    walk_forward_daily, generate_watchlist, FORWARD_DAYS,
    compute_daily_labels,
)
from hedge_fund.execution import simulate_hybrid_trades
from hedge_fund.data import RateLimiter
from hedge_fund.ensemble import EnsembleModel

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

KEYS = {
    "POLY": os.environ.get("POLYGON_API_KEY", ""),
}

IO_WORKERS = 16

_HAS_POLYGON_KEY = bool(os.environ.get("POLYGON_API_KEY", ""))
LOOKBACK_DAYS = 1500 if _HAS_POLYGON_KEY else 730  # ~6 years for more walk-forward windows

TICKERS = [
    # High-beta momentum
    'RKLB', 'ASTS', 'AMD', 'NVDA', 'PLTR', 'COIN',
    # Large-cap value
    'GS', 'GE', 'COST', 'JPM', 'UNH', 'CAT',
    # Defensive
    'XOM', 'JNJ', 'PG',
]

# Walk-forward settings for daily model
DAILY_TRAIN_DAYS = 250    # ~1 year
DAILY_TEST_DAYS = 60      # ~3 months
DAILY_STEP_DAYS = 60      # non-overlapping
HOLDOUT_DAYS = 90         # last 90 trading days

# Optuna
OPTUNA_N_TRIALS = 80
OPTUNA_TIMEOUT = None

# Cost
COST_PCT = 0.0005         # 0.05% per side = 0.10% round-trip

# Monte Carlo
MONTE_CARLO_RUNS = 1000

console = Console()


# ==============================================================================
# INFRASTRUCTURE (data fetching)
# ==============================================================================

class Polygon_Helper:
    def __init__(self):
        self.sess = requests.Session()
        self.base = "https://api.polygon.io"
        self.last_429 = 0
        self._lock = threading.Lock()
        self._rate_limiter = RateLimiter(rate_per_sec=12.0, burst=20)

    def _throttle(self):
        self._rate_limiter.acquire()

    def fetch_data(self, t, days=365, mult=1, timespan='day'):
        with self._lock:
            if time.time() - self.last_429 < 60:
                time.sleep(60 - (time.time() - self.last_429))

        end = datetime.datetime.now(datetime.timezone.utc)
        start = end - datetime.timedelta(days=days)

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
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms', utc=True)
        if timespan == 'day':
            # Daily bars: Polygon timestamps at midnight UTC = correct trading date
            # Converting to ET shifts to previous day — use UTC date instead
            df['Datetime'] = df['Datetime'].dt.tz_localize(None).dt.normalize()
        else:
            # Intraday bars: convert to ET normally
            df['Datetime'] = df['Datetime'].dt.tz_convert("America/New_York")
        df = df.set_index('Datetime').sort_index()
        df = df[~df.index.duplicated()]
        return df


# ==============================================================================
# RISK METRICS (adapted for V12 trade tuples)
# ==============================================================================

def compute_risk_metrics(trades):
    """
    Compute risk metrics from trade tuples: (pnl_r, resolved, size, ticker, direction).
    """
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

    if len(outcomes) > 1:
        normalized = [o * 0.02 for o in outcomes]
        mean_r = np.mean(normalized)
        std_r = np.std(normalized, ddof=1)
        sharpe_per_trade = mean_r / std_r if std_r > 1e-10 else 0
        trading_days = 252
        trades_per_day = max(1, n_total / max(trading_days, 1))
        sharpe_annual = sharpe_per_trade * np.sqrt(252 * trades_per_day)
    else:
        sharpe_annual = 0

    total_return = equity[-1] if len(equity) > 0 else 0

    avg_win = gross_win / wins if wins > 0 else 0
    avg_loss = gross_loss / (n_total - wins) if (n_total - wins) > 0 else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Direction breakdown
    long_trades = [t for t in trades if len(t) >= 5 and t[4] == 'LONG']
    short_trades = [t for t in trades if len(t) >= 5 and t[4] == 'SHORT']

    return {
        'PF_Raw': round(pf_raw, 3),
        'PF_Res': round(pf_res, 3),
        'WR_Raw': round(wr_raw, 3),
        'WR_Res': round(wr_res, 3),
        'Trades': n_total,
        'Resolved': len(resolved),
        'MaxDD_R': round(max_dd, 2),
        'Sharpe': round(sharpe_annual, 2),
        'TotalReturn_R': round(total_return, 2),
        'AvgWin_R': round(avg_win, 2),
        'AvgLoss_R': round(avg_loss, 2),
        'PayoffRatio': round(payoff_ratio, 2),
        'LongTrades': len(long_trades),
        'ShortTrades': len(short_trades),
    }


def _empty_metrics():
    return {
        'PF_Raw': 0, 'PF_Res': 0, 'WR_Raw': 0, 'WR_Res': 0,
        'Trades': 0, 'Resolved': 0, 'MaxDD_R': 0, 'Sharpe': 0,
        'TotalReturn_R': 0, 'AvgWin_R': 0, 'AvgLoss_R': 0, 'PayoffRatio': 0,
        'LongTrades': 0, 'ShortTrades': 0,
    }


# ==============================================================================
# SIGNAL ACCURACY
# ==============================================================================

def compute_signal_accuracy(watchlist, daily_data, forward_days=5):
    """
    Measure daily signal accuracy:
      - What % of top-N longs actually went up over 5 days?
      - What % of bottom-N shorts actually went down?
    """
    long_correct = 0
    long_total = 0
    short_correct = 0
    short_total = 0

    for d in sorted(watchlist.keys()):
        signals = watchlist[d]

        for ticker, conv in signals.get('longs', []):
            if ticker not in daily_data:
                continue
            df = daily_data[ticker]
            try:
                if hasattr(df.index, 'date'):
                    mask = df.index.date == d
                    if mask.any():
                        entry_price = float(df.loc[mask, 'Close'].iloc[-1])
                        entry_loc = df.index.get_loc(df.loc[mask].index[-1])
                        if isinstance(entry_loc, slice):
                            entry_loc = entry_loc.stop - 1
                        exit_loc = min(entry_loc + forward_days, len(df) - 1)
                        exit_price = float(df.iloc[exit_loc]['Close'])
                        long_total += 1
                        if exit_price > entry_price:
                            long_correct += 1
            except Exception:
                continue

        for ticker, conv in signals.get('shorts', []):
            if ticker not in daily_data:
                continue
            df = daily_data[ticker]
            try:
                if hasattr(df.index, 'date'):
                    mask = df.index.date == d
                    if mask.any():
                        entry_price = float(df.loc[mask, 'Close'].iloc[-1])
                        entry_loc = df.index.get_loc(df.loc[mask].index[-1])
                        if isinstance(entry_loc, slice):
                            entry_loc = entry_loc.stop - 1
                        exit_loc = min(entry_loc + forward_days, len(df) - 1)
                        exit_price = float(df.iloc[exit_loc]['Close'])
                        short_total += 1
                        if exit_price < entry_price:
                            short_correct += 1
            except Exception:
                continue

    return {
        'long_accuracy': long_correct / max(long_total, 1),
        'long_total': long_total,
        'short_accuracy': short_correct / max(short_total, 1),
        'short_total': short_total,
    }


# ==============================================================================
# COST ANALYSIS
# ==============================================================================

def compute_cost_analysis(trades, daily_data, sl_atr_mult=1.5):
    """
    Per-ticker cost analysis.
    Returns dict of {ticker: {'avg_cost_r': float, 'n_trades': int}}.
    """
    ticker_costs = {}
    for t in trades:
        ticker = t[3]  # (pnl_r, resolved, size, ticker, direction)
        if ticker not in ticker_costs:
            ticker_costs[ticker] = {'costs': [], 'n_trades': 0}
        ticker_costs[ticker]['n_trades'] += 1

    # Estimate cost_r from daily ATR and cost_pct
    result = {}
    for ticker, info in ticker_costs.items():
        if ticker in daily_data:
            df = daily_data[ticker]
            if 'Daily_ATR' in df.columns and 'Close' in df.columns:
                avg_atr_pct = (df['Daily_ATR'] / df['Close']).mean()
                cost_r = (COST_PCT * 2) / (sl_atr_mult * avg_atr_pct) if avg_atr_pct > 0 else 0
                result[ticker] = {
                    'avg_cost_r': round(cost_r, 4),
                    'n_trades': info['n_trades'],
                    'atr_pct': round(avg_atr_pct * 100, 2),
                }
            else:
                result[ticker] = {'avg_cost_r': 0, 'n_trades': info['n_trades'], 'atr_pct': 0}
        else:
            result[ticker] = {'avg_cost_r': 0, 'n_trades': info['n_trades'], 'atr_pct': 0}

    return result


# ==============================================================================
# MONTE CARLO
# ==============================================================================

def monte_carlo_test(trades, observed_pf, n_simulations=1000):
    """Sign-randomization test for profit factor significance."""
    if len(trades) < 10:
        return 1.0

    outcomes = [t[0] for t in trades]
    magnitudes = [abs(x) for x in outcomes]
    beat_count = 0

    for _ in range(n_simulations):
        randomized = [m * random.choice([1, -1]) for m in magnitudes]
        gw = sum(x for x in randomized if x > 0)
        gl = abs(sum(x for x in randomized if x < 0))
        pf_rand = gw / gl if gl > 0 else 0
        if pf_rand >= observed_pf:
            beat_count += 1

    return beat_count / n_simulations


# ==============================================================================
# OPTUNA OBJECTIVE
# ==============================================================================

def _filter_watchlist(watchlist, top_n, min_spread=0.0):
    """Filter watchlist by top_n and minimum prediction spread."""
    filtered = {}
    for d, signals in watchlist.items():
        longs = signals.get('longs', [])[:top_n]
        shorts = signals.get('shorts', [])[:top_n]

        # Skip low-conviction days
        if min_spread > 0 and longs and shorts:
            best_score = longs[0][1] if longs else 0
            worst_score = shorts[0][1] if shorts else 0
            spread = abs(best_score) + abs(worst_score)
            if spread < min_spread:
                continue

        if longs or shorts:
            filtered[d] = {'longs': longs, 'shorts': shorts}
    return filtered


def create_hybrid_objective(watchlist, intraday_data, daily_data):
    """Create Optuna objective for hybrid system."""

    def objective(trial):
        sl_atr_mult = trial.suggest_float("sl_atr_mult", 1.3, 2.0)
        tp_rr = trial.suggest_float("tp_rr", 2.0, 4.0)
        tp_atr_mult = sl_atr_mult * tp_rr
        max_hold_days = trial.suggest_int("max_hold_days", 5, 12)
        entry_threshold = trial.suggest_float("entry_threshold", 0.25, 0.50)
        top_n = trial.suggest_int("top_n", 2, 3)
        partial_exit_atr = trial.suggest_float("partial_exit_atr", 1.0, 2.0)
        min_spread = trial.suggest_float("min_spread", 0.0, 0.02)
        short_size_mult = trial.suggest_float("short_size_mult", 0.3, 1.0)

        # Regenerate watchlist with this trial's top_n and min_spread
        trial_watchlist = _filter_watchlist(watchlist, top_n, min_spread)

        trades = simulate_hybrid_trades(
            trial_watchlist, intraday_data, daily_data,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            max_hold_days=max_hold_days,
            entry_threshold=entry_threshold,
            partial_exit_atr=partial_exit_atr,
            cost_pct=COST_PCT,
            short_size_mult=short_size_mult,
        )

        n_trades = len(trades)

        # Diagnostic for first 3 trials
        if trial.number < 3:
            long_n = len([t for t in trades if len(t) >= 5 and t[4] == 'LONG'])
            short_n = len([t for t in trades if len(t) >= 5 and t[4] == 'SHORT'])
            print(f"\n      [DIAG Trial {trial.number}] n_trades={n_trades} (L:{long_n} S:{short_n})")
            if n_trades > 0:
                outcomes = [t[0] for t in trades]
                gw = sum(x for x in outcomes if x > 0)
                gl = abs(sum(x for x in outcomes if x < 0))
                pf = gw / gl if gl > 0 else 0
                print(f"      [DIAG] Mean PnL={np.mean(outcomes):.4f}, PF={pf:.3f}")
                # Per-ticker breakdown
                diag_tickers = {}
                for t in trades:
                    tk = t[3]
                    if tk not in diag_tickers:
                        diag_tickers[tk] = []
                    diag_tickers[tk].append(t)
                for tk, tt in sorted(diag_tickers.items()):
                    if len(tt) >= 3:
                        tm = compute_risk_metrics(tt)
                        print(f"        {tk}: PF={tm['PF_Raw']:.2f} Trades={tm['Trades']}")

        if n_trades < 30:
            return -5.0

        metrics = compute_risk_metrics(trades)
        pf = metrics.get('PF_Raw', 0)
        sharpe = metrics['Sharpe']
        wr = metrics['WR_Raw']

        # Per-ticker consistency
        ticker_trades = {}
        for t in trades:
            tk = t[3]
            if tk not in ticker_trades:
                ticker_trades[tk] = []
            ticker_trades[tk].append(t)

        ticker_pfs = {}
        for tk, tt in ticker_trades.items():
            if len(tt) >= 5:
                tm = compute_risk_metrics(tt)
                ticker_pfs[tk] = tm.get('PF_Raw', 0)

        profitable = sum(1 for v in ticker_pfs.values() if v >= 0.90)
        consistency = profitable / max(len(ticker_pfs), 1)

        # Graduated scoring — WR-prioritized
        pf_score = np.clip(np.log(max(pf, 0.01)) / np.log(2.0), -1.5, 1.0)
        sharpe_score = np.clip(sharpe / 2.0, -1.0, 1.0)
        wr_score = np.clip((wr - 0.40) / 0.15, -1.0, 1.0)
        trade_score = np.clip(np.log(max(n_trades, 1) / 30) / np.log(300 / 30), 0, 1)

        # Penalize configs where WR < 42%
        if wr < 0.42:
            pf_score *= 0.5

        score = (
            0.25 * pf_score +
            0.20 * sharpe_score +
            0.25 * wr_score +
            0.15 * consistency +
            0.15 * trade_score
        )

        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("pf", round(pf, 4))
        trial.set_user_attr("sharpe", round(sharpe, 4))
        trial.set_user_attr("long_n", len([t for t in trades if len(t) >= 5 and t[4] == 'LONG']))
        trial.set_user_attr("short_n", len([t for t in trades if len(t) >= 5 and t[4] == 'SHORT']))
        trial.set_user_attr("ticker_pfs", {k: round(v, 3) for k, v in ticker_pfs.items()})

        return score

    return objective


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    console.print("[bold green]HYBRID BACKTESTER V12.0 (DAILY ALPHA + INTRADAY EXECUTION)[/bold green]")
    console.print(f"[dim]Lookback: {LOOKBACK_DAYS}d | Daily WF: {DAILY_TRAIN_DAYS}/{DAILY_TEST_DAYS}/{DAILY_STEP_DAYS}d[/dim]")
    console.print(f"[dim]Universe: {len(TICKERS)} tickers | Cost: {COST_PCT*100:.3f}% per side[/dim]")
    console.print(f"[dim]Daily features: {len(DAILY_FEATURES)} | Forward: {FORWARD_DAYS}d[/dim]")
    console.print(f"[dim]Optuna: {OPTUNA_N_TRIALS} trials, 6 params[/dim]\n")

    # ── 1. Download data ──
    daily_cache = {}
    intraday_cache = {}

    if _HAS_POLYGON_KEY:
        poly = Polygon_Helper()
        print(f"Downloading daily + 15m data from Polygon ({LOOKBACK_DAYS}d)...")

        def _fetch_daily(t):
            try:
                raw = poly.fetch_data(t, days=LOOKBACK_DAYS, mult=1, timespan='day')
                if len(raw) > 200:
                    return t, raw
                print(f"   {t}: Insufficient daily data ({len(raw)} bars)")
                return t, None
            except Exception as e:
                print(f"   {t}: Daily fetch failed ({e})")
                return t, None

        def _fetch_intraday(t):
            try:
                raw = poly.fetch_data(t, days=LOOKBACK_DAYS, mult=15, timespan='minute')
                if len(raw) > 1000:
                    return t, raw
                print(f"   {t}: Insufficient 15m data ({len(raw)} bars)")
                return t, None
            except Exception as e:
                print(f"   {t}: 15m fetch failed ({e})")
                return t, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
            daily_futures = {executor.submit(_fetch_daily, t): t for t in TICKERS}
            for future in concurrent.futures.as_completed(daily_futures):
                t, raw = future.result()
                if raw is not None:
                    daily_cache[t] = raw
                    print(f"   {t}: {len(raw)} daily bars")

        with concurrent.futures.ThreadPoolExecutor(max_workers=IO_WORKERS) as executor:
            intra_futures = {executor.submit(_fetch_intraday, t): t for t in TICKERS}
            for future in concurrent.futures.as_completed(intra_futures):
                t, raw = future.result()
                if raw is not None:
                    intraday_cache[t] = raw
                    print(f"   {t}: {len(raw)} 15m bars")
    else:
        # yfinance fallback
        try:
            import yfinance as yf
        except ImportError:
            print("No Polygon API key and yfinance not available.")
            return

        print(f"Downloading daily data from yfinance...")
        for t in TICKERS:
            try:
                raw = yf.download(t, period='2y', interval='1d',
                                  auto_adjust=True, progress=False)
                if raw is not None and len(raw) > 200:
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                    daily_cache[t] = raw
                    print(f"   {t}: {len(raw)} daily bars")
            except Exception as e:
                print(f"   {t}: Failed ({e})")

        print(f"Downloading 15m data from yfinance (57d max)...")
        for t in TICKERS:
            try:
                raw = yf.download(t, period='57d', interval='15m',
                                  auto_adjust=True, progress=False)
                if raw is not None and len(raw) > 200:
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                    intraday_cache[t] = raw
                    print(f"   {t}: {len(raw)} 15m bars")
            except Exception as e:
                print(f"   {t}: Failed ({e})")

    if not daily_cache:
        print("No daily data available.")
        return

    # Verify daily bar dates are correct (no timezone shift)
    for t in list(daily_cache.keys())[:1]:
        sample_dates = daily_cache[t].index[:3]
        print(f"   Date check ({t}): {[str(d) for d in sample_dates]}")
        if hasattr(sample_dates[0], 'dayofweek'):
            dow = sample_dates[0].dayofweek
            print(f"   Day of week: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]}")

    # ── 2. Compute daily features ──
    print("\nComputing daily features...")
    daily_featured = {}
    for t, raw in daily_cache.items():
        try:
            df = compute_daily_features(raw, ticker=t, universe_daily=daily_cache)
            daily_featured[t] = df
            print(f"   {t}: {len(df)} bars after features")
        except Exception as e:
            print(f"   {t}: Feature computation failed ({e})")

    if not daily_featured:
        print("No data after feature computation.")
        return

    # ── 2b. Prepare intraday data with execution features ──
    print("\nComputing intraday execution features (VPIN, OFI, VWAP_ZScore)...")
    intraday_featured = {}
    for t, raw in intraday_cache.items():
        try:
            from hedge_fund.features import calculate_vpin, compute_ofi
            df = raw.copy()

            # VPIN
            df['VPIN'] = calculate_vpin(df)

            # OFI
            df['OFI'] = compute_ofi(df)

            # VWAP_ZScore
            typical = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum().clip(lower=1)
            vwap_std = (df['Close'] - vwap).rolling(50).std().clip(lower=1e-10)
            df['VWAP_ZScore'] = (df['Close'] - vwap) / vwap_std

            df.dropna(subset=['VPIN', 'OFI', 'VWAP_ZScore'], inplace=True)
            intraday_featured[t] = df
            print(f"   {t}: {len(df)} 15m bars with execution features")
        except Exception as e:
            print(f"   {t}: Intraday features failed ({e}), using raw")
            raw = raw.copy()
            for col in ['VPIN', 'OFI', 'VWAP_ZScore']:
                if col not in raw.columns:
                    raw[col] = 0.0
            intraday_featured[t] = raw

    # ── 3. Split holdout ──
    print(f"\nSplitting holdout ({HOLDOUT_DAYS} trading days)...")
    train_daily = {}
    holdout_daily = {}
    for t, df in daily_featured.items():
        if len(df) > HOLDOUT_DAYS + DAILY_TRAIN_DAYS:
            train_daily[t] = df.iloc[:-HOLDOUT_DAYS]
            holdout_daily[t] = df.iloc[-HOLDOUT_DAYS:]
            print(f"   {t}: Train {len(train_daily[t])} | Holdout {len(holdout_daily[t])}")
        else:
            train_daily[t] = df
            holdout_daily[t] = None
            print(f"   {t}: All {len(df)} for training (no holdout)")

    # ── 4. Walk-forward daily model ──
    print(f"\nRunning daily walk-forward ({DAILY_TRAIN_DAYS}d train, {DAILY_TEST_DAYS}d test)...")
    daily_predictions = {}
    for t, df in train_daily.items():
        avail_feats = [f for f in DAILY_FEATURES if f in df.columns]
        result = walk_forward_daily(
            df, avail_feats,
            train_days=DAILY_TRAIN_DAYS,
            test_days=DAILY_TEST_DAYS,
            step_days=DAILY_STEP_DAYS,
        )
        if result is not None:
            daily_predictions[t] = result
            print(f"   {t}: {len(result)} test predictions")
        else:
            print(f"   {t}: No predictions (insufficient data)")

    if not daily_predictions:
        print("No daily predictions generated.")
        return

    # ── 5. Generate full watchlist (max top_n=5 for Optuna to subset) ──
    print("\nGenerating daily watchlist...")
    full_watchlist = generate_watchlist(daily_predictions, top_n=5, bottom_n=5, min_spread=0.0)
    print(f"   {len(full_watchlist)} trading days with signals")

    if not full_watchlist:
        print("No watchlist generated.")
        return

    # ── Signal accuracy on training period ──
    signal_acc = compute_signal_accuracy(full_watchlist, daily_featured)
    console.print(f"\n[bold cyan]DAILY SIGNAL ACCURACY (training):[/bold cyan]")
    console.print(f"   Long accuracy:  {signal_acc['long_accuracy']:.1%} "
                  f"({signal_acc['long_total']} signals)")
    console.print(f"   Short accuracy: {signal_acc['short_accuracy']:.1%} "
                  f"({signal_acc['short_total']} signals)")

    # ── 6. Split watchlist 75/25 for opt/val ──
    sorted_wl_dates = sorted(full_watchlist.keys())
    split_idx = int(len(sorted_wl_dates) * 0.75)
    opt_dates = set(sorted_wl_dates[:split_idx])
    val_dates = set(sorted_wl_dates[split_idx:])

    opt_watchlist = {d: full_watchlist[d] for d in opt_dates}
    val_watchlist = {d: full_watchlist[d] for d in val_dates}

    print(f"\n   Opt watchlist: {len(opt_watchlist)} days | Val watchlist: {len(val_watchlist)} days")

    # ── 7. Optuna optimization ──
    print(f"\nStarting Optuna ({OPTUNA_N_TRIALS} trials, 6 params)...")

    study = optuna.create_study(
        study_name="hedge_fund_v12",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=15, multivariate=True, seed=42),
    )

    # Seed trials
    seed_configs = [
        {'sl_atr_mult': 1.5, 'tp_rr': 3.0, 'max_hold_days': 6,
         'entry_threshold': 0.40, 'top_n': 2, 'partial_exit_atr': 1.5,
         'min_spread': 0.005, 'short_size_mult': 0.5},
        {'sl_atr_mult': 1.5, 'tp_rr': 2.0, 'max_hold_days': 8,
         'entry_threshold': 0.35, 'top_n': 2, 'partial_exit_atr': 1.5,
         'min_spread': 0.01, 'short_size_mult': 0.5},
        {'sl_atr_mult': 1.7, 'tp_rr': 2.5, 'max_hold_days': 7,
         'entry_threshold': 0.30, 'top_n': 3, 'partial_exit_atr': 1.5,
         'min_spread': 0.005, 'short_size_mult': 0.7},
        {'sl_atr_mult': 2.0, 'tp_rr': 2.0, 'max_hold_days': 10,
         'entry_threshold': 0.35, 'top_n': 2, 'partial_exit_atr': 2.0,
         'min_spread': 0.0, 'short_size_mult': 1.0},
    ]
    for sp in seed_configs:
        study.enqueue_trial(sp)

    objective = create_hybrid_objective(opt_watchlist, intraday_featured, daily_featured)
    study.optimize(
        objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: (
                print(
                    f"\n  Trial {trial.number}: "
                    f"score={trial.value:.4f} | "
                    f"PF={trial.user_attrs.get('pf', '?')} | "
                    f"Sharpe={trial.user_attrs.get('sharpe', '?')} | "
                    f"Trades={trial.user_attrs.get('n_trades', '?')}"
                )
                if trial.number % 5 == 0 and trial.value is not None else None
            )
        ],
    )

    # ── 8. Collect results ──
    valid_trials = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.value is None or trial.value <= -4.0:
            continue
        valid_trials.append(trial)

    valid_trials.sort(key=lambda t: t.value, reverse=True)

    if not valid_trials:
        console.print("[red]No valid results. Check data and parameters.[/red]")
        return

    # Show trial score distribution
    all_scores = [t.value for t in study.trials if t.value is not None]
    n_positive = sum(1 for s in all_scores if s > 0)
    n_negative = sum(1 for s in all_scores if s <= -4.0)
    console.print(f"\n[dim]Trial distribution: {n_positive} positive, "
                  f"{len(all_scores) - n_positive - n_negative} mixed, "
                  f"{n_negative} zero-trade[/dim]")

    # ── Display top results ──
    print("\n" + "=" * 100)
    print("TOP 10 CONFIGS — V12 HYBRID (DAILY ALPHA + INTRADAY EXECUTION)")
    print("=" * 100)

    table = Table(show_header=True, header_style="bold magenta",
                  title=f"Top 10 ({len(TICKERS)} tickers, {LOOKBACK_DAYS}d)")
    for col in ["Rank", "Score", "PF", "Sharpe", "Trades", "SL", "R:R",
                "Hold", "TopN", "Thresh", "Spread", "ShSz"]:
        table.add_column(col, justify="right" if col != "Rank" else "left")

    best_trial = valid_trials[0]
    for idx_r, trial in enumerate(valid_trials[:10]):
        p = trial.params
        table.add_row(
            str(idx_r + 1),
            f"{trial.value:.4f}",
            f"{trial.user_attrs.get('pf', 0):.2f}",
            f"{trial.user_attrs.get('sharpe', 0):.1f}",
            str(trial.user_attrs.get('n_trades', 0)),
            f"{p.get('sl_atr_mult', 0):.1f}",
            f"{p.get('tp_rr', 0):.1f}",
            str(p.get('max_hold_days', 0)),
            str(p.get('top_n', 0)),
            f"{p.get('entry_threshold', 0):.2f}",
            f"{p.get('min_spread', 0):.3f}",
            f"{p.get('short_size_mult', 1):.1f}",
        )
    console.print(table)

    # ── 9. Run best config on FULL watchlist for detailed analysis ──
    bp = best_trial.params
    bp_tp_atr = bp['sl_atr_mult'] * bp['tp_rr']
    console.print(f"\n[bold green]BEST CONFIG:[/bold green]")
    console.print(f"   SL={bp['sl_atr_mult']:.2f} ATR | TP={bp_tp_atr:.2f} ATR (R:R 1:{bp['tp_rr']:.1f}) | "
                  f"Hold={bp['max_hold_days']}d | Thresh={bp['entry_threshold']:.2f} | "
                  f"TopN={bp['top_n']} | Spread>{bp['min_spread']:.3f} | "
                  f"ShortSize={bp['short_size_mult']:.2f}")

    # Rebuild watchlist with best top_n and min_spread
    best_watchlist = _filter_watchlist(full_watchlist, bp['top_n'], bp['min_spread'])

    best_trades = simulate_hybrid_trades(
        best_watchlist, intraday_featured, daily_featured,
        sl_atr_mult=bp['sl_atr_mult'],
        tp_atr_mult=bp_tp_atr,
        max_hold_days=bp['max_hold_days'],
        entry_threshold=bp['entry_threshold'],
        partial_exit_atr=bp['partial_exit_atr'],
        cost_pct=COST_PCT,
        short_size_mult=bp['short_size_mult'],
    )

    if not best_trades:
        console.print("[red]No trades from best config.[/red]")
        return

    best_metrics = compute_risk_metrics(best_trades)
    console.print(f"\n[bold cyan]FULL TRAINING RESULTS:[/bold cyan]")
    console.print(f"   PF={best_metrics['PF_Raw']:.2f} | WR={best_metrics['WR_Raw']:.1%} | "
                  f"Sharpe={best_metrics['Sharpe']:.2f} | MaxDD={best_metrics['MaxDD_R']:.1f}R")
    console.print(f"   Trades: {best_metrics['Trades']} "
                  f"(L:{best_metrics['LongTrades']} S:{best_metrics['ShortTrades']})")
    console.print(f"   TotalReturn: {best_metrics['TotalReturn_R']:.1f}R | "
                  f"PayoffRatio: {best_metrics['PayoffRatio']:.2f}")

    # ── LONG vs SHORT breakdown ──
    long_trades = [t for t in best_trades if len(t) >= 5 and t[4] == 'LONG']
    short_trades = [t for t in best_trades if len(t) >= 5 and t[4] == 'SHORT']

    if long_trades:
        lm = compute_risk_metrics(long_trades)
        console.print(f"\n   LONG PF:  {lm['PF_Raw']:.2f} | WR: {lm['WR_Raw']:.1%} | Trades: {lm['Trades']}")
    if short_trades:
        sm = compute_risk_metrics(short_trades)
        console.print(f"   SHORT PF: {sm['PF_Raw']:.2f} | WR: {sm['WR_Raw']:.1%} | Trades: {sm['Trades']}")

    # ── Per-ticker breakdown ──
    ticker_trades = {}
    for t in best_trades:
        tk = t[3]
        if tk not in ticker_trades:
            ticker_trades[tk] = []
        ticker_trades[tk].append(t)

    console.print(f"\n[bold cyan]PER-TICKER BREAKDOWN:[/bold cyan]")
    ticker_table = Table(show_header=True, header_style="bold cyan")
    for col in ["Ticker", "PF", "WR", "Trades", "Return_R"]:
        ticker_table.add_column(col, justify="right" if col != "Ticker" else "left")

    for tk in sorted(ticker_trades.keys()):
        tt = ticker_trades[tk]
        tm = compute_risk_metrics(tt)
        ticker_table.add_row(
            tk, f"{tm['PF_Raw']:.2f}", f"{tm['WR_Raw']:.1%}",
            str(tm['Trades']), f"{tm['TotalReturn_R']:.1f}R")
    console.print(ticker_table)

    # Trade outcome breakdown
    n = len(best_trades)
    if n > 0:
        sl_hits = sum(1 for t in best_trades if t[1] and t[0] < -0.5)
        tp_hits = sum(1 for t in best_trades if t[1] and t[0] > 1.5)
        trail_wins = sum(1 for t in best_trades if t[1] and 0 < t[0] <= 1.5)
        timeouts_pos = sum(1 for t in best_trades if not t[1] and t[0] > 0)
        timeouts_neg = sum(1 for t in best_trades if not t[1] and t[0] <= 0)

        console.print(f"\n[bold cyan]TRADE OUTCOME BREAKDOWN:[/bold cyan]")
        console.print(f"   Full SL hits:    {sl_hits:>4} ({sl_hits/n*100:>5.1f}%)")
        console.print(f"   Full TP hits:    {tp_hits:>4} ({tp_hits/n*100:>5.1f}%)")
        console.print(f"   Trail/Partial:   {trail_wins:>4} ({trail_wins/n*100:>5.1f}%)")
        console.print(f"   Timeout (win):   {timeouts_pos:>4} ({timeouts_pos/n*100:>5.1f}%)")
        console.print(f"   Timeout (loss):  {timeouts_neg:>4} ({timeouts_neg/n*100:>5.1f}%)")

    # ── Cost analysis ──
    console.print(f"\n[bold cyan]COST ANALYSIS:[/bold cyan]")
    cost_info = compute_cost_analysis(best_trades, daily_featured, sl_atr_mult=bp['sl_atr_mult'])
    cost_table = Table(show_header=True, header_style="bold cyan")
    for col in ["Ticker", "ATR%", "Cost_R", "Trades"]:
        cost_table.add_column(col, justify="right" if col != "Ticker" else "left")

    for tk in sorted(cost_info.keys()):
        ci = cost_info[tk]
        cost_table.add_row(
            tk, f"{ci['atr_pct']:.1f}%", f"{ci['avg_cost_r']:.4f}R",
            str(ci['n_trades']))
    console.print(cost_table)

    avg_cost = np.mean([ci['avg_cost_r'] for ci in cost_info.values()]) if cost_info else 0
    console.print(f"   Average cost_in_r: {avg_cost:.4f}R")

    # ── Monte Carlo ──
    p_value = monte_carlo_test(best_trades, best_metrics['PF_Raw'], MONTE_CARLO_RUNS)
    console.print(f"\n[bold cyan]MONTE CARLO:[/bold cyan]")
    console.print(f"   p-value: {p_value:.4f} "
                  f"({'SIGNIFICANT' if p_value < 0.10 else 'NOT significant'} at 10%)")

    # ── 10. Validation on held-out watchlist dates ──
    if val_watchlist:
        console.print(f"\n[bold magenta]VALIDATION ({len(val_watchlist)} days, 25% of watchlist):[/bold magenta]")
        val_wl = _filter_watchlist(val_watchlist, bp['top_n'], bp['min_spread'])

        val_trades = simulate_hybrid_trades(
            val_wl, intraday_featured, daily_featured,
            sl_atr_mult=bp['sl_atr_mult'],
            tp_atr_mult=bp_tp_atr,
            max_hold_days=bp['max_hold_days'],
            entry_threshold=bp['entry_threshold'],
            partial_exit_atr=bp['partial_exit_atr'],
            cost_pct=COST_PCT,
            short_size_mult=bp['short_size_mult'],
        )

        if val_trades:
            vm = compute_risk_metrics(val_trades)
            console.print(f"   Trades: {vm['Trades']} | PF: {vm['PF_Raw']:.2f} | "
                          f"WR: {vm['WR_Raw']:.1%} | Sharpe: {vm['Sharpe']:.2f}")

            val_sig = compute_signal_accuracy(val_wl, daily_featured)
            console.print(f"   Signal accuracy: Long {val_sig['long_accuracy']:.1%} | "
                          f"Short {val_sig['short_accuracy']:.1%}")
        else:
            console.print("   [yellow]No validation trades[/yellow]")

    # ── 11. Holdout validation ──
    holdout_tickers = {t: h for t, h in holdout_daily.items()
                       if h is not None and len(h) > 20}
    if holdout_tickers:
        console.print(f"\n[bold magenta]HOLDOUT VALIDATION ({HOLDOUT_DAYS} days):[/bold magenta]")

        # Train final model on all training data, predict on holdout
        holdout_predictions = {}
        for t, h_df in holdout_tickers.items():
            if t not in train_daily:
                continue
            full_train = train_daily[t]
            avail_feats = [f for f in DAILY_FEATURES
                          if f in full_train.columns and f in h_df.columns]
            if len(avail_feats) < 5:
                continue

            labels = compute_daily_labels(full_train)
            full_train_copy = full_train.copy()
            full_train_copy['DailyTarget'] = labels
            train_clean = full_train_copy.dropna(subset=['DailyTarget'])

            if len(train_clean) < 100:
                continue

            model = EnsembleModel(use_daily=True)
            model.fit(train_clean[avail_feats], train_clean['DailyTarget'])

            h_df = h_df.copy()
            h_df['DailyPrediction'] = model.predict(h_df[avail_feats])
            holdout_predictions[t] = h_df

        if holdout_predictions:
            holdout_watchlist = generate_watchlist(
                holdout_predictions,
                top_n=bp.get('top_n', 3),
                bottom_n=bp.get('top_n', 3),
                min_spread=bp.get('min_spread', 0.0),
            )

            if holdout_watchlist:
                # Subset to best top_n and min_spread
                h_wl = _filter_watchlist(holdout_watchlist, bp['top_n'], bp['min_spread'])

                holdout_trades = simulate_hybrid_trades(
                    h_wl, intraday_featured, daily_featured,
                    sl_atr_mult=bp['sl_atr_mult'],
                    tp_atr_mult=bp_tp_atr,
                    max_hold_days=bp['max_hold_days'],
                    entry_threshold=bp['entry_threshold'],
                    partial_exit_atr=bp['partial_exit_atr'],
                    cost_pct=COST_PCT,
                    short_size_mult=bp['short_size_mult'],
                )

                if holdout_trades:
                    h_metrics = compute_risk_metrics(holdout_trades)
                    console.print(f"   Trades: {h_metrics['Trades']} | "
                                  f"PF: {h_metrics['PF_Raw']:.2f} | "
                                  f"WR: {h_metrics['WR_Raw']:.1%} | "
                                  f"Sharpe: {h_metrics['Sharpe']:.2f}")

                    h_signal_acc = compute_signal_accuracy(h_wl, daily_featured)
                    console.print(f"   Signal accuracy: "
                                  f"Long {h_signal_acc['long_accuracy']:.1%} | "
                                  f"Short {h_signal_acc['short_accuracy']:.1%}")

                    h_pval = monte_carlo_test(holdout_trades, h_metrics['PF_Raw'])
                    console.print(f"   Monte Carlo p-value: {h_pval:.4f}")
                else:
                    console.print("   [yellow]No holdout trades[/yellow]")
            else:
                console.print("   [yellow]No holdout watchlist[/yellow]")
        else:
            console.print("   [yellow]No holdout predictions[/yellow]")

    console.print(f"\n[dim]Universe: {', '.join(TICKERS)}[/dim]")
    console.print(f"[dim]Features ({len(DAILY_FEATURES)}): {', '.join(DAILY_FEATURES)}[/dim]")


if __name__ == "__main__":
    main()
