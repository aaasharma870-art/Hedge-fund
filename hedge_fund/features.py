"""
Feature engineering for institutional-grade trading signals.

Provides microstructure features (VPIN, VWAP, Amihud), regime detection
(GEX proxy, volatility regime), alpha signals (RRS, liquidity sweeps),
and enhanced triple-barrier labeling.
"""

import logging
import threading
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

EXPECTED_FEATURES = [
    # Microstructure / Flow
    'OFI', 'VPIN', 'Amihud_Illiquidity', 'GEX_Proxy', 'VPT_Acceleration',
    # Momentum
    'Kalman_Trend', 'Kalman_Velocity', 'Overnight_Gap', 'Intraday_Mom',
    'Beta_Alpha', 'Relative_Return_Strength',
    # Regime / Structure
    'Hurst_Exponent', 'Efficiency_Ratio', 'RV_Ratio', 'ADX', 'VWAP_ZScore',
    'ATR_Channel_Pos',
    # Bar Microstructure
    'Upper_Wick', 'Lower_Wick', 'Body_Ratio',
    # Normalizers
    'ATR_Pct',
]


def calculate_vpin(df, volume_bucket_size=None, window=50):
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).

    Measures order flow toxicity. High VPIN indicates informed traders
    are present and adverse selection risk is elevated.

    Uses Bulk Volume Classification (BVC) to estimate buy/sell volume
    from price changes, then computes rolling imbalance across
    volume-bucketed bars.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns.
        volume_bucket_size: Volume per bucket. If None, auto-determined
            as median(20-day avg volume) / 50.
        window: Number of volume buckets for the rolling VPIN average.

    Returns:
        Series of VPIN values clipped to [0, 1]. Higher = more toxic.
    """
    if df is None or len(df) < window:
        return pd.Series(0.0, index=df.index if df is not None else [])

    df = df.copy()

    if volume_bucket_size is None:
        avg_daily_volume = df["Volume"].rolling(20).mean().median()
        volume_bucket_size = max(1, int(avg_daily_volume / 50))

    price_change = df["Close"].diff()
    price_std = price_change.rolling(20).std()
    z_score = price_change / (price_std + 1e-9)
    buy_prob = norm.cdf(z_score)

    df["Buy_Volume"] = df["Volume"] * buy_prob
    df["Sell_Volume"] = df["Volume"] * (1 - buy_prob)

    cumulative_volume = df["Volume"].cumsum()
    bucket_id = (cumulative_volume / volume_bucket_size).astype(int)

    bucket_imbalance = df.groupby(bucket_id).apply(
        lambda x: abs(x["Buy_Volume"].sum() - x["Sell_Volume"].sum())
    )
    bucket_volume = df.groupby(bucket_id)["Volume"].sum()

    vpin_buckets = bucket_imbalance / (bucket_volume + 1)
    vpin_rolling = vpin_buckets.rolling(window, min_periods=10).mean()

    bucket_to_vpin = vpin_rolling.to_dict()
    result = df.groupby(bucket_id).ngroup().map(bucket_to_vpin).fillna(0).clip(0, 1)
    result.index = df.index
    return result


def calculate_enhanced_vwap_features(df):
    """
    Enhanced VWAP features for institutional mean-reversion signals.

    Returns:
        Dict with keys:
            VWAP_ZScore: Standardized distance from 20-period VWAP.
            VWAP_Slope: Rate of VWAP change (trending vs ranging).
            VWAP_Volume_Ratio: Current volume vs VWAP-period average volume.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (
        (typical_price * df["Volume"]).rolling(20).sum()
        / df["Volume"].rolling(20).sum()
    )

    vwap_dist = df["Close"] - vwap
    vwap_std = vwap_dist.rolling(20).std()
    vwap_zscore = vwap_dist / (vwap_std + 1e-9)

    vwap_roc = vwap.pct_change(5)
    vwap_slope = vwap_roc.rolling(10).mean()

    avg_vwap_volume = df["Volume"].rolling(20).mean()
    vwap_volume_ratio = df["Volume"] / (avg_vwap_volume + 1)

    return {
        "VWAP_ZScore": vwap_zscore,
        "VWAP_Slope": vwap_slope,
        "VWAP_Volume_Ratio": vwap_volume_ratio,
    }


def calculate_volatility_regime(df, atr_func=None):
    """
    GEX Proxy via volatility regime detection.

    Since real GEX requires options data, this proxies it using realized
    volatility patterns:
      - Low RV + contracting ATR = Positive GEX (mean reversion)
      - High RV + expanding ATR = Negative GEX (trending/breakouts)

    Args:
        df: DataFrame with 'Close', 'High', 'Low' columns.
        atr_func: Optional callable(high, low, close, length) -> Series.
            If None, computes ATR internally.

    Returns:
        Tuple of (regime_gex, vol_regime_label):
            regime_gex: Series of -1 (negative GEX), 0 (neutral), +1 (positive GEX).
            vol_regime_label: Series of 'LOW', 'MEDIUM', 'HIGH'.
    """
    returns = df["Close"].pct_change()
    realized_vol = returns.rolling(20).std() * np.sqrt(252)

    if atr_func is not None:
        atr_20 = atr_func(df["High"], df["Low"], df["Close"], length=20)
    else:
        from hedge_fund.indicators import ManualTA
        atr_20 = ManualTA.atr(df["High"], df["Low"], df["Close"], length=20)

    atr_change = atr_20.pct_change(5)

    vol_percentile = realized_vol.rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )

    regime = pd.Series(0, index=df.index)
    regime[(vol_percentile < 0.3) & (atr_change < 0)] = 1
    regime[(vol_percentile > 0.7) & (atr_change > 0)] = -1

    vol_regime = pd.Series("MEDIUM", index=df.index)
    vol_regime[vol_percentile < 0.3] = "LOW"
    vol_regime[vol_percentile > 0.7] = "HIGH"

    return regime, vol_regime


def calculate_amihud_illiquidity(df, window=20):
    """
    Amihud Illiquidity Ratio: price impact per dollar traded.

    High values indicate low liquidity and high slippage risk.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns.
        window: Rolling window for the illiquidity average.

    Returns:
        Series of illiquidity percentile ranks (0-1).
    """
    returns = df["Close"].pct_change().abs()
    dollar_volume = df["Volume"] * df["Close"]

    illiquidity = returns / (dollar_volume + 1e-9)
    illiquidity_ratio = illiquidity.rolling(window).mean()

    illiquidity_rank = illiquidity_ratio.rolling(100).apply(
        lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
    )
    return illiquidity_rank


def calculate_real_relative_strength(stock_df, spy_df=None):
    """
    Real Relative Strength (RRS) vs the market.

    Measures how much a stock outperforms its beta-adjusted expected return.
    Positive RRS means institutional accumulation (stronger than expected).

    Args:
        stock_df: DataFrame with 'Close' for the stock.
        spy_df: DataFrame with 'Close' for SPY. If None, falls back to
            5-period momentum as a proxy.

    Returns:
        Series of cumulative 5-period RRS values.
    """
    if spy_df is None or len(spy_df) < 50:
        return stock_df["Close"].pct_change(5)

    stock_returns = stock_df["Close"].pct_change()
    spy_returns = spy_df["Close"].pct_change()

    spy_returns_aligned = spy_returns.reindex(stock_df.index, method="ffill")

    covariance = stock_returns.rolling(50).cov(spy_returns_aligned)
    spy_variance = spy_returns_aligned.rolling(50).var()
    beta = covariance / (spy_variance + 1e-9)
    beta = beta.fillna(1.0).clip(-3, 3)

    expected_return = beta * spy_returns_aligned
    rrs = stock_returns - expected_return

    rrs_cumulative = rrs.rolling(5).sum()
    return rrs_cumulative.fillna(0.0)


def calculate_liquidity_sweep(df, lookback=16):
    """
    Institutional liquidity sweep detection.

    Detects "fakeouts" where price breaks a key level to trigger retail
    stops, then reverses. High win-rate mean reversion signal.

    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns.
        lookback: Periods to look back for high/low levels.

    Returns:
        Series of signals: +1 (bullish sweep), -1 (bearish sweep), 0 (none).
    """
    signals = pd.Series(0, index=df.index)

    rolling_high = df["High"].rolling(lookback).max()
    rolling_low = df["Low"].rolling(lookback).min()

    avg_volume = df["Volume"].rolling(20).mean()
    volume_surge = df["Volume"] > (avg_volume * 1.5)

    for i in range(lookback + 1, len(df)):
        current_bar = df.iloc[i]
        prev_high = rolling_high.iloc[i - 1]
        prev_low = rolling_low.iloc[i - 1]

        if (
            current_bar["High"] > prev_high
            and current_bar["Close"] < prev_high
            and volume_surge.iloc[i]
        ):
            signals.iloc[i] = -1  # Bearish sweep
        elif (
            current_bar["Low"] < prev_low
            and current_bar["Close"] > prev_low
            and volume_surge.iloc[i]
        ):
            signals.iloc[i] = 1  # Bullish sweep

    return signals


class CrossSectionalRanker:
    """
    Cross-sectional factor ranking: rank stocks vs peers by momentum, value, quality.

    Produces percentile scores (0-1) per ticker for use as model features
    and score boosters. Recomputed once per scan cycle from snapshot data.
    """

    def __init__(self):
        self._ranks = {}  # ticker -> {'momentum_rank': 0.75, ...}
        self._last_update = 0
        self._lock = threading.Lock()

    def update(self, snap_data, fundamental_getter, universe):
        """
        Rank universe cross-sectionally.

        Args:
            snap_data: Dict from market snapshot
                {ticker: {'price': float, 'dayVol': int, 'dayOpen': float, ...}}
            fundamental_getter: Callable(ticker) -> dict with at least 'pe_ratio'.
                Can be None to skip value factor.
            universe: List of ticker strings to rank.
        """
        if not snap_data or len(snap_data) < 5:
            return

        tickers = [t for t in universe if t in snap_data]
        if len(tickers) < 5:
            return

        # Momentum factor: intraday return
        mom = {}
        for t in tickers:
            s = snap_data[t]
            day_open = s.get('dayOpen', 0)
            price = s.get('price', 0)
            if day_open > 0 and price > 0:
                mom[t] = (price - day_open) / day_open
            else:
                mom[t] = 0.0

        # Volume factor: relative dollar volume (liquidity + attention)
        vol = {}
        for t in tickers:
            s = snap_data[t]
            vol[t] = s.get('dayVol', 0) * s.get('price', 1)

        # Value factor: inverse P/E (high = cheap = good value)
        value = {}
        for t in tickers:
            try:
                if fundamental_getter is not None:
                    feats = fundamental_getter(t)
                    pe = feats.get('pe_ratio', 20)
                    if pe and pe > 0:
                        value[t] = 1.0 / pe
                    else:
                        value[t] = 0.05
                else:
                    value[t] = 0.05
            except Exception:
                value[t] = 0.05

        # Compute percentile ranks
        def _pct_rank(data_dict):
            if not data_dict:
                return {}
            items = sorted(data_dict.items(), key=lambda x: x[1])
            n = len(items)
            return {t: (i / max(1, n - 1)) for i, (t, _) in enumerate(items)}

        mom_ranks = _pct_rank(mom)
        vol_ranks = _pct_rank(vol)
        val_ranks = _pct_rank(value)

        # Composite score: 50% momentum + 25% volume + 25% value
        ranks = {}
        for t in tickers:
            mr = mom_ranks.get(t, 0.5)
            vr = vol_ranks.get(t, 0.5)
            vlr = val_ranks.get(t, 0.5)
            composite = 0.50 * mr + 0.25 * vr + 0.25 * vlr
            ranks[t] = {
                'momentum_rank': round(mr, 3),
                'volume_rank': round(vr, 3),
                'value_rank': round(vlr, 3),
                'composite_rank': round(composite, 3),
            }

        with self._lock:
            self._ranks = ranks
            self._last_update = time.time()

        logging.info(f"Cross-sectional ranking: {len(ranks)} tickers ranked")

    def get_ranks(self, ticker):
        """Return factor ranks for a ticker, or neutral defaults."""
        with self._lock:
            return self._ranks.get(ticker, {
                'momentum_rank': 0.5, 'volume_rank': 0.5,
                'value_rank': 0.5, 'composite_rank': 0.5,
            })

    def get_score_boost(self, ticker):
        """
        Score multiplier based on composite rank.

        Top-ranked stocks get up to 1.3x score boost, bottom get 0.7x.
        """
        r = self.get_ranks(ticker)
        comp = r['composite_rank']
        return float(np.clip(0.7 + 0.6 * comp, 0.7, 1.3))


import json
import os

FEATURE_STATS = {}
_FEATURE_STATS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feature_stats.json')

def load_feature_stats(path=None):
    """Load training feature statistics for live validation."""
    global FEATURE_STATS
    p = path or _FEATURE_STATS_PATH
    if os.path.exists(p):
        with open(p) as f:
            FEATURE_STATS = json.load(f)
    return FEATURE_STATS

def save_feature_stats(feature_df, path=None):
    """Save feature statistics from training for live distribution validation."""
    p = path or _FEATURE_STATS_PATH
    stats = {}
    for col in feature_df.columns:
        if col == 'Target':
            continue
        try:
            vals = feature_df[col].dropna()
            if len(vals) > 0:
                stats[col] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                    'median': float(vals.median()),
                }
        except Exception:
            pass
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(stats, f, indent=2)
    return stats

def validate_feature_distributions(live_features, training_stats=None):
    """Check if live features are within expected ranges from training."""
    stats = training_stats or FEATURE_STATS
    if not stats:
        return
    warnings = []
    for feat_name, live_val in live_features.items():
        if feat_name not in stats:
            continue
        try:
            train_mean = stats[feat_name]['mean']
            train_std = stats[feat_name]['std']
            if train_std > 0:
                z_score = abs((live_val - train_mean) / (train_std + 1e-8))
                if z_score > 4.0:
                    warnings.append(f"Feature {feat_name} is {z_score:.1f} std devs from training mean. Live={live_val:.4f}, Train mean={train_mean:.4f}")
        except (KeyError, TypeError):
            pass
    for w in warnings:
        logging.warning(w)
    return warnings


def compute_ofi(df, smooth_span=10, norm_window=50):
    """Order Flow Imbalance from OHLCV."""
    bar_range = (df['High'] - df['Low']).clip(lower=1e-10)
    close_pos = (df['Close'] - df['Low']) / bar_range
    buy_vol = df['Volume'] * close_pos
    sell_vol = df['Volume'] * (1.0 - close_pos)
    total_vol = buy_vol + sell_vol + 1e-10
    ofi_raw = (buy_vol - sell_vol) / total_vol
    ofi_smooth = ofi_raw.ewm(span=smooth_span, min_periods=3).mean()
    roll_mean = ofi_smooth.rolling(norm_window, min_periods=20).mean()
    roll_std = ofi_smooth.rolling(norm_window, min_periods=20).std().clip(lower=1e-10)
    ofi_z = (ofi_smooth - roll_mean) / roll_std
    return ofi_z.clip(-4, 4).rename('OFI')


def compute_rv_ratio(df, short=5, long=20):
    """Ratio of short-term to long-term realized volatility."""
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    rv_short = log_returns.rolling(short, min_periods=3).std() * np.sqrt(252 * 6.5)
    rv_long = log_returns.rolling(long, min_periods=10).std() * np.sqrt(252 * 6.5)
    rv_ratio = rv_short / rv_long.clip(lower=1e-6)
    return rv_ratio.clip(0, 5).rename('RV_Ratio')


def compute_momentum_decomp(df, session_bars=6):
    """Decompose momentum into overnight gap and intraday components."""
    prev_session_close = df['Close'].shift(session_bars)
    overnight_gap = (df['Open'] - prev_session_close) / prev_session_close.clip(lower=1e-10)
    overnight_gap = overnight_gap.rolling(3, min_periods=1).mean()
    session_open = df['Open'].shift(session_bars - 1)
    intraday_mom = (df['Close'] - session_open) / session_open.clip(lower=1e-10)
    intraday_mom = intraday_mom.rolling(3, min_periods=1).mean()
    return (overnight_gap.clip(-0.10, 0.10).rename('Overnight_Gap'),
            intraday_mom.clip(-0.10, 0.10).rename('Intraday_Mom'))


def compute_efficiency_ratio(df, window=10):
    """Kaufman Efficiency Ratio: net_change / path_length."""
    price = df['Close']
    net_change = (price - price.shift(window)).abs()
    bar_changes = price.diff().abs()
    path_length = bar_changes.rolling(window, min_periods=3).sum().clip(lower=1e-10)
    er = net_change / path_length
    return er.clip(0, 1).rename('Efficiency_Ratio')


def compute_vpt_acceleration(df, slope_window=5, accel_window=3, norm_window=50):
    """Volume Price Trend acceleration - institutional flow signal."""
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    vpt = (log_returns * df['Volume']).cumsum()
    vpt_slope = vpt.diff(slope_window) / slope_window
    vpt_accel = vpt_slope.diff(accel_window)
    roll_mean = vpt_accel.rolling(norm_window, min_periods=20).mean()
    roll_std = vpt_accel.rolling(norm_window, min_periods=20).std().clip(lower=1e-10)
    vpt_accel_z = (vpt_accel - roll_mean) / roll_std
    return vpt_accel_z.clip(-4, 4).rename('VPT_Acceleration')


def compute_bar_patterns(df, smooth=3):
    """Bar anatomy: upper wick, lower wick, body ratio."""
    bar_range = (df['High'] - df['Low']).clip(lower=1e-10)
    bar_top = df[['Open', 'Close']].max(axis=1)
    bar_bottom = df[['Open', 'Close']].min(axis=1)
    upper_wick = (df['High'] - bar_top) / bar_range
    lower_wick = (bar_bottom - df['Low']) / bar_range
    body_ratio = (bar_top - bar_bottom) / bar_range
    upper_wick_s = upper_wick.rolling(smooth, min_periods=1).mean()
    lower_wick_s = lower_wick.rolling(smooth, min_periods=1).mean()
    body_ratio_s = body_ratio.rolling(smooth, min_periods=1).mean()
    return (upper_wick_s.rename('Upper_Wick'),
            lower_wick_s.rename('Lower_Wick'),
            body_ratio_s.rename('Body_Ratio'))


def compute_beta_alpha(df, proxy_returns, beta_window=20, alpha_window=20, norm_window=60):
    """Beta-adjusted alpha (excess return over market proxy)."""
    ticker_returns = df['Close'].pct_change()
    cov = ticker_returns.rolling(beta_window, min_periods=10).cov(proxy_returns)
    var = proxy_returns.rolling(beta_window, min_periods=10).var().clip(lower=1e-10)
    beta = (cov / var).clip(-3, 3)
    expected_return = beta * proxy_returns
    alpha = ticker_returns - expected_return
    cum_alpha = alpha.rolling(alpha_window, min_periods=5).sum()
    roll_mean = cum_alpha.rolling(norm_window, min_periods=20).mean()
    roll_std = cum_alpha.rolling(norm_window, min_periods=20).std().clip(lower=1e-10)
    return ((cum_alpha - roll_mean) / roll_std).clip(-4, 4).rename('Beta_Alpha')


def compute_atr_channel_pos(df, atr_window=14, channel_window=20, atr_multiplier=1.5):
    """Close position within ATR-adaptive channel, centered at 0."""
    from hedge_fund.indicators import ManualTA
    atr = ManualTA.atr(df['High'], df['Low'], df['Close'], length=atr_window)
    midline = df['Close'].rolling(channel_window, min_periods=5).mean()
    upper = midline + atr_multiplier * atr
    lower = midline - atr_multiplier * atr
    channel_width = (upper - lower).clip(lower=1e-10)
    position = (df['Close'] - lower) / channel_width
    return (position.clip(0, 1) - 0.5).rename('ATR_Channel_Pos')


class FeatureImportanceTracker:
    """Track rolling feature importances across XGBoost training cycles.

    Persists a history of top-10 feature rankings to disk and warns when
    a feature's rank shifts by more than 3 positions between consecutive
    training cycles, which may indicate data drift or regime change.
    """

    TOP_N = 10
    DRIFT_THRESHOLD = 3

    def __init__(self, state_dir='data'):
        self._state_dir = state_dir
        self._state_path = os.path.join(state_dir, 'feature_importance_history.json')
        self._history = []  # list of dicts: [{"feature_name": importance, ...}, ...]
        self._logger = logging.getLogger(__name__)
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, feature_names, importances):
        """Record importances from the latest training cycle.

        Args:
            feature_names: list of feature name strings.
            importances: list/array of importance values (same order).
        """
        if len(feature_names) != len(importances):
            self._logger.warning(
                "feature_names length (%d) != importances length (%d); skipping update",
                len(feature_names), len(importances),
            )
            return

        current = {name: float(imp) for name, imp in zip(feature_names, importances)}
        self._history.append(current)

        # Log drift warnings when we have at least two snapshots
        if len(self._history) >= 2:
            drift = self.get_drift_report()
            for feat, info in drift.items():
                self._logger.warning(
                    "Feature importance drift: '%s' moved from rank %d to %d (shift=%d)",
                    feat, info['prev_rank'], info['curr_rank'], info['shift'],
                )

        self._save_state()

    def get_drift_report(self):
        """Compare the two most recent snapshots and return significant rank changes.

        Returns:
            dict keyed by feature name with values:
                {'prev_rank': int, 'curr_rank': int, 'shift': int}
            Only features whose rank changed by more than DRIFT_THRESHOLD
            are included.  Ranks are 1-based within the top-N union.
        """
        if len(self._history) < 2:
            return {}

        prev_ranking = self._top_n_ranking(self._history[-2])
        curr_ranking = self._top_n_ranking(self._history[-1])

        # Build a unified set of features that appeared in either top-N
        all_features = set(prev_ranking.keys()) | set(curr_ranking.keys())

        # Features absent from a top-N list get a sentinel rank of TOP_N + 1
        sentinel = self.TOP_N + 1
        report = {}
        for feat in all_features:
            prev_rank = prev_ranking.get(feat, sentinel)
            curr_rank = curr_ranking.get(feat, sentinel)
            shift = abs(curr_rank - prev_rank)
            if shift > self.DRIFT_THRESHOLD:
                report[feat] = {
                    'prev_rank': prev_rank,
                    'curr_rank': curr_rank,
                    'shift': shift,
                }
        return report

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_state(self):
        """Persist the importance history to disk."""
        os.makedirs(self._state_dir, exist_ok=True)
        try:
            with open(self._state_path, 'w') as f:
                json.dump(self._history, f, indent=2)
        except OSError as exc:
            self._logger.error("Failed to save feature importance history: %s", exc)

    def _load_state(self):
        """Load previous importance history from disk if available."""
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._history = data
                    self._logger.info(
                        "Loaded %d feature importance snapshots from %s",
                        len(self._history), self._state_path,
                    )
            except (OSError, json.JSONDecodeError) as exc:
                self._logger.error("Failed to load feature importance history: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _top_n_ranking(self, importance_dict):
        """Return {feature: 1-based rank} for the top-N features by importance."""
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return {name: rank for rank, (name, _) in enumerate(sorted_features[:self.TOP_N], start=1)}
