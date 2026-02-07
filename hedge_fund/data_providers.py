"""
hedge_fund.data_providers - External data source wrappers.

Extracted from bot.py to reduce monolith size. Each class accepts its
dependencies (API keys, drive root, etc.) via constructor parameters
instead of relying on module-level globals.

Classes:
    VIX_Helper        - VIX with multi-source fallback (yfinance -> FMP)
    Polygon_Helper    - Polygon.io REST bars, snapshots, daily data
    FMP_Helper        - FMP screener, news scoring, fundamentals
    FundamentalGuard  - Filter toxic fundamentals (debt spiral, revenue collapse)
    SeekingAlpha_Helper - RapidAPI Seeking Alpha news/sentiment
"""

import os
import time
import datetime
import logging
import threading
import json
import concurrent.futures

import numpy as np
import pandas as pd
import requests

from hedge_fund.data import RateLimiter
from hedge_fund.reliability import (
    FailureThresholds,
    ReliabilityMonitor,
    structured_failure_log,
)


# ---------------------------------------------------------------------------
# VIX Helper
# ---------------------------------------------------------------------------

class VIX_Helper:
    """VIX with multi-source fallback (yfinance -> FMP)."""

    def __init__(self, keys, error_tracker=None):
        """
        Args:
            keys: dict with at least 'FMP' key for API access.
            error_tracker: object with record_success/record_failure methods (optional).
        """
        self._keys = keys
        self._error_tracker = error_tracker
        self.cache = {'value': None, 'ts': 0}
        self.cache_ttl = 60
        self.data_valid = False
        self._reliability = ReliabilityMonitor("vix", FailureThresholds(degraded_after=2, safe_stop_after=5))

    def get_vix(self):
        now = time.time()
        if self.cache['value'] is not None and now - self.cache['ts'] < self.cache_ttl:
            return self.cache['value']

        vix = self._try_yfinance() or self._try_fmp()

        if vix is not None and 5 < vix < 100:
            self.cache = {'value': vix, 'ts': now}
            self.data_valid = True
            if self._error_tracker:
                self._error_tracker.record_success("VIX")
            return vix
        else:
            if self._error_tracker:
                self._error_tracker.record_failure("VIX", f"Invalid: {vix}")
            self.data_valid = False
            return 20.0

    def _try_yfinance(self):
        try:
            import yfinance as yf
            tk = yf.Ticker('^VIX')
            hist = tk.history(period='2d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            retries = self._reliability.record_failure("^VIX")
            structured_failure_log(
                component="vix_yfinance",
                symbol="^VIX",
                endpoint="yfinance://ticker/^VIX/history",
                retry_count=retries,
                error=e,
                logger=logging.debug,
            )
        return None

    def _try_fmp(self):
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/%5EVIX?apikey={self._keys['FMP']}"
            res = requests.get(url, timeout=5).json()
            if res:
                return float(res[0].get('price', 0))
        except Exception as e:
            retries = self._reliability.record_failure("^VIX")
            structured_failure_log(
                component="vix_fmp",
                symbol="^VIX",
                endpoint="https://financialmodelingprep.com/api/v3/quote/%5EVIX",
                retry_count=retries,
                error=e,
                logger=logging.debug,
            )
        return None


# ---------------------------------------------------------------------------
# Polygon Helper
# ---------------------------------------------------------------------------

class Polygon_Helper:
    """Polygon.io REST API wrapper with failover to FMP and yfinance."""

    def __init__(self, keys, drive_root, error_tracker=None, io_workers=16):
        """
        Args:
            keys: dict with 'POLY' and 'FMP' keys.
            drive_root: path for market_cache directory.
            error_tracker: object with record_failure method (optional).
            io_workers: thread pool size for concurrent fetches.
        """
        self._keys = keys
        self._drive_root = drive_root
        self._error_tracker = error_tracker
        self._io_workers = io_workers
        self.sess = requests.Session()
        self.base = "https://api.polygon.io"
        self.last_429 = 0
        self._lock = threading.Lock()
        self._mem_cache = {}
        self._rate_limiter = RateLimiter(rate_per_sec=6.0, burst=10)
        self._reliability = ReliabilityMonitor("polygon", FailureThresholds(degraded_after=3, safe_stop_after=8))

    def _throttle(self):
        self._rate_limiter.acquire()

    def fetch_snapshot_prices(self, tickers, ttl=5):
        """
        Polygon Snapshot: 1 call returns latest price & day stats for many tickers.
        Returns dict: { "AAPL": {"price": 123.4, "dayVol": 123, ...}, ... }
        """
        if not tickers:
            return {}

        key = ("snap", tuple(sorted(tickers)))
        now = time.time()
        if key in self._mem_cache and (now - self._mem_cache[key][0] < ttl):
            return self._mem_cache[key][1]

        out = {}
        chunk_size = 200
        chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]

        def fetch_chunk(chunk):
            try:
                url = f"{self.base}/v2/snapshot/locale/us/markets/stocks/tickers"
                params = {"tickers": ",".join(chunk), "apiKey": self._keys["POLY"]}
                self._throttle()
                r = self.sess.get(url, params=params, timeout=10)
                if r.status_code == 429:
                    with self._lock:
                        self.last_429 = time.time()
                    return []
                if r.status_code != 200:
                    return []
                return r.json().get("tickers", [])
            except Exception as e:
                retries = self._reliability.record_failure("snapshot")
                structured_failure_log(
                    component="polygon_snapshot",
                    symbol=",".join(chunk[:3]) if chunk else "unknown",
                    endpoint=f"{self.base}/v2/snapshot/locale/us/markets/stocks/tickers",
                    retry_count=retries,
                    error=e,
                    logger=logging.debug,
                )
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._io_workers) as executor:
            futures = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                for item in data:
                    sym = item.get("ticker")
                    if not sym:
                        continue
                    last_trade = item.get("lastTrade") or {}
                    day = item.get("day") or {}
                    price = last_trade.get("p") or day.get("c")
                    if price is None:
                        continue
                    out[sym] = {
                        "price": float(price),
                        "dayVol": float(day.get("v") or 0),
                        "dayVWAP": float(day.get("vw") or 0),
                        "dayOpen": float(day.get("o") or 0),
                        "dayHigh": float(day.get("h") or 0),
                        "dayLow": float(day.get("l") or 0),
                        "dayClose": float(day.get("c") or 0),
                    }

        self._mem_cache[key] = (now, out)
        return out

    def fetch_data(self, t, days=5, mult=1):
        """
        Fetch Polygon aggregate bars with pagination.
        Falls back to FMP then yfinance on failure.
        """
        with self._lock:
            if time.time() - self.last_429 < 60:
                time.sleep(60 - (time.time() - self.last_429))

        cache = os.path.join(self._drive_root, f"market_cache/{t}_{mult}min_{days}d.parquet")
        if os.path.exists(cache):
            age = time.time() - os.path.getmtime(cache)
            max_age = 60 if days <= 5 else 2592000
            if age < max_age:
                try:
                    df = pd.read_parquet(cache)
                    if len(df) > 0:
                        logging.info(f"📂 Cached {t}: {len(df)} bars (Age: {age/3600:.1f}h)")
                        return df
                except Exception as e:
                    logging.debug(f"Cache read failed for {t}: {e}")

        end = datetime.datetime.now(datetime.timezone.utc)
        start = end - datetime.timedelta(days=days)

        url = (
            f"{self.base}/v2/aggs/ticker/{t}/range/{mult}/minute/"
            f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
            f"?adjusted=true&limit=50000&sort=asc&apiKey={self._keys['POLY']}"
        )

        all_rows = []
        max_retries = 12

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
                            url = url + ("&" if "?" in url else "?") + f"apiKey={self._keys['POLY']}"
                        break
                    elif r.status_code == 429:
                        retries += 1
                        with self._lock:
                            self.last_429 = time.time()
                        wait_time = 60 + (retries * 5)
                        logging.warning(f"⚠️ Polygon 429 {t} (Attempt {retries}/{max_retries}) -> Sleeping {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.warning(f"Polygon {t} status {r.status_code}: {r.text[:200] if r.text else 'no body'}")
                        url = None
                        break
                except Exception as e:
                    retries += 1
                    failure_count = self._reliability.record_failure(t)
                    structured_failure_log(
                        component="polygon_aggs",
                        symbol=t,
                        endpoint=url,
                        retry_count=retries,
                        error=e,
                        logger=logging.warning,
                    )
                    if self._reliability.is_degraded(t):
                        logging.warning(
                            "Polygon monitor degraded",
                            extra={"component": "polygon_aggs", "symbol": t, "failure_count": failure_count},
                        )
                    if self._reliability.should_safe_stop(t):
                        logging.critical(
                            "Polygon monitor safe-stop threshold reached",
                            extra={"component": "polygon_aggs", "symbol": t, "failure_count": failure_count},
                        )
                    time.sleep(3)

            if retries >= max_retries:
                url = None

        if not all_rows:
            # FAILOVER 1: FMP
            try:
                logging.warning(f"⚠️ Polygon {t}: No data. Failing over to FMP...")
                fmp_from = start.strftime('%Y-%m-%d')
                fmp_to = end.strftime('%Y-%m-%d')
                fmp_url = (
                    f"https://financialmodelingprep.com/api/v3/historical-chart/15min/{t}"
                    f"?from={fmp_from}&to={fmp_to}&apikey={self._keys['FMP']}"
                )
                r_fmp = self.sess.get(fmp_url, timeout=15)
                if r_fmp.status_code == 200:
                    data_fmp = r_fmp.json()
                    if data_fmp and isinstance(data_fmp, list):
                        df_fmp = pd.DataFrame(data_fmp)
                        if 'date' in df_fmp.columns:
                            df_fmp = df_fmp.rename(columns={
                                'date': 'Datetime', 'open': 'Open', 'high': 'High',
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                            })
                            df_fmp['Datetime'] = pd.to_datetime(df_fmp['Datetime'])
                            df_fmp = df_fmp.set_index('Datetime').sort_index()
                            if len(df_fmp) > 100:
                                logging.info(f"✅ FMP {t}: Fetched {len(df_fmp)} bars ({fmp_from} to {fmp_to})")
                                try:
                                    df_fmp.to_parquet(cache)
                                except Exception:
                                    pass
                                return df_fmp
                        elif r_fmp.status_code == 403:
                            logging.warning(f"⚠️ FMP {t}: 403 Forbidden - check FMP plan/key")
                        else:
                            logging.debug(f"FMP {t}: status {r_fmp.status_code}")
            except Exception as ef:
                retries = self._reliability.record_failure(t)
                structured_failure_log(
                    component="polygon_failover_fmp",
                    symbol=t,
                    endpoint="https://financialmodelingprep.com/api/v3/historical-chart/15min",
                    retry_count=retries,
                    error=ef,
                )

            # FAILOVER 2: yfinance
            try:
                import yfinance as yf
                logging.warning(f"⚠️ Polygon {t}: No data. Failing over to Yahoo Finance (60d max)...")
                df_yf = yf.download(t, period="59d", interval="15m", progress=False, auto_adjust=True)
                if not df_yf.empty:
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = df_yf.columns.get_level_values(0)
                    df_yf = df_yf.reset_index()
                    df_yf.columns = [c.lower() for c in df_yf.columns]
                    rename_map = {
                        'date': 'Datetime', 'datetime': 'Datetime',
                        'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    }
                    df_yf = df_yf.rename(columns=rename_map)
                    if 'Datetime' in df_yf.columns:
                        df_yf = df_yf.set_index('Datetime')
                    if df_yf.index.tz is not None:
                        df_yf.index = df_yf.index.tz_convert("America/New_York")
                    df_yf = df_yf.sort_index()
                    df_yf = df_yf[~df_yf.index.duplicated()]
                    logging.info(f"✅ YFinance {t}: Fetched {len(df_yf)} bars")
                    return df_yf
            except Exception as ey:
                retries = self._reliability.record_failure(t)
                structured_failure_log(
                    component="polygon_failover_yfinance",
                    symbol=t,
                    endpoint="yfinance://download",
                    retry_count=retries,
                    error=ey,
                    logger=logging.debug,
                )

            if self._error_tracker:
                self._error_tracker.record_failure(f"Data_{t}", "No data (Poly+YF)")
            return pd.DataFrame()

        self._reliability.record_success(t)

        df = pd.DataFrame(all_rows).rename(columns={
            't': 'Datetime', 'c': 'Close', 'o': 'Open',
            'h': 'High', 'l': 'Low', 'v': 'Volume'
        })
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms', utc=True).dt.tz_convert("America/New_York")
        df = df.set_index('Datetime').sort_index()
        df = df[~df.index.duplicated()]

        try:
            df.to_parquet(cache)
        except Exception as e:
            logging.debug(f"Cache write failed for {t}: {e}")

        return df

    def fetch_daily_data(self, t, days=365):
        """Fetch daily bars from Polygon. Cached for 1 hour."""
        key = ("daily", t, days)
        now = time.time()
        with self._lock:
            if key in self._mem_cache and (now - self._mem_cache[key][0] < 3600):
                return self._mem_cache[key][1]

        end = datetime.datetime.now(datetime.timezone.utc).date()
        start = end - datetime.timedelta(days=days)

        url = (
            f"{self.base}/v2/aggs/ticker/{t}/range/1/day/{start}/{end}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={self._keys['POLY']}"
        )
        try:
            self._throttle()
            r = self.sess.get(url, timeout=15)
            if r.status_code != 200:
                return pd.DataFrame()

            rows = r.json().get("results", [])
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows).rename(columns={
                't': 'Datetime', 'c': 'Close', 'o': 'Open',
                'h': 'High', 'l': 'Low', 'v': 'Volume'
            })
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms', utc=True).dt.tz_convert("America/New_York")
            df = df.set_index('Datetime').sort_index()

            with self._lock:
                self._mem_cache[key] = (now, df)
            return df
        except Exception as e:
            retries = self._reliability.record_failure(t)
            structured_failure_log(
                component="polygon_daily",
                symbol=t,
                endpoint=url,
                retry_count=retries,
                error=e,
                logger=logging.debug,
            )
            return pd.DataFrame()

    def fetch_batch_bars(self, tickers, days=30):
        """
        Fetch daily close prices for multiple tickers as a single DataFrame.
        Used by the portfolio optimizer for covariance estimation.
        """
        frames = {}
        for t in tickers:
            try:
                df = self.fetch_daily_data(t, days=days)
                if not df.empty and 'Close' in df.columns:
                    frames[t] = df['Close']
            except Exception as e:
                logging.debug(f"fetch_batch_bars({t}) failed: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        result = result.sort_index().ffill().dropna(how='all')
        return result


# ---------------------------------------------------------------------------
# FMP Helper
# ---------------------------------------------------------------------------

class FMP_Helper:
    """Financial Modeling Prep API wrapper for screener, news, and fundamentals."""

    def __init__(self, keys):
        """
        Args:
            keys: dict with 'FMP' key.
        """
        self._keys = keys
        self.news_cache = {}

    def get_dynamic_universe(self, exclude=None, max_tickers=30):
        """
        Supertrend Gold Screener -- finds high-momentum trending stocks.
        Returns top max_tickers ranked by YTD performance.
        """
        exclude = exclude or set()

        now = time.time()
        if hasattr(self, '_stg_cache') and (now - self._stg_cache[0] < 1800):
            cached = [s for s in self._stg_cache[1] if s not in exclude]
            logging.info(f"Supertrend Gold: returning {len(cached[:max_tickers])} from cache")
            return cached[:max_tickers]

        try:
            TARGET_SECTORS = ['Technology', 'Industrials', 'Healthcare']
            raw = []
            for sector in TARGET_SECTORS:
                try:
                    url = (
                        f"https://financialmodelingprep.com/api/v3/stock-screener?"
                        f"marketCapMoreThan=2000000000&volumeMoreThan=1000000"
                        f"&betaMoreThan=1.5&priceMoreThan=5"
                        f"&sector={sector}&limit=100"
                        f"&apikey={self._keys['FMP']}"
                    )
                    resp = requests.get(url, timeout=10).json()
                    if isinstance(resp, list):
                        raw.extend(resp)
                except Exception as e:
                    logging.warning(f"Supertrend Gold screener failed for {sector}: {e}")

            seen = set()
            candidates = []
            for x in raw:
                sym = x.get('symbol', '')
                if (sym and sym not in seen
                        and x.get('exchangeShortName') in ('NASDAQ', 'NYSE')
                        and not x.get('isEtf', False)):
                    seen.add(sym)
                    candidates.append(x)

            logging.info(f"Supertrend Gold: {len(candidates)} passed hard filters")

            candidates.sort(key=lambda x: x.get('volume', 0), reverse=True)
            to_check = [c['symbol'] for c in candidates[:60]]

            if not to_check:
                self._stg_cache = (now, [])
                return []

            qualified = []
            for sym in to_check:
                try:
                    url = (
                        f"https://financialmodelingprep.com/api/v3/"
                        f"historical-price-full/{sym}?timeseries=250"
                        f"&apikey={self._keys['FMP']}"
                    )
                    data = requests.get(url, timeout=8).json()
                    bars = data.get('historical', [])
                    if len(bars) < 200:
                        continue

                    bars = list(reversed(bars))
                    closes = pd.Series([b['close'] for b in bars], dtype=float)
                    highs = pd.Series([b['high'] for b in bars], dtype=float)
                    lows = pd.Series([b['low'] for b in bars], dtype=float)
                    price = closes.iloc[-1]

                    sma20 = closes.rolling(20).mean().iloc[-1]
                    sma50 = closes.rolling(50).mean().iloc[-1]
                    sma200 = closes.rolling(200).mean().iloc[-1]
                    if not (price > sma50 and price > sma200 and sma20 > sma50):
                        continue

                    delta = closes.diff()
                    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
                    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
                    rs = gain / loss.replace(0, 1e-10)
                    rsi_val = float((100 - 100 / (1 + rs)).iloc[-1])
                    if not (50 <= rsi_val <= 70):
                        continue

                    tr = pd.concat([
                        highs - lows,
                        (highs - closes.shift(1)).abs(),
                        (lows - closes.shift(1)).abs()
                    ], axis=1).max(axis=1)
                    up = highs.diff()
                    down = -lows.diff()
                    import numpy as _np
                    plus_dm = pd.Series(
                        _np.where((up > down) & (up > 0), up, 0.0),
                        index=closes.index)
                    minus_dm = pd.Series(
                        _np.where((down > up) & (down > 0), down, 0.0),
                        index=closes.index)
                    atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
                    plus_di = 100 * plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14
                    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14
                    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
                    adx_val = float(dx.ewm(alpha=1/14, min_periods=14).mean().iloc[-1])
                    if adx_val < 25:
                        continue

                    year_str = str(pd.Timestamp.now().year)
                    dates = [b['date'] for b in bars]
                    ytd_idx = None
                    for i, d in enumerate(dates):
                        if d.startswith(year_str):
                            ytd_idx = i
                            break
                    if ytd_idx is None:
                        continue
                    ytd_perf = (price / closes.iloc[ytd_idx] - 1) * 100
                    if ytd_perf < 20:
                        continue

                    qualified.append((sym, ytd_perf, rsi_val, adx_val))
                except Exception:
                    continue

            logging.info(
                f"Supertrend Gold: {len(qualified)}/{len(to_check)} passed "
                f"technical filters"
            )

            qualified.sort(key=lambda x: x[1], reverse=True)
            result = [x[0] for x in qualified]
            self._stg_cache = (now, result)

            filtered = [s for s in result if s not in exclude]
            return filtered[:max_tickers]

        except Exception as e:
            logging.error(f"Supertrend Gold screener failed: {e}")
            return []

    def news_score(self, ticker, lookback_hours=24, limit=30):
        """Returns integer news severity score (0=neutral, 1=mild, 2=strong, 3+=severe)."""
        key = (ticker, lookback_hours)
        now = time.time()

        if key in self.news_cache and (now - self.news_cache[key][0] < 300):
            return self.news_cache[key][1]

        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={self._keys['FMP']}"
            items = requests.get(url, timeout=5).json() or []
        except Exception:
            items = []

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=lookback_hours)

        severe = ["fraud", "subpoena", "sec", "doj", "indict", "bankrupt", "restatement", "delist"]
        strong = ["lawsuit", "investigation", "probe", "recall", "halt", "crash", "explosion", "guidance cut"]
        mild = ["downgrade", "miss", "cuts", "warns", "weak", "slows", "delay"]

        score = 0
        for it in items:
            title = (it.get("title") or "").lower()
            dt_str = it.get("publishedDate") or it.get("published_date") or ""
            try:
                dt = pd.to_datetime(dt_str, utc=True)
            except Exception:
                dt = None
            if dt is not None and dt < cutoff:
                continue

            if any(w in title for w in severe):
                score = max(score, 3)
            elif any(w in title for w in strong):
                score = max(score, 2)
            elif any(w in title for w in mild):
                score = max(score, 1)

        self.news_cache[key] = (now, score)
        return score

    def news_scores_batch(self, tickers, lookback_hours=24, limit=30, cache_ttl=300, chunk_size=20):
        """Batch news scoring. Returns {ticker: score_int}."""
        tickers = list(dict.fromkeys([t for t in tickers if t]))
        if not tickers:
            return {}

        key = ("news_batch", tuple(sorted(tickers)), lookback_hours, limit)
        now = time.time()

        if key in self.news_cache and (now - self.news_cache[key][0] < cache_ttl):
            return self.news_cache[key][1]

        severe = ["fraud", "subpoena", "sec", "doj", "indict", "bankrupt", "restatement", "delist"]
        strong = ["lawsuit", "investigation", "probe", "recall", "halt", "crash", "explosion", "guidance cut"]
        mild = ["downgrade", "miss", "cuts", "warns", "weak", "slows", "delay"]

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=lookback_hours)
        scores = {t: 0 for t in tickers}

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            try:
                url = "https://financialmodelingprep.com/api/v3/stock_news"
                params = {"tickers": ",".join(chunk), "limit": limit, "apikey": self._keys["FMP"]}
                items = requests.get(url, params=params, timeout=5).json() or []
            except Exception:
                items = []

            for it in items:
                tkr = (it.get("symbol") or it.get("ticker") or "").upper()
                if tkr not in scores:
                    continue

                dt_str = it.get("publishedDate") or it.get("published_date") or ""
                try:
                    dt = pd.to_datetime(dt_str, utc=True)
                except Exception:
                    dt = None
                if dt is not None and dt < cutoff:
                    continue

                title = (it.get("title") or "").lower()
                txt = (it.get("text") or "").lower()
                content = title + " " + txt

                if any(w in content for w in severe):
                    scores[tkr] = max(scores[tkr], 3)
                elif any(w in content for w in strong):
                    scores[tkr] = max(scores[tkr], 2)
                elif any(w in content for w in mild):
                    scores[tkr] = max(scores[tkr], 1)

        self.news_cache[key] = (now, scores)
        return scores

    def get_ratios(self, ticker):
        """Fetch Key TTM Ratios (P/E, Debt/Eq)."""
        try:
            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={self._keys['FMP']}"
            res = requests.get(url, timeout=5).json()
            if res and isinstance(res, list):
                return res[0]
        except Exception:
            pass
        return {}

    def get_growth(self, ticker):
        """Fetch Financial Growth (Revenue, Income)."""
        try:
            url = f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?limit=1&apikey={self._keys['FMP']}"
            res = requests.get(url, timeout=5).json()
            if res and isinstance(res, list):
                return res[0]
        except Exception:
            pass
        return {}

    def get_fundamental_features(self, ticker):
        """Fetch fundamental features for ML. Cached for 24h per ticker."""
        cache_key = ("fundamentals", ticker)
        now = time.time()
        if cache_key in self.news_cache and (now - self.news_cache[cache_key][0] < 86400):
            return self.news_cache[cache_key][1]

        result = {'earnings_surprise': 0.0, 'revenue_growth_yoy': 0.0, 'pe_ratio': 20.0, 'news_impact_weight': 0.0}
        try:
            ratios = self.get_ratios(ticker)
            pe = float(ratios.get('peRatioTTM', 0) or 0)
            result['pe_ratio'] = max(-100, min(500, pe)) if pe else 20.0

            growth = self.get_growth(ticker)
            rev_g = float(growth.get('revenueGrowth', 0) or 0)
            result['revenue_growth_yoy'] = max(-1.0, min(5.0, rev_g))

            url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={self._keys['FMP']}"
            res = requests.get(url, timeout=5).json()
            if res and isinstance(res, list) and len(res) > 0:
                latest = res[0]
                actual = float(latest.get('actualEarningResult', 0) or 0)
                estimated = float(latest.get('estimatedEarning', 0) or 0)
                if estimated != 0:
                    result['earnings_surprise'] = max(-5.0, min(5.0, (actual - estimated) / abs(estimated)))
        except Exception as e:
            logging.debug(f"Fundamental features {ticker}: {e}")

        self.news_cache[cache_key] = (now, result)
        return result


# ---------------------------------------------------------------------------
# Fundamental Guard
# ---------------------------------------------------------------------------

class FundamentalGuard:
    """Filter out companies with toxic fundamentals."""

    def __init__(self, fmp_helper):
        self.fmp = fmp_helper
        self.cache = {}

    def check_healthy(self, ticker):
        now = datetime.datetime.now()
        if ticker in self.cache:
            exp, safe = self.cache[ticker]
            if now < exp:
                return safe

        try:
            ratios = self.fmp.get_ratios(ticker)
            debt_eq = float(ratios.get('debtEquityRatio', 0) or 0)

            growth = self.fmp.get_growth(ticker)
            rev_growth = float(growth.get('revenueGrowth', 0) or 0)

            is_safe = True
            if debt_eq > 5.0:
                is_safe = False
            if rev_growth < -0.20:
                is_safe = False

            self.cache[ticker] = (now + datetime.timedelta(hours=24), is_safe)
            return is_safe
        except Exception:
            self.cache[ticker] = (now + datetime.timedelta(hours=6), True)
            return True


# ---------------------------------------------------------------------------
# SeekingAlpha Helper
# ---------------------------------------------------------------------------

class SeekingAlpha_Helper:
    """RapidAPI Seeking Alpha wrapper for news/sentiment features."""

    def __init__(self, keys, drive_root):
        """
        Args:
            keys: dict with 'RAPIDAPI_KEY' key.
            drive_root: path for cache directory.
        """
        self._keys = keys
        self.host = "seeking-alpha-finance.p.rapidapi.com"
        self.key = keys.get('RAPIDAPI_KEY')
        self.base = f"https://{self.host}"
        self.sess = requests.Session()
        self.cache_dir = os.path.join(drive_root, 'market_cache', 'sa_cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get(self, path, params=None, cache_key=None):
        if not self.key:
            return {}

        if cache_key:
            cp = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cp):
                try:
                    with open(cp, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logging.debug(f"SA Cache Read Error: {e}")

        headers = {
            "x-rapidapi-key": self.key,
            "x-rapidapi-host": self.host
        }
        try:
            url = f"{self.base}{path}"
            time.sleep(0.25)
            r = self.sess.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if cache_key:
                    with open(cp, 'w') as f:
                        json.dump(data, f)
                return data
            else:
                logging.debug(f"SA {path} error: {r.status_code}")
                return {}
        except Exception as e:
            logging.debug(f"SA request failed: {e}")
            return {}

    def get_news_features(self, ticker):
        """Return dict of news features: count_3d, count_7d, sentiment_score."""
        if not self.key:
            return {'sa_news_count_3d': 0, 'sa_news_count_7d': 0, 'sa_sentiment_score': 0}

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        data = self._get("/v1/symbols/news",
                         {"ticker_slug": ticker.lower()},
                         cache_key=f"news_{ticker}_{today}")

        now = datetime.datetime.now(datetime.timezone.utc)
        count_3d = 0
        count_7d = 0
        sentiment_score = 0.0

        pos_words = {'beat', 'jump', 'surge', 'buy', 'strong', 'upgrade', 'record', 'growth', 'bull', 'top', 'winner'}
        neg_words = {'miss', 'fall', 'drop', 'sell', 'weak', 'downgrade', 'loss', 'bear', 'crash', 'risk', 'warn'}

        items = data.get('data', [])
        scored_items = 0

        for item in items:
            try:
                attr = item.get('attributes', {})
                pub_str = attr.get('publishOn')
                if not pub_str:
                    continue

                pub = pd.to_datetime(pub_str, utc=True)
                age = (now - pub).days

                if age <= 7:
                    if age <= 3:
                        count_3d += 1
                    count_7d += 1

                    title = (attr.get('title') or "").lower()
                    title_words = set(title.split())

                    s_val = 0
                    if title_words & pos_words:
                        s_val += 1
                    if title_words & neg_words:
                        s_val -= 1

                    sentiment_score += s_val
                    scored_items += 1
            except Exception:
                pass

        avg_sentiment = (sentiment_score / scored_items) if scored_items > 0 else 0

        return {
            'sa_news_count_3d': count_3d,
            'sa_news_count_7d': count_7d,
            'sa_sentiment_score': avg_sentiment
        }

    def get_ratings(self, ticker):
        """Fetch Quant and Analyst Ratings. Returns 1-5 scale."""
        if not self.key:
            return {'sa_quant_rating': 3, 'sa_analyst_rating': 3}

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        data = self._get("/v1/symbols/summary",
                         {"ticker_slug": ticker.lower()},
                         cache_key=f"summary_{ticker}_{today}")

        try:
            attr = data.get('data', {}).get('attributes', {})

            def map_rating(val):
                if not val:
                    return 3
                val = float(val)
                return max(1, min(5, val))

            quant = map_rating(attr.get('quantRating', 3))
            authors = map_rating(attr.get('authorsRating', 3))

            return {
                'sa_quant_rating': quant,
                'sa_analyst_rating': authors
            }
        except Exception:
            return {'sa_quant_rating': 3, 'sa_analyst_rating': 3}
