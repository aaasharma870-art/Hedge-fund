"""
hedge_fund.websocket - WebSocket bar stream and scan cache.

Extracted from bot.py. Provides:
    ScanBarCache     - In-memory 15-min bar cache to reduce API calls ~93%
    PolygonBarStream - WebSocket stream for Polygon aggregate minute bars
"""

import time
import datetime
import logging
import threading
import json


# ---------------------------------------------------------------------------
# Scan Bar Cache
# ---------------------------------------------------------------------------

class ScanBarCache:
    """
    In-memory cache for live-scan 15-min bar data.
    Tracks which 15-minute slot each ticker was last fetched in.
    If the slot hasn't changed, returns cached bars (no API call).
    Reduces API calls by ~93% (14 of every 15 scan cycles skip fetch).
    """

    def __init__(self, tz=None):
        """
        Args:
            tz: timezone object for current time (e.g. ZoneInfo('America/New_York')).
                If None, uses system local time.
        """
        self._data = {}
        self._last_slot = {}
        self._lock = threading.Lock()
        self._tz = tz

    def _current_slot(self):
        """Current 15-min slot as (hour, quarter). E.g. 9:37 -> (9, 2)."""
        try:
            now = datetime.datetime.now(self._tz) if self._tz else datetime.datetime.now()
        except Exception:
            now = datetime.datetime.now()
        return (now.hour, now.minute // 15)

    def get_if_same_slot(self, ticker):
        """Return cached df if we're in the same 15-min slot as last fetch."""
        slot = self._current_slot()
        with self._lock:
            if ticker in self._last_slot and self._last_slot[ticker] == slot:
                return self._data.get(ticker)
        return None

    def put(self, ticker, df):
        """Cache bar data and record current slot."""
        with self._lock:
            self._data[ticker] = df
            self._last_slot[ticker] = self._current_slot()

    def invalidate(self, ticker=None):
        """Clear cache for a ticker (or all)."""
        with self._lock:
            if ticker:
                self._data.pop(ticker, None)
                self._last_slot.pop(ticker, None)
            else:
                self._data.clear()
                self._last_slot.clear()


# ---------------------------------------------------------------------------
# Polygon Bar Stream
# ---------------------------------------------------------------------------

class PolygonBarStream:
    """
    WebSocket stream for Polygon aggregate minute bars.
    Detects 15-min bar closes and exposes ready tickers to the main loop.
    Hybrid: REST stays for snapshots/history, WS only triggers scan timing.
    """

    def __init__(self, api_key, tz=None, ws_module=None):
        """
        Args:
            api_key: Polygon API key for WS auth.
            tz: timezone object for bar boundary detection.
            ws_module: websocket module (websocket-client). If None, attempts import.
        """
        self._api_key = api_key
        self._tz = tz
        self._ws_module = ws_module
        self._ws = None
        self._thread = None
        self._running = False
        self._subscribed = set()
        self._lock = threading.Lock()
        self._ready = set()
        self._event = threading.Event()
        self._connected = False

        if self._ws_module is None:
            try:
                import websocket as _ws
                self._ws_module = _ws
            except ImportError:
                self._ws_module = None

    @property
    def has_ws(self):
        return self._ws_module is not None

    @property
    def is_connected(self):
        return self._connected

    def start(self):
        """Start the WS listener in a background daemon thread."""
        if not self.has_ws:
            logging.warning("websocket-client not installed. WS bar stream disabled.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PolyWS")
        self._thread.start()
        logging.info("Polygon WebSocket bar stream started")

    def _run_loop(self):
        """Reconnecting event loop."""
        while self._running:
            try:
                self._connect()
            except Exception as e:
                logging.debug(f"WS connect error: {e}")
            self._connected = False
            if self._running:
                time.sleep(5)

    def _connect(self):
        url = "wss://socket.polygon.io/stocks"

        def on_open(ws):
            ws.send(json.dumps({"action": "auth", "params": self._api_key}))

        def on_message(ws, raw):
            try:
                msgs = json.loads(raw)
                if not isinstance(msgs, list):
                    msgs = [msgs]
                for msg in msgs:
                    ev = msg.get('ev')
                    if ev == 'status':
                        st = msg.get('status', '')
                        if st == 'auth_success':
                            self._connected = True
                            logging.info("Polygon WS authenticated")
                            self._send_subscriptions(ws)
                        elif st == 'auth_failed':
                            logging.error("Polygon WS auth FAILED")
                            self._running = False
                    elif ev == 'AM':
                        sym = msg.get('sym', '')
                        end_ms = msg.get('e', 0)
                        if sym and end_ms:
                            try:
                                tz = self._tz or datetime.timezone.utc
                                bar_end = datetime.datetime.fromtimestamp(end_ms / 1000, tz=tz)
                                if bar_end.minute % 15 == 0:
                                    with self._lock:
                                        self._ready.add(sym)
                                    self._event.set()
                            except Exception:
                                pass
            except Exception:
                pass

        def on_error(ws, error):
            logging.debug(f"WS error: {error}")
            self._connected = False

        def on_close(ws, code, msg):
            self._connected = False
            logging.info(f"WS closed ({code})")

        self._ws = self._ws_module.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def _send_subscriptions(self, ws):
        with self._lock:
            tickers = list(self._subscribed)
        if tickers:
            params = ",".join(f"AM.{t}" for t in tickers)
            ws.send(json.dumps({"action": "subscribe", "params": params}))
            logging.info(f"Subscribed to AM.* for {len(tickers)} tickers")

    def subscribe(self, tickers):
        """Update subscription list. Safe to call anytime."""
        with self._lock:
            new = set(tickers) - self._subscribed
            self._subscribed = set(tickers)
        if new and self._ws and self._connected:
            try:
                params = ",".join(f"AM.{t}" for t in new)
                self._ws.send(json.dumps({"action": "subscribe", "params": params}))
            except Exception:
                pass

    def get_ready_tickers(self):
        """Return tickers with fresh 15-min bars and clear the set."""
        with self._lock:
            ready = set(self._ready)
            self._ready.clear()
        self._event.clear()
        return ready

    def wait_for_bars(self, timeout=60.0):
        """Block until bar-close event or timeout. Returns True if bars ready."""
        return self._event.wait(timeout=timeout)

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
