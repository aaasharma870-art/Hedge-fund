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

from hedge_fund.reliability import FailureThresholds, ReliabilityMonitor, structured_failure_log


class ScanBarCache:
    """In-memory cache for live-scan 15-min bar data."""

    def __init__(self, tz=None):
        self._data = {}
        self._last_slot = {}
        self._lock = threading.Lock()
        self._tz = tz
        self._reliability = ReliabilityMonitor("scan_bar_cache", FailureThresholds(degraded_after=5, safe_stop_after=20))

    def _current_slot(self):
        try:
            now = datetime.datetime.now(self._tz) if self._tz else datetime.datetime.now()
        except Exception as e:
            retries = self._reliability.record_failure("slot")
            structured_failure_log(
                component="scan_bar_cache",
                symbol="slot",
                endpoint="datetime.now",
                retry_count=retries,
                error=e,
                logger=logging.debug,
            )
            now = datetime.datetime.now()
        return (now.hour, now.minute // 15)

    def get_if_same_slot(self, ticker):
        slot = self._current_slot()
        with self._lock:
            if ticker in self._last_slot and self._last_slot[ticker] == slot:
                return self._data.get(ticker)
        return None

    def put(self, ticker, df):
        with self._lock:
            self._data[ticker] = df
            self._last_slot[ticker] = self._current_slot()

    def invalidate(self, ticker=None):
        with self._lock:
            if ticker:
                self._data.pop(ticker, None)
                self._last_slot.pop(ticker, None)
            else:
                self._data.clear()
                self._last_slot.clear()


class PolygonBarStream:
    """WebSocket stream for Polygon aggregate minute bars."""

    def __init__(self, api_key, tz=None, ws_module=None):
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
        self._reliability = ReliabilityMonitor("polygon_ws", FailureThresholds(degraded_after=3, safe_stop_after=10))

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
        if not self.has_ws:
            logging.warning("websocket-client not installed. WS bar stream disabled.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PolyWS")
        self._thread.start()
        logging.info("Polygon WebSocket bar stream started")

    def _run_loop(self):
        while self._running:
            try:
                self._connect()
            except Exception as e:
                retries = self._reliability.record_failure("connect")
                structured_failure_log(
                    component="polygon_ws_connect",
                    symbol="all",
                    endpoint="wss://socket.polygon.io/stocks",
                    retry_count=retries,
                    error=e,
                    logger=logging.debug,
                )
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
                            self._reliability.record_success("connect")
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
                                    self._reliability.record_success(sym)
                            except Exception as e:
                                retries = self._reliability.record_failure(sym or "unknown")
                                structured_failure_log(
                                    component="polygon_ws_bar_parse",
                                    symbol=sym or "unknown",
                                    endpoint="AM event",
                                    retry_count=retries,
                                    error=e,
                                    logger=logging.debug,
                                )
            except Exception as e:
                retries = self._reliability.record_failure("message")
                structured_failure_log(
                    component="polygon_ws_message",
                    symbol="all",
                    endpoint="on_message",
                    retry_count=retries,
                    error=e,
                    logger=logging.debug,
                )

        def on_error(ws, error):
            self._connected = False
            retries = self._reliability.record_failure("socket")
            structured_failure_log(
                component="polygon_ws_error",
                symbol="all",
                endpoint="on_error",
                retry_count=retries,
                error=error if isinstance(error, Exception) else Exception(str(error)),
                logger=logging.debug,
            )

        def on_close(ws, code, msg):
            self._connected = False
            logging.info(f"WS closed ({code})")

        self._ws = self._ws_module.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
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
        with self._lock:
            new = set(tickers) - self._subscribed
            self._subscribed = set(tickers)
        if new and self._ws and self._connected:
            try:
                params = ",".join(f"AM.{t}" for t in new)
                self._ws.send(json.dumps({"action": "subscribe", "params": params}))
                self._reliability.record_success("subscribe")
            except Exception as e:
                retries = self._reliability.record_failure("subscribe")
                structured_failure_log(
                    component="polygon_ws_subscribe",
                    symbol=",".join(sorted(new))[:64] or "all",
                    endpoint="subscribe",
                    retry_count=retries,
                    error=e,
                    logger=logging.debug,
                )

    def get_ready_tickers(self):
        with self._lock:
            ready = set(self._ready)
            self._ready.clear()
        self._event.clear()
        return ready

    def wait_for_bars(self, timeout=60.0):
        return self._event.wait(timeout=timeout)

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception as e:
                retries = self._reliability.record_failure("stop")
                structured_failure_log(
                    component="polygon_ws_stop",
                    symbol="all",
                    endpoint="ws.close",
                    retry_count=retries,
                    error=e,
                    logger=logging.debug,
                )
