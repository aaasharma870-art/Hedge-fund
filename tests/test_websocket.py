"""Tests for hedge_fund.websocket - WebSocket bar stream and scan cache."""

import datetime
import threading
import pytest

from hedge_fund.websocket import ScanBarCache, PolygonBarStream


class TestScanBarCache:
    def test_empty_cache_returns_none(self):
        cache = ScanBarCache()
        assert cache.get_if_same_slot("AAPL") is None

    def test_put_and_get(self):
        cache = ScanBarCache()
        cache.put("AAPL", "mock_df")
        result = cache.get_if_same_slot("AAPL")
        assert result == "mock_df"

    def test_invalidate_ticker(self):
        cache = ScanBarCache()
        cache.put("AAPL", "df1")
        cache.put("TSLA", "df2")
        cache.invalidate("AAPL")
        assert cache.get_if_same_slot("AAPL") is None
        assert cache.get_if_same_slot("TSLA") == "df2"

    def test_invalidate_all(self):
        cache = ScanBarCache()
        cache.put("AAPL", "df1")
        cache.put("TSLA", "df2")
        cache.invalidate()
        assert cache.get_if_same_slot("AAPL") is None
        assert cache.get_if_same_slot("TSLA") is None

    def test_thread_safety(self):
        cache = ScanBarCache()
        errors = []

        def writer():
            for i in range(100):
                cache.put(f"T{i}", f"df_{i}")

        def reader():
            for i in range(100):
                cache.get_if_same_slot(f"T{i}")

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No crash = pass


class TestPolygonBarStream:
    def test_init_no_ws(self):
        stream = PolygonBarStream(api_key="test", ws_module=None)
        assert stream.has_ws is False
        assert stream.is_connected is False

    def test_start_without_ws_module(self):
        stream = PolygonBarStream(api_key="test", ws_module=None)
        stream.start()  # Should log warning, not crash
        assert stream._running is False

    def test_subscribe(self):
        stream = PolygonBarStream(api_key="test", ws_module=None)
        stream.subscribe(["AAPL", "TSLA"])
        assert "AAPL" in stream._subscribed
        assert "TSLA" in stream._subscribed

    def test_get_ready_tickers_empty(self):
        stream = PolygonBarStream(api_key="test", ws_module=None)
        assert stream.get_ready_tickers() == set()

    def test_stop(self):
        stream = PolygonBarStream(api_key="test", ws_module=None)
        stream.stop()  # Should not crash
        assert stream._running is False

    def test_ready_tickers_manual(self):
        stream = PolygonBarStream(api_key="test", ws_module=None)
        stream._ready.add("AAPL")
        stream._ready.add("TSLA")
        ready = stream.get_ready_tickers()
        assert ready == {"AAPL", "TSLA"}
        # Should be cleared after get
        assert stream.get_ready_tickers() == set()
