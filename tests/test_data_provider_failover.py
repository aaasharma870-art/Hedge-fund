import types

import pandas as pd
import requests

from hedge_fund.data_providers import Polygon_Helper
from hedge_fund.reliability import ReliabilityMonitor, classify_failure, retry_with_backoff


class FakeErrorTracker:
    def __init__(self):
        self.failures = []

    def record_failure(self, component, error):
        self.failures.append((component, error))


def test_failure_classifier_transient_vs_fatal():
    assert classify_failure(requests.exceptions.Timeout("slow")) == "transient"
    assert classify_failure(ValueError("bad payload")) == "fatal"


def test_retry_with_backoff_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.Timeout("network")
        return "ok"

    assert retry_with_backoff(flaky, retries=4, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_polygon_failover_records_failures_and_tracker(tmp_path, monkeypatch):
    keys = {"POLY": "poly", "FMP": "fmp"}
    tracker = FakeErrorTracker()
    helper = Polygon_Helper(keys=keys, drive_root=str(tmp_path), error_tracker=tracker)

    # Avoid long sleeps and rate-limiter waits.
    monkeypatch.setattr("time.sleep", lambda *_: None)
    helper._throttle = lambda: None

    # Polygon + FMP calls timeout.
    def always_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("downstream timeout")

    helper.sess.get = always_timeout

    # yfinance fallback returns empty frame.
    fake_yf = types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)

    out = helper.fetch_data("AAPL", days=1, mult=1)

    assert out.empty
    assert any(comp == "Data_AAPL" for comp, _ in tracker.failures)
    assert helper._reliability.failure_count("AAPL") >= 3
    assert helper._reliability.is_degraded("AAPL")


def test_reliability_monitor_safe_stop_threshold():
    monitor = ReliabilityMonitor("provider")
    for _ in range(6):
        monitor.record_failure("AAPL")
    assert monitor.should_safe_stop("AAPL")
