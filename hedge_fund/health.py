"""Runtime health monitoring and heartbeat emission."""

import datetime
import json
import logging
import os
import threading
from collections import deque


class RuntimeHealth:
    """Tracks runtime health and emits a probe-friendly heartbeat JSON file."""

    def __init__(
        self,
        heartbeat_path="/tmp/hedge_fund_health.json",
        market_data_stale_s=180,
        broker_sync_stale_s=180,
        latency_window=200,
        exception_window=200,
    ):
        self.heartbeat_path = heartbeat_path
        self.market_data_stale_s = float(market_data_stale_s)
        self.broker_sync_stale_s = float(broker_sync_stale_s)
        self._latencies = deque(maxlen=int(latency_window))
        self._exceptions = deque(maxlen=int(exception_window))
        self._lock = threading.Lock()

        self._last_market_data_update = None
        self._last_broker_sync = None
        self._safe_mode = False
        self._safe_reason = ""

    @staticmethod
    def _utc_now():
        return datetime.datetime.now(datetime.timezone.utc)

    @staticmethod
    def _iso(ts):
        return ts.isoformat() if ts else None

    def mark_market_data_update(self, ts=None):
        with self._lock:
            self._last_market_data_update = ts or self._utc_now()

    def mark_broker_sync(self, ts=None):
        with self._lock:
            self._last_broker_sync = ts or self._utc_now()

    def record_order_submission_latency(self, latency_s):
        with self._lock:
            self._latencies.append(max(0.0, float(latency_s)))

    def record_exception(self, has_exception=True):
        with self._lock:
            self._exceptions.append(1 if has_exception else 0)

    def _compute_mode_locked(self, now):
        stale_reasons = []

        if self._last_market_data_update is None:
            stale_reasons.append("market_data_never_updated")
        else:
            age = (now - self._last_market_data_update).total_seconds()
            if age > self.market_data_stale_s:
                stale_reasons.append(f"market_data_stale:{age:.1f}s")

        if self._last_broker_sync is None:
            stale_reasons.append("broker_sync_never_updated")
        else:
            age = (now - self._last_broker_sync).total_seconds()
            if age > self.broker_sync_stale_s:
                stale_reasons.append(f"broker_sync_stale:{age:.1f}s")

        if stale_reasons:
            self._safe_mode = True
            self._safe_reason = ",".join(stale_reasons)
        else:
            self._safe_mode = False
            self._safe_reason = ""

    def snapshot(self):
        with self._lock:
            now = self._utc_now()
            self._compute_mode_locked(now)

            avg_latency = (sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0
            exception_rate = (sum(self._exceptions) / len(self._exceptions)) if self._exceptions else 0.0

            return {
                "timestamp": self._iso(now),
                "mode": "SAFE_MODE" if self._safe_mode else "LIVE",
                "safe_mode_reason": self._safe_reason,
                "thresholds": {
                    "market_data_stale_s": self.market_data_stale_s,
                    "broker_sync_stale_s": self.broker_sync_stale_s,
                },
                "last_market_data_update": self._iso(self._last_market_data_update),
                "last_broker_sync": self._iso(self._last_broker_sync),
                "order_submission_latency": {
                    "sample_count": len(self._latencies),
                    "avg_seconds": round(avg_latency, 6),
                    "last_seconds": round(self._latencies[-1], 6) if self._latencies else None,
                },
                "open_exception_rate": {
                    "sample_count": len(self._exceptions),
                    "rate": round(exception_rate, 6),
                },
            }

    def write_heartbeat(self, extra=None):
        payload = self.snapshot()
        if extra:
            payload["runtime"] = extra

        tmp = f"{self.heartbeat_path}.tmp"
        os.makedirs(os.path.dirname(self.heartbeat_path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, self.heartbeat_path)
        return payload

    @property
    def in_safe_mode(self):
        return self.snapshot()["mode"] == "SAFE_MODE"

    def current_reason(self):
        return self.snapshot().get("safe_mode_reason", "")
