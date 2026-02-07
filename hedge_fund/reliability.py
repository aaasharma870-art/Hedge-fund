"""Reliability helpers for retries, failure classification, and circuit-breaker state."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import requests


TRANSIENT_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)


@dataclass
class FailureThresholds:
    degraded_after: int = 3
    safe_stop_after: int = 6


class ReliabilityMonitor:
    """Track failures and expose degrade/safe-stop state."""

    def __init__(self, component: str, thresholds: Optional[FailureThresholds] = None):
        self.component = component
        self.thresholds = thresholds or FailureThresholds()
        self._counts = defaultdict(int)

    def record_success(self, symbol: str = "global") -> None:
        self._counts[symbol] = 0

    def record_failure(self, symbol: str = "global") -> int:
        self._counts[symbol] += 1
        return self._counts[symbol]

    def failure_count(self, symbol: str = "global") -> int:
        return self._counts[symbol]

    def is_degraded(self, symbol: str = "global") -> bool:
        return self.failure_count(symbol) >= self.thresholds.degraded_after

    def should_safe_stop(self, symbol: str = "global") -> bool:
        return self.failure_count(symbol) >= self.thresholds.safe_stop_after


def classify_failure(error: Exception) -> str:
    return "transient" if isinstance(error, TRANSIENT_EXCEPTIONS) else "fatal"


def structured_failure_log(
    *,
    component: str,
    symbol: str,
    endpoint: str,
    retry_count: int,
    error: Exception,
    logger: Callable[[str], None] = logging.warning,
) -> Dict[str, Any]:
    """Log JSON-formatted failure details and return the payload."""
    payload = {
        "component": component,
        "symbol": symbol,
        "endpoint": endpoint,
        "retry_count": retry_count,
        "classification": classify_failure(error),
        "error": str(error),
        "error_type": type(error).__name__,
    }
    logger(f"reliability_failure={json.dumps(payload, sort_keys=True)}")
    return payload


def retry_with_backoff(
    func: Callable[[], Any],
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter: float = 0.2,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Any:
    """Execute a callable with retry/backoff for transient errors."""
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:  # intentionally broad for wrapper behavior
            attempt += 1
            if classify_failure(exc) == "fatal" or attempt > retries:
                raise
            if on_retry:
                on_retry(attempt, exc)
            sleep_for = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_for += random.uniform(0, jitter)
            time.sleep(sleep_for)
