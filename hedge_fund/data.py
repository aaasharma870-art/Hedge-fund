"""
Data infrastructure utilities.

Provides rate limiting and caching for API data fetching.
"""

import time
import threading


class RateLimiter:
    """
    Token-bucket rate limiter for API requests.

    Allows short bursts while capping the average request rate.
    Thread-safe.

    Args:
        rate_per_sec: Sustained requests per second.
        burst: Maximum burst capacity (tokens).
    """

    def __init__(self, rate_per_sec=6.0, burst=10):
        self._rate = rate_per_sec
        self._burst = float(burst)
        self._tokens = float(burst)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, timeout=30.0):
        """
        Block until a token is available.

        Args:
            timeout: Maximum seconds to wait before giving up.

        Returns:
            True if a token was acquired, False on timeout.
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                now = time.time()
                self._tokens = min(
                    self._burst,
                    self._tokens + (now - self._last_refill) * self._rate,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.time() > deadline:
                return False
            time.sleep(0.02)
