"""Tests for hedge_fund.data_providers - External data source wrappers."""

import datetime
import pytest

from hedge_fund.data_providers import (
    VIX_Helper,
    Polygon_Helper,
    FMP_Helper,
    FundamentalGuard,
    SeekingAlpha_Helper,
)


class FakeErrorTracker:
    def __init__(self):
        self.successes = []
        self.failures = []

    def record_success(self, key):
        self.successes.append(key)

    def record_failure(self, key, msg=""):
        self.failures.append((key, msg))


class TestVIXHelper:
    def test_init(self):
        keys = {"FMP": "test_key"}
        v = VIX_Helper(keys=keys)
        assert v.cache['value'] is None
        assert v.data_valid is False

    def test_cache_returns_on_hit(self):
        keys = {"FMP": "test_key"}
        v = VIX_Helper(keys=keys)
        import time
        v.cache = {'value': 18.5, 'ts': time.time()}
        assert v.get_vix() == 18.5

    def test_fallback_on_invalid_vix(self):
        keys = {"FMP": "fake"}
        tracker = FakeErrorTracker()
        v = VIX_Helper(keys=keys, error_tracker=tracker)
        # Both yfinance and FMP will fail with fake keys
        result = v.get_vix()
        assert result == 20.0  # default fallback
        assert v.data_valid is False

    def test_error_tracker_called(self):
        keys = {"FMP": "fake"}
        tracker = FakeErrorTracker()
        v = VIX_Helper(keys=keys, error_tracker=tracker)
        v.get_vix()
        assert len(tracker.failures) > 0


class TestPolygonHelper:
    def test_init(self):
        keys = {"POLY": "test", "FMP": "test"}
        p = Polygon_Helper(keys=keys, drive_root="/tmp")
        assert p.base == "https://api.polygon.io"
        assert p.last_429 == 0

    def test_snapshot_empty_tickers(self):
        keys = {"POLY": "test", "FMP": "test"}
        p = Polygon_Helper(keys=keys, drive_root="/tmp")
        assert p.fetch_snapshot_prices([]) == {}

    def test_mem_cache_structure(self):
        keys = {"POLY": "test", "FMP": "test"}
        p = Polygon_Helper(keys=keys, drive_root="/tmp")
        assert isinstance(p._mem_cache, dict)


class TestFMPHelper:
    def test_init(self):
        keys = {"FMP": "test_key"}
        f = FMP_Helper(keys=keys)
        assert f.news_cache == {}

    def test_news_score_cache(self):
        """Cached news score should be returned."""
        import time
        keys = {"FMP": "test"}
        f = FMP_Helper(keys=keys)
        f.news_cache[("AAPL", 24)] = (time.time(), 2)
        assert f.news_score("AAPL") == 2

    def test_news_scores_batch_empty(self):
        keys = {"FMP": "test"}
        f = FMP_Helper(keys=keys)
        assert f.news_scores_batch([]) == {}

    def test_get_ratios_returns_dict(self):
        keys = {"FMP": "fake_key_wont_work"}
        f = FMP_Helper(keys=keys)
        # API call will fail, returns empty dict
        result = f.get_ratios("FAKE_TICKER")
        assert isinstance(result, dict)

    def test_get_fundamental_features_defaults(self):
        """Should return defaults when API fails."""
        keys = {"FMP": "fake"}
        f = FMP_Helper(keys=keys)
        result = f.get_fundamental_features("FAKE")
        assert result['pe_ratio'] == 20.0
        assert result['earnings_surprise'] == 0.0

    def test_fundamental_features_cached(self):
        """Second call should hit cache."""
        import time
        keys = {"FMP": "fake"}
        f = FMP_Helper(keys=keys)
        cached = {'earnings_surprise': 1.0, 'revenue_growth_yoy': 0.5, 'pe_ratio': 15.0, 'news_impact_weight': 0.0}
        f.news_cache[("fundamentals", "AAPL")] = (time.time(), cached)
        result = f.get_fundamental_features("AAPL")
        assert result['pe_ratio'] == 15.0


class TestFundamentalGuard:
    def test_healthy_on_failure(self):
        """Should return True (fail-safe) when API call fails."""
        keys = {"FMP": "fake"}
        fmp = FMP_Helper(keys=keys)
        guard = FundamentalGuard(fmp)
        assert guard.check_healthy("FAKE") is True

    def test_cache_works(self):
        keys = {"FMP": "fake"}
        fmp = FMP_Helper(keys=keys)
        guard = FundamentalGuard(fmp)
        # First call populates cache
        guard.check_healthy("AAPL")
        # Second call should use cache
        assert guard.check_healthy("AAPL") is True
        # Only one entry in cache
        assert "AAPL" in guard.cache


class TestSeekingAlphaHelper:
    def test_init_no_key(self):
        keys = {}
        sa = SeekingAlpha_Helper(keys=keys, drive_root="/tmp")
        assert sa.key is None

    def test_news_features_no_key(self):
        keys = {}
        sa = SeekingAlpha_Helper(keys=keys, drive_root="/tmp")
        result = sa.get_news_features("AAPL")
        assert result == {'sa_news_count_3d': 0, 'sa_news_count_7d': 0, 'sa_sentiment_score': 0}

    def test_ratings_no_key(self):
        keys = {}
        sa = SeekingAlpha_Helper(keys=keys, drive_root="/tmp")
        result = sa.get_ratings("AAPL")
        assert result == {'sa_quant_rating': 3, 'sa_analyst_rating': 3}
