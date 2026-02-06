"""Tests for hedge_fund.scanner - candidate evaluation and filtering."""

import pytest

from hedge_fund.scanner import CandidateScanner


DEFAULT_SETTINGS = {
    "MIN_CONFIDENCE": 0.02,
    "MIN_HURST": 0.38,
    "NEWS_HARD_SKIP_SCORE": 3,
    "NEWS_SOFT_PENALTY_SCORE": 1,
    "NEWS_PENALTY_SIZE_MULT": 0.75,
    "TIER_1": {
        "NAME": "SPECIALIST",
        "MIN_PROB": 0.30,
        "MAX_HURST": 0.55,
        "MIN_ADX": 25,
        "RISK_MULT": 2.0,
    },
    "TIER_2": {
        "NAME": "GRINDER",
        "MIN_PROB": 0.20,
        "MIN_ADX": 20,
        "RISK_MULT": 1.0,
    },
}

GOOD_FEATURES = {
    'RSI': 55.0,
    'ADX': 30.0,
    'Hurst': 0.45,
    'VPIN': 0.3,
    'Amihud_Illiquidity': 0.2,
    'VWAP_ZScore': 0.5,
    'Volatility_Rank': 0.6,
    'EMA_200': 100.0,
    'ATR': 2.0,
    'Close': 105.0,
}


class TestCandidateScanner:
    def test_good_candidate_passes(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        result = scanner.evaluate_candidate("AAPL", 0.25, GOOD_FEATURES)
        assert result['passed'] is True
        assert result['side'] == 'LONG'
        assert result['score'] > 0

    def test_weak_prediction_blocked(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        result = scanner.evaluate_candidate("AAPL", 0.01, GOOD_FEATURES)
        assert result['passed'] is False
        assert "weak" in result['reason'].lower()

    def test_high_vpin_blocked(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        toxic = {**GOOD_FEATURES, 'VPIN': 0.90}
        result = scanner.evaluate_candidate("AAPL", 0.25, toxic)
        assert result['passed'] is False
        assert "VPIN" in result['reason']

    def test_illiquid_blocked(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        illiquid = {**GOOD_FEATURES, 'Amihud_Illiquidity': 0.95}
        result = scanner.evaluate_candidate("AAPL", 0.25, illiquid)
        assert result['passed'] is False
        assert "lliquid" in result['reason']

    def test_low_adx_blocked(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        low_adx = {**GOOD_FEATURES, 'ADX': 10}
        result = scanner.evaluate_candidate("AAPL", 0.25, low_adx)
        assert result['passed'] is False
        assert "ADX" in result['reason']

    def test_extreme_vwap_blocked(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        extreme = {**GOOD_FEATURES, 'VWAP_ZScore': 4.0}
        result = scanner.evaluate_candidate("AAPL", 0.25, extreme)
        assert result['passed'] is False
        assert "VWAP" in result['reason']

    def test_low_vol_blocked(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        low_vol = {**GOOD_FEATURES, 'Volatility_Rank': 0.1}
        result = scanner.evaluate_candidate("AAPL", 0.25, low_vol)
        assert result['passed'] is False
        assert "olatility" in result['reason']

    def test_negative_prediction_gives_short(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        result = scanner.evaluate_candidate("AAPL", -0.25, GOOD_FEATURES)
        assert result['passed'] is True
        assert result['side'] == 'SHORT'

    def test_specialist_tier_for_strong_signal(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        result = scanner.evaluate_candidate("AAPL", 0.35, GOOD_FEATURES)
        assert result['passed'] is True
        assert result['type'] == 'SPECIALIST'
        assert result['tier_mult'] == 2.0

    def test_grinder_tier_for_moderate_signal(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        result = scanner.evaluate_candidate("AAPL", 0.15, GOOD_FEATURES)
        assert result['passed'] is True
        assert 'GRINDER' in result['type']
        assert result['tier_mult'] == 1.0

    def test_news_hard_skip(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        result = scanner.evaluate_candidate("AAPL", 0.25, GOOD_FEATURES, news_score=5)
        assert result['passed'] is False
        assert "news" in result['reason'].lower()

    def test_news_soft_penalty(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        no_news = scanner.evaluate_candidate("AAPL", 0.25, GOOD_FEATURES, news_score=0)
        with_news = scanner.evaluate_candidate("AAPL", 0.25, GOOD_FEATURES, news_score=2)
        assert with_news['score'] < no_news['score']

    def test_regime_boost(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        result = scanner.evaluate_candidate("AAPL", 0.25, GOOD_FEATURES, regime='BULL')
        assert '+REGIME' in result['type']

    def test_rank_candidates(self):
        scanner = CandidateScanner(DEFAULT_SETTINGS)
        candidates = [
            scanner.evaluate_candidate("A", 0.35, GOOD_FEATURES),
            scanner.evaluate_candidate("B", 0.10, GOOD_FEATURES),
            scanner.evaluate_candidate("C", 0.25, GOOD_FEATURES),
            scanner.evaluate_candidate("D", 0.01, GOOD_FEATURES),  # blocked
        ]
        ranked = scanner.rank_candidates(candidates, max_candidates=3)
        assert len(ranked) <= 3
        assert all(c['passed'] for c in ranked)
        # Should be sorted by score descending
        scores = [c['score'] for c in ranked]
        assert scores == sorted(scores, reverse=True)
