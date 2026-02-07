"""
Candidate scanning and trade filtering pipeline.

Evaluates potential trades through institutional-grade entry gates
before they reach the order execution stage. Centralizes all
pre-trade filtering logic that was previously inlined in bot.py.
"""

import logging

import numpy as np


class CandidateScanner:
    """
    Multi-gate trade candidate evaluation.

    Applies 10+ institutional entry filters to raw model predictions,
    producing a ranked list of trade candidates with scores.
    """

    def __init__(self, settings, earnings_guard=None, fundamental_guard=None):
        """
        Args:
            settings: Bot SETTINGS dict with filter thresholds.
            earnings_guard: Optional EarningsGuard instance.
            fundamental_guard: Optional FundamentalGuard instance.
        """
        self.settings = settings
        self.earnings_guard = earnings_guard
        self.fundamental_guard = fundamental_guard

    def evaluate_candidate(self, ticker, prediction, features, regime=None,
                           news_score=0, cross_rank=None):
        """
        Run all entry gates on a single candidate.

        Args:
            ticker: Stock symbol.
            prediction: Model prediction (signed R-multiple).
            features: Dict of computed features for this bar:
                RSI, ADX, Hurst, VPIN, Amihud_Illiquidity, VWAP_ZScore,
                Volatility_Rank, EMA_200, ATR, Close, etc.
            regime: Market regime string (BULL/BEAR/NEUTRAL).
            news_score: News sentiment score (higher = worse).
            cross_rank: Dict from CrossSectionalRanker.get_ranks().

        Returns:
            Dict with keys: passed (bool), reason (str), score (float),
            side (str), ev (float), tier_mult (float), type (str).
            Returns None if candidate is blocked by hard gates.
        """
        result = {
            'symbol': ticker,
            'passed': False,
            'reason': '',
            'score': 0.0,
            'side': 'LONG' if prediction > 0 else 'SHORT',
            'ev': abs(prediction),
            'tier_mult': 1.0,
            'type': 'STANDARD',
            'prediction': prediction,
        }

        abs_pred = abs(prediction)
        min_conf = self.settings.get('MIN_CONFIDENCE', 0.02)

        # Gate 1: Minimum prediction strength
        if abs_pred < min_conf:
            result['reason'] = f"Prediction too weak ({abs_pred:.3f} < {min_conf})"
            return result

        # Gate 2: VPIN toxicity
        vpin = features.get('VPIN', 0)
        if vpin > 0.85:
            result['reason'] = f"VPIN too high ({vpin:.2f} > 0.85)"
            return result

        # Gate 3: Amihud illiquidity
        amihud = features.get('Amihud_Illiquidity', 0)
        if amihud > 0.90:
            result['reason'] = f"Illiquid ({amihud:.2f} > 0.90)"
            return result

        # Gate 4: Hurst exponent
        hurst = features.get('Hurst', 0.5)
        min_hurst = self.settings.get('MIN_HURST', 0.38)
        max_hurst = self.settings.get('TIER_1', {}).get('MAX_HURST', 0.55)
        if hurst > max_hurst and abs_pred < 0.3:
            result['reason'] = f"Hurst too high ({hurst:.2f} > {max_hurst})"
            return result

        # Gate 5: ADX trend strength
        adx = features.get('ADX', 20)
        min_adx = self.settings.get('TIER_2', {}).get('MIN_ADX', 20)
        if adx < min_adx:
            result['reason'] = f"ADX too low ({adx:.0f} < {min_adx})"
            return result

        # Gate 6: VWAP Z-score bounds
        vwap_z = features.get('VWAP_ZScore', 0)
        if abs(vwap_z) > 3.0:
            result['reason'] = f"VWAP Z-score extreme ({vwap_z:.2f})"
            return result

        # Gate 7: Volatility rank
        vol_rank = features.get('Volatility_Rank', 0.5)
        if vol_rank < 0.30:
            result['reason'] = f"Volatility too low ({vol_rank:.2f})"
            return result

        # Gate 8: Earnings guard
        if self.earnings_guard:
            try:
                if not self.earnings_guard.check_safe(ticker):
                    result['reason'] = "Earnings proximity"
                    return result
            except Exception:
                pass

        # Gate 9: Fundamental guard
        if self.fundamental_guard:
            try:
                if not self.fundamental_guard.check_healthy(ticker):
                    result['reason'] = "Weak fundamentals"
                    return result
            except Exception:
                pass

        # Gate 10: News hard skip
        news_hard = self.settings.get('NEWS_HARD_SKIP_SCORE', 3)
        if news_score >= news_hard:
            result['reason'] = f"Negative news ({news_score} >= {news_hard})"
            return result

        # --- Passed all gates ---
        result['passed'] = True

        # Tier classification
        tier1 = self.settings.get('TIER_1', {})
        if abs_pred >= tier1.get('MIN_PROB', 0.30) and adx >= tier1.get('MIN_ADX', 25):
            result['tier_mult'] = tier1.get('RISK_MULT', 2.0)
            result['type'] = 'SPECIALIST'
        else:
            tier2 = self.settings.get('TIER_2', {})
            result['tier_mult'] = tier2.get('RISK_MULT', 1.0)
            result['type'] = 'GRINDER'

        # Regime-aware type
        if regime == 'BULL' and result['side'] == 'LONG':
            result['type'] += ' +REGIME'
        elif regime == 'BEAR' and result['side'] == 'SHORT':
            result['type'] += ' +REGIME'

        # Score calculation
        score = abs_pred * result['tier_mult']

        # Cross-sectional rank boost
        if cross_rank:
            composite = cross_rank.get('composite_rank', 0.5)
            score *= float(np.clip(0.7 + 0.6 * composite, 0.7, 1.3))

        # News soft penalty
        news_soft = self.settings.get('NEWS_SOFT_PENALTY_SCORE', 1)
        if news_score >= news_soft:
            penalty = self.settings.get('NEWS_PENALTY_SIZE_MULT', 0.75)
            score *= penalty

        result['score'] = score
        result['p_win'] = min(0.95, 0.5 + abs_pred * 0.5)  # approximate
        return result

    def rank_candidates(self, candidates, max_candidates=10):
        """
        Sort and filter evaluated candidates by score.

        Args:
            candidates: List of dicts from evaluate_candidate().
            max_candidates: Maximum number to return.

        Returns:
            List of top candidates sorted by score descending.
        """
        passed = [c for c in candidates if c and c.get('passed')]
        passed.sort(key=lambda x: x['score'], reverse=True)
        return passed[:max_candidates]
