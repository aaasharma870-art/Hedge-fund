"""Tests for feature parity between training and prediction paths.

Ensures that bot.py's WalkForwardAI.train() and check_candidate() compute
the same set of features, preventing silent distribution shift at inference.
"""

import ast
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The canonical feature list from WalkForwardAI.__init__
EXPECTED_FEATURES = [
    # Original features
    'RSI', 'ADX', 'ATR_Pct', 'Vol_Rel', 'Kalman_Dist', 'Hurst',
    # Price action features
    'BB_Width', 'BB_Position', 'VWAP_Dist', 'HL_Range',
    # Momentum features
    'ROC_5', 'ROC_20',
    # Volume features
    'Vol_Surge', 'Money_Flow',
    # Regime features
    'Volatility_Rank', 'Trend_Consistency',
    # Time features
    'Hour', 'Day_of_Week',
    # Market context
    'SPY_ROC_5', 'SPY_ROC_20', 'VIX_Level', 'VIX_ROC',
    # Cross-sectional ranks
    'momentum_rank', 'volume_rank', 'value_rank', 'composite_rank',
    # Institutional microstructure
    'VPIN', 'VWAP_ZScore', 'VWAP_Slope', 'VWAP_Volume_Ratio',
    'Regime_GEX_Proxy', 'Amihud_Illiquidity', 'Volatility_Regime_Score',
    # Alpha features
    'RRS_Cumulative', 'Liquidity_Sweep', 'Beta_To_SPY',
]

# Features that were removed (constant during training -> distribution shift)
REMOVED_FEATURES = [
    'sa_news_count_3d', 'sa_sentiment_score', 'earnings_surprise',
    'revenue_growth_yoy', 'pe_ratio', 'news_impact_weight',
]


def _read_bot_source():
    """Read bot.py source code."""
    bot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bot.py')
    with open(bot_path, 'r') as f:
        return f.read()


def _extract_feature_assignments(source, start_marker, end_marker):
    """Extract df['feature_name'] = ... assignments between markers in source."""
    # Find the section between markers
    start_idx = source.find(start_marker)
    end_idx = source.find(end_marker, start_idx) if start_idx >= 0 else -1
    if start_idx < 0 or end_idx < 0:
        return set()

    section = source[start_idx:end_idx]

    # Match df['feature'] = or df_15m['feature'] = patterns
    pattern = re.compile(r"(?:df|df_15m)\['(\w+)'\]\s*=")
    return set(pattern.findall(section))


class TestTrainingFeaturesMatchPredictionFeatures:
    """Verify training and prediction compute the same feature set."""

    def test_self_cols_matches_expected(self):
        """WalkForwardAI.cols should match the canonical feature list."""
        source = _read_bot_source()
        # Extract self.cols list from source
        match = re.search(r'self\.cols\s*=\s*\[([^\]]+)\]', source, re.DOTALL)
        assert match, "Could not find self.cols definition in bot.py"

        # Parse feature names from the list
        cols_str = match.group(1)
        features = re.findall(r"'(\w+)'", cols_str)

        assert set(features) == set(EXPECTED_FEATURES), (
            f"self.cols mismatch.\n"
            f"  Missing: {set(EXPECTED_FEATURES) - set(features)}\n"
            f"  Extra:   {set(features) - set(EXPECTED_FEATURES)}"
        )

    def test_training_computes_all_cols(self):
        """Training path must assign every feature in self.cols."""
        source = _read_bot_source()

        # Training features are computed in the process() function inside train()
        # Look for df['feature'] = assignments in the training section
        train_features = _extract_feature_assignments(
            source,
            "# ENHANCED FEATURES: Original",  # Start of training feature computation
            "# FIX: Attach daily ATR",  # End marker (after all features computed)
        )

        # Some features are set as constants or injected differently
        # 'Hour' and 'Day_of_Week' use .index.hour / .index.dayofweek
        # Market context features may be merged from spy_ctx
        time_and_context = {'Hour', 'Day_of_Week', 'SPY_ROC_5', 'SPY_ROC_20', 'VIX_Level', 'VIX_ROC'}

        for feat in EXPECTED_FEATURES:
            if feat in time_and_context:
                continue  # These use different assignment patterns
            assert feat in train_features, (
                f"Feature '{feat}' is in self.cols but not computed in training path"
            )

    def test_prediction_computes_all_cols(self):
        """check_candidate() must assign every feature in self.cols."""
        source = _read_bot_source()

        # Prediction features are computed in check_candidate()
        pred_features = _extract_feature_assignments(
            source,
            "def check_candidate(t):",
            "row = df_15m.iloc[-2].to_dict()",
        )

        # Time and context features use different patterns in prediction too
        time_and_context = {'Hour', 'Day_of_Week', 'SPY_ROC_5', 'SPY_ROC_20', 'VIX_Level', 'VIX_ROC'}

        for feat in EXPECTED_FEATURES:
            if feat in time_and_context:
                continue
            assert feat in pred_features, (
                f"Feature '{feat}' is in self.cols but not computed in check_candidate()"
            )

    def test_no_extra_features_in_prediction(self):
        """Prediction path should not compute features absent from self.cols."""
        source = _read_bot_source()

        pred_features = _extract_feature_assignments(
            source,
            "def check_candidate(t):",
            "row = df_15m.iloc[-2].to_dict()",
        )

        # Allow helper columns that are used to derive features but aren't model inputs
        allowed_helpers = {
            'ATR', 'EMA_200', 'Kalman', 'BB_Upper', 'BB_Mid', 'BB_Lower',
            'VWAP', 'ATR_D', 'Vol_Rel',  # intermediate columns
        }

        extra = pred_features - set(EXPECTED_FEATURES) - allowed_helpers
        assert not extra, (
            f"check_candidate() computes features not in self.cols: {extra}\n"
            f"These will be ignored by the model but waste computation."
        )


class TestNoConstantFeaturesInTraining:
    """Ensure removed features don't reappear."""

    def test_removed_features_not_in_cols(self):
        """Features that cause distribution shift must not be in self.cols."""
        source = _read_bot_source()
        match = re.search(r'self\.cols\s*=\s*\[([^\]]+)\]', source, re.DOTALL)
        assert match, "Could not find self.cols in bot.py"

        cols_str = match.group(1)
        features = re.findall(r"'(\w+)'", cols_str)

        for feat in REMOVED_FEATURES:
            assert feat not in features, (
                f"Removed feature '{feat}' found in self.cols - "
                f"this feature is constant during training, causing distribution shift"
            )

    def test_expected_feature_count(self):
        """Verify we have the expected number of features (sanity check)."""
        assert len(EXPECTED_FEATURES) == 36, (
            f"Expected 36 features, got {len(EXPECTED_FEATURES)}. "
            f"Update this test if features are intentionally added/removed."
        )
