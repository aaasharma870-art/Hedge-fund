"""Tests for percentile-based prediction threshold."""

import numpy as np
import pytest


def test_percentile_threshold_correct_trade_rate():
    """With pred_threshold_pct=0.80, exactly 20% should exceed threshold."""
    np.random.seed(42)
    preds = np.random.randn(1000)
    pct = 0.80
    threshold = np.percentile(np.abs(preds), pct * 100)
    trade_rate = (np.abs(preds) > threshold).mean()
    assert abs(trade_rate - 0.20) < 0.02, f"Expected ~20%, got {trade_rate:.2%}"


def test_percentile_threshold_no_zero_trades():
    """With any pct < 1.0, should never get zero trades."""
    np.random.seed(42)
    preds = np.random.randn(500) * 0.05  # small values like real model output
    for pct in [0.55, 0.70, 0.85]:
        threshold = np.percentile(np.abs(preds), pct * 100)
        trade_rate = (np.abs(preds) > threshold).mean()
        assert trade_rate > 0, f"pct={pct} produced zero trades (threshold={threshold:.6f})"


def test_percentile_threshold_monotonic():
    """Higher percentile -> higher threshold -> fewer trades."""
    np.random.seed(42)
    preds = np.random.randn(1000)
    thresholds = []
    for pct in [0.55, 0.65, 0.75, 0.85]:
        threshold = np.percentile(np.abs(preds), pct * 100)
        thresholds.append(threshold)
    for i in range(len(thresholds) - 1):
        assert thresholds[i] < thresholds[i + 1]
