"""Purged walk-forward utilities to prevent data leakage."""
import numpy as np


def purged_embargo_split(n_samples, n_folds=5, embargo_pct=0.02):
    """Generate train/test indices with purging and embargo.

    Purging: remove training samples whose labels overlap with test period.
    Embargo: add buffer between train end and test start.
    """
    fold_size = n_samples // n_folds
    embargo_size = max(1, int(n_samples * embargo_pct))

    splits = []
    for i in range(1, n_folds):
        train_end = i * fold_size
        test_start = train_end + embargo_size
        test_end = min(test_start + fold_size, n_samples)

        if test_end <= test_start:
            continue

        train_idx = np.arange(0, train_end - embargo_size)  # purge near boundary
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) > 50 and len(test_idx) > 10:
            splits.append((train_idx, test_idx))

    return splits
