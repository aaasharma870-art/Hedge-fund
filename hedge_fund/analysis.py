"""
Attribution analysis for trade outcome patterns.

Uses decision tree regression to identify which feature combinations
lead to winning vs losing trades, providing interpretable "rules"
for improving strategy parameters.
"""

import logging
import os

import numpy as np
import pandas as pd


def run_attribution_analysis(db_path=None, limit=1000):
    """
    Analyze past trade outcomes to find winning and losing patterns.

    Trains a shallow DecisionTree on recorded trade features to extract
    interpretable rules about which conditions produce the best/worst
    outcomes.

    Args:
        db_path: Path to the godmode.db SQLite database. If None, searches
            common locations.
        limit: Maximum number of recent trades to analyze.

    Returns:
        Dict with keys 'winning_rules', 'losing_rules', 'importances',
        or None if analysis cannot be performed.
    """
    try:
        import sqlite3
        from sklearn.tree import DecisionTreeRegressor, _tree
    except ImportError:
        logging.error("Attribution analysis requires scikit-learn. "
                       "Install with: pip install scikit-learn")
        return None

    # Resolve DB path
    if db_path is None:
        candidates = [
            "./data/db/godmode.db",
            "data/db/godmode.db",
        ]
        for p in candidates:
            if os.path.exists(p):
                db_path = p
                break

    if db_path is None or not os.path.exists(db_path):
        logging.error(f"Database not found at {db_path}")
        return None

    # Load data
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(
            f"SELECT * FROM trade_outcomes ORDER BY ts DESC LIMIT {limit}", conn
        )
        conn.close()
    except Exception as e:
        logging.error(f"DB connection failed: {e}")
        return None

    if df.empty:
        logging.warning("No trade history found.")
        return None

    # Ensure target column
    if 'pnl_r' not in df.columns:
        df['pnl_r'] = df['pnl'] / df['entry_price']

    # Identify feature columns
    ignore_cols = {
        'id', 'symbol', 'side', 'entry_price', 'exit_price',
        'pnl', 'pnl_r', 'outcome', 'ts', 'reason',
    }
    feature_cols = [
        c for c in df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_clean = df.dropna(subset=feature_cols + ['pnl_r'])

    if len(df_clean) < 10:
        logging.warning("Not enough data points for ML analysis (need > 10).")
        return None

    # Train interpretable model
    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)
    X = df_clean[feature_cols]
    y = df_clean['pnl_r']
    dt.fit(X, y)

    # Extract rules
    rules = _get_rules(dt, feature_cols, _tree)

    # Feature importance
    importances = pd.Series(
        dt.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    winning_rules = rules[:3]
    losing_rules = rules[-3:]

    # Print results
    print(f"QUANT ATTRIBUTION ANALYSIS ({len(df_clean)} trades)")
    print("=" * 50)

    print("\nWINNING PATTERNS (Where PnL is highest):")
    for val, rule_path in winning_rules:
        print(f"   Avg Return: {val:.2f}R | Condition: {' AND '.join(rule_path)}")

    print("\nLOSING PATTERNS (Where PnL is lowest):")
    for val, rule_path in losing_rules:
        print(f"   Avg Return: {val:.2f}R | Condition: {' AND '.join(rule_path)}")

    print("\nMOST IMPORTANT FACTORS:")
    for feat, imp in importances.head(5).items():
        print(f"   {feat}: {imp:.3f}")

    return {
        'winning_rules': winning_rules,
        'losing_rules': losing_rules,
        'importances': importances,
        'n_trades': len(df_clean),
    }


def _get_rules(tree_model, feature_names, _tree_module):
    """Extract decision rules from a fitted DecisionTreeRegressor."""
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree_module.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []

    def recurse(node, path, names):
        if tree_.feature[node] != _tree_module.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1 = path + [f"{name} <= {threshold:.2f}"]
            n1 = names + [name]
            recurse(tree_.children_left[node], p1, n1)
            p2 = path + [f"{name} > {threshold:.2f}"]
            n2 = names + [name]
            recurse(tree_.children_right[node], p2, n2)
        else:
            value = tree_.value[node][0][0]
            paths.append((value, path))

    recurse(0, [], [])
    return sorted(paths, key=lambda x: x[0], reverse=True)
