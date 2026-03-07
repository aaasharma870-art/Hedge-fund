"""
Ensemble model stacking for more robust predictions.

Combines XGBoost, LightGBM (optional), and Ridge regression into a
meta-learner that reduces single-model fragility.
"""

import logging

import numpy as np


class EnsembleModel:
    """
    Stacked ensemble: XGBoost + LightGBM + Ridge -> Ridge meta-learner.

    Each base model is trained on the same data. Their out-of-fold
    predictions are used as features for a Ridge meta-learner that
    produces the final prediction.

    Falls back gracefully if LightGBM is not installed (XGBoost + Ridge
    only).
    """

    def __init__(self, xgb_params=None, lgb_params=None, ridge_alpha=1.0):
        """
        Args:
            xgb_params: Dict of XGBRegressor parameters. Uses defaults if None.
            lgb_params: Dict of LGBMRegressor parameters. Uses defaults if None.
            ridge_alpha: Regularization strength for Ridge and meta-learner.
        """
        import xgboost as xgb
        from sklearn.linear_model import Ridge

        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.50,
            "colsample_bytree": 0.50,
            "min_child_weight": 10,
            "reg_alpha": 5.0,
            "reg_lambda": 10.0,
            "gamma": 0.5,
            "n_jobs": -1,
            "verbosity": 0,
        }

        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.ridge_model = Ridge(alpha=ridge_alpha)
        self.meta_model = Ridge(alpha=ridge_alpha)

        self.lgb_model = None
        self._has_lgb = False
        try:
            import lightgbm as lgb
            self.lgb_params = lgb_params or {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.50,
                "colsample_bytree": 0.50,
                "min_child_weight": 10,
                "reg_alpha": 5.0,
                "reg_lambda": 10.0,
                "n_jobs": -1,
                "verbosity": -1,
            }
            self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
            self._has_lgb = True
        except ImportError:
            logging.info("LightGBM not available, using XGBoost + Ridge ensemble")

        self._is_fitted = False
        self._feature_importances = None

    def fit(self, X, y):
        """
        Train all base models and the meta-learner.

        Uses 3-fold time-series split to generate out-of-fold predictions
        for the meta-learner (prevents data leakage).

        Args:
            X: Feature matrix (numpy array or DataFrame).
            y: Target values (numpy array or Series).
        """
        from sklearn.model_selection import TimeSeriesSplit

        X_arr = np.array(X)
        y_arr = np.array(y)

        n_splits = 3
        if len(X_arr) < n_splits * 50:
            # Not enough data for cross-validation, train directly
            self._fit_direct(X_arr, y_arr)
            return

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_preds = np.zeros((len(X_arr), self._n_base_models()))

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr = y_arr[train_idx]

            self.xgb_model.fit(X_tr, y_tr)
            oof_preds[val_idx, 0] = self.xgb_model.predict(X_val)

            self.ridge_model.fit(X_tr, y_tr)
            oof_preds[val_idx, 1] = self.ridge_model.predict(X_val)

            if self._has_lgb:
                self.lgb_model.fit(X_tr, y_tr)
                oof_preds[val_idx, 2] = self.lgb_model.predict(X_val)

        # Train meta-learner on OOF predictions (exclude first fold which has no OOF)
        first_val_start = list(tscv.split(X_arr))[0][1][0]
        meta_X = oof_preds[first_val_start:]
        meta_y = y_arr[first_val_start:]

        # Remove rows where all OOF preds are zero (unfilled folds)
        mask = np.any(meta_X != 0, axis=1)
        if mask.sum() > 20:
            self.meta_model.fit(meta_X[mask], meta_y[mask])
        else:
            # Fallback: equal weights
            self.meta_model.fit(
                np.column_stack([np.ones(100)] * self._n_base_models()),
                np.ones(100),
            )

        # Refit base models on ALL data for prediction
        self.xgb_model.fit(X_arr, y_arr)
        self.ridge_model.fit(X_arr, y_arr)
        if self._has_lgb:
            self.lgb_model.fit(X_arr, y_arr)

        self._is_fitted = True
        self._feature_importances = self.xgb_model.feature_importances_

    def _fit_direct(self, X, y):
        """Train all models directly when data is too small for OOF splits.

        FIX: Previously trained meta-learner on in-sample base predictions,
        which is data leakage (meta sees same data base models trained on).
        Now uses equal-weight averaging via fixed meta coefficients instead.
        """
        self.xgb_model.fit(X, y)
        self.ridge_model.fit(X, y)
        if self._has_lgb:
            self.lgb_model.fit(X, y)

        # FIX: Set meta-learner to equal-weight averaging instead of fitting
        # on in-sample predictions (which leaks information).
        n_base = self._n_base_models()
        dummy_X = np.eye(n_base)
        dummy_y = np.ones(n_base) / n_base
        self.meta_model.fit(dummy_X, dummy_y)
        # Override with explicit equal weights
        self.meta_model.coef_ = np.ones(n_base) / n_base
        self.meta_model.intercept_ = 0.0

        self._is_fitted = True
        self._feature_importances = self.xgb_model.feature_importances_

    def predict(self, X):
        """
        Generate ensemble predictions.

        Args:
            X: Feature matrix.

        Returns:
            Numpy array of predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("EnsembleModel not fitted. Call fit() first.")

        base_preds = self._base_predictions(np.array(X))
        return self.meta_model.predict(base_preds)

    def score(self, X, y):
        """Return R² score on given data."""
        from sklearn.metrics import r2_score
        preds = self.predict(X)
        return r2_score(y, preds)

    def _base_predictions(self, X):
        """Get predictions from all base models as a matrix."""
        preds = [
            self.xgb_model.predict(X),
            self.ridge_model.predict(X),
        ]
        if self._has_lgb:
            preds.append(self.lgb_model.predict(X))
        return np.column_stack(preds)

    def _n_base_models(self):
        """Number of base models in the ensemble."""
        return 3 if self._has_lgb else 2

    @property
    def feature_importances_(self):
        """Return XGBoost feature importances (primary model)."""
        if self._feature_importances is None:
            raise RuntimeError("Model not fitted yet.")
        return self._feature_importances
