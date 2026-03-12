"""
Ensemble model stacking for more robust predictions.

Combines XGBoost and LightGBM via Ridge meta-learner using
Out-of-Fold (OOF) stacking to prevent data leakage.
"""

import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")


class EnsembleModel:
    """
    Stacked ensemble: XGBoost + LightGBM -> Ridge meta-learner.

    Uses OOF stacking by default. Falls back to equal-weight averaging
    when stacking doesn't improve over best base model.
    """

    def __init__(self, xgb_params=None, lgb_params=None, ridge_alpha=1.0,
                 use_oof=True, use_daily=False):
        import xgboost as xgb
        from sklearn.linear_model import Ridge

        if use_daily:
            # Daily models: moderate depth, less regularization
            # (fewer samples than 15-min, but higher SNR)
            default_xgb = {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.70,
                "colsample_bytree": 0.70,
                "min_child_weight": 10,
                "reg_alpha": 1.0,
                "reg_lambda": 3.0,
                "gamma": 0.1,
                "n_jobs": -1,
                "verbosity": 0,
            }
            default_lgb = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.70,
                "colsample_bytree": 0.70,
                "min_child_weight": 10,
                "reg_alpha": 1.0,
                "reg_lambda": 3.0,
                "n_jobs": -1,
                "verbosity": -1,
            }
        else:
            # Intraday models (V11 config)
            default_xgb = {
                "n_estimators": 80,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.60,
                "colsample_bytree": 0.60,
                "min_child_weight": 15,
                "reg_alpha": 3.0,
                "reg_lambda": 5.0,
                "gamma": 0.3,
                "n_jobs": -1,
                "verbosity": 0,
            }
            default_lgb = {
                "n_estimators": 80,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.60,
                "colsample_bytree": 0.60,
                "min_child_weight": 15,
                "reg_alpha": 3.0,
                "reg_lambda": 5.0,
                "n_jobs": -1,
                "verbosity": -1,
            }

        self.xgb_params = xgb_params or default_xgb

        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.meta_model = Ridge(alpha=ridge_alpha)
        # Daily data has too few samples for reliable OOF stacking
        self._use_oof = False if use_daily else use_oof

        self.lgb_model = None
        self._has_lgb = False
        try:
            import lightgbm as lgb
            self.lgb_params = lgb_params or default_lgb
            self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
            self._has_lgb = True
        except ImportError:
            logging.info("LightGBM not available, using XGBoost only")

        self._is_fitted = False
        self._feature_importances = None
        self.stacking_method_ = 'equal_weight'
        self.oof_predictions_ = {}

    def fit(self, X, y, sample_weight=None):
        """Train ensemble using OOF stacking or equal-weight fallback."""
        if hasattr(X, 'columns'):
            self._feature_names = list(X.columns)
        X_arr = np.array(X)
        y_arr = np.array(y)

        if self._use_oof and len(X_arr) >= 200:
            try:
                self._fit_oof_stacking(X_arr, y_arr, n_folds=5, sample_weight=sample_weight)
                return
            except Exception as e:
                logging.warning(f"OOF stacking failed ({e}), falling back to equal-weight")

        self._fit_direct(X_arr, y_arr, sample_weight=sample_weight)

    def _fit_oof_stacking(self, X_train, y_train, n_folds=5, sample_weight=None):
        """Fit ensemble using Out-of-Fold stacking. Zero data leakage."""
        from sklearn.model_selection import TimeSeriesSplit
        from scipy.stats import spearmanr

        n_samples = len(X_train)
        n_base = self._n_base_models()

        oof_preds = np.full((n_samples, n_base), np.nan)
        tscv = TimeSeriesSplit(n_splits=n_folds)
        fold_ics = {'xgb': [], 'lgb': []}

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            if len(train_idx) < 30 or len(val_idx) < 10:
                continue

            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            sw_tr = sample_weight[train_idx] if sample_weight is not None else None

            # XGBoost
            import xgboost as xgb
            xgb_fold = xgb.XGBRegressor(**self.xgb_params)
            xgb_fold.fit(X_tr, y_tr, sample_weight=sw_tr)
            oof_preds[val_idx, 0] = xgb_fold.predict(X_val)

            if len(val_idx) >= 10:
                ic, _ = spearmanr(oof_preds[val_idx, 0], y_val)
                if not np.isnan(ic):
                    fold_ics['xgb'].append(ic)

            # LightGBM
            if self._has_lgb:
                import lightgbm as lgb
                lgb_fold = lgb.LGBMRegressor(**self.lgb_params)
                lgb_fold.fit(X_tr, y_tr, sample_weight=sw_tr)
                oof_preds[val_idx, 1] = lgb_fold.predict(X_val)

                if len(val_idx) >= 10:
                    ic, _ = spearmanr(oof_preds[val_idx, 1], y_val)
                    if not np.isnan(ic):
                        fold_ics['lgb'].append(ic)

        # Build meta-feature matrix
        nan_mask = np.isnan(oof_preds).any(axis=1)
        valid_mask = ~nan_mask

        if valid_mask.sum() < 20:
            raise ValueError(f"Only {valid_mask.sum()} valid OOF samples")

        meta_X = np.nan_to_num(oof_preds[valid_mask])
        meta_y = y_train[valid_mask]

        self.meta_model.fit(meta_X, meta_y)

        # Compute OOF ensemble IC
        oof_ensemble = self.meta_model.predict(meta_X)
        ic_ensemble, _ = spearmanr(oof_ensemble, meta_y)

        ic_xgb_mean = np.mean(fold_ics['xgb']) if fold_ics['xgb'] else 0
        ic_lgb_mean = np.mean(fold_ics['lgb']) if fold_ics['lgb'] else 0
        best_base_ic = max(ic_xgb_mean, ic_lgb_mean)

        logging.info(f"OOF IC - XGB: {ic_xgb_mean:.4f} | LGB: {ic_lgb_mean:.4f} | Ensemble: {ic_ensemble:.4f}")

        if ic_ensemble > best_base_ic * 0.9:
            self.stacking_method_ = 'oof'
        else:
            self.stacking_method_ = 'equal_weight'

        self.oof_predictions_ = {
            'ic': float(ic_ensemble) if not np.isnan(ic_ensemble) else 0.0,
            'fold_ics_xgb': fold_ics['xgb'],
            'fold_ics_lgb': fold_ics['lgb'],
        }

        # Retrain base models on FULL training set
        self.xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
        if self._has_lgb:
            self.lgb_model.fit(X_train, y_train, sample_weight=sample_weight)

        self._is_fitted = True
        self._feature_importances = self.xgb_model.feature_importances_

    def _fit_direct(self, X, y, sample_weight=None):
        """Train all models directly with equal-weight meta."""
        self.xgb_model.fit(X, y, sample_weight=sample_weight)
        if self._has_lgb:
            self.lgb_model.fit(X, y, sample_weight=sample_weight)

        n_base = self._n_base_models()
        dummy_X = np.eye(n_base)
        dummy_y = np.ones(n_base) / n_base
        self.meta_model.fit(dummy_X, dummy_y)
        self.meta_model.coef_ = np.ones(n_base) / n_base
        self.meta_model.intercept_ = 0.0
        self.stacking_method_ = 'equal_weight'

        self._is_fitted = True
        self._feature_importances = self.xgb_model.feature_importances_

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("EnsembleModel not fitted. Call fit() first.")

        X_arr = np.array(X)  # Always use numpy to avoid feature name warnings
        if self.stacking_method_ == 'oof':
            base_preds = self._base_predictions(X_arr)
            return self.meta_model.predict(base_preds)
        else:
            # Equal-weight averaging
            preds = self.xgb_model.predict(X_arr)
            if self._has_lgb:
                preds = (preds + self.lgb_model.predict(X_arr)) / 2.0
            return preds

    def score(self, X, y):
        from sklearn.metrics import r2_score
        preds = self.predict(X)
        return r2_score(y, preds)

    def _base_predictions(self, X):
        preds = [self.xgb_model.predict(X)]
        if self._has_lgb:
            preds.append(self.lgb_model.predict(X))
        return np.column_stack(preds)

    def _n_base_models(self):
        return 2 if self._has_lgb else 1

    @property
    def feature_importances_(self):
        if self._feature_importances is None:
            raise RuntimeError("Model not fitted yet.")
        return self._feature_importances
