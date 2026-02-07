"""
Shared configuration management for backtester and live bot.

The backtester writes optimal parameters to a JSON config file.
The bot reads them on startup to use the latest optimized settings.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def _normalize_path(path: str) -> str:
    """Normalize user-provided and env-provided paths."""
    return os.path.abspath(os.path.expanduser(path))


def _env_or_default(name: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    value = os.getenv(name, default)
    return _normalize_path(value)


@dataclass(frozen=True)
class AppConfig:
    """Canonical app configuration loaded once at startup."""

    data_root: str
    log_dir: str
    optimal_params_path: str

    @property
    def db_dir(self) -> str:
        return os.path.join(self.data_root, "db")

    @property
    def db_path(self) -> str:
        return os.path.join(self.db_dir, "godmode.db")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.data_root, "models")

    @property
    def market_cache_dir(self) -> str:
        return os.path.join(self.data_root, "market_cache")

    def ensure_directories(self) -> None:
        """Create required directories for runtime state and logs."""
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.market_cache_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)


_DEFAULT_CONFIG = AppConfig(
    data_root=_env_or_default("DATA_ROOT", "./data"),
    log_dir=_env_or_default("LOG_DIR", "./data/logs"),
    optimal_params_path=_env_or_default("OPTIMAL_PARAMS_PATH", "./data/optimal_params.json"),
)


def load_app_config() -> AppConfig:
    """Load and return the canonical app config from environment variables."""
    config = AppConfig(
        data_root=_env_or_default("DATA_ROOT", _DEFAULT_CONFIG.data_root),
        log_dir=_env_or_default("LOG_DIR", _DEFAULT_CONFIG.log_dir),
        optimal_params_path=_env_or_default("OPTIMAL_PARAMS_PATH", _DEFAULT_CONFIG.optimal_params_path),
    )
    config.ensure_directories()
    return config


def _config_paths(config: Optional[AppConfig] = None):
    """Compute candidate optimal-params paths in lookup order."""
    cfg = config or load_app_config()
    return [
        cfg.optimal_params_path,
        os.path.join(cfg.data_root, "optimal_params.json"),
        os.path.join(os.getcwd(), "optimal_params.json"),
    ]


def _resolve_path(path=None, config: Optional[AppConfig] = None):
    """Resolve config file path."""
    if path:
        return _normalize_path(path)

    for p in _config_paths(config=config):
        if os.path.exists(p):
            return p

    return _config_paths(config=config)[0]


def save_optimal_params(best_config, metrics=None, holdout_metrics=None,
                        path=None, config: Optional[AppConfig] = None):
    """
    Save the best parameters from backtester optimization to JSON.

    Called by the backtester after Optuna finishes. The bot reads this
    file on startup to use the latest optimized settings.

    Args:
        best_config: Dict from the Optuna results with parameter keys.
        metrics: Optional in-sample metrics dict.
        holdout_metrics: Optional holdout validation metrics dict.
        path: File path to write. Uses default if None.
        config: Optional AppConfig object.
    """
    path = _resolve_path(path, config=config)

    params = {
        # Trading parameters (what the bot uses)
        "STOP_MULT": float(best_config.get("SL", 1.5)),
        "TP_MULT": float(best_config.get("SL", 1.5)) * float(
            best_config.get("R:R", "1:2.0").split(":")[1]
        ),
        "MIN_CONFIDENCE": float(best_config.get("Thresh", 0.20)),
        "MAX_BARS": int(best_config.get("MB", 10)),
        "TRAIL_ATR_MULT": float(best_config.get("Trail", "1.0")),
        "MIN_HURST": float(best_config.get("Hurst", "0.50")),
        "MIN_ADX": int(best_config.get("ADX", "20")),
        "FILTER_MODE": best_config.get("Mode", "STRICT"),
        "SCALE_OUT_R": float(best_config.get("ScaleOut", "1.5")),
        "REGIME_HURST_FILTER": best_config.get("Regime", "True") == "True",

        # Derived
        "RR_RATIO": float(
            best_config.get("R:R", "1:2.0").split(":")[1]
        ),
    }

    output = {
        "version": "v6.1",
        "timestamp": datetime.now().isoformat(),
        "params": params,
    }

    # In-sample performance
    if metrics or best_config:
        output["in_sample"] = {
            "profit_factor": best_config.get("PF_Res", 0),
            "win_rate": best_config.get("WR_Res", 0),
            "trades": best_config.get("Trades", 0),
            "sharpe": best_config.get("Sharpe", 0),
            "max_drawdown_r": best_config.get("MaxDD_R", 0),
            "calmar": best_config.get("Calmar", 0),
            "total_return_r": best_config.get("TotalReturn_R", 0),
            "avg_win_r": best_config.get("AvgWin_R", 0),
            "avg_loss_r": best_config.get("AvgLoss_R", 0),
            "payoff_ratio": best_config.get("PayoffRatio", 0),
        }

    # Holdout performance
    if holdout_metrics:
        output["holdout"] = {
            "profit_factor": holdout_metrics.get("PF_Res", 0),
            "win_rate": holdout_metrics.get("WR_Res", 0),
            "trades": holdout_metrics.get("Trades", 0),
            "sharpe": holdout_metrics.get("Sharpe", 0),
            "max_drawdown_r": holdout_metrics.get("MaxDD_R", 0),
            "total_return_r": holdout_metrics.get("TotalReturn_R", 0),
        }

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    logging.info(f"Optimal parameters saved to {path}")
    return path


def load_optimal_params(path=None, config: Optional[AppConfig] = None):
    """
    Load optimal parameters from JSON config file.

    Called by the bot on startup. Returns the params dict that can be
    merged into SETTINGS.

    Args:
        path: File path to read. Searches defaults if None.
        config: Optional AppConfig object.

    Returns:
        Dict with parameter keys, or empty dict if no config found.
    """
    path = _resolve_path(path, config=config)

    if not os.path.exists(path):
        logging.warning(f"No optimal params file found at {path}")
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)

        params = data.get("params", {})
        version = data.get("version", "unknown")
        timestamp = data.get("timestamp", "unknown")

        # Log what we loaded
        logging.info(f"Loaded optimal params (v={version}, ts={timestamp})")
        if "in_sample" in data:
            is_data = data["in_sample"]
            logging.info(
                f"  In-sample: PF={is_data.get('profit_factor', 0):.2f} "
                f"WR={is_data.get('win_rate', 0):.1%} "
                f"Sharpe={is_data.get('sharpe', 0):.2f}"
            )
        if "holdout" in data:
            ho = data["holdout"]
            logging.info(
                f"  Holdout: PF={ho.get('profit_factor', 0):.2f} "
                f"WR={ho.get('win_rate', 0):.1%}"
            )

        return params

    except Exception as e:
        logging.error(f"Failed to load optimal params: {e}")
        return {}


def apply_to_settings(settings, optimal_params):
    """
    Merge optimal parameters into the bot's SETTINGS dict.

    Only overwrites keys that exist in optimal_params. Preserves all
    other settings.

    Args:
        settings: The bot's SETTINGS dict (modified in-place).
        optimal_params: Dict from load_optimal_params().

    Returns:
        List of keys that were updated.
    """
    updated = []
    key_map = {
        "STOP_MULT": "STOP_MULT",
        "TP_MULT": "TP_MULT",
        "MIN_CONFIDENCE": "MIN_CONFIDENCE",
        "TRAIL_ATR_MULT": "TRAIL_ATR_MULT",
        "MIN_HURST": "MIN_HURST",
        "MIN_ADX": "MIN_ADX",
        "FILTER_MODE": "FILTER_MODE",
        "SCALE_OUT_R": "SCALE_OUT_R",
        "REGIME_HURST_FILTER": "REGIME_HURST_FILTER",
    }

    for opt_key, settings_key in key_map.items():
        if opt_key in optimal_params:
            old = settings.get(settings_key)
            new = optimal_params[opt_key]
            if old != new:
                settings[settings_key] = new
                updated.append(f"{settings_key}: {old} -> {new}")

    # Update tier settings if RR changed
    if "RR_RATIO" in optimal_params:
        rr = optimal_params["RR_RATIO"]
        if "TIER_1" in settings:
            settings["TIER_1"]["RR"] = rr
        if "TIER_2" in settings:
            settings["TIER_2"]["RR"] = max(1.5, rr - 0.5)

    if updated:
        logging.info(f"Applied {len(updated)} optimized params: {', '.join(updated)}")

    return updated
