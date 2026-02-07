"""Tests for hedge_fund.config - parameter sync between backtester and bot."""

import json
import os
import tempfile
import pytest

from hedge_fund.config import (
    AppConfig,
    load_app_config,
    save_optimal_params,
    load_optimal_params,
    apply_to_settings,
)


class TestSaveOptimalParams:
    def test_saves_json_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config = {
                "SL": 1.5, "R:R": "1:2.0", "MB": 10, "Thresh": 0.20,
                "Mode": "STRICT", "Trail": "1.0", "Hurst": "0.50",
                "ADX": "20", "ScaleOut": "1.5", "Regime": "True",
                "PF_Res": 1.8, "WR_Res": 0.55, "Trades": 100,
                "Sharpe": 1.5, "MaxDD_R": -3.0, "Calmar": 2.0,
                "TotalReturn_R": 15.0, "AvgWin_R": 1.8, "AvgLoss_R": 0.9,
                "PayoffRatio": 2.0,
            }
            result = save_optimal_params(config, path=path)
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)
            assert "params" in data
            assert "version" in data
            assert data["params"]["STOP_MULT"] == 1.5
            assert data["params"]["TP_MULT"] == 3.0
            assert data["params"]["MIN_CONFIDENCE"] == 0.20
        finally:
            os.unlink(path)

    def test_includes_in_sample_metrics(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config = {
                "SL": 2.0, "R:R": "1:2.5", "MB": 12, "Thresh": 0.15,
                "Mode": "MODERATE", "Trail": "1.5", "Hurst": "0.45",
                "ADX": "25", "ScaleOut": "1.0", "Regime": "False",
                "PF_Res": 2.0, "WR_Res": 0.60, "Trades": 80,
                "Sharpe": 2.0, "MaxDD_R": -2.0, "Calmar": 3.0,
                "TotalReturn_R": 20.0, "AvgWin_R": 2.0, "AvgLoss_R": 0.8,
                "PayoffRatio": 2.5,
            }
            save_optimal_params(config, path=path)

            with open(path) as f:
                data = json.load(f)
            assert data["in_sample"]["profit_factor"] == 2.0
            assert data["in_sample"]["win_rate"] == 0.60
        finally:
            os.unlink(path)

    def test_includes_holdout_metrics(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config = {"SL": 1.5, "R:R": "1:2.0", "MB": 10, "Thresh": 0.20,
                      "Mode": "STRICT", "Trail": "1.0", "Hurst": "0.50",
                      "ADX": "20", "ScaleOut": "1.5", "Regime": "True"}
            holdout = {"PF_Res": 1.5, "WR_Res": 0.52, "Trades": 30,
                       "Sharpe": 1.2, "MaxDD_R": -2.5, "TotalReturn_R": 8.0}
            save_optimal_params(config, holdout_metrics=holdout, path=path)

            with open(path) as f:
                data = json.load(f)
            assert "holdout" in data
            assert data["holdout"]["profit_factor"] == 1.5
        finally:
            os.unlink(path)


class TestLoadOptimalParams:
    def test_loads_saved_params(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config = {"SL": 2.0, "R:R": "1:3.0", "MB": 14, "Thresh": 0.25,
                      "Mode": "MINIMAL", "Trail": "0.75", "Hurst": "0.60",
                      "ADX": "15", "ScaleOut": "2.0", "Regime": "True"}
            save_optimal_params(config, path=path)

            params = load_optimal_params(path=path)
            assert params["STOP_MULT"] == 2.0
            assert params["TP_MULT"] == 6.0  # 2.0 * 3.0
            assert params["MIN_CONFIDENCE"] == 0.25
            assert params["FILTER_MODE"] == "MINIMAL"
            assert params["REGIME_HURST_FILTER"] is True
        finally:
            os.unlink(path)

    def test_returns_empty_for_missing_file(self):
        params = load_optimal_params(path="/tmp/nonexistent_params_xyz.json")
        assert params == {}


class TestApplyToSettings:
    def test_updates_matching_keys(self):
        settings = {
            "STOP_MULT": 1.5,
            "TP_MULT": 3.0,
            "MIN_CONFIDENCE": 0.02,
            "OTHER_KEY": "unchanged",
        }
        optimal = {
            "STOP_MULT": 2.0,
            "TP_MULT": 5.0,
            "MIN_CONFIDENCE": 0.15,
        }
        updated = apply_to_settings(settings, optimal)
        assert settings["STOP_MULT"] == 2.0
        assert settings["TP_MULT"] == 5.0
        assert settings["MIN_CONFIDENCE"] == 0.15
        assert settings["OTHER_KEY"] == "unchanged"
        assert len(updated) == 3

    def test_preserves_unrelated_settings(self):
        settings = {"STOP_MULT": 1.5, "SCAN_INTERVAL": 60, "MAX_POSITIONS": 5}
        optimal = {"STOP_MULT": 2.0}
        apply_to_settings(settings, optimal)
        assert settings["SCAN_INTERVAL"] == 60
        assert settings["MAX_POSITIONS"] == 5

    def test_updates_tier_rr(self):
        settings = {
            "TIER_1": {"RR": 2.0, "NAME": "SPECIALIST"},
            "TIER_2": {"RR": 1.5, "NAME": "GRINDER"},
        }
        optimal = {"RR_RATIO": 3.0}
        apply_to_settings(settings, optimal)
        assert settings["TIER_1"]["RR"] == 3.0
        assert settings["TIER_2"]["RR"] == 2.5

    def test_empty_optimal_changes_nothing(self):
        settings = {"STOP_MULT": 1.5}
        updated = apply_to_settings(settings, {})
        assert settings["STOP_MULT"] == 1.5
        assert updated == []


class TestAppConfig:
    def test_loads_env_driven_paths(self, monkeypatch, tmp_path):
        data_root = tmp_path / "runtime"
        log_dir = tmp_path / "logs"
        params_path = tmp_path / "cfg" / "optimal.json"

        monkeypatch.setenv("DATA_ROOT", str(data_root))
        monkeypatch.setenv("LOG_DIR", str(log_dir))
        monkeypatch.setenv("OPTIMAL_PARAMS_PATH", str(params_path))

        config = load_app_config()

        assert isinstance(config, AppConfig)
        assert config.data_root == str(data_root.resolve())
        assert config.log_dir == str(log_dir.resolve())
        assert config.optimal_params_path == str(params_path.resolve())
        assert os.path.isdir(config.db_dir)
        assert os.path.isdir(config.model_dir)

    def test_uses_config_path_for_optimal_params_roundtrip(self, tmp_path):
        config = AppConfig(
            data_root=str((tmp_path / "data").resolve()),
            log_dir=str((tmp_path / "logs").resolve()),
            optimal_params_path=str((tmp_path / "data" / "optimal_params.json").resolve()),
        )
        config.ensure_directories()

        payload = {"SL": 1.7, "R:R": "1:2.0", "Thresh": 0.21}
        save_optimal_params(payload, config=config)
        loaded = load_optimal_params(config=config)

        assert loaded["STOP_MULT"] == 1.7
        assert loaded["TP_MULT"] == 3.4
        assert loaded["MIN_CONFIDENCE"] == 0.21
