"""Tests for the configuration loader.

Validates design decisions:
- Environment variable loading with typed defaults.
- Type conversions: float, int, bool, list.
- Default values match architecture spec.
- Empty/missing env vars fall back to defaults.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.config import get_config, _env, _env_float, _env_int, _env_bool, _env_list


# ------------------------------------------------------------------
# Primitive Helpers
# ------------------------------------------------------------------


class TestEnvHelper:
    """_env returns the value or default."""

    @patch.dict(os.environ, {"TEST_KEY": "hello"})
    def test_env_reads_value(self):
        assert _env("TEST_KEY") == "hello"

    @patch.dict(os.environ, {}, clear=True)
    def test_env_missing_returns_default(self):
        os.environ.pop("TEST_KEY", None)
        assert _env("TEST_KEY", "fallback") == "fallback"

    @patch.dict(os.environ, {"TEST_KEY": ""})
    def test_env_empty_returns_default(self):
        assert _env("TEST_KEY", "fallback") == "fallback"


class TestEnvFloat:
    """_env_float converts to float."""

    @patch.dict(os.environ, {"F_KEY": "3.14"})
    def test_reads_float(self):
        assert _env_float("F_KEY", 0.0) == pytest.approx(3.14)

    @patch.dict(os.environ, {}, clear=True)
    def test_default_float(self):
        os.environ.pop("F_KEY", None)
        assert _env_float("F_KEY", 1.5) == 1.5


class TestEnvInt:
    """_env_int converts to int."""

    @patch.dict(os.environ, {"I_KEY": "42"})
    def test_reads_int(self):
        assert _env_int("I_KEY", 0) == 42

    @patch.dict(os.environ, {}, clear=True)
    def test_default_int(self):
        os.environ.pop("I_KEY", None)
        assert _env_int("I_KEY", 10) == 10


class TestEnvBool:
    """_env_bool converts to bool."""

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "Yes"])
    @patch.dict(os.environ, {})
    def test_truthy_values(self, value):
        with patch.dict(os.environ, {"B_KEY": value}):
            assert _env_bool("B_KEY", False) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "anything"])
    @patch.dict(os.environ, {})
    def test_falsy_values(self, value):
        with patch.dict(os.environ, {"B_KEY": value}):
            assert _env_bool("B_KEY", True) is False

    @patch.dict(os.environ, {}, clear=True)
    def test_default_bool(self):
        os.environ.pop("B_KEY", None)
        assert _env_bool("B_KEY", True) is True
        assert _env_bool("B_KEY", False) is False


class TestEnvList:
    """_env_list splits comma-separated values."""

    @patch.dict(os.environ, {"L_KEY": "a, b, c"})
    def test_splits_and_strips(self):
        assert _env_list("L_KEY") == ["a", "b", "c"]

    @patch.dict(os.environ, {"L_KEY": "single"})
    def test_single_value(self):
        assert _env_list("L_KEY") == ["single"]

    @patch.dict(os.environ, {"L_KEY": ""})
    def test_empty_string_returns_default(self):
        assert _env_list("L_KEY", ["x"]) == ["x"]

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_returns_default(self):
        os.environ.pop("L_KEY", None)
        assert _env_list("L_KEY", ["default"]) == ["default"]

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_no_default_returns_empty(self):
        os.environ.pop("L_KEY", None)
        assert _env_list("L_KEY") == []


# ------------------------------------------------------------------
# Full Config Defaults
# ------------------------------------------------------------------


class TestGetConfigDefaults:
    """get_config returns correct defaults matching architecture spec."""

    @patch.dict(os.environ, {}, clear=True)
    def test_hysteresis_defaults(self):
        # Clear all potential env vars
        for key in list(os.environ.keys()):
            if key.startswith(("HYSTERESIS", "MAX_", "MIN_", "ECOBEE", "AIRNOW",
                               "PURPLEAIR", "USER_", "NTFY_", "UPSTAIRS", "DOWNSTAIRS",
                               "POLLING", "NOTIFICATION", "ALLOWED", "AQ_", "ENABLE_")):
                os.environ.pop(key, None)

        cfg = get_config()
        assert cfg["hysteresis_open_diff"] == 1.0
        assert cfg["hysteresis_close_diff"] == 1.0

    @patch.dict(os.environ, {}, clear=True)
    def test_aqi_threshold_defaults(self):
        for key in list(os.environ.keys()):
            if key.startswith(("MAX_AQI", "MIN_AQI")):
                os.environ.pop(key, None)
        cfg = get_config()
        assert cfg["max_aqi_threshold"] == 100
        assert cfg["min_aqi_for_opening"] == 50

    @patch.dict(os.environ, {}, clear=True)
    def test_humidity_default(self):
        os.environ.pop("MAX_OUTDOOR_HUMIDITY", None)
        cfg = get_config()
        assert cfg["max_outdoor_humidity"] == 80

    @patch.dict(os.environ, {}, clear=True)
    def test_humidity_deadband_default_is_5(self):
        os.environ.pop("MAX_OUTDOOR_HUMIDITY_DEADBAND", None)
        cfg = get_config()
        assert cfg["humidity_deadband"] == 5

    @patch.dict(os.environ, {"MAX_OUTDOOR_HUMIDITY_DEADBAND": "10"})
    def test_humidity_deadband_override_honored(self):
        cfg = get_config()
        assert cfg["humidity_deadband"] == 10

    @patch.dict(os.environ, {}, clear=True)
    def test_notification_cooldown_default(self):
        os.environ.pop("NOTIFICATION_COOLDOWN_HOURS", None)
        cfg = get_config()
        assert cfg["notification_cooldown_hours"] == 1

    @patch.dict(os.environ, {}, clear=True)
    def test_purpleair_sensor_cache_hours_default(self):
        os.environ.pop("PURPLEAIR_SENSOR_CACHE_HOURS", None)
        cfg = get_config()
        assert cfg["purpleair_sensor_cache_hours"] == 12.0

    @patch.dict(os.environ, {"PURPLEAIR_SENSOR_CACHE_HOURS": "6"})
    def test_purpleair_sensor_cache_hours_override(self):
        cfg = get_config()
        assert cfg["purpleair_sensor_cache_hours"] == 6.0

    @patch.dict(os.environ, {}, clear=True)
    def test_allowed_hvac_modes_default(self):
        os.environ.pop("ALLOWED_HVAC_MODES", None)
        cfg = get_config()
        assert cfg["allowed_hvac_modes"] == ["cool", "heatCool", "auto"]

    @patch.dict(os.environ, {}, clear=True)
    def test_feature_flags_defaults(self):
        os.environ.pop("ENABLE_HUMIDITY_GATE", None)
        os.environ.pop("ENABLE_AQI_GATE", None)
        os.environ.pop("ENABLE_WIND_CHECK", None)
        cfg = get_config()
        assert cfg["enable_humidity_gate"] is True
        assert cfg["enable_aqi_gate"] is True
        assert cfg["enable_wind_check"] is False


# ------------------------------------------------------------------
# Config Overrides
# ------------------------------------------------------------------


class TestGetConfigOverrides:
    """Environment variables override defaults."""

    @patch.dict(os.environ, {
        "HYSTERESIS_OPEN_DIFF": "2.5",
        "HYSTERESIS_CLOSE_DIFF": "3.0",
        "MAX_OUTDOOR_HUMIDITY": "90",
        "MAX_AQI_THRESHOLD": "150",
        "MIN_AQI_FOR_OPENING": "75",
    })
    def test_numeric_overrides(self):
        cfg = get_config()
        assert cfg["hysteresis_open_diff"] == 2.5
        assert cfg["hysteresis_close_diff"] == 3.0
        assert cfg["max_outdoor_humidity"] == 90
        assert cfg["max_aqi_threshold"] == 150
        assert cfg["min_aqi_for_opening"] == 75

    @patch.dict(os.environ, {"ALLOWED_HVAC_MODES": "cool,heat"})
    def test_list_override(self):
        cfg = get_config()
        assert cfg["allowed_hvac_modes"] == ["cool", "heat"]

    @patch.dict(os.environ, {"ENABLE_HUMIDITY_GATE": "false"})
    def test_bool_override(self):
        cfg = get_config()
        assert cfg["enable_humidity_gate"] is False


# ------------------------------------------------------------------
# Outdoor source stickiness (Phase 2)
# ------------------------------------------------------------------


class TestOutdoorSourceStickinessConfig:
    """``outdoor_source_stickiness`` gates the conservative median-pool
    retention feature (env ``OUTDOOR_SOURCE_STICKINESS``, default on)."""

    @patch.dict(os.environ, {}, clear=True)
    def test_outdoor_source_stickiness_default_is_true(self):
        os.environ.pop("OUTDOOR_SOURCE_STICKINESS", None)
        cfg = get_config()
        assert cfg["outdoor_source_stickiness"] is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "No"])
    def test_outdoor_source_stickiness_falsy_disables(self, value):
        with patch.dict(os.environ, {"OUTDOOR_SOURCE_STICKINESS": value}):
            cfg = get_config()
            assert cfg["outdoor_source_stickiness"] is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "Yes"])
    def test_outdoor_source_stickiness_truthy_enables(self, value):
        with patch.dict(os.environ, {"OUTDOOR_SOURCE_STICKINESS": value}):
            cfg = get_config()
            assert cfg["outdoor_source_stickiness"] is True


# ------------------------------------------------------------------
# Outdoor signal-confidence gate (revert R2 + confidence gate)
# ------------------------------------------------------------------


class TestOutdoorConfidenceConfig:
    """The four confidence-gate tuning keys and their env overrides.

    Defaults (architecture decision #1 addendum): the gate is ON, needs 2
    corroborating sources, holds an uncorroborated churn move whose contributor
    spread exceeds 3.0°F, and releases a persistent hold after 2 cycles.
    """

    @patch.dict(os.environ, {}, clear=True)
    def test_confidence_defaults(self):
        for key in (
            "OUTDOOR_CONFIDENCE_ENABLED",
            "OUTDOOR_MIN_CORROBORATING_SOURCES",
            "OUTDOOR_CONFIDENCE_MAX_SPREAD_F",
            "OUTDOOR_CONFIDENCE_HOLD_MAX_CYCLES",
        ):
            os.environ.pop(key, None)
        cfg = get_config()
        assert cfg["outdoor_confidence_enabled"] is True
        assert cfg["outdoor_min_corroborating_sources"] == 2
        assert cfg["outdoor_confidence_max_spread_f"] == pytest.approx(3.0)
        assert cfg["outdoor_confidence_hold_max_cycles"] == 2

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "No"])
    def test_confidence_enabled_falsy_disables(self, value):
        with patch.dict(os.environ, {"OUTDOOR_CONFIDENCE_ENABLED": value}):
            cfg = get_config()
            assert cfg["outdoor_confidence_enabled"] is False

    @pytest.mark.parametrize("value", ["true", "True", "1", "yes"])
    def test_confidence_enabled_truthy_enables(self, value):
        with patch.dict(os.environ, {"OUTDOOR_CONFIDENCE_ENABLED": value}):
            cfg = get_config()
            assert cfg["outdoor_confidence_enabled"] is True

    @patch.dict(os.environ, {"OUTDOOR_MIN_CORROBORATING_SOURCES": "3"})
    def test_min_corroborating_sources_int_override(self):
        cfg = get_config()
        assert cfg["outdoor_min_corroborating_sources"] == 3
        assert isinstance(cfg["outdoor_min_corroborating_sources"], int)

    @patch.dict(os.environ, {"OUTDOOR_CONFIDENCE_MAX_SPREAD_F": "4.5"})
    def test_max_spread_float_override(self):
        cfg = get_config()
        assert cfg["outdoor_confidence_max_spread_f"] == pytest.approx(4.5)

    @patch.dict(os.environ, {"OUTDOOR_CONFIDENCE_HOLD_MAX_CYCLES": "5"})
    def test_hold_max_cycles_int_override(self):
        cfg = get_config()
        assert cfg["outdoor_confidence_hold_max_cycles"] == 5

