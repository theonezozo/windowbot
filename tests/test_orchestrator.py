"""Tests for the WindowBot orchestrator.

Validates design decisions:
- Full pipeline: fetch → decide → notify with mocked dependencies.
- Notification cooldown: 1 hour between non-urgent notifications.
- Cooldown bypass: urgent AQI close ALWAYS sends immediately.
- Auth failure: EcobeeAuthError triggers urgent notification to user.
- Per-floor iteration: upstairs and downstairs evaluated independently.
- AQI source preference: PurpleAir first, AirNow fallback on failure.
- Graceful handling of NWS/AQI failures.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from src.orchestrator import run_check, _fetch_aqi, _evaluate_floor, _NOTIFICATION_COOLDOWN
from src.ecobee_client import EcobeeAuthError, EcobeeApiError
from src.nws_client import NWSError
from src.decision_engine import DecisionEngine, FloorDecision, InsufficientDataError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _base_config(**overrides):
    """Build a minimal config dict for orchestrator tests."""
    cfg = {
        "ecobee_client_id": "cid",
        "ecobee_refresh_token": "rtoken",
        "airnow_api_key": "akey",
        "purpleair_api_key": "pkey",
        "user_latitude": 40.0,
        "user_longitude": -74.0,
        "ntfy_topic": "test",
        "upstairs_sensors": ["sensor_up"],
        "downstairs_sensors": ["sensor_down"],
        "hysteresis_open_diff": 1.0,
        "hysteresis_close_diff": 1.0,
        "max_outdoor_humidity": 80,
        "max_aqi_threshold": 100,
        "min_aqi_for_opening": 50,
        "allowed_hvac_modes": ["cool", "heatCool", "auto"],
        "enable_humidity_gate": True,
        "enable_aqi_gate": True,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------
# Auth Failure Notification
# ------------------------------------------------------------------


class TestAuthFailure:
    """EcobeeAuthError triggers urgent notification to user."""

    @patch("src.orchestrator.send_notification")
    @patch("src.orchestrator.StateManager")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_config")
    def test_auth_error_sends_urgent_notification(
        self, mock_config, mock_ecobee_cls, mock_state_cls, mock_notify
    ):
        """Ecobee auth failure → urgent notification sent."""
        mock_config.return_value = _base_config()
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.side_effect = EcobeeAuthError("Token revoked")
        mock_ecobee_cls.return_value = mock_ecobee

        run_check()

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["urgent"] is True
        assert kwargs["priority"] == "urgent"
        assert "Auth" in kwargs["title"] or "auth" in kwargs["title"].lower()

    @patch("src.orchestrator.send_notification")
    @patch("src.orchestrator.StateManager")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_config")
    def test_api_error_does_not_notify(
        self, mock_config, mock_ecobee_cls, mock_state_cls, mock_notify
    ):
        """EcobeeApiError (non-auth) → no notification sent, just returns."""
        mock_config.return_value = _base_config()
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.side_effect = EcobeeApiError("Server error")
        mock_ecobee_cls.return_value = mock_ecobee

        run_check()

        mock_notify.assert_not_called()


# ------------------------------------------------------------------
# AQI Source Preference
# ------------------------------------------------------------------


class TestAQISourcePreference:
    """PurpleAir first, AirNow fallback."""

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_purpleair_used_when_available(self, mock_pa_cls, mock_an_cls):
        """PurpleAir succeeds → AirNow not called."""
        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 42, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        config = _base_config()
        result = _fetch_aqi(config)

        assert result["aqi"] == 42
        assert result["source"] == "purpleair"
        mock_an_cls.assert_not_called()

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_fallback_on_purpleair_failure(self, mock_pa_cls, mock_an_cls):
        """PurpleAir raises → AirNow used as fallback."""
        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = Exception("PurpleAir down")
        mock_pa_cls.return_value = mock_pa

        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 55, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        config = _base_config()
        result = _fetch_aqi(config)

        assert result["aqi"] == 55
        assert result["source"] == "airnow"

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_both_fail_returns_zero(self, mock_pa_cls, mock_an_cls):
        """Both providers fail → AQI defaults to 0."""
        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = Exception("PA down")
        mock_pa_cls.return_value = mock_pa

        mock_an = MagicMock()
        mock_an.get_aqi.side_effect = Exception("AN down")
        mock_an_cls.return_value = mock_an

        config = _base_config()
        result = _fetch_aqi(config)

        assert result["aqi"] == 0
        assert result["source"] == "none"

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_no_purpleair_key_skips_to_airnow(self, mock_pa_cls, mock_an_cls):
        """No PurpleAir API key → skips PurpleAir, goes to AirNow."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 30, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        config = _base_config(purpleair_api_key="")
        result = _fetch_aqi(config)

        assert result["source"] == "airnow"
        mock_pa_cls.assert_not_called()


# ------------------------------------------------------------------
# Notification Cooldown
# ------------------------------------------------------------------


class TestNotificationCooldown:
    """1-hour cooldown between non-urgent notifications."""

    def test_cooldown_constant_is_3600(self):
        """Cooldown is 1 hour = 3600 seconds."""
        assert _NOTIFICATION_COOLDOWN == 3600

    @patch("src.orchestrator.send_notification")
    def test_first_notification_always_sent(self, mock_notify):
        """First notification ever (no LastNotificationTime) → always sent."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()

    @patch("src.orchestrator.send_notification")
    def test_cooldown_blocks_notification(self, mock_notify):
        """Non-urgent notification within cooldown → suppressed."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()

        # Last notification was 30 minutes ago (within 1-hour cooldown)
        recent = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": recent,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_not_called()

    @patch("src.orchestrator.send_notification")
    def test_cooldown_expired_allows_notification(self, mock_notify):
        """Non-urgent notification after cooldown expires → sent."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()

        # Last notification was 2 hours ago (beyond 1-hour cooldown)
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": old,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()


# ------------------------------------------------------------------
# Urgent AQI Cooldown Bypass
# ------------------------------------------------------------------


class TestUrgentCooldownBypass:
    """Urgent AQI close ALWAYS sends immediately regardless of cooldown."""

    @patch("src.orchestrator.send_notification")
    def test_urgent_bypasses_cooldown(self, mock_notify):
        """AQI ≥100 on OPEN windows → urgent notification even during cooldown."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()

        # Last notification was just 5 minutes ago (within cooldown)
        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "OPEN",
            "LastNotificationTime": recent,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 150}  # triggers urgent close
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["urgent"] is True


# ------------------------------------------------------------------
# Per-Floor Iteration
# ------------------------------------------------------------------


class TestPerFloorIteration:
    """Each floor evaluated independently in run_check."""

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.StateManager")
    @patch("src.orchestrator.get_config")
    def test_both_floors_evaluated(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """run_check calls _evaluate_floor for both upstairs and downstairs."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": 70.0, "humidity": 50.0
        }
        mock_nws_cls.return_value = mock_nws

        mock_fetch_aqi.return_value = {"aqi": 30}

        run_check()

        # Should be called for both floors
        assert mock_eval_floor.call_count == 2
        floor_names = [c.args[0] for c in mock_eval_floor.call_args_list]
        assert "upstairs" in floor_names
        assert "downstairs" in floor_names

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.StateManager")
    @patch("src.orchestrator.get_config")
    def test_empty_floor_skipped(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Floor with empty sensor list → skipped."""
        mock_config.return_value = _base_config(downstairs_sensors=[])

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": 70.0, "humidity": 50.0
        }
        mock_nws_cls.return_value = mock_nws

        mock_fetch_aqi.return_value = {"aqi": 30}

        run_check()

        # Only upstairs evaluated (downstairs has empty sensor list)
        assert mock_eval_floor.call_count == 1
        assert mock_eval_floor.call_args.args[0] == "upstairs"


# ------------------------------------------------------------------
# State Persistence
# ------------------------------------------------------------------


class TestStatePersistence:
    """State is always updated after evaluation."""

    @patch("src.orchestrator.send_notification")
    def test_state_updated_on_change(self, mock_notify):
        """State change → update_floor_state called with new state."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        state_mgr.update_floor_state.assert_called_once()
        update_args = state_mgr.update_floor_state.call_args
        assert update_args.args[0] == "upstairs"
        assert update_args.args[1]["CurrentState"] == "OPEN"

    @patch("src.orchestrator.send_notification")
    def test_state_updated_even_without_change(self, mock_notify):
        """No state change → still updates state (timestamp, reason, etc.)."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 80.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        state_mgr.update_floor_state.assert_called_once()


# ------------------------------------------------------------------
# NWS Failure Handling
# ------------------------------------------------------------------


class TestNWSFailure:
    """NWS failure gracefully aborts the check."""

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.StateManager")
    @patch("src.orchestrator.get_config")
    def test_nws_failure_aborts(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """NWSError → floors not evaluated."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("API down")
        mock_nws_cls.return_value = mock_nws

        run_check()

        mock_eval_floor.assert_not_called()


# ------------------------------------------------------------------
# InsufficientDataError Handling
# ------------------------------------------------------------------


class TestInsufficientData:
    """InsufficientDataError for a floor is caught and logged."""

    @patch("src.orchestrator.send_notification")
    def test_insufficient_data_caught(self, mock_notify):
        """Floor with all offline sensors → InsufficientDataError caught."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        # All sensors offline
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": False}]

        # Should not raise — InsufficientDataError is caught internally by the engine
        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        # State still gets updated (engine returns _keep)
        state_mgr.update_floor_state.assert_called_once()
