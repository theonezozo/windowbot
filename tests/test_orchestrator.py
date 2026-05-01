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
from src.openmeteo_client import OpenMeteoError
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
    @patch("src.orchestrator.get_state_manager")
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
    @patch("src.orchestrator.get_state_manager")
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
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_both_floors_evaluated(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
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
        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("test")

        mock_fetch_aqi.return_value = {"aqi": 30}

        run_check()

        # Should be called for both floors
        assert mock_eval_floor.call_count == 2
        floor_names = [c.args[0] for c in mock_eval_floor.call_args_list]
        assert "upstairs" in floor_names
        assert "downstairs" in floor_names

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_empty_floor_skipped(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
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
        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("test")

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
    """NWS failure when it's the only remaining source gracefully aborts the check."""

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_failure_with_openmeteo_failure_aborts(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """NWSError + OpenMeteoError (peer and fallback both fail) → floors not evaluated."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("API down")
        mock_get_nws.return_value = mock_nws

        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("OM down")
        mock_om.get_outdoor_conditions.side_effect = OpenMeteoError("OM down")
        mock_om_cls.return_value = mock_om

        run_check()

        mock_eval_floor.assert_not_called()


# ------------------------------------------------------------------
# InsufficientDataError Handling
# ------------------------------------------------------------------


class TestInsufficientData:
    """InsufficientDataError for a floor is caught and logged."""

    @patch("src.orchestrator.send_notification")
    def test_offline_sensor_with_valid_temp_works(self, mock_notify):
        """Floor with offline sensor but valid temp → decision proceeds normally."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        # Sensor is offline but has valid temperature
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": False}]

        # Should proceed normally — offline status doesn't block decisions anymore
        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        # State gets updated with normal decision (warmest=74, threshold=73, outdoor 68 < 73 → OPEN)
        state_mgr.update_floor_state.assert_called_once()
        call_args = state_mgr.update_floor_state.call_args[0]
        assert call_args[1]["CurrentState"] == "OPEN"


# ------------------------------------------------------------------
# Conditional / Lazy AQI Polling
# ------------------------------------------------------------------


class TestConditionalAqiPolling:
    """Lazy AQI fetching — orchestrator skips PurpleAir/AirNow when unnecessary.

    Validates:
    - AQI skipped when windows CLOSED + indoor comfortable
    - AQI skipped when HVAC mode blocks action
    - AQI fetched when windows OPEN (safety net)
    - AQI fetched when temperature favors opening
    - AQI cached across floors in same run_check cycle
    - Fallback PurpleAir→AirNow still works in lazy path
    - Skip reason logged correctly
    """

    # -- Helpers --

    @staticmethod
    def _setup_run_check_mocks(
        mock_config,
        mock_state_cls,
        mock_ecobee_cls,
        mock_nws_cls,
        mock_om_cls=None,
        *,
        sensor_temps=None,
        hvac_mode="cool",
        outdoor_temp=65.0,
        outdoor_humidity=50.0,
        current_state="CLOSED",
    ):
        """Wire up the standard mocks for a run_check integration test."""
        if sensor_temps is None:
            sensor_temps = {"sensor_up": 70.0, "sensor_down": 70.0}

        mock_config.return_value = _base_config()

        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {
            "CurrentState": current_state,
        }
        mock_state_cls.return_value = mock_state

        sensors = [
            {"name": name, "temperature_f": temp, "is_online": True}
            for name, temp in sensor_temps.items()
        ]
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = sensors
        mock_ecobee.get_hvac_mode.return_value = hvac_mode
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": outdoor_temp,
            "humidity": outdoor_humidity,
        }
        mock_nws_cls.return_value = mock_nws

        if mock_om_cls is not None:
            mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")

    # ------------------------------------------------------------------
    # 1. AQI skipped when windows CLOSED + indoor comfortable (≤72°F)
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_skipped_when_closed_and_comfortable(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Windows CLOSED + indoor 70°F (≤ 72°F comfort max) → AQI not fetched."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        mock_fetch_aqi.assert_not_called()

    # ------------------------------------------------------------------
    # 2. AQI skipped when HVAC mode not in allowed list
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_skipped_when_hvac_mode_not_allowed(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """HVAC mode 'heat' not in allowed modes → AQI not fetched."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 78.0, "sensor_down": 78.0},
            hvac_mode="heat",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        mock_fetch_aqi.assert_not_called()

    # ------------------------------------------------------------------
    # 3. AQI fetched when windows are OPEN
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_fetched_when_windows_open(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Windows OPEN → AQI always fetched (urgent close safety net)."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 74.0, "sensor_down": 74.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )
        mock_fetch_aqi.return_value = {"aqi": 30, "source": "purpleair"}

        run_check()

        mock_fetch_aqi.assert_called_once()

    # ------------------------------------------------------------------
    # 4. AQI fetched when windows closed but temperature favors opening
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_fetched_when_temp_favors_opening(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Indoor 78°F, outdoor 65°F, CLOSED → AQI fetched to confirm safe."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 78.0, "sensor_down": 78.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )
        mock_fetch_aqi.return_value = {"aqi": 25, "source": "purpleair"}

        run_check()

        mock_fetch_aqi.assert_called_once()

    # ------------------------------------------------------------------
    # 5. AQI cached across floors in same run_check cycle
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_cached_across_floors(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Both floors need AQI (OPEN) → _fetch_aqi called exactly once."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 74.0, "sensor_down": 74.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )
        aqi_result = {"aqi": 42, "source": "purpleair"}
        mock_fetch_aqi.return_value = aqi_result

        run_check()

        # Single fetch despite two floors needing AQI
        mock_fetch_aqi.assert_called_once()
        # Both floors evaluated with the cached result
        assert mock_eval_floor.call_count == 2
        for eval_call in mock_eval_floor.call_args_list:
            assert eval_call.args[4] == aqi_result  # aqi_data positional arg

    # ------------------------------------------------------------------
    # 6. Fallback PurpleAir→AirNow still works in the lazy path
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_fallback_works_when_aqi_needed(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_pa_cls, mock_an_cls, mock_eval_floor,
    ):
        """PurpleAir fails when AQI IS needed → AirNow fallback used."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 74.0, "sensor_down": 74.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )

        # PurpleAir fails
        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = Exception("PurpleAir down")
        mock_pa_cls.return_value = mock_pa

        # AirNow succeeds
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 55, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        run_check()

        # AirNow was used as fallback
        mock_an.get_aqi.assert_called_once()
        # _evaluate_floor received the AirNow result
        assert mock_eval_floor.call_count == 2
        for eval_call in mock_eval_floor.call_args_list:
            assert eval_call.args[4]["source"] == "airnow"

    # ------------------------------------------------------------------
    # 7. AQI skip reason logged
    # ------------------------------------------------------------------

    @patch("src.orchestrator.logger")
    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_skipped_log_message(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor, mock_logger,
    ):
        """When AQI is skipped, logger records the floor name and skip reason."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        # At least one "Skipping AQI fetch" log per skipped floor
        skip_calls = [
            c for c in mock_logger.info.call_args_list
            if len(c.args) >= 2 and "Skipping AQI fetch" in str(c.args[0])
        ]
        assert len(skip_calls) >= 1
        # The log message includes the floor name
        assert any("upstairs" in str(c) for c in skip_calls)

    # ------------------------------------------------------------------
    # 8. Skipped AQI passes sentinel to _evaluate_floor
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_skipped_aqi_passes_sentinel_to_evaluate(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """When AQI is skipped, _evaluate_floor receives {"aqi": 0, "source": "skipped"}."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        mock_fetch_aqi.assert_not_called()
        for eval_call in mock_eval_floor.call_args_list:
            aqi_arg = eval_call.args[4]
            assert aqi_arg == {"aqi": 0, "source": "skipped"}


# ------------------------------------------------------------------
# Weather Fallback Chain: NWS (+ OM peer) → Open-Meteo sole fallback
# ------------------------------------------------------------------


class TestWeatherFallbackChain:
    """Outdoor weather: Open-Meteo always attempted as a peer alongside NWS.

    Flow:
      1. om.get_observation() called first (peer blending attempt).
      2. nws.get_outdoor_conditions(peer_observations=...) called.
      3. If NWS fails + OM peer fresh → OM used as sole source.
      4. If NWS fails + OM peer stale → om.get_outdoor_conditions() (no age check).
      5. If everything fails → return early.
    """

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_succeeds_floors_evaluated(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS succeeds (OM peer optional) → floors evaluated."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": 68.0, "humidity": 55.0, "source": "nws",
        }
        mock_get_nws.return_value = mock_nws

        # OM peer fails — NWS should still work alone
        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("stale")
        mock_om_cls.return_value = mock_om

        run_check()

        mock_nws.get_outdoor_conditions.assert_called_once()
        assert mock_eval_floor.call_count >= 1

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_fails_om_peer_used_as_sole_source(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS fails → fresh OM peer used as sole outdoor source; get_outdoor_conditions not called."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("NWS down")
        mock_get_nws.return_value = mock_nws

        from datetime import datetime, timezone
        mock_om = MagicMock()
        mock_om.get_observation.return_value = {
            "station_id": "OPENMETEO",
            "temperature_f": 66.0,
            "humidity": 60.0,
            "wind_speed_mph": 5.0,
            "timestamp": datetime.now(timezone.utc),
        }
        mock_om_cls.return_value = mock_om

        run_check()

        mock_nws.get_outdoor_conditions.assert_called_once()
        mock_om.get_observation.assert_called_once()
        mock_om.get_outdoor_conditions.assert_not_called()
        assert mock_eval_floor.call_count >= 1

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_fails_peer_stale_falls_back_to_om_conditions(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS fails, OM peer stale → get_outdoor_conditions() last-resort fallback."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("NWS down")
        mock_get_nws.return_value = mock_nws

        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("stale")
        mock_om.get_outdoor_conditions.return_value = {
            "temperature_f": 66.0, "humidity": 60.0, "wind_speed_mph": 5.0,
            "source": "openmeteo", "is_fallback": True,
            "station_count": 1, "used_cache": False,
        }
        mock_om_cls.return_value = mock_om

        run_check()

        mock_om.get_outdoor_conditions.assert_called_once()
        assert mock_eval_floor.call_count >= 1

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_both_fail_returns_early(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS fails, OM peer stale, OM fallback fails → returns early, no floor evaluation."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("NWS down")
        mock_get_nws.return_value = mock_nws

        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("stale")
        mock_om.get_outdoor_conditions.side_effect = OpenMeteoError("OM down")
        mock_om_cls.return_value = mock_om

        run_check()

        mock_eval_floor.assert_not_called()

