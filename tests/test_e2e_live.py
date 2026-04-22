"""End-to-end integration tests using LIVE API calls.

These tests exercise the real WindowBot pipeline against production APIs.
They are excluded from default test runs and only execute when explicitly
requested:

    pytest -m e2e -v --tb=short

Requires environment variables from local.settings.json to be set.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Load local.settings.json into env BEFORE importing any src modules
# ---------------------------------------------------------------------------

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "local.settings.json"


def _load_local_settings() -> bool:
    """Inject local.settings.json values into the environment.

    Returns True if the settings file was found and loaded.
    """
    if not _SETTINGS_PATH.exists():
        return False
    with open(_SETTINGS_PATH) as f:
        data = json.load(f)
    for key, value in data.get("Values", {}).items():
        if key not in os.environ or os.environ[key] == "":
            os.environ[key] = str(value)
    return True


_SETTINGS_LOADED = _load_local_settings()

# Now safe to import src modules (they read env at import or call time)
from src.beestat_client import BeestatClient  # noqa: E402
from src.nws_client import NWSClient  # noqa: E402
from src.purpleair_client import PurpleAirClient  # noqa: E402
from src.airnow_client import AirNowClient, AirNowError  # noqa: E402
from src.notifier import send_notification  # noqa: E402
from src.config import get_config  # noqa: E402
from src.decision_engine import DecisionEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.e2e

logger = logging.getLogger("windowbot.e2e")

_LIVE_TIMEOUT = 30  # seconds — generous for flaky APIs


def _require_env(*keys: str) -> None:
    """Skip the test if any required environment variable is missing."""
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")


def _require_settings() -> None:
    """Skip the entire module if local.settings.json wasn't loaded."""
    if not _SETTINGS_LOADED:
        pytest.skip("local.settings.json not found — cannot run live E2E tests")


# ---------------------------------------------------------------------------
# Beestat (indoor sensors)
# ---------------------------------------------------------------------------


class TestBeestatLive:
    """Live API tests for the Beestat indoor sensor provider."""

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()
        _require_env("BEESTAT_API_KEY")

    def test_get_sensors_returns_list(self):
        """Beestat returns a non-empty list of sensor dicts."""
        client = BeestatClient(api_key=os.environ["BEESTAT_API_KEY"])
        sensors = client.get_sensors()

        assert isinstance(sensors, list)
        assert len(sensors) > 0, "Expected at least one sensor from Beestat"

    def test_sensors_have_temperature(self):
        """At least one returned sensor has a valid temperature_f reading."""
        client = BeestatClient(api_key=os.environ["BEESTAT_API_KEY"])
        sensors = client.get_sensors()

        temps = [s["temperature_f"] for s in sensors if s.get("temperature_f") is not None]
        assert len(temps) > 0, "No sensors reported a temperature_f value"
        for t in temps:
            assert 30.0 <= t <= 120.0, f"Suspicious temperature: {t}°F"

    def test_sensors_have_expected_names(self):
        """Configured sensor names appear in the Beestat response."""
        client = BeestatClient(api_key=os.environ["BEESTAT_API_KEY"])
        sensors = client.get_sensors()
        names = {s["name"] for s in sensors}

        expected = {"Kid's Room", "Bedroom", "Office", "Downstairs"}
        found = expected & names
        assert len(found) >= 1, (
            f"None of the expected sensors {expected} found in Beestat response: {names}"
        )

    def test_get_hvac_mode_returns_string(self):
        """HVAC mode is one of the known Ecobee mode strings."""
        client = BeestatClient(api_key=os.environ["BEESTAT_API_KEY"])
        mode = client.get_hvac_mode()

        valid_modes = {"heat", "cool", "heatCool", "auto", "off", "auxHeatOnly"}
        assert mode in valid_modes, f"Unexpected HVAC mode: {mode!r}"


# ---------------------------------------------------------------------------
# NWS (outdoor weather)
# ---------------------------------------------------------------------------


class TestNWSLive:
    """Live API tests for the National Weather Service client."""

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()
        _require_env("USER_LATITUDE", "USER_LONGITUDE")

    def _client(self) -> NWSClient:
        return NWSClient(
            float(os.environ["USER_LATITUDE"]),
            float(os.environ["USER_LONGITUDE"]),
        )

    def test_discover_stations_returns_list(self):
        """NWS discovers at least one weather station near the configured location."""
        nws = self._client()
        stations = nws.discover_stations()

        assert isinstance(stations, list)
        assert len(stations) > 0, "No NWS stations discovered"

    def test_outdoor_conditions_has_temperature(self):
        """Aggregated outdoor conditions include a plausible temperature_f."""
        nws = self._client()
        outdoor = nws.get_outdoor_conditions()

        assert "temperature_f" in outdoor
        temp = outdoor["temperature_f"]
        assert isinstance(temp, (int, float))
        assert -20.0 <= temp <= 130.0, f"Outdoor temp looks wrong: {temp}°F"

    def test_outdoor_conditions_has_station_count(self):
        """Response reports how many stations contributed to the median."""
        nws = self._client()
        outdoor = nws.get_outdoor_conditions()

        assert "station_count" in outdoor
        assert outdoor["station_count"] >= 1


# ---------------------------------------------------------------------------
# PurpleAir (AQI primary)
# ---------------------------------------------------------------------------


class TestPurpleAirLive:
    """Live API tests for the PurpleAir AQI client."""

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()
        _require_env("PURPLEAIR_API_KEY", "USER_LATITUDE", "USER_LONGITUDE")

    def _client(self) -> PurpleAirClient:
        return PurpleAirClient(
            float(os.environ["USER_LATITUDE"]),
            float(os.environ["USER_LONGITUDE"]),
            api_key=os.environ["PURPLEAIR_API_KEY"],
        )

    def test_find_nearby_sensors_returns_results(self):
        """PurpleAir finds outdoor sensors within 5 km."""
        pa = self._client()
        sensors = pa.find_nearby_sensors()

        assert isinstance(sensors, list)
        assert len(sensors) > 0, "No PurpleAir sensors found nearby"

    def test_sensors_sorted_by_distance(self):
        """Returned sensors are sorted closest-first."""
        pa = self._client()
        sensors = pa.find_nearby_sensors()

        if len(sensors) >= 2:
            distances = [s["distance_km"] for s in sensors]
            assert distances == sorted(distances), "Sensors not sorted by distance"

    def test_get_aqi_returns_valid_value(self):
        """AQI computation returns an integer in the 0–500 range."""
        pa = self._client()
        result = pa.get_aqi()

        assert "aqi" in result
        assert isinstance(result["aqi"], int)
        assert 0 <= result["aqi"] <= 500, f"AQI out of range: {result['aqi']}"
        assert result.get("source") == "purpleair"

    def test_get_aqi_reports_sensor_count(self):
        """AQI result includes the number of contributing sensors (up to 3)."""
        pa = self._client()
        result = pa.get_aqi()

        assert "sensor_count" in result
        assert 1 <= result["sensor_count"] <= 3


# ---------------------------------------------------------------------------
# AirNow (AQI fallback) — xfail because 504s are common
# ---------------------------------------------------------------------------


class TestAirNowLive:
    """Live API tests for the AirNow fallback AQI client."""

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()
        _require_env("AIRNOW_API_KEY", "USER_LATITUDE", "USER_LONGITUDE")

    @pytest.mark.xfail(reason="AirNow frequently returns 504 Gateway Timeout", strict=False)
    def test_get_aqi_returns_valid_value(self):
        """AirNow AQI result contains a plausible integer."""
        client = AirNowClient(
            os.environ["AIRNOW_API_KEY"],
            float(os.environ["USER_LATITUDE"]),
            float(os.environ["USER_LONGITUDE"]),
        )
        result = client.get_aqi()

        assert "aqi" in result
        assert isinstance(result["aqi"], int)
        assert 0 <= result["aqi"] <= 500
        assert result.get("source") == "airnow"


# ---------------------------------------------------------------------------
# Notifier (ntfy.sh)
# ---------------------------------------------------------------------------


class TestNotifierLive:
    """Live notification delivery via ntfy.sh."""

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()
        _require_env("NTFY_TOPIC")

    def test_send_test_notification(self):
        """A test notification is delivered successfully to ntfy.sh."""
        result = send_notification(
            title="🧪 WindowBot E2E Test",
            message="This is an automated test notification. Safe to ignore.",
            priority="low",
        )
        assert result is True


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestConfigLive:
    """Verify config loads correctly from the live environment."""

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()

    def test_config_loads_all_keys(self):
        """get_config() returns a dict with all expected keys populated."""
        config = get_config()

        assert config["beestat_api_key"], "BEESTAT_API_KEY missing from config"
        assert config["user_latitude"] != 0.0, "USER_LATITUDE not set"
        assert config["user_longitude"] != 0.0, "USER_LONGITUDE not set"
        assert config["ntfy_topic"], "NTFY_TOPIC missing from config"
        assert len(config["upstairs_sensors"]) > 0, "No upstairs sensors configured"
        assert len(config["downstairs_sensors"]) > 0, "No downstairs sensors configured"

    def test_config_thresholds_match_architecture(self):
        """Config defaults match the architecture spec (symmetric 1°F hysteresis, etc.)."""
        config = get_config()

        assert config["hysteresis_open_diff"] == 1.0
        assert config["hysteresis_close_diff"] == 1.0
        assert config["max_outdoor_humidity"] == 80
        assert config["max_aqi_threshold"] == 100
        assert config["min_aqi_for_opening"] == 50
        assert set(config["allowed_hvac_modes"]) == {"cool", "heatCool", "auto"}


# ---------------------------------------------------------------------------
# Decision engine with LIVE data
# ---------------------------------------------------------------------------


class TestDecisionEngineLiveData:
    """Feed real API data into the decision engine to verify integration."""

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()
        _require_env("BEESTAT_API_KEY", "USER_LATITUDE", "USER_LONGITUDE", "PURPLEAIR_API_KEY")

    def test_decide_with_live_data_completes(self):
        """Decision engine processes real sensor data without raising."""
        config = get_config()

        # Fetch real data
        beestat = BeestatClient(api_key=config["beestat_api_key"])
        sensors = beestat.get_sensors()
        hvac_mode = beestat.get_hvac_mode()

        nws = NWSClient(config["user_latitude"], config["user_longitude"])
        outdoor = nws.get_outdoor_conditions()

        pa = PurpleAirClient(
            config["user_latitude"], config["user_longitude"],
            api_key=config["purpleair_api_key"],
        )
        aqi_data = pa.get_aqi()

        engine = DecisionEngine(config)

        # Run decision for upstairs
        decision = engine.decide(
            floor="upstairs",
            floor_sensors=sensors,
            outdoor=outdoor,
            aqi=aqi_data,
            hvac_mode=hvac_mode,
            last_state="CLOSED",
            floor_group=config["upstairs_sensors"],
        )

        assert decision.floor == "upstairs"
        assert decision.new_state in ("OPEN", "CLOSED")
        assert isinstance(decision.reason, str) and len(decision.reason) > 0
        assert isinstance(decision.urgent, bool)
        assert isinstance(decision.changed, bool)

    def test_decide_both_floors(self):
        """Both upstairs and downstairs produce valid decisions."""
        config = get_config()

        beestat = BeestatClient(api_key=config["beestat_api_key"])
        sensors = beestat.get_sensors()
        hvac_mode = beestat.get_hvac_mode()

        nws = NWSClient(config["user_latitude"], config["user_longitude"])
        outdoor = nws.get_outdoor_conditions()

        pa = PurpleAirClient(
            config["user_latitude"], config["user_longitude"],
            api_key=config["purpleair_api_key"],
        )
        aqi_data = pa.get_aqi()

        engine = DecisionEngine(config)

        for floor_name, sensor_group in [
            ("upstairs", config["upstairs_sensors"]),
            ("downstairs", config["downstairs_sensors"]),
        ]:
            decision = engine.decide(
                floor=floor_name,
                floor_sensors=sensors,
                outdoor=outdoor,
                aqi=aqi_data,
                hvac_mode=hvac_mode,
                last_state="CLOSED",
                floor_group=sensor_group,
            )
            assert decision.floor == floor_name
            assert decision.new_state in ("OPEN", "CLOSED")
            logger.info(
                "Floor %s: %s — %s", floor_name, decision.new_state, decision.reason,
            )


# ---------------------------------------------------------------------------
# Full orchestrator pipeline (minus Azure Table Storage)
# ---------------------------------------------------------------------------


class TestOrchestratorPipeline:
    """End-to-end pipeline test using real APIs.

    The orchestrator's run_check() depends on Azure Table Storage for
    state persistence. Since we don't have Azurite running, we mock only
    the StateManager while keeping ALL API calls real.
    """

    @pytest.fixture(autouse=True)
    def _check_env(self):
        _require_settings()
        _require_env(
            "BEESTAT_API_KEY", "USER_LATITUDE", "USER_LONGITUDE",
            "PURPLEAIR_API_KEY", "NTFY_TOPIC",
        )

    def test_run_check_completes(self, monkeypatch):
        """Full run_check() completes with real APIs and a mock StateManager."""
        from src import orchestrator

        # Stub StateManager to avoid Azure Table Storage dependency
        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        monkeypatch.setattr(orchestrator, "StateManager", lambda: mock_state)

        # This calls real Beestat, NWS, PurpleAir, and ntfy APIs
        orchestrator.run_check()

        # Verify state was written for at least one floor
        assert mock_state.update_floor_state.called, (
            "run_check() did not persist any floor state"
        )

        # Verify we got meaningful state updates
        calls = mock_state.update_floor_state.call_args_list
        for call in calls:
            floor_name = call[0][0]
            state_dict = call[0][1]
            assert floor_name in ("upstairs", "downstairs")
            assert state_dict["CurrentState"] in ("OPEN", "CLOSED")
            assert "DecisionReason" in state_dict
            logger.info(
                "Pipeline result — %s: %s (%s)",
                floor_name, state_dict["CurrentState"], state_dict["DecisionReason"],
            )

    def test_run_check_evaluates_both_floors(self, monkeypatch):
        """Pipeline evaluates both upstairs and downstairs."""
        from src import orchestrator

        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }
        monkeypatch.setattr(orchestrator, "StateManager", lambda: mock_state)

        orchestrator.run_check()

        floor_names = {call[0][0] for call in mock_state.update_floor_state.call_args_list}
        assert "upstairs" in floor_names, "Upstairs was not evaluated"
        assert "downstairs" in floor_names, "Downstairs was not evaluated"
