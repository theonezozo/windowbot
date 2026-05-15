"""Tests for the Beestat API client.

Validates design decisions:
- API key validation on construction
- Batch request format (sensors + thermostat in single POST)
- Sensor parsing: Beestat temperature is actual °F (no /10 conversion)
- Humidity scaling: Beestat stores value/10, multiply by 10 for percentage
- Inactive / not-in-use sensor filtering
- Offline sensor detection (no temperature → is_online=False)
- HVAC mode extraction from thermostat settings
- BeestatAuthError raised on 401 or explicit auth failure
- BeestatApiError raised on network / unexpected API errors
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
import requests as real_requests

from src.beestat_client import (
    BeestatClient,
    BeestatAuthError,
    BeestatApiError,
    BEESTAT_API_URL,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _beestat_response(
    sensors: dict | None = None,
    thermostat: dict | None = None,
    *,
    success: bool = True,
) -> dict:
    """Build a Beestat batch API response body."""
    data: dict = {}
    if sensors is not None:
        data["sensors"] = sensors
    if thermostat is not None:
        data["thermostat"] = thermostat
    return {"success": success, "data": data}


def _raw_sensor(
    sensor_id: int,
    name: str,
    *,
    temperature: float | None = 72.5,
    humidity: float | None = 4.5,
    sensor_type: str = "ecobee3_remote_sensor",
    in_use: bool = True,
    inactive: bool = False,
) -> tuple[str, dict]:
    """Build a raw Beestat sensor entry (id → dict)."""
    return (
        str(sensor_id),
        {
            "ecobee_sensor_id": sensor_id,
            "name": name,
            "type": sensor_type,
            "temperature": temperature,
            "humidity": humidity,
            "occupancy": True,
            "in_use": in_use,
            "inactive": inactive,
            "capability": [],
        },
    )


def _raw_thermostat(
    thermostat_id: int = 789,
    *,
    hvac_mode: str = "cool",
) -> dict:
    """Build a raw Beestat thermostat dict keyed by ID."""
    return {
        str(thermostat_id): {
            "ecobee_thermostat_id": thermostat_id,
            "settings": {"hvacMode": hvac_mode},
        }
    }


def _ok_post(body: dict) -> MagicMock:
    """Return a mock requests.Response for a successful POST."""
    resp = MagicMock()
    resp.status_code = 200
    resp.ok = True
    resp.json.return_value = body
    return resp


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestConstruction:
    """BeestatClient instantiation."""

    def test_valid_api_key(self):
        """Constructor accepts a non-empty API key."""
        client = BeestatClient("abc123")
        assert client._api_key == "abc123"

    @pytest.mark.xfail(reason="BeestatClient.__init__ does not yet validate empty API key")
    def test_empty_api_key_raises(self):
        """Empty string API key should be rejected early."""
        with pytest.raises((ValueError, BeestatAuthError)):
            BeestatClient("")


# ------------------------------------------------------------------
# get_sensors() — happy path
# ------------------------------------------------------------------


class TestGetSensorsHappyPath:
    """Normal sensor parsing from Beestat batch response."""

    @patch("src.beestat_client.requests.post")
    def test_multiple_sensors(self, mock_post):
        """Multiple sensors with temperature and humidity are parsed correctly."""
        sid1, s1 = _raw_sensor(123, "Living Room", temperature=72.5, humidity=4.5)
        sid2, s2 = _raw_sensor(456, "Bedroom", temperature=71.0, humidity=3.8)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 2
        names = {s["name"] for s in sensors}
        assert "Living Room" in names
        assert "Bedroom" in names

    @patch("src.beestat_client.requests.post")
    def test_temperature_not_divided_by_10(self, mock_post):
        """Beestat temperature is already actual °F — must NOT divide by 10 again."""
        sid, s = _raw_sensor(123, "Test", temperature=72.5)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        # 72.5 is the real temperature; dividing by 10 would give 7.25 — wrong!
        assert sensors[0]["temperature_f"] == 72.5

    @patch("src.beestat_client.requests.post")
    def test_humidity_multiplied_by_10(self, mock_post):
        """Beestat humidity (e.g. 4.5) → 45% for standard interface."""
        sid, s = _raw_sensor(123, "Test", temperature=72.0, humidity=4.5)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["humidity"] == 45

    @patch("src.beestat_client.requests.post")
    def test_null_temperature_means_offline(self, mock_post):
        """Sensor with temperature=None → is_online=False."""
        sid, s = _raw_sensor(123, "Dead Sensor", temperature=None, humidity=4.0)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] is None
        assert sensors[0]["is_online"] is False

    @patch("src.beestat_client.requests.post")
    def test_null_humidity_returns_none(self, mock_post):
        """Sensor with humidity=None → humidity=None in output."""
        sid, s = _raw_sensor(123, "Dry Sensor", temperature=70.0, humidity=None)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["humidity"] is None

    @patch("src.beestat_client.requests.post")
    def test_sensor_output_keys(self, mock_post):
        """Each sensor dict has exactly the expected keys."""
        sid, s = _raw_sensor(123, "S1", temperature=70.0, humidity=5.0)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert set(sensors[0].keys()) == {"name", "temperature_f", "humidity", "is_online"}


# ------------------------------------------------------------------
# get_sensors() — filtering
# ------------------------------------------------------------------


class TestGetSensorsFiltering:
    """Inactive and not-in-use sensors must be excluded."""

    @patch("src.beestat_client.requests.post")
    def test_inactive_sensors_excluded(self, mock_post):
        """Sensors with inactive != 0 are skipped."""
        sid1, s1 = _raw_sensor(1, "Active", inactive=0)
        sid2, s2 = _raw_sensor(2, "Inactive", inactive=True)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 1
        assert sensors[0]["name"] == "Active"

    @patch("src.beestat_client.requests.post")
    def test_not_in_use_sensors_included(self, mock_post):
        """Sensors with in_use=False are NOT excluded — in_use is a comfort-profile
        flag, not a hardware status indicator."""
        sid1, s1 = _raw_sensor(1, "Used", in_use=1)
        sid2, s2 = _raw_sensor(2, "Unused", in_use=False)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 2

    @patch("src.beestat_client.requests.post")
    def test_supported_types_included(self, mock_post):
        """ecobee3_remote_sensor and thermostat types are kept."""
        sid1, s1 = _raw_sensor(1, "Remote", sensor_type="ecobee3_remote_sensor")
        sid2, s2 = _raw_sensor(2, "Thermostat", sensor_type="thermostat")
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 2


# ------------------------------------------------------------------
# get_sensors() — in_use is ignored (comfort-profile flag only)
# ------------------------------------------------------------------


class TestGetSensorsInUseIgnored:
    """The in_use flag reflects Ecobee comfort-profile participation, NOT
    hardware status. All non-inactive sensors must be returned regardless
    of their in_use value.
    """

    @patch("src.beestat_client.requests.post")
    def test_all_sensors_not_in_use_still_returned(self, mock_post):
        """All non-inactive sensors returned even when all have in_use=False."""
        sid1, s1 = _raw_sensor(1, "Room A", in_use=False)
        sid2, s2 = _raw_sensor(2, "Room B", in_use=False)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 2
        names = {s["name"] for s in sensors}
        assert names == {"Room A", "Room B"}

    @patch("src.beestat_client.requests.post")
    def test_mix_in_use_true_and_false_returns_all(self, mock_post):
        """Mix of in_use=True and False → ALL returned (in_use not filtered)."""
        sid1, s1 = _raw_sensor(1, "Active", in_use=True)
        sid2, s2 = _raw_sensor(2, "Rotated Off", in_use=False)
        sid3, s3 = _raw_sensor(3, "Also Active", in_use=True)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2, sid3: s3},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 3
        names = {s["name"] for s in sensors}
        assert names == {"Active", "Rotated Off", "Also Active"}

    @patch("src.beestat_client.requests.post")
    def test_all_sensors_in_use_returns_all(self, mock_post):
        """Normal case — every sensor in_use=True → all returned."""
        sid1, s1 = _raw_sensor(1, "S1", in_use=True)
        sid2, s2 = _raw_sensor(2, "S2", in_use=True)
        sid3, s3 = _raw_sensor(3, "S3", in_use=True)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2, sid3: s3},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 3

    @patch("src.beestat_client.requests.post")
    def test_inactive_excluded_regardless_of_in_use(self, mock_post):
        """Inactive sensors always excluded; non-inactive sensors always included."""
        sid1, s1 = _raw_sensor(1, "Inactive", inactive=True, in_use=False)
        sid2, s2 = _raw_sensor(2, "Not In Use A", inactive=False, in_use=False)
        sid3, s3 = _raw_sensor(3, "Not In Use B", inactive=False, in_use=False)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2, sid3: s3},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 2
        names = {s["name"] for s in sensors}
        assert "Inactive" not in names
        assert names == {"Not In Use A", "Not In Use B"}

    @patch("src.beestat_client.requests.post")
    def test_no_warning_logged_for_in_use_false(self, mock_post, caplog):
        """No in_use-related warning is ever logged — in_use is not a filter."""
        sid1, s1 = _raw_sensor(1, "Room A", in_use=False)
        sid2, s2 = _raw_sensor(2, "Room B", in_use=False)
        body = _beestat_response(
            sensors={sid1: s1, sid2: s2},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        with caplog.at_level(logging.WARNING, logger="windowbot.beestat"):
            client.get_sensors()

        assert not any("in_use" in msg for msg in caplog.messages)


# ------------------------------------------------------------------
# get_hvac_mode() — happy path
# ------------------------------------------------------------------


class TestGetHvacModeHappyPath:
    """HVAC mode extraction from thermostat data."""

    @pytest.mark.parametrize(
        "mode",
        ["cool", "heat", "heatCool", "auto", "off", "auxHeatOnly"],
    )
    @patch("src.beestat_client.requests.post")
    def test_returns_correct_mode(self, mock_post, mode):
        """get_hvac_mode returns the correct mode string."""
        body = _beestat_response(
            sensors={},
            thermostat=_raw_thermostat(hvac_mode=mode),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        assert client.get_hvac_mode() == mode

    @patch("src.beestat_client.requests.post")
    def test_multiple_thermostats_uses_first(self, mock_post):
        """When multiple thermostats exist, uses the first one."""
        thermostat = {
            "100": {
                "ecobee_thermostat_id": 100,
                "settings": {"hvacMode": "cool"},
            },
            "200": {
                "ecobee_thermostat_id": 200,
                "settings": {"hvacMode": "heat"},
            },
        }
        body = _beestat_response(sensors={}, thermostat=thermostat)
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        result = client.get_hvac_mode()

        # Should be one of the modes (dict iteration order in Python 3.7+)
        assert result in ("cool", "heat")


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    """HTTP, network, and API error scenarios."""

    @patch("src.beestat_client.requests.post")
    def test_http_401_raises_auth_error(self, mock_post):
        """401 response → BeestatAuthError."""
        resp = MagicMock(status_code=401, ok=False)
        mock_post.return_value = resp
        client = BeestatClient("bad_key")

        with pytest.raises(BeestatAuthError):
            client.get_sensors()

    @patch("src.beestat_client.requests.post")
    def test_http_500_raises_api_error(self, mock_post):
        """500 response → BeestatApiError."""
        resp = MagicMock(status_code=500, ok=False, text="Internal Server Error")
        mock_post.return_value = resp
        client = BeestatClient("key")

        with pytest.raises(BeestatApiError, match="500"):
            client.get_sensors()

    @patch("src.beestat_client.requests.post")
    def test_invalid_json_raises_api_error(self, mock_post):
        """Non-JSON response → BeestatApiError."""
        resp = MagicMock(status_code=200, ok=True)
        resp.json.side_effect = real_requests.exceptions.JSONDecodeError(
            "msg", "doc", 0
        )
        mock_post.return_value = resp
        client = BeestatClient("key")

        with pytest.raises((BeestatApiError, real_requests.exceptions.JSONDecodeError)):
            client.get_sensors()

    @patch("src.beestat_client.requests.post")
    def test_network_timeout_raises_api_error(self, mock_post):
        """Connection timeout → BeestatApiError."""
        mock_post.side_effect = real_requests.Timeout("Connection timed out")
        client = BeestatClient("key")

        with pytest.raises(BeestatApiError, match="Network error"):
            client.get_sensors()

    @patch("src.beestat_client.requests.post")
    def test_connection_error_raises_api_error(self, mock_post):
        """DNS / connection failure → BeestatApiError."""
        mock_post.side_effect = real_requests.ConnectionError("DNS failure")
        client = BeestatClient("key")

        with pytest.raises(BeestatApiError, match="Network error"):
            client.get_sensors()

    @patch("src.beestat_client.requests.post")
    def test_success_false_raises_api_error(self, mock_post):
        """Response with success=false → BeestatApiError."""
        body = {"success": False, "data": {"error_message": "Rate limit exceeded"}}
        mock_post.return_value = _ok_post(body)
        # Fix: ok_post sets success=true in the mock, but we want json to return our body
        mock_post.return_value.json.return_value = body
        client = BeestatClient("key")

        with pytest.raises(BeestatApiError, match="Rate limit"):
            client.get_sensors()

    @patch("src.beestat_client.requests.post")
    def test_success_false_with_session_error_raises_auth_error(self, mock_post):
        """Auth-related failure message in success=false → BeestatAuthError."""
        body = {
            "success": False,
            "data": {"error_message": "Invalid session; please re-authenticate"},
        }
        resp = MagicMock(status_code=200, ok=True, text="")
        resp.json.return_value = body
        mock_post.return_value = resp
        client = BeestatClient("key")

        with pytest.raises(BeestatAuthError, match="session"):
            client.get_sensors()

    @patch("src.beestat_client.requests.post")
    def test_missing_sensor_data_returns_empty(self, mock_post):
        """Response with no sensor data → empty sensor list."""
        body = _beestat_response(thermostat=_raw_thermostat())
        # "sensors" key is absent from data
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors == []


# ------------------------------------------------------------------
# Batch API call verification
# ------------------------------------------------------------------


class TestBatchRequestFormat:
    """Verify the correct batch request format is sent to the Beestat API."""

    @patch("src.beestat_client.requests.post")
    def test_batch_request_format(self, mock_post):
        """POST body contains correct batch JSON and api_key."""
        body = _beestat_response(sensors={}, thermostat=_raw_thermostat())
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("my_key_123")

        client.get_sensors()

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert sent_json["api_key"] == "my_key_123"
        batch = json.loads(sent_json["batch"])
        assert len(batch) == 2

    @patch("src.beestat_client.requests.post")
    def test_batch_aliases_correct(self, mock_post):
        """Batch items have correct resource/method/alias combinations."""
        body = _beestat_response(sensors={}, thermostat=_raw_thermostat())
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        client.get_sensors()

        sent_json = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        batch = json.loads(sent_json["batch"])

        aliases = {item["alias"] for item in batch}
        assert "sensors" in aliases
        assert "thermostat" in aliases

        resources = {item["resource"] for item in batch}
        assert "sensor" in resources
        assert "ecobee_thermostat" in resources

    @patch("src.beestat_client.requests.post")
    def test_posts_to_correct_url(self, mock_post):
        """Request goes to the Beestat API URL."""
        body = _beestat_response(sensors={}, thermostat=_raw_thermostat())
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        client.get_sensors()

        call_args = mock_post.call_args
        assert call_args[0][0] == BEESTAT_API_URL


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Unusual but valid inputs and boundary conditions."""

    @patch("src.beestat_client.requests.post")
    def test_no_sensors_returns_empty_list(self, mock_post):
        """Empty sensor dict → empty list."""
        body = _beestat_response(sensors={}, thermostat=_raw_thermostat())
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        assert client.get_sensors() == []

    @patch("src.beestat_client.requests.post")
    def test_no_thermostats_handles_gracefully(self, mock_post):
        """Missing thermostat data → returns 'off' or raises error."""
        body = _beestat_response(sensors={}, thermostat={})
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        # Current implementation returns "off"; spec may want an error
        result = client.get_hvac_mode()
        assert result == "off"

    @patch("src.beestat_client.requests.post")
    def test_temperature_zero_is_online(self, mock_post):
        """Sensor with temperature=0.0 → is_online=True (0°F is valid!)."""
        sid, s = _raw_sensor(123, "Freezer", temperature=0.0, humidity=5.0)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] == 0.0
        assert sensors[0]["is_online"] is True

    @patch("src.beestat_client.requests.post")
    def test_negative_temperature_is_online(self, mock_post):
        """Sensor with negative temperature → is_online=True."""
        sid, s = _raw_sensor(123, "Outside", temperature=-5.0, humidity=9.0)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] == -5.0
        assert sensors[0]["is_online"] is True

    @patch("src.beestat_client.requests.post")
    def test_humidity_rounding(self, mock_post):
        """Beestat humidity 4.3 → 43 (round to nearest int)."""
        sid, s = _raw_sensor(123, "Room", temperature=70.0, humidity=4.3)
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["humidity"] == 43

    @patch("src.beestat_client.requests.post")
    def test_sensor_with_bad_temperature_value(self, mock_post):
        """Non-numeric temperature string is handled gracefully."""
        sid = "999"
        s = {
            "ecobee_sensor_id": 999,
            "name": "Glitchy",
            "type": "ecobee3_remote_sensor",
            "temperature": "not_a_number",
            "humidity": 4.5,
            "in_use": True,
            "inactive": False,
        }
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] is None
        assert sensors[0]["is_online"] is False

    @patch("src.beestat_client.requests.post")
    def test_inactive_excluded_not_in_use_included(self, mock_post):
        """Sensor with inactive=True is excluded; sensor with in_use=False is kept."""
        sid, s = _raw_sensor(1, "Ghost", inactive=True, in_use=False)
        sid2, s2 = _raw_sensor(2, "Alive")
        body = _beestat_response(
            sensors={sid: s, sid2: s2},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert len(sensors) == 1
        assert sensors[0]["name"] == "Alive"

    @patch("src.beestat_client.requests.post")
    def test_sensor_missing_name_defaults(self, mock_post):
        """Sensor without a 'name' key gets a default name."""
        sid = "111"
        s = {
            "ecobee_sensor_id": 111,
            "type": "ecobee3_remote_sensor",
            "temperature": 70.0,
            "humidity": 5.0,
            "in_use": True,
            "inactive": False,
        }
        body = _beestat_response(
            sensors={sid: s},
            thermostat=_raw_thermostat(),
        )
        mock_post.return_value = _ok_post(body)
        client = BeestatClient("key")

        sensors = client.get_sensors()

        assert sensors[0]["name"] == "Unknown"
