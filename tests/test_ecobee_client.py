"""Tests for the Ecobee API client.

Validates design decisions:
- OAuth2 token refresh flow (persists new tokens via state_manager)
- Automatic retry on 401 / code-14 responses
- Sensor parsing: Ecobee F×10 integer → float °F conversion
- Offline sensor detection (no valid temperature → is_online=False)
- HVAC mode extraction from thermostat settings
- EcobeeAuthError raised on irrecoverable 401
- EcobeeApiError raised on network / unexpected API errors
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest

from src.ecobee_client import (
    EcobeeClient,
    EcobeeAuthError,
    EcobeeApiError,
    TOKEN_URL,
    THERMOSTAT_URL,
    _TEMP_SCALE,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_state_manager(access="", refresh=""):
    """Return a mock StateManager that stores/retrieves OAuth tokens."""
    sm = MagicMock()
    sm.get_oauth_tokens.return_value = {
        "AccessToken": access,
        "RefreshToken": refresh,
    }
    return sm


def _thermostat_body(sensors=None, hvac_mode="cool"):
    """Build a minimal Ecobee thermostat response body."""
    if sensors is None:
        sensors = []
    return {
        "status": {"code": 0},
        "thermostatList": [
            {
                "remoteSensors": sensors,
                "settings": {"hvacMode": hvac_mode},
            }
        ],
    }


def _raw_sensor(name, temp_f10=None, humidity=None):
    """Build a raw Ecobee remoteSensor dict with F×10 temperature."""
    caps = []
    if temp_f10 is not None:
        caps.append({"type": "temperature", "value": str(temp_f10)})
    if humidity is not None:
        caps.append({"type": "humidity", "value": str(humidity)})
    return {"name": name, "capability": caps}


# ------------------------------------------------------------------
# OAuth / Token Refresh
# ------------------------------------------------------------------


class TestTokenRefresh:
    """OAuth2 refresh flow and token persistence."""

    @patch("src.ecobee_client.requests.post")
    def test_refresh_stores_new_tokens(self, mock_post):
        """Successful refresh persists both access and refresh tokens."""
        mock_post.return_value = MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
            },
        )
        sm = _make_state_manager(refresh="old_refresh")
        client = EcobeeClient("cid", "init_refresh", sm)

        token = client._refresh_access_token()

        assert token == "new_access"
        sm.update_oauth_tokens.assert_called_once()
        stored = sm.update_oauth_tokens.call_args[0][0]
        assert stored["AccessToken"] == "new_access"
        assert stored["RefreshToken"] == "new_refresh"

    @patch("src.ecobee_client.requests.post")
    def test_refresh_401_raises_auth_error(self, mock_post):
        """401 from token endpoint → EcobeeAuthError."""
        mock_post.return_value = MagicMock(status_code=401, ok=False)
        sm = _make_state_manager(refresh="bad_token")
        client = EcobeeClient("cid", "bad_token", sm)

        with pytest.raises(EcobeeAuthError, match="revoked"):
            client._refresh_access_token()

    @patch("src.ecobee_client.requests.post")
    def test_refresh_500_raises_api_error(self, mock_post):
        """Non-401 failure from token endpoint → EcobeeApiError."""
        mock_post.return_value = MagicMock(
            status_code=500, ok=False, text="Internal Server Error"
        )
        sm = _make_state_manager(refresh="tok")
        client = EcobeeClient("cid", "tok", sm)

        with pytest.raises(EcobeeApiError, match="500"):
            client._refresh_access_token()

    @patch("src.ecobee_client.requests.post")
    def test_refresh_network_error_raises_api_error(self, mock_post):
        """Network exception during refresh → EcobeeApiError."""
        import requests as real_requests

        mock_post.side_effect = real_requests.ConnectionError("DNS failure")
        sm = _make_state_manager(refresh="tok")
        client = EcobeeClient("cid", "tok", sm)

        with pytest.raises(EcobeeApiError, match="Network error"):
            client._refresh_access_token()


# ------------------------------------------------------------------
# Authed GET with retry
# ------------------------------------------------------------------


class TestAuthedGet:
    """_authed_get retries on 401 and code-14, and raises on persistent failures."""

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_authed_get_refreshes_when_no_access_token(self, mock_post, mock_get):
        """First call with empty access_token triggers refresh before GET."""
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "fresh", "refresh_token": "r2"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"status": {"code": 0}, "data": "ok"},
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "init_r", sm)

        result = client._authed_get("https://example.com")
        assert result["data"] == "ok"
        mock_post.assert_called_once()

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_authed_get_retries_on_401(self, mock_post, mock_get):
        """First GET returns 401 → refresh → retry GET succeeds."""
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a2", "refresh_token": "r2"},
        )
        resp_401 = MagicMock(status_code=401, ok=False)
        resp_ok = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"status": {"code": 0}, "result": True},
        )
        mock_get.side_effect = [resp_401, resp_ok]
        sm = _make_state_manager(access="stale")
        client = EcobeeClient("cid", "r", sm)

        result = client._authed_get("https://example.com")
        assert result["result"] is True
        assert mock_get.call_count == 2

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_authed_get_code14_triggers_retry(self, mock_post, mock_get):
        """Ecobee code-14 (expired token) → refresh → retry."""
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a2", "refresh_token": "r2"},
        )
        resp_code14 = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"status": {"code": 14}},
        )
        resp_ok = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"status": {"code": 0}, "ok": 1},
        )
        mock_get.side_effect = [resp_code14, resp_ok]
        sm = _make_state_manager(access="valid")
        client = EcobeeClient("cid", "r", sm)

        result = client._authed_get("https://example.com")
        assert result["ok"] == 1

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_authed_get_persistent_401_raises_auth_error(self, mock_post, mock_get):
        """Two consecutive 401s → EcobeeAuthError."""
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a2", "refresh_token": "r2"},
        )
        mock_get.return_value = MagicMock(status_code=401, ok=False)
        sm = _make_state_manager(access="a")
        client = EcobeeClient("cid", "r", sm)

        with pytest.raises(EcobeeAuthError, match="Failed to authenticate"):
            client._authed_get("https://example.com")


# ------------------------------------------------------------------
# Sensor Parsing: F×10 → float
# ------------------------------------------------------------------


class TestSensorParsing:
    """Ecobee temperature conversion (F×10 integer → °F float)."""

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_temp_f10_to_float(self, mock_post, mock_get):
        """740 → 74.0°F, 685 → 68.5°F."""
        body = _thermostat_body(
            sensors=[
                _raw_sensor("Living Room", temp_f10=740, humidity=45),
                _raw_sensor("Bedroom", temp_f10=685),
            ]
        )
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)
        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] == 74.0
        assert sensors[0]["humidity"] == 45
        assert sensors[0]["is_online"] is True
        assert sensors[1]["temperature_f"] == 68.5
        assert sensors[1]["humidity"] is None

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_direct_ecobee_source_tag_and_unknown_age(self, mock_post, mock_get):
        """Direct Ecobee readings carry ``source="ecobee:direct"`` and
        ``data_age_seconds=None`` (Ecobee's API returns the live cloud-side
        value with no upstream sync timestamp). Keeps the Ecobee/Beestat
        client dicts symmetric so the orchestrator and status page don't
        need provider branching. ``None`` (unknown) is the correct semantic
        for the age — NOT zero.
        """
        body = _thermostat_body(
            sensors=[
                _raw_sensor("Living Room", temp_f10=740, humidity=45),
                _raw_sensor("Bedroom", temp_f10=685),
            ]
        )
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)

        sensors = client.get_sensors()

        assert len(sensors) == 2
        for s in sensors:
            assert s["source"] == "ecobee:direct"
            assert s["data_age_seconds"] is None

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_sensor_offline_no_temp(self, mock_post, mock_get):
        """Sensor with no temperature capability → is_online=False."""
        body = _thermostat_body(sensors=[_raw_sensor("Garage")])
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)
        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] is None
        assert sensors[0]["is_online"] is False

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_sensor_empty_temp_value_offline(self, mock_post, mock_get):
        """Sensor with empty string temperature → is_online=False."""
        body = _thermostat_body(
            sensors=[{"name": "Bad", "capability": [{"type": "temperature", "value": ""}]}]
        )
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)
        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] is None
        assert sensors[0]["is_online"] is False

    @pytest.mark.parametrize(
        "f10_value, expected_f",
        [
            (700, 70.0),
            (735, 73.5),
            (1000, 100.0),
            (0, 0.0),
            (-50, -5.0),
        ],
    )
    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_f10_conversion_parametrized(self, mock_post, mock_get, f10_value, expected_f):
        """Parametrized F×10 → °F conversion."""
        body = _thermostat_body(sensors=[_raw_sensor("S", temp_f10=f10_value)])
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)
        sensors = client.get_sensors()

        assert sensors[0]["temperature_f"] == expected_f


# ------------------------------------------------------------------
# HVAC Mode Extraction
# ------------------------------------------------------------------


class TestHVACMode:
    """HVAC mode is read from thermostat settings."""

    @pytest.mark.parametrize("mode", ["cool", "heat", "heatCool", "auto", "off", "auxHeatOnly"])
    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_hvac_mode_returned(self, mock_post, mock_get, mode):
        """get_hvac_mode returns the settings.hvacMode value."""
        body = _thermostat_body(hvac_mode=mode)
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)

        assert client.get_hvac_mode() == mode

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_hvac_mode_default_off(self, mock_post, mock_get):
        """Missing hvacMode defaults to 'off'."""
        body = {"status": {"code": 0}, "thermostatList": [{"settings": {}, "remoteSensors": []}]}
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)

        assert client.get_hvac_mode() == "off"


# ------------------------------------------------------------------
# No Thermostats
# ------------------------------------------------------------------


class TestNoThermostats:
    """Edge case: account has no thermostats."""

    @patch("src.ecobee_client.requests.get")
    @patch("src.ecobee_client.requests.post")
    def test_no_thermostats_raises(self, mock_post, mock_get):
        """Empty thermostatList → EcobeeApiError."""
        body = {"status": {"code": 0}, "thermostatList": []}
        mock_post.return_value = MagicMock(
            status_code=200, ok=True,
            json=lambda: {"access_token": "a", "refresh_token": "r"},
        )
        mock_get.return_value = MagicMock(
            status_code=200, ok=True, json=lambda: body,
        )
        sm = _make_state_manager()
        client = EcobeeClient("cid", "r", sm)

        with pytest.raises(EcobeeApiError, match="No thermostats"):
            client.get_thermostat_data()
