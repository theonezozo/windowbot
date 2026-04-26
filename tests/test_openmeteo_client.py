"""Tests for the Open-Meteo free weather peer/fallback client.

Validates design decisions:
- No API key required — zero-auth grid-based weather data.
- Single API call (no station discovery step).
- get_outdoor_conditions() returns NWS-compatible aggregated dict.
- get_observation() returns NWS-compatible single-station dict with timestamp.
- Freshness check: get_observation() raises OpenMeteoError when obs > 30 min old.
- Timestamp parsed from current["time"] + utc_offset_seconds → UTC datetime.
- Always: source="openmeteo", is_fallback=True, station_count=1, used_cache=False.
- Null humidity / wind → None in result (not error).
- HTTP errors → OpenMeteoError.
- Network errors → OpenMeteoError.
- Missing "current" block → OpenMeteoError.
- Missing temperature → OpenMeteoError.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.openmeteo_client import OpenMeteoClient, OpenMeteoError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_LAT = 37.40
_LON = -122.08


def _api_response(
    temp_f: float = 65.3,
    humidity: float | None = 58,
    wind_mph: float | None = 7.2,
    utc_offset_seconds: int = 0,
    time_offset_minutes: int = -5,  # default: timestamp 5 minutes ago (fresh)
):
    """Build a mock Open-Meteo current-weather JSON response.

    Args:
        utc_offset_seconds: UTC offset in seconds (0 = UTC).
        time_offset_minutes: Offset from now in minutes (negative = in the past).
    """
    # Build a timestamp that, after UTC conversion, is `time_offset_minutes` old.
    # local_time = UTC + utc_offset_seconds
    # so: local_naive = (now + offset_delta) + time_offset
    now_utc = datetime.now(timezone.utc)
    local_naive = (
        now_utc
        + timedelta(seconds=utc_offset_seconds)
        + timedelta(minutes=time_offset_minutes)
    ).replace(tzinfo=None)
    ts_str = local_naive.strftime("%Y-%m-%dT%H:%M")

    current = {
        "time": ts_str,
        "interval": 900,
        "temperature_2m": temp_f,
    }
    if humidity is not None:
        current["relative_humidity_2m"] = humidity
    if wind_mph is not None:
        current["wind_speed_10m"] = wind_mph
    return {"current": current, "utc_offset_seconds": utc_offset_seconds}


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestInitialization:
    """OpenMeteoClient stores lat/lon on construction."""

    def test_stores_latitude(self):
        client = OpenMeteoClient(_LAT, _LON)
        assert client._lat == _LAT

    def test_stores_longitude(self):
        client = OpenMeteoClient(_LAT, _LON)
        assert client._lon == _LON


# ------------------------------------------------------------------
# Successful Fetch
# ------------------------------------------------------------------


class TestSuccessfulFetch:
    """Valid API response returns correct dict with all fields."""

    @patch("src.openmeteo_client.requests.get")
    def test_returns_complete_dict(self, mock_get):
        """Full response → all fields present and correct."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(temp_f=65.3, humidity=58, wind_mph=7.2),
        )
        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_outdoor_conditions()

        assert result["temperature_f"] == pytest.approx(65.3, abs=0.1)
        assert result["humidity"] == pytest.approx(58.0, abs=0.1)
        assert result["wind_speed_mph"] == pytest.approx(7.2, abs=0.1)
        assert result["station_count"] == 1
        assert result["is_fallback"] is True
        assert result["used_cache"] is False
        assert result["source"] == "openmeteo"

    @patch("src.openmeteo_client.requests.get")
    def test_passes_correct_params(self, mock_get):
        """API call includes lat, lon, Fahrenheit, mph units."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(),
        )
        client = OpenMeteoClient(_LAT, _LON)
        client.get_outdoor_conditions()

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        params = kwargs["params"]
        assert params["latitude"] == _LAT
        assert params["longitude"] == _LON
        assert params["temperature_unit"] == "fahrenheit"
        assert params["wind_speed_unit"] == "mph"
        assert "temperature_2m" in params["current"]

    @patch("src.openmeteo_client.requests.get")
    def test_rounds_to_one_decimal(self, mock_get):
        """Values with many decimal places are rounded to 1 decimal."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(temp_f=65.3456, humidity=58.789, wind_mph=7.234),
        )
        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_outdoor_conditions()

        assert result["temperature_f"] == 65.3
        assert result["humidity"] == 58.8
        assert result["wind_speed_mph"] == 7.2


# ------------------------------------------------------------------
# Missing / Null Fields
# ------------------------------------------------------------------


class TestMissingFields:
    """Null humidity or wind speed → None in result (not error)."""

    @patch("src.openmeteo_client.requests.get")
    def test_null_humidity(self, mock_get):
        """Humidity missing from response → humidity is None."""
        resp = _api_response(temp_f=70.0, humidity=None, wind_mph=5.0)
        # Explicitly remove the key to simulate missing field
        resp["current"].pop("relative_humidity_2m", None)
        mock_get.return_value = MagicMock(ok=True, json=lambda: resp)

        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_outdoor_conditions()

        assert result["humidity"] is None
        assert result["temperature_f"] == pytest.approx(70.0, abs=0.1)
        assert result["wind_speed_mph"] == pytest.approx(5.0, abs=0.1)

    @patch("src.openmeteo_client.requests.get")
    def test_null_wind_speed(self, mock_get):
        """Wind speed missing from response → wind_speed_mph is None."""
        resp = _api_response(temp_f=70.0, humidity=50.0, wind_mph=None)
        resp["current"].pop("wind_speed_10m", None)
        mock_get.return_value = MagicMock(ok=True, json=lambda: resp)

        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_outdoor_conditions()

        assert result["wind_speed_mph"] is None
        assert result["humidity"] == pytest.approx(50.0, abs=0.1)

    @patch("src.openmeteo_client.requests.get")
    def test_both_optional_fields_null(self, mock_get):
        """Both humidity and wind missing → both None, still succeeds."""
        resp = _api_response(temp_f=72.0, humidity=None, wind_mph=None)
        resp["current"].pop("relative_humidity_2m", None)
        resp["current"].pop("wind_speed_10m", None)
        mock_get.return_value = MagicMock(ok=True, json=lambda: resp)

        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_outdoor_conditions()

        assert result["humidity"] is None
        assert result["wind_speed_mph"] is None
        assert result["temperature_f"] == pytest.approx(72.0, abs=0.1)


# ------------------------------------------------------------------
# API Error
# ------------------------------------------------------------------


class TestAPIError:
    """HTTP error responses raise OpenMeteoError."""

    @patch("src.openmeteo_client.requests.get")
    def test_http_400_raises(self, mock_get):
        """400 Bad Request → OpenMeteoError."""
        mock_get.return_value = MagicMock(
            ok=False, status_code=400, text="Bad Request"
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="API error.*400"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_http_500_raises(self, mock_get):
        """500 Internal Server Error → OpenMeteoError."""
        mock_get.return_value = MagicMock(
            ok=False, status_code=500, text="Internal Server Error"
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="API error.*500"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_http_429_raises(self, mock_get):
        """429 Rate Limit → OpenMeteoError."""
        mock_get.return_value = MagicMock(
            ok=False, status_code=429, text="Too Many Requests"
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="API error.*429"):
            client.get_outdoor_conditions()


# ------------------------------------------------------------------
# Network Error
# ------------------------------------------------------------------


class TestNetworkError:
    """requests.RequestException raises OpenMeteoError."""

    @patch("src.openmeteo_client.requests.get")
    def test_connection_error_raises(self, mock_get):
        """ConnectionError → OpenMeteoError."""
        import requests as req
        mock_get.side_effect = req.ConnectionError("DNS resolution failed")
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="Network error"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_timeout_raises(self, mock_get):
        """Timeout → OpenMeteoError."""
        import requests as req
        mock_get.side_effect = req.Timeout("Request timed out")
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="Network error"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_generic_request_exception_raises(self, mock_get):
        """Generic RequestException → OpenMeteoError."""
        import requests as req
        mock_get.side_effect = req.RequestException("Something went wrong")
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="Network error"):
            client.get_outdoor_conditions()


# ------------------------------------------------------------------
# Response Format Invariants
# ------------------------------------------------------------------


class TestResponseFormat:
    """Constant fields are always present with fixed values."""

    @patch("src.openmeteo_client.requests.get")
    def test_source_always_openmeteo(self, mock_get):
        mock_get.return_value = MagicMock(ok=True, json=lambda: _api_response())
        result = OpenMeteoClient(_LAT, _LON).get_outdoor_conditions()
        assert result["source"] == "openmeteo"

    @patch("src.openmeteo_client.requests.get")
    def test_is_fallback_always_true(self, mock_get):
        mock_get.return_value = MagicMock(ok=True, json=lambda: _api_response())
        result = OpenMeteoClient(_LAT, _LON).get_outdoor_conditions()
        assert result["is_fallback"] is True

    @patch("src.openmeteo_client.requests.get")
    def test_station_count_always_one(self, mock_get):
        mock_get.return_value = MagicMock(ok=True, json=lambda: _api_response())
        result = OpenMeteoClient(_LAT, _LON).get_outdoor_conditions()
        assert result["station_count"] == 1

    @patch("src.openmeteo_client.requests.get")
    def test_used_cache_always_false(self, mock_get):
        mock_get.return_value = MagicMock(ok=True, json=lambda: _api_response())
        result = OpenMeteoClient(_LAT, _LON).get_outdoor_conditions()
        assert result["used_cache"] is False


# ------------------------------------------------------------------
# Empty / Malformed Response
# ------------------------------------------------------------------


class TestMalformedResponse:
    """Malformed or missing data raises OpenMeteoError."""

    @patch("src.openmeteo_client.requests.get")
    def test_missing_current_key_raises(self, mock_get):
        """Response without 'current' block → OpenMeteoError."""
        mock_get.return_value = MagicMock(
            ok=True, json=lambda: {"hourly": {"time": []}}
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="current"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_empty_current_block_raises(self, mock_get):
        """Empty 'current' dict → OpenMeteoError (falsy block treated as missing)."""
        mock_get.return_value = MagicMock(
            ok=True, json=lambda: {"current": {}}
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_missing_temperature_raises(self, mock_get):
        """Current block present but no temperature_2m → OpenMeteoError."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {"current": {"relative_humidity_2m": 50, "wind_speed_10m": 5}},
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="temperature"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_invalid_json_raises(self, mock_get):
        """Response that isn't valid JSON → OpenMeteoError."""
        resp = MagicMock(ok=True)
        resp.json.side_effect = ValueError("No JSON object could be decoded")
        mock_get.return_value = resp

        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="Invalid JSON"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_null_current_block_raises(self, mock_get):
        """current: null → OpenMeteoError."""
        mock_get.return_value = MagicMock(
            ok=True, json=lambda: {"current": None}
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="current"):
            client.get_outdoor_conditions()

    @patch("src.openmeteo_client.requests.get")
    def test_temperature_null_raises(self, mock_get):
        """temperature_2m explicitly null → OpenMeteoError."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {
                "current": {
                    "temperature_2m": None,
                    "relative_humidity_2m": 50,
                    "wind_speed_10m": 5,
                }
            },
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="temperature"):
            client.get_outdoor_conditions()


# ------------------------------------------------------------------
# get_observation() — NWS-compatible single-station dict
# ------------------------------------------------------------------


class TestGetObservation:
    """get_observation() returns an NWS-compatible dict and enforces freshness."""

    @patch("src.openmeteo_client.requests.get")
    def test_returns_nws_compatible_dict(self, mock_get):
        """Fresh observation → dict with station_id, temperature_f, humidity,
        wind_speed_mph, and timestamp."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(temp_f=65.3, humidity=58, wind_mph=7.2),
        )
        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_observation()

        assert result["station_id"] == "OPENMETEO"
        assert result["temperature_f"] == pytest.approx(65.3, abs=0.1)
        assert result["humidity"] == pytest.approx(58.0, abs=0.1)
        assert result["wind_speed_mph"] == pytest.approx(7.2, abs=0.1)
        assert isinstance(result["timestamp"], datetime)
        assert result["timestamp"].tzinfo == timezone.utc

    @patch("src.openmeteo_client.requests.get")
    def test_fresh_observation_does_not_raise(self, mock_get):
        """Timestamp 5 minutes old → no error."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(time_offset_minutes=-5),
        )
        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_observation()  # should not raise
        assert result["station_id"] == "OPENMETEO"

    @patch("src.openmeteo_client.requests.get")
    def test_stale_observation_raises(self, mock_get):
        """Timestamp 45 minutes old → OpenMeteoError (stale)."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(time_offset_minutes=-45),
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="stale"):
            client.get_observation()

    @patch("src.openmeteo_client.requests.get")
    def test_exactly_30_minute_boundary(self, mock_get):
        """Timestamp exactly 30 minutes old → raises (age > 30m required to be fresh)."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(time_offset_minutes=-30),
        )
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="stale"):
            client.get_observation()

    @patch("src.openmeteo_client.requests.get")
    def test_timestamp_utc_conversion_positive_offset(self, mock_get):
        """UTC+5 offset: local time converted to correct UTC datetime."""
        utc_offset = 5 * 3600  # UTC+5
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(
                utc_offset_seconds=utc_offset, time_offset_minutes=-10
            ),
        )
        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_observation()
        ts = result["timestamp"]
        # Timestamp should be within 2 minutes of 10 minutes ago (UTC)
        expected = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert abs((ts - expected).total_seconds()) < 120

    @patch("src.openmeteo_client.requests.get")
    def test_timestamp_utc_conversion_negative_offset(self, mock_get):
        """UTC-8 offset: local time converted to correct UTC datetime."""
        utc_offset = -8 * 3600  # UTC-8
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _api_response(
                utc_offset_seconds=utc_offset, time_offset_minutes=-10
            ),
        )
        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_observation()
        ts = result["timestamp"]
        expected = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert abs((ts - expected).total_seconds()) < 120

    @patch("src.openmeteo_client.requests.get")
    def test_null_humidity_allowed(self, mock_get):
        """Missing humidity → humidity is None, no error."""
        resp = _api_response(humidity=None)
        resp["current"].pop("relative_humidity_2m", None)
        mock_get.return_value = MagicMock(ok=True, json=lambda: resp)
        client = OpenMeteoClient(_LAT, _LON)
        result = client.get_observation()
        assert result["humidity"] is None

    @patch("src.openmeteo_client.requests.get")
    def test_network_error_raises(self, mock_get):
        """Network error → OpenMeteoError."""
        import requests as req
        mock_get.side_effect = req.ConnectionError("DNS failed")
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="Network error"):
            client.get_observation()

    @patch("src.openmeteo_client.requests.get")
    def test_http_error_raises(self, mock_get):
        """HTTP 500 → OpenMeteoError."""
        mock_get.return_value = MagicMock(ok=False, status_code=500, text="error")
        client = OpenMeteoClient(_LAT, _LON)
        with pytest.raises(OpenMeteoError, match="API error"):
            client.get_observation()

