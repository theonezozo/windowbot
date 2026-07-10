"""Tests for the AirNow API client.

Validates design decisions:
- Worst-pollutant selection: picks the observation with the highest AQI.
- Response parsing: extracts AQI, category name, and dominant pollutant.
- Fallback role: AirNow is used when PurpleAir is unavailable.
- AirNowError raised on network failures and empty responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.airnow_client import AirNowClient, AirNowError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _airnow_obs(parameter, aqi, category="Moderate"):
    """Build a single AirNow observation dict (new API schema)."""
    return {
        "nowcastAQI": aqi,
        "parameterName": parameter,
        "AQICategoryName": category,
    }


# ------------------------------------------------------------------
# Worst-Pollutant Selection
# ------------------------------------------------------------------


class TestWorstPollutant:
    """AirNow may report multiple pollutants; we pick the worst."""

    @patch("src.airnow_client.requests.get")
    def test_picks_highest_aqi(self, mock_get):
        """Two pollutants: PM2.5=75, O3=45 → PM2.5 selected (highest)."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: [
                _airnow_obs("O3", 45, "Good"),
                _airnow_obs("PM2.5", 75, "Moderate"),
            ],
        )
        client = AirNowClient("key", 40.0, -74.0)
        result = client.get_aqi()

        assert result["aqi"] == 75
        assert result["parameter"] == "PM2.5"
        assert result["category"] == "Moderate"
        assert result["source"] == "airnow"

    @patch("src.airnow_client.requests.get")
    def test_single_pollutant(self, mock_get):
        """Single pollutant → that one is returned."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: [_airnow_obs("PM2.5", 42, "Good")],
        )
        client = AirNowClient("key", 40.0, -74.0)
        result = client.get_aqi()

        assert result["aqi"] == 42
        assert result["parameter"] == "PM2.5"

    @patch("src.airnow_client.requests.get")
    def test_three_pollutants_worst_wins(self, mock_get):
        """Three pollutants: PM2.5=60, O3=120, PM10=30 → O3 selected."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: [
                _airnow_obs("PM2.5", 60),
                _airnow_obs("O3", 120, "USG"),
                _airnow_obs("PM10", 30, "Good"),
            ],
        )
        client = AirNowClient("key", 40.0, -74.0)
        result = client.get_aqi()

        assert result["aqi"] == 120
        assert result["parameter"] == "O3"


# ------------------------------------------------------------------
# Response Parsing
# ------------------------------------------------------------------


class TestResponseParsing:
    """Correct extraction of fields from AirNow JSON."""

    @patch("src.airnow_client.requests.get")
    def test_category_name_extracted(self, mock_get):
        """Category.Name is returned as 'category' field."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: [_airnow_obs("PM2.5", 155, "Unhealthy")],
        )
        client = AirNowClient("key", 40.0, -74.0)
        result = client.get_aqi()

        assert result["category"] == "Unhealthy"

    @patch("src.airnow_client.requests.get")
    def test_missing_category_defaults_unknown(self, mock_get):
        """Missing AQICategoryName → 'Unknown'."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: [{"nowcastAQI": 50, "parameterName": "PM2.5"}],
        )
        client = AirNowClient("key", 40.0, -74.0)
        result = client.get_aqi()

        assert result["category"] == "Unknown"

    @patch("src.airnow_client.requests.get")
    def test_missing_parameter_defaults_unknown(self, mock_get):
        """Missing parameterName → 'Unknown'."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: [{"nowcastAQI": 50, "AQICategoryName": "Good"}],
        )
        client = AirNowClient("key", 40.0, -74.0)
        result = client.get_aqi()

        assert result["parameter"] == "Unknown"

    @patch("src.airnow_client.requests.get")
    def test_api_params_include_key_and_location(self, mock_get):
        """Request params include api_key, lat, lon; distance removed."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: [_airnow_obs("PM2.5", 50)],
        )
        client = AirNowClient("my_key", 40.123, -74.456)
        client.get_aqi()

        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params", {})
        assert params["api_key"] == "my_key"
        assert params["latitude"] == "40.123"
        assert params["longitude"] == "-74.456"
        assert "distance" not in params

    @patch("src.airnow_client.requests.get")
    def test_observation_time_from_timestamp_fields(self, mock_get):
        """observation_time is populated when timestamp fields present, else None."""
        obs_with_time = _airnow_obs("PM2.5", 50)
        obs_with_time.update(
            {
                "dateObserved": "2026-02-01",
                "hourObserved": "05:00",
                "localTimeZone": "PDT",
            }
        )
        mock_get.return_value = MagicMock(ok=True, json=lambda: [obs_with_time])
        client = AirNowClient("key", 40.0, -74.0)
        result = client.get_aqi()

        assert result["observation_time"] is not None
        obs_time = result["observation_time"]
        assert "2026-02-01" in obs_time
        assert "05:00" in obs_time
        assert "PDT" in obs_time

        # Absent timestamp fields → None.
        mock_get.return_value = MagicMock(
            ok=True, json=lambda: [_airnow_obs("PM2.5", 50)]
        )
        result_no_time = client.get_aqi()
        assert result_no_time["observation_time"] is None


# ------------------------------------------------------------------
# Error Handling
# ------------------------------------------------------------------


class TestAirNowErrors:
    """Network failures and empty responses raise AirNowError."""

    @patch("src.airnow_client.requests.get")
    def test_empty_response_raises(self, mock_get):
        """Empty observation list → AirNowError."""
        mock_get.return_value = MagicMock(ok=True, json=lambda: [])
        client = AirNowClient("key", 40.0, -74.0)

        with pytest.raises(AirNowError, match="no observations"):
            client.get_aqi()

    @patch("src.airnow_client.requests.get")
    def test_http_error_raises(self, mock_get):
        """Non-200 response → AirNowError."""
        mock_get.return_value = MagicMock(
            ok=False, status_code=500, text="Server Error"
        )
        client = AirNowClient("key", 40.0, -74.0)

        with pytest.raises(AirNowError, match="500"):
            client.get_aqi()

    @patch("src.airnow_client.requests.get")
    def test_network_error_raises(self, mock_get):
        """Network exception → AirNowError."""
        import requests as real_requests

        mock_get.side_effect = real_requests.ConnectionError("timeout")
        client = AirNowClient("key", 40.0, -74.0)

        with pytest.raises(AirNowError, match="Network error"):
            client.get_aqi()
