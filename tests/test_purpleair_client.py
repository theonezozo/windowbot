"""Tests for the PurpleAir API client.

Validates design decisions:
- Haversine great-circle distance calculation.
- PM2.5 → AQI conversion using the EPA breakpoint table.
- Median of 3 closest sensors for robustness.
- Bounding box parameter generation from radius.
- Negative PM2.5 readings discarded.
- Stale sensor readings (>30 min) discarded.
- PurpleAirError when no sensors available.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.purpleair_client import PurpleAirClient, PurpleAirError


# ------------------------------------------------------------------
# Haversine Distance
# ------------------------------------------------------------------


class TestHaversine:
    """Great-circle distance calculation."""

    def test_same_point_zero_distance(self):
        """Same coordinates → 0 km."""
        assert PurpleAirClient._haversine_km(40.0, -74.0, 40.0, -74.0) == 0.0

    def test_known_distance_nyc_to_la(self):
        """NYC (40.7128, -74.0060) → LA (34.0522, -118.2437) ≈ 3944 km."""
        d = PurpleAirClient._haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < d < 4000

    def test_symmetric(self):
        """distance(A, B) == distance(B, A)."""
        d1 = PurpleAirClient._haversine_km(40.0, -74.0, 41.0, -75.0)
        d2 = PurpleAirClient._haversine_km(41.0, -75.0, 40.0, -74.0)
        assert d1 == pytest.approx(d2)

    def test_short_distance(self):
        """~1 degree of latitude ≈ 111 km."""
        d = PurpleAirClient._haversine_km(40.0, -74.0, 41.0, -74.0)
        assert 110 < d < 112


# ------------------------------------------------------------------
# PM2.5 → AQI Conversion (EPA breakpoints)
# ------------------------------------------------------------------


class TestPM25ToAQI:
    """EPA breakpoint table conversion."""

    @pytest.mark.parametrize(
        "pm25, expected_aqi",
        [
            (0.0, 0),
            (6.0, 25),       # midpoint of 0-12.0 → 0-50
            (12.0, 50),       # top of "Good" range
            (12.1, 51),       # start of "Moderate"
            (35.4, 100),      # top of "Moderate"
            (35.5, 101),      # start of "USG"
            (55.4, 150),      # top of "USG"
            (55.5, 151),      # start of "Unhealthy"
            (150.4, 200),     # top of "Unhealthy"
            (150.5, 201),     # start of "Very Unhealthy"
            (250.4, 300),     # top of "Very Unhealthy"
            (250.5, 301),     # start of "Hazardous"
            (350.4, 400),     # top of "Hazardous 1"
            (350.5, 401),     # start of "Hazardous 2"
            (500.4, 500),     # top of table
        ],
    )
    def test_breakpoint_boundaries(self, pm25, expected_aqi):
        """Breakpoint boundary values produce exact AQI."""
        assert PurpleAirClient.pm25_to_aqi(pm25) == expected_aqi

    def test_negative_pm25_returns_zero(self):
        """Negative concentration → AQI 0."""
        assert PurpleAirClient.pm25_to_aqi(-5.0) == 0

    def test_above_max_capped_at_500(self):
        """PM2.5 > 500.4 → AQI capped at 500."""
        assert PurpleAirClient.pm25_to_aqi(600.0) == 500
        assert PurpleAirClient.pm25_to_aqi(999.9) == 500

    def test_pm25_truncation(self):
        """EPA methodology truncates PM2.5 to 1 decimal place."""
        # 12.04 truncates to 12.0 → AQI 50
        assert PurpleAirClient.pm25_to_aqi(12.04) == 50
        # 12.09 truncates to 12.0 → AQI 50
        assert PurpleAirClient.pm25_to_aqi(12.09) == 50
        # 12.14 truncates to 12.1 → AQI 51
        assert PurpleAirClient.pm25_to_aqi(12.14) == 51


# ------------------------------------------------------------------
# Median of 3 Sensors
# ------------------------------------------------------------------


class TestMedianOf3:
    """AQI is computed from the median PM2.5 of the 3 closest sensors."""

    @patch.object(PurpleAirClient, "find_nearby_sensors")
    def test_median_of_three(self, mock_find):
        """Three sensors → median PM2.5 used for AQI."""
        mock_find.return_value = [
            {"sensor_index": 1, "pm25": 10.0, "distance_km": 0.5},
            {"sensor_index": 2, "pm25": 20.0, "distance_km": 1.0},
            {"sensor_index": 3, "pm25": 30.0, "distance_km": 1.5},
        ]
        client = PurpleAirClient(40.0, -74.0)
        result = client.get_aqi()

        assert result["pm25"] == 20.0  # median of [10, 20, 30]
        assert result["source"] == "purpleair"
        assert result["sensor_count"] == 3

    @patch.object(PurpleAirClient, "find_nearby_sensors")
    def test_uses_only_closest_three(self, mock_find):
        """Five sensors available → only closest 3 used."""
        mock_find.return_value = [
            {"sensor_index": 1, "pm25": 5.0, "distance_km": 0.5},
            {"sensor_index": 2, "pm25": 10.0, "distance_km": 1.0},
            {"sensor_index": 3, "pm25": 15.0, "distance_km": 1.5},
            {"sensor_index": 4, "pm25": 100.0, "distance_km": 2.0},
            {"sensor_index": 5, "pm25": 200.0, "distance_km": 3.0},
        ]
        client = PurpleAirClient(40.0, -74.0)
        result = client.get_aqi()

        assert result["pm25"] == 10.0  # median of [5, 10, 15], not influenced by 100/200
        assert result["sensor_count"] == 3

    @patch.object(PurpleAirClient, "find_nearby_sensors")
    def test_single_sensor(self, mock_find):
        """One sensor → that value is the "median"."""
        mock_find.return_value = [
            {"sensor_index": 1, "pm25": 8.0, "distance_km": 0.5},
        ]
        client = PurpleAirClient(40.0, -74.0)
        result = client.get_aqi()

        assert result["pm25"] == 8.0
        assert result["sensor_count"] == 1

    @patch.object(PurpleAirClient, "find_nearby_sensors")
    def test_no_sensors_raises(self, mock_find):
        """No sensors → PurpleAirError."""
        mock_find.return_value = []
        client = PurpleAirClient(40.0, -74.0)

        with pytest.raises(PurpleAirError, match="No PurpleAir sensors"):
            client.get_aqi()


# ------------------------------------------------------------------
# Bounding Box Parameter Generation
# ------------------------------------------------------------------


class TestBoundingBox:
    """Bounding box built from radius around user location."""

    @patch("src.purpleair_client.requests.get")
    def test_bounding_box_params(self, mock_get):
        """Bounding box parameters are correctly generated from radius."""
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {"fields": [], "data": []},
        )
        client = PurpleAirClient(40.0, -74.0, api_key="test_key")
        client.find_nearby_sensors(radius_km=5.0)

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})

        # Verify bounding box exists and is numerically reasonable
        nwlat = float(params["nwlat"])
        selat = float(params["selat"])
        nwlng = float(params["nwlng"])
        selng = float(params["selng"])

        assert nwlat > 40.0  # north of user
        assert selat < 40.0  # south of user
        assert nwlng < -74.0  # west of user
        assert selng > -74.0  # east of user

    @patch("src.purpleair_client.requests.get")
    def test_api_key_sent_in_header(self, mock_get):
        """API key is included in X-API-Key header."""
        mock_get.return_value = MagicMock(
            ok=True, json=lambda: {"fields": [], "data": []},
        )
        client = PurpleAirClient(40.0, -74.0, api_key="my_key")
        client.find_nearby_sensors()

        headers = mock_get.call_args.kwargs.get("headers") or mock_get.call_args[1].get("headers", {})
        assert headers.get("X-API-Key") == "my_key"

    @patch("src.purpleair_client.requests.get")
    def test_outdoor_only_filter(self, mock_get):
        """location_type=0 ensures only outdoor sensors."""
        mock_get.return_value = MagicMock(
            ok=True, json=lambda: {"fields": [], "data": []},
        )
        client = PurpleAirClient(40.0, -74.0)
        client.find_nearby_sensors()

        params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params", {})
        assert params["location_type"] == "0"


# ------------------------------------------------------------------
# Sensor Filtering
# ------------------------------------------------------------------


class TestSensorFiltering:
    """Negative PM2.5 and stale sensors are discarded."""

    @patch("src.purpleair_client.requests.get")
    def test_negative_pm25_discarded(self, mock_get):
        """Negative PM2.5 readings are filtered out."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {
                "fields": ["sensor_index", "pm2.5", "latitude", "longitude", "last_seen"],
                "data": [
                    [1, -5.0, 40.001, -74.001, now_ts],
                    [2, 10.0, 40.002, -74.002, now_ts],
                ],
            },
        )
        client = PurpleAirClient(40.0, -74.0)
        sensors = client.find_nearby_sensors()

        assert len(sensors) == 1
        assert sensors[0]["pm25"] == 10.0

    @patch("src.purpleair_client.requests.get")
    def test_stale_sensor_discarded(self, mock_get):
        """Sensor last seen >30 min ago is filtered out."""
        now = datetime.now(timezone.utc)
        fresh_ts = int(now.timestamp())
        stale_ts = int((now - timedelta(minutes=45)).timestamp())

        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {
                "fields": ["sensor_index", "pm2.5", "latitude", "longitude", "last_seen"],
                "data": [
                    [1, 10.0, 40.001, -74.001, stale_ts],
                    [2, 15.0, 40.002, -74.002, fresh_ts],
                ],
            },
        )
        client = PurpleAirClient(40.0, -74.0)
        sensors = client.find_nearby_sensors()

        assert len(sensors) == 1
        assert sensors[0]["sensor_index"] == 2

    @patch("src.purpleair_client.requests.get")
    def test_sorted_by_distance(self, mock_get):
        """Sensors are returned sorted by distance (closest first)."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {
                "fields": ["sensor_index", "pm2.5", "latitude", "longitude", "last_seen"],
                "data": [
                    [1, 10.0, 40.05, -74.0, now_ts],    # farther
                    [2, 15.0, 40.001, -74.001, now_ts],  # closer
                ],
            },
        )
        client = PurpleAirClient(40.0, -74.0)
        sensors = client.find_nearby_sensors()

        assert sensors[0]["sensor_index"] == 2  # closer one first

    @patch("src.purpleair_client.requests.get")
    def test_network_error_raises(self, mock_get):
        """Network failure → PurpleAirError."""
        import requests as real_requests

        mock_get.side_effect = real_requests.ConnectionError("timeout")
        client = PurpleAirClient(40.0, -74.0)

        with pytest.raises(PurpleAirError, match="Network error"):
            client.find_nearby_sensors()


# ------------------------------------------------------------------
# HTTP Error Diagnostics
# ------------------------------------------------------------------


class TestHttpErrorDiagnostics:
    """Payment/auth failures surface actionable, status-specific messages.

    Regression guard: a depleted PurpleAir points balance returns HTTP 402.
    Previously this was raised as a generic error and swallowed by the
    orchestrator's fallback, so a genuine smoke event went ungated. The error
    message must now name the payment problem explicitly.
    """

    @patch("src.purpleair_client.requests.get")
    def test_payment_required_402_message(self, mock_get):
        """HTTP 402 → PurpleAirError mentioning payment/points depletion."""
        mock_get.return_value = MagicMock(
            ok=False,
            status_code=402,
            text='{"error":"PaymentRequiredError","description":"balance is -1595 points"}',
        )
        client = PurpleAirClient(40.0, -74.0, api_key="k")

        with pytest.raises(PurpleAirError, match="Payment Required"):
            client.find_nearby_sensors()

    @patch("src.purpleair_client.requests.get")
    def test_402_message_is_actionable(self, mock_get):
        """402 message includes the top-up URL and points/credits guidance."""
        mock_get.return_value = MagicMock(
            ok=False, status_code=402, text="{}",
        )
        client = PurpleAirClient(40.0, -74.0, api_key="k")

        with pytest.raises(PurpleAirError) as exc_info:
            client.find_nearby_sensors()

        msg = str(exc_info.value)
        assert "402" in msg
        assert "points" in msg.lower() or "credits" in msg.lower()
        assert "develop.purpleair.com" in msg

    @patch("src.purpleair_client.requests.get")
    def test_unauthorized_401_message(self, mock_get):
        """HTTP 401 → PurpleAirError pointing at the API key."""
        mock_get.return_value = MagicMock(
            ok=False, status_code=401, text="Unauthorized",
        )
        client = PurpleAirClient(40.0, -74.0, api_key="bad")

        with pytest.raises(PurpleAirError, match="PURPLEAIR_API_KEY"):
            client.find_nearby_sensors()

    @patch("src.purpleair_client.requests.get")
    def test_unknown_status_still_raises_with_code(self, mock_get):
        """An unmapped error status still raises with the status code included."""
        mock_get.return_value = MagicMock(
            ok=False, status_code=500, text="server error",
        )
        client = PurpleAirClient(40.0, -74.0, api_key="k")

        with pytest.raises(PurpleAirError, match="500"):
            client.find_nearby_sensors()


# ------------------------------------------------------------------
# Nearby-sensor ID caching (metered-cost optimization)
# ------------------------------------------------------------------


def _pa_response(body: dict):
    """Build a MagicMock HTTP response returning *body* as JSON."""
    return MagicMock(ok=True, status_code=200, json=lambda: body, text="")


def _is_live_read(call) -> bool:
    """True if a requests.get call used the cheap show_only live-read path."""
    params = call.kwargs.get("params") or (call[1].get("params") if len(call) > 1 else {})
    return "show_only" in (params or {})


class TestSensorIdCaching:
    """Discovery runs once, then cheap show_only live reads reuse cached IDs."""

    @staticmethod
    def _fresh_discovery_body():
        now_ts = int(datetime.now(timezone.utc).timestamp())
        return {
            "fields": ["sensor_index", "pm2.5", "latitude", "longitude", "last_seen"],
            "data": [
                [101, 10.0, 40.001, -74.001, now_ts],
                [102, 20.0, 40.002, -74.002, now_ts],
                [103, 30.0, 40.003, -74.003, now_ts],
            ],
        }

    @staticmethod
    def _fresh_live_body():
        now_ts = int(datetime.now(timezone.utc).timestamp())
        return {
            "fields": ["sensor_index", "pm2.5", "last_seen"],
            "data": [
                [101, 12.0, now_ts],
                [102, 22.0, now_ts],
                [103, 32.0, now_ts],
            ],
        }

    @patch("src.purpleair_client.requests.get")
    def test_discovery_once_then_cached_reads(self, mock_get):
        """First get_aqi discovers; subsequent calls reuse cached IDs (no re-discovery)."""
        discovery = self._fresh_discovery_body()
        live = self._fresh_live_body()

        def side_effect(url, params=None, headers=None, timeout=None):
            if "show_only" in (params or {}):
                return _pa_response(live)
            return _pa_response(discovery)

        mock_get.side_effect = side_effect

        client = PurpleAirClient(40.0, -74.0, api_key="k")
        r1 = client.get_aqi()  # cold start → discovery
        r2 = client.get_aqi()  # cached → show_only
        r3 = client.get_aqi()  # cached → show_only

        discovery_calls = [c for c in mock_get.call_args_list if not _is_live_read(c)]
        live_calls = [c for c in mock_get.call_args_list if _is_live_read(c)]

        assert len(discovery_calls) == 1, "discovery should run exactly once"
        assert len(live_calls) == 2, "later cycles use cheap show_only reads"

        # First result derived from discovery data (median of [10,20,30]).
        assert r1["pm25"] == 20.0
        # Cached reads use the live values (median of [12,22,32]).
        assert r2["pm25"] == 22.0
        assert r3["pm25"] == 22.0

    @patch("src.purpleair_client.requests.get")
    def test_live_reads_request_trimmed_fields(self, mock_get):
        """Live reads request only pm2.5,last_seen (no lat/lon); discovery keeps lat/lon."""
        discovery = self._fresh_discovery_body()
        live = self._fresh_live_body()

        def side_effect(url, params=None, headers=None, timeout=None):
            if "show_only" in (params or {}):
                return _pa_response(live)
            return _pa_response(discovery)

        mock_get.side_effect = side_effect

        client = PurpleAirClient(40.0, -74.0, api_key="k")
        client.get_aqi()  # discovery
        client.get_aqi()  # live read

        live_call = next(c for c in mock_get.call_args_list if _is_live_read(c))
        live_params = live_call.kwargs["params"]
        assert live_params["fields"] == "pm2.5,last_seen"
        assert "latitude" not in live_params["fields"]
        assert "longitude" not in live_params["fields"]
        assert live_params["show_only"]  # non-empty ID list

        disc_call = next(c for c in mock_get.call_args_list if not _is_live_read(c))
        disc_params = disc_call.kwargs["params"]
        assert "latitude" in disc_params["fields"]
        assert "longitude" in disc_params["fields"]

    @patch("src.purpleair_client.requests.get")
    def test_ttl_expiry_triggers_rediscovery(self, mock_get):
        """When the cache TTL lapses, discovery re-runs."""
        discovery = self._fresh_discovery_body()
        live = self._fresh_live_body()

        def side_effect(url, params=None, headers=None, timeout=None):
            if "show_only" in (params or {}):
                return _pa_response(live)
            return _pa_response(discovery)

        mock_get.side_effect = side_effect

        client = PurpleAirClient(40.0, -74.0, api_key="k", sensor_cache_ttl_hours=12.0)
        client.get_aqi()  # discovery
        client.get_aqi()  # cached
        assert len([c for c in mock_get.call_args_list if not _is_live_read(c)]) == 1

        # Force the cache to look expired.
        client._cache_expiry = datetime.now(timezone.utc) - timedelta(seconds=1)
        client.get_aqi()  # should re-discover

        discovery_calls = [c for c in mock_get.call_args_list if not _is_live_read(c)]
        assert len(discovery_calls) == 2, "expired TTL forces a fresh discovery"

    @patch("src.purpleair_client.requests.get")
    def test_stale_cached_sensors_trigger_rediscovery(self, mock_get):
        """If cached sensors return only stale readings, discovery refreshes the cache."""
        now = datetime.now(timezone.utc)
        stale_ts = int((now - timedelta(minutes=45)).timestamp())
        discovery = self._fresh_discovery_body()
        stale_live = {
            "fields": ["sensor_index", "pm2.5", "last_seen"],
            "data": [
                [101, 12.0, stale_ts],
                [102, 22.0, stale_ts],
                [103, 32.0, stale_ts],
            ],
        }

        def side_effect(url, params=None, headers=None, timeout=None):
            if "show_only" in (params or {}):
                return _pa_response(stale_live)
            return _pa_response(discovery)

        mock_get.side_effect = side_effect

        client = PurpleAirClient(40.0, -74.0, api_key="k")
        client.get_aqi()  # discovery, caches IDs
        r = client.get_aqi()  # cached read all stale → re-discover

        discovery_calls = [c for c in mock_get.call_args_list if not _is_live_read(c)]
        assert len(discovery_calls) == 2, "stale cached sensors force re-discovery"
        assert r["source"] == "purpleair"
        assert r["pm25"] == 20.0  # falls back to fresh discovery data

    @patch("src.purpleair_client.requests.get")
    def test_cached_read_preserves_median_and_freshness(self, mock_get):
        """Cached live reads keep median-of-3 and drop stale/negative sensors."""
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        stale_ts = int((now - timedelta(minutes=45)).timestamp())
        discovery = self._fresh_discovery_body()
        # One negative, one stale, three fresh → median of the three fresh.
        mixed_live = {
            "fields": ["sensor_index", "pm2.5", "last_seen"],
            "data": [
                [101, -1.0, now_ts],     # negative → discarded
                [102, 40.0, stale_ts],   # stale → discarded
                [103, 30.0, now_ts],     # fresh
            ],
        }

        def side_effect(url, params=None, headers=None, timeout=None):
            if "show_only" in (params or {}):
                return _pa_response(mixed_live)
            return _pa_response(discovery)

        mock_get.side_effect = side_effect

        client = PurpleAirClient(40.0, -74.0, api_key="k")
        client.get_aqi()  # discovery
        r = client.get_aqi()  # cached read with mixed quality

        # Only sensor 103 survived freshness/negative filters.
        assert r["pm25"] == 30.0
        assert r["sensor_count"] == 1


