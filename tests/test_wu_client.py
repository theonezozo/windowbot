"""Tests for the Weather Underground PWS client.

Validates design decisions:
- Station discovery via /v3/location/near with parallel arrays.
- Stations sorted by distance, capped to 10-mile radius.
- Caching: second discover_stations() call does not hit API.
- Pre-computed distance_miles on station dicts (haversine).
- Observation staleness rejection (>30 min old).
- Missing temperature → rejected; missing optional fields → accepted.
- Median aggregation of temperature, humidity, wind speed.
- LKG (last-known-good) cache: fresh reading stored; expired (>2h) evicted.
- Batch fetching: collects 3 valid readings, stops early.
- WUError raised on API failures and zero valid observations.
- Result dict: source="wu", is_fallback=False, used_cache flag.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from src.wu_client import WUClient, WUError
from src.nws_client import NWSError
from src.openmeteo_client import OpenMeteoError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_LAT = 37.40
_LON = -122.08
_API_KEY = "test-wu-key-12345"


def _wu_obs(
    station_id: str,
    temp_f: float,
    humidity: float | None = 55,
    wind_mph: float | None = 5.2,
    age_minutes: int = 5,
):
    """Build a mock WU current-observation response."""
    ts = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    obs = {
        "stationID": station_id,
        "obsTimeUtc": ts,
        "imperial": {"temp": temp_f},
    }
    if humidity is not None:
        obs["humidity"] = humidity
    if wind_mph is not None:
        obs["imperial"]["windSpeed"] = wind_mph
    return {"observations": [obs]}


def _discovery_response(
    station_ids: list[str],
    latitudes: list[float] | None = None,
    longitudes: list[float] | None = None,
):
    """Build a mock WU /v3/location/near response with parallel arrays."""
    n = len(station_ids)
    if latitudes is None:
        latitudes = [_LAT + 0.01 * i for i in range(n)]
    if longitudes is None:
        longitudes = [_LON + 0.01 * i for i in range(n)]
    return {
        "location": {
            "stationId": station_ids,
            "stationName": [f"Station {s}" for s in station_ids],
            "latitude": latitudes,
            "longitude": longitudes,
        }
    }


def _mock_get_ok(json_data: dict) -> MagicMock:
    """Return a MagicMock response with ok=True and given json."""
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = json_data
    return resp


def _mock_get_fail(status_code: int = 500) -> MagicMock:
    resp = MagicMock()
    resp.ok = False
    resp.status_code = status_code
    resp.text = "Internal Server Error"
    return resp


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestInitialization:
    """Constructor stores lat/lon/api_key."""

    def test_stores_lat_lon_api_key(self):
        client = WUClient(37.40, -122.08, "mykey")
        assert client._lat == 37.40
        assert client._lon == -122.08
        assert client._api_key == "mykey"

    def test_empty_stations_on_init(self):
        client = WUClient(_LAT, _LON, _API_KEY)
        assert client._stations == []

    @pytest.mark.xfail(
        reason="WUClient.__init__ does not validate empty API key — tracking for future fix",
        strict=False,
    )
    def test_empty_api_key_raises(self):
        """Empty API key should ideally raise ValueError."""
        with pytest.raises(ValueError):
            WUClient(_LAT, _LON, "")


# ------------------------------------------------------------------
# Station Discovery
# ------------------------------------------------------------------


class TestStationDiscovery:
    """Discovery via /v3/location/near with parallel arrays."""

    @patch("src.wu_client.requests.get")
    def test_discover_returns_station_ids(self, mock_get):
        """Successful discovery returns list of station IDs."""
        mock_get.return_value = _mock_get_ok(
            _discovery_response(["KCASUNNY1", "KCAMOUNT2", "KCAOTHER3"])
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        ids = client.discover_stations()
        assert isinstance(ids, list)
        assert len(ids) == 3
        assert all(isinstance(s, str) for s in ids)

    @patch("src.wu_client.requests.get")
    def test_stations_sorted_by_distance(self, mock_get):
        """Stations returned sorted closest-first by haversine distance."""
        # Place stations at different distances: far, near, medium
        mock_get.return_value = _mock_get_ok(
            _discovery_response(
                ["FAR1", "NEAR1", "MID1"],
                latitudes=[_LAT + 0.1, _LAT + 0.001, _LAT + 0.03],
                longitudes=[_LON, _LON, _LON],
            )
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        ids = client.discover_stations()
        assert ids[0] == "NEAR1"
        assert ids[1] == "MID1"
        assert ids[2] == "FAR1"

    @patch("src.wu_client.requests.get")
    def test_distance_miles_precomputed(self, mock_get):
        """Each cached station dict has a distance_miles field."""
        mock_get.return_value = _mock_get_ok(
            _discovery_response(["KCASUNNY1"], latitudes=[_LAT + 0.01], longitudes=[_LON])
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        client.discover_stations()
        assert len(client._stations) == 1
        stn = client._stations[0]
        assert "distance_miles" in stn
        assert isinstance(stn["distance_miles"], float)
        assert stn["distance_miles"] > 0

    @patch("src.wu_client.requests.get")
    def test_caching_second_call_no_api(self, mock_get):
        """Second discover_stations() call returns cached results, no extra API call."""
        mock_get.return_value = _mock_get_ok(
            _discovery_response(["KCASUNNY1", "KCAMOUNT2"])
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        ids1 = client.discover_stations()
        ids2 = client.discover_stations()
        assert ids1 == ids2
        assert mock_get.call_count == 1  # single API call for both discover_stations() calls

    @patch("src.wu_client.requests.get")
    def test_api_error_raises_wuerror(self, mock_get):
        """API error on discovery raises WUError."""
        mock_get.return_value = _mock_get_fail(500)
        client = WUClient(_LAT, _LON, _API_KEY)
        with pytest.raises(WUError):
            client.discover_stations()

    @patch("src.wu_client.requests.get")
    def test_empty_response_no_stations(self, mock_get):
        """Empty location response → no stations discovered."""
        mock_get.return_value = _mock_get_ok({"location": {}})
        client = WUClient(_LAT, _LON, _API_KEY)
        ids = client.discover_stations()
        assert ids == []

    @patch("src.wu_client.requests.get")
    def test_network_error_raises_wuerror(self, mock_get):
        """requests.RequestException → WUError."""
        import requests as req
        mock_get.side_effect = req.ConnectionError("DNS failure")
        client = WUClient(_LAT, _LON, _API_KEY)
        with pytest.raises(WUError, match="Network error"):
            client.discover_stations()


# ------------------------------------------------------------------
# Haversine Distance
# ------------------------------------------------------------------


class TestHaversineDistance:
    """Great-circle distance calculation."""

    def test_same_point_is_zero(self):
        assert WUClient._haversine_miles(37.0, -122.0, 37.0, -122.0) == 0.0

    def test_known_distance(self):
        # San Francisco to Los Angeles ≈ 347 miles
        dist = WUClient._haversine_miles(37.7749, -122.4194, 34.0522, -118.2437)
        assert 340 < dist < 355

    def test_symmetric(self):
        d1 = WUClient._haversine_miles(37.0, -122.0, 38.0, -121.0)
        d2 = WUClient._haversine_miles(38.0, -121.0, 37.0, -122.0)
        assert d1 == pytest.approx(d2)


# ------------------------------------------------------------------
# Single Observation Fetching
# ------------------------------------------------------------------


class TestSingleObservation:
    """Fetch and validate a single station's observation."""

    @patch("src.wu_client.requests.get")
    def test_valid_observation_all_fields(self, mock_get):
        """Fresh observation with all fields → valid dict returned."""
        mock_get.return_value = _mock_get_ok(
            _wu_obs("KCASUNNY1", temp_f=68.5, humidity=55, wind_mph=5.2, age_minutes=5)
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)

        assert result is not None
        assert result["station_id"] == "KCASUNNY1"
        assert result["temperature_f"] == pytest.approx(68.5, abs=0.1)
        assert result["humidity"] == pytest.approx(55.0, abs=0.1)
        assert result["wind_speed_mph"] == pytest.approx(5.2, abs=0.1)
        assert isinstance(result["timestamp"], datetime)

    @patch("src.wu_client.requests.get")
    def test_stale_observation_rejected(self, mock_get):
        """41-minute-old observation → returns None."""
        mock_get.return_value = _mock_get_ok(
            _wu_obs("KCASUNNY1", temp_f=68.5, age_minutes=41)
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is None

    @patch("src.wu_client.requests.get")
    def test_fresh_observation_at_boundary(self, mock_get):
        """29-minute-old observation → accepted."""
        mock_get.return_value = _mock_get_ok(
            _wu_obs("KCASUNNY1", temp_f=68.5, age_minutes=29)
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is not None

    @patch("src.wu_client.requests.get")
    def test_missing_temperature_rejected(self, mock_get):
        """No temperature in imperial block → None."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        mock_get.return_value = _mock_get_ok({
            "observations": [{
                "stationID": "KCASUNNY1",
                "obsTimeUtc": ts,
                "humidity": 55,
                "imperial": {},  # no temp
            }]
        })
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is None

    @patch("src.wu_client.requests.get")
    def test_missing_humidity_accepted(self, mock_get):
        """Missing humidity → observation accepted with humidity=None."""
        mock_get.return_value = _mock_get_ok(
            _wu_obs("KCASUNNY1", temp_f=68.5, humidity=None, wind_mph=5.0, age_minutes=5)
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is not None
        assert result["humidity"] is None

    @patch("src.wu_client.requests.get")
    def test_missing_wind_accepted(self, mock_get):
        """Missing wind → observation accepted with wind_speed_mph=None."""
        mock_get.return_value = _mock_get_ok(
            _wu_obs("KCASUNNY1", temp_f=68.5, humidity=55, wind_mph=None, age_minutes=5)
        )
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is not None
        assert result["wind_speed_mph"] is None

    @patch("src.wu_client.requests.get")
    def test_no_timestamp_rejected(self, mock_get):
        """Observation with no obsTimeUtc → None."""
        mock_get.return_value = _mock_get_ok({
            "observations": [{
                "stationID": "KCASUNNY1",
                "humidity": 55,
                "imperial": {"temp": 68.5},
            }]
        })
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is None

    @patch("src.wu_client.requests.get")
    def test_empty_observations_array(self, mock_get):
        """observations: [] → None."""
        mock_get.return_value = _mock_get_ok({"observations": []})
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is None

    @patch("src.wu_client.requests.get")
    def test_api_error_raises_wuerror(self, mock_get):
        """API error on observation fetch → WUError."""
        mock_get.return_value = _mock_get_fail(503)
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        with pytest.raises(WUError):
            client._fetch_single_observation("KCASUNNY1", now)

    @patch("src.wu_client.requests.get")
    def test_invalid_timestamp_rejected(self, mock_get):
        """Unparseable timestamp → None."""
        mock_get.return_value = _mock_get_ok({
            "observations": [{
                "stationID": "KCASUNNY1",
                "obsTimeUtc": "not-a-date",
                "humidity": 55,
                "imperial": {"temp": 68.5},
            }]
        })
        client = WUClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KCASUNNY1", now)
        assert result is None


# ------------------------------------------------------------------
# Batch Fetching
# ------------------------------------------------------------------


class TestBatchFetching:
    """_fetch_batch collects target valid readings, handles LKG cache."""

    def _make_client_with_stations(self, station_ids, distances=None):
        """Create a WUClient with pre-loaded station metadata."""
        client = WUClient(_LAT, _LON, _API_KEY)
        if distances is None:
            distances = [1.0 * (i + 1) for i in range(len(station_ids))]
        client._stations = [
            {"id": sid, "lat": _LAT, "lon": _LON, "distance_miles": dist}
            for sid, dist in zip(station_ids, distances)
        ]
        return client

    @patch.object(WUClient, "_fetch_single_observation")
    def test_collects_three_valid_stops_early(self, mock_fetch):
        """Collects 3 valid readings, doesn't query remaining stations."""
        client = self._make_client_with_stations(
            ["S1", "S2", "S3", "S4", "S5"]
        )
        now = datetime.now(timezone.utc)

        def side_effect(sid, _now):
            return {
                "station_id": sid,
                "temperature_f": 70.0,
                "humidity": 50.0,
                "wind_speed_mph": 5.0,
                "timestamp": now,
            }

        mock_fetch.side_effect = side_effect
        results = client._fetch_batch(client._stations, now, target=3)

        assert len(results) == 3
        assert mock_fetch.call_count == 3  # stopped after getting 3

    @patch.object(WUClient, "_fetch_single_observation")
    def test_skips_stale_continues_to_next(self, mock_fetch):
        """Stale stations skipped, continues to find valid ones."""
        client = self._make_client_with_stations(["STALE1", "STALE2", "FRESH1", "FRESH2", "FRESH3"])
        now = datetime.now(timezone.utc)

        def side_effect(sid, _now):
            if sid.startswith("STALE"):
                return None
            return {
                "station_id": sid,
                "temperature_f": 72.0,
                "humidity": 50.0,
                "wind_speed_mph": 5.0,
                "timestamp": now,
            }

        mock_fetch.side_effect = side_effect
        results = client._fetch_batch(client._stations, now, target=3)

        assert len(results) == 3
        assert all(r["station_id"].startswith("FRESH") for r in results)

    @patch.object(WUClient, "_fetch_single_observation")
    def test_lkg_cache_stores_fresh_readings(self, mock_fetch):
        """Fresh readings are stored in _station_cache."""
        client = self._make_client_with_stations(["S1"])
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = {
            "station_id": "S1",
            "temperature_f": 72.0,
            "humidity": 50.0,
            "wind_speed_mph": 5.0,
            "timestamp": now,
        }
        client._fetch_batch(client._stations, now, target=3)

        assert "S1" in client._station_cache
        assert client._station_cache["S1"]["temperature_f"] == 72.0

    @patch.object(WUClient, "_fetch_single_observation")
    def test_lkg_cache_used_on_failure(self, mock_fetch):
        """After caching a reading, a subsequent failure uses the cached value."""
        client = self._make_client_with_stations(["S1"])
        now = datetime.now(timezone.utc)

        # First call: successful reading cached
        good_obs = {
            "station_id": "S1",
            "temperature_f": 72.0,
            "humidity": 50.0,
            "wind_speed_mph": 5.0,
            "timestamp": now,
        }
        client._station_cache["S1"] = good_obs

        # Now fetch fails
        mock_fetch.return_value = None
        results = client._fetch_batch(client._stations, now, target=3)

        assert len(results) == 1
        assert results[0]["temperature_f"] == 72.0
        assert results[0].get("is_cached") is True

    @patch.object(WUClient, "_fetch_single_observation")
    def test_lkg_cache_expired_not_used(self, mock_fetch):
        """Cached reading older than 2 hours is NOT used."""
        client = self._make_client_with_stations(["S1"])
        now = datetime.now(timezone.utc)

        # Expired cache entry (3 hours old)
        client._station_cache["S1"] = {
            "station_id": "S1",
            "temperature_f": 72.0,
            "humidity": 50.0,
            "wind_speed_mph": 5.0,
            "timestamp": now - timedelta(hours=3),
        }

        mock_fetch.return_value = None
        results = client._fetch_batch(client._stations, now, target=3)

        assert len(results) == 0  # expired cache not used

    @patch.object(WUClient, "_fetch_single_observation")
    def test_all_stations_fail_empty_list(self, mock_fetch):
        """All stations return None → empty list."""
        client = self._make_client_with_stations(["S1", "S2", "S3"])
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = None
        results = client._fetch_batch(client._stations, now, target=3)
        assert results == []

    @patch.object(WUClient, "_fetch_single_observation")
    def test_api_error_caught_continues(self, mock_fetch):
        """WUError on one station doesn't stop batch; continues to next."""
        client = self._make_client_with_stations(["ERR1", "OK1", "OK2", "OK3"])
        now = datetime.now(timezone.utc)

        def side_effect(sid, _now):
            if sid == "ERR1":
                raise WUError("API down")
            return {
                "station_id": sid,
                "temperature_f": 70.0,
                "humidity": 50.0,
                "wind_speed_mph": 5.0,
                "timestamp": now,
            }

        mock_fetch.side_effect = side_effect
        results = client._fetch_batch(client._stations, now, target=3)
        assert len(results) == 3
        assert results[0]["station_id"] == "OK1"

    @patch.object(WUClient, "_fetch_single_observation")
    def test_used_cache_flag_set(self, mock_fetch):
        """Cached observations have is_cached=True."""
        client = self._make_client_with_stations(["S1", "S2"])
        now = datetime.now(timezone.utc)

        # S1 succeeds fresh
        # S2 fails but has cache
        client._station_cache["S2"] = {
            "station_id": "S2",
            "temperature_f": 74.0,
            "humidity": 55.0,
            "wind_speed_mph": 8.0,
            "timestamp": now - timedelta(minutes=10),
        }

        def side_effect(sid, _now):
            if sid == "S1":
                return {
                    "station_id": "S1",
                    "temperature_f": 72.0,
                    "humidity": 50.0,
                    "wind_speed_mph": 5.0,
                    "timestamp": now,
                }
            return None

        mock_fetch.side_effect = side_effect
        results = client._fetch_batch(client._stations, now, target=3)
        assert len(results) == 2
        fresh = [r for r in results if not r.get("is_cached")]
        cached = [r for r in results if r.get("is_cached")]
        assert len(fresh) == 1
        assert len(cached) == 1


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------


class TestAggregation:
    """Median aggregation of readings."""

    def test_median_of_three(self):
        obs = [
            {"temperature_f": 70.0, "humidity": 40.0, "wind_speed_mph": 5.0},
            {"temperature_f": 75.0, "humidity": 50.0, "wind_speed_mph": 10.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = WUClient._aggregate(obs)
        assert result["temperature_f"] == 75.0
        assert result["humidity"] == 50.0
        assert result["wind_speed_mph"] == 10.0
        assert result["station_count"] == 3

    def test_median_is_not_mean(self):
        """With outlier, median differs from mean."""
        obs = [
            {"temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0},
            {"temperature_f": 72.0, "humidity": 50.0, "wind_speed_mph": 5.0},
            {"temperature_f": 120.0, "humidity": 50.0, "wind_speed_mph": 5.0},  # outlier
        ]
        result = WUClient._aggregate(obs)
        assert result["temperature_f"] == 72.0  # median, not ~87.3 mean

    def test_median_of_two(self):
        """Two observations → average (median of two)."""
        obs = [
            {"temperature_f": 70.0, "humidity": 40.0, "wind_speed_mph": 5.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = WUClient._aggregate(obs)
        assert result["temperature_f"] == 75.0

    def test_single_observation(self):
        obs = [{"temperature_f": 72.5, "humidity": 55.0, "wind_speed_mph": 8.0}]
        result = WUClient._aggregate(obs)
        assert result["temperature_f"] == 72.5
        assert result["station_count"] == 1

    def test_missing_humidity_excluded(self):
        """Observations with None humidity excluded from humidity median."""
        obs = [
            {"temperature_f": 70.0, "humidity": None, "wind_speed_mph": 5.0},
            {"temperature_f": 75.0, "humidity": 50.0, "wind_speed_mph": 10.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = WUClient._aggregate(obs)
        assert result["humidity"] == 55.0  # median of [50, 60]

    def test_all_humidity_none(self):
        obs = [{"temperature_f": 70.0, "humidity": None, "wind_speed_mph": 5.0}]
        result = WUClient._aggregate(obs)
        assert result["humidity"] is None

    def test_missing_wind_excluded(self):
        obs = [
            {"temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": None},
            {"temperature_f": 75.0, "humidity": 50.0, "wind_speed_mph": 10.0},
        ]
        result = WUClient._aggregate(obs)
        assert result["wind_speed_mph"] == 10.0

    def test_is_fallback_always_false(self):
        """WU aggregation always returns is_fallback=False."""
        obs = [{"temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0}]
        result = WUClient._aggregate(obs)
        assert result["is_fallback"] is False


# ------------------------------------------------------------------
# get_outdoor_conditions() — Full Integration
# ------------------------------------------------------------------


class TestGetOutdoorConditions:
    """End-to-end: discover → fetch → aggregate."""

    @patch.object(WUClient, "_fetch_single_observation")
    @patch.object(WUClient, "discover_stations")
    def test_happy_path(self, mock_discover, mock_fetch):
        """3 nearby stations → median returned with source='wu'."""
        client = WUClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {"id": "S1", "distance_miles": 1.0},
            {"id": "S2", "distance_miles": 2.0},
            {"id": "S3", "distance_miles": 3.0},
        ]
        now = datetime.now(timezone.utc)
        temps = [70.0, 75.0, 80.0]
        idx = [0]

        def fetch_side(sid, _now):
            t = temps[idx[0]]
            idx[0] += 1
            return {
                "station_id": sid,
                "temperature_f": t,
                "humidity": 50.0,
                "wind_speed_mph": 5.0,
                "timestamp": now,
            }

        mock_fetch.side_effect = fetch_side

        result = client.get_outdoor_conditions()

        assert result["temperature_f"] == 75.0  # median
        assert result["source"] == "wu"
        assert result["is_fallback"] is False
        assert result["station_count"] == 3

    @patch.object(WUClient, "_fetch_single_observation")
    @patch.object(WUClient, "discover_stations")
    def test_used_cache_flag(self, mock_discover, mock_fetch):
        """used_cache=True when a cached reading contributes."""
        client = WUClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {"id": "S1", "distance_miles": 1.0},
            {"id": "S2", "distance_miles": 2.0},
        ]
        now = datetime.now(timezone.utc)

        # S2 has cached reading
        client._station_cache["S2"] = {
            "station_id": "S2",
            "temperature_f": 74.0,
            "humidity": 55.0,
            "wind_speed_mph": 8.0,
            "timestamp": now - timedelta(minutes=10),
        }

        def fetch_side(sid, _now):
            if sid == "S1":
                return {
                    "station_id": "S1",
                    "temperature_f": 72.0,
                    "humidity": 50.0,
                    "wind_speed_mph": 5.0,
                    "timestamp": now,
                }
            return None

        mock_fetch.side_effect = fetch_side

        result = client.get_outdoor_conditions()
        assert result["used_cache"] is True

    @patch.object(WUClient, "_fetch_single_observation")
    @patch.object(WUClient, "discover_stations")
    def test_no_valid_readings_raises(self, mock_discover, mock_fetch):
        """Zero valid observations → WUError."""
        client = WUClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {"id": "S1", "distance_miles": 1.0},
            {"id": "S2", "distance_miles": 2.0},
        ]
        mock_fetch.return_value = None

        with pytest.raises(WUError, match="No valid weather observations"):
            client.get_outdoor_conditions()

    @patch.object(WUClient, "_fetch_single_observation")
    @patch.object(WUClient, "discover_stations")
    def test_distance_cap_filters_far_stations(self, mock_discover, mock_fetch):
        """Stations beyond 10 miles are filtered from nearby pool."""
        client = WUClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {"id": "NEAR1", "distance_miles": 2.0},
            {"id": "NEAR2", "distance_miles": 5.0},
            {"id": "FAR1", "distance_miles": 15.0},
            {"id": "FAR2", "distance_miles": 20.0},
        ]
        now = datetime.now(timezone.utc)

        def fetch_side(sid, _now):
            return {
                "station_id": sid,
                "temperature_f": 70.0,
                "humidity": 50.0,
                "wind_speed_mph": 5.0,
                "timestamp": now,
            }

        mock_fetch.side_effect = fetch_side

        result = client.get_outdoor_conditions()
        # Only NEAR1 and NEAR2 are within 10 mi
        assert result["station_count"] == 2
        queried_ids = [c.args[0] for c in mock_fetch.call_args_list]
        assert "FAR1" not in queried_ids
        assert "FAR2" not in queried_ids

    @patch.object(WUClient, "_fetch_single_observation")
    @patch.object(WUClient, "discover_stations")
    def test_all_beyond_distance_cap_uses_all(self, mock_discover, mock_fetch):
        """If ALL stations exceed 10 mi, falls back to using all."""
        client = WUClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {"id": "FAR1", "distance_miles": 15.0},
            {"id": "FAR2", "distance_miles": 20.0},
        ]
        now = datetime.now(timezone.utc)

        def fetch_side(sid, _now):
            return {
                "station_id": sid,
                "temperature_f": 70.0,
                "humidity": 50.0,
                "wind_speed_mph": 5.0,
                "timestamp": now,
            }

        mock_fetch.side_effect = fetch_side

        result = client.get_outdoor_conditions()
        assert result["station_count"] == 2  # used all stations as fallback

    @patch.object(WUClient, "_fetch_single_observation")
    @patch.object(WUClient, "discover_stations")
    def test_used_cache_false_when_all_fresh(self, mock_discover, mock_fetch):
        """used_cache=False when all readings are fresh."""
        client = WUClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {"id": "S1", "distance_miles": 1.0},
        ]
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = {
            "station_id": "S1",
            "temperature_f": 72.0,
            "humidity": 50.0,
            "wind_speed_mph": 5.0,
            "timestamp": now,
        }

        result = client.get_outdoor_conditions()
        assert result["used_cache"] is False


