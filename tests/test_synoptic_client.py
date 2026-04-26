"""Tests for the MesoWest/Synoptic Data weather client.

Validates design decisions:
- Station discovery via /stations/nearest with distance sorting.
- Caching: second discover_stations() call does not hit API.
- Single batch fetch: all stations in ONE API call (key advantage).
- Observation staleness rejection (>60 min old).
- Missing temperature → skipped; missing optional fields → accepted.
- Median aggregation of temperature, humidity, wind speed.
- LKG (last-known-good) cache: fresh reading stored; expired (>2h) evicted.
- Distance cap: only stations within 10 miles.
- SynopticError on API failures (RESPONSE_CODE != 1, network errors).
- Result dict: source="synoptic", is_fallback=False, used_cache flag.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.synoptic_client import SynopticClient, SynopticError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_LAT = 37.40
_LON = -122.08
_API_KEY = "test-synoptic-key-12345"


def _station_discovery_response(stations: list[dict]) -> dict:
    """Build a mock /stations/nearest response.

    Each station dict should have: stid, name, distance, mnet_id.
    """
    return {
        "SUMMARY": {
            "RESPONSE_CODE": 1,
            "RESPONSE_MESSAGE": "OK",
            "NUMBER_OF_OBJECTS": len(stations),
        },
        "STATION": [
            {
                "STID": s.get("stid", "KNUQ"),
                "NAME": s.get("name", "TEST STATION"),
                "LATITUDE": str(s.get("lat", _LAT)),
                "LONGITUDE": str(s.get("lon", _LON)),
                "DISTANCE": s.get("distance", 1.0),
                "STATUS": "ACTIVE",
                "MNET_ID": s.get("mnet_id", "1"),
            }
            for s in stations
        ],
    }


def _observations_response(stations: list[dict]) -> dict:
    """Build a mock /stations/latest response.

    Each station dict should have: stid, temp_f, humidity, wind_mph, age_minutes.
    """
    station_list = []
    for s in stations:
        age = s.get("age_minutes", 5)
        ts = (datetime.now(timezone.utc) - timedelta(minutes=age)).isoformat()

        obs = {}
        if s.get("temp_f") is not None:
            obs["air_temp_value_1"] = {
                "value": s["temp_f"],
                "date_time": ts,
            }

        if s.get("humidity") is not None:
            obs["relative_humidity_value_1"] = {
                "value": s["humidity"],
                "date_time": ts,
            }

        if s.get("wind_mph") is not None:
            obs["wind_speed_value_1"] = {
                "value": s["wind_mph"],
                "date_time": ts,
            }

        station_list.append({
            "STID": s.get("stid", "KNUQ"),
            "NAME": s.get("name", "TEST STATION"),
            "OBSERVATIONS": obs,
        })

    return {
        "SUMMARY": {"RESPONSE_CODE": 1, "NUMBER_OF_OBJECTS": len(stations)},
        "STATION": station_list,
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
        client = SynopticClient(37.40, -122.08, "mykey")
        assert client._lat == 37.40
        assert client._lon == -122.08
        assert client._api_key == "mykey"

    def test_empty_stations_on_init(self):
        client = SynopticClient(_LAT, _LON, _API_KEY)
        assert client._stations == []

    def test_empty_cache_on_init(self):
        client = SynopticClient(_LAT, _LON, _API_KEY)
        assert client._station_cache == {}

    @pytest.mark.xfail(
        reason="SynopticClient.__init__ does not validate empty API key — tracking for future fix",
        strict=False,
    )
    def test_empty_api_key_raises(self):
        """Empty API key should ideally raise ValueError."""
        with pytest.raises(ValueError):
            SynopticClient(_LAT, _LON, "")


# ------------------------------------------------------------------
# Station Discovery
# ------------------------------------------------------------------


class TestStationDiscovery:
    """Discovery via /stations/nearest."""

    @patch("src.synoptic_client.requests.get")
    def test_discover_returns_station_ids(self, mock_get):
        """Successful discovery returns list of station IDs."""
        mock_get.return_value = _mock_get_ok(
            _station_discovery_response([
                {"stid": "KNUQ", "distance": 1.2, "mnet_id": "1"},
                {"stid": "KSJC", "distance": 3.5, "mnet_id": "1"},
                {"stid": "MMTN", "distance": 5.0, "mnet_id": "42"},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        ids = client.discover_stations()
        assert isinstance(ids, list)
        assert len(ids) == 3
        assert all(isinstance(s, str) for s in ids)

    @patch("src.synoptic_client.requests.get")
    def test_stations_sorted_by_distance(self, mock_get):
        """Stations returned sorted closest-first by distance."""
        mock_get.return_value = _mock_get_ok(
            _station_discovery_response([
                {"stid": "FAR", "distance": 8.0},
                {"stid": "NEAR", "distance": 0.5},
                {"stid": "MID", "distance": 3.0},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        ids = client.discover_stations()
        assert ids == ["NEAR", "MID", "FAR"]

    @patch("src.synoptic_client.requests.get")
    def test_caching_second_call_no_api(self, mock_get):
        """Second discover_stations() returns cached results, no extra API call."""
        mock_get.return_value = _mock_get_ok(
            _station_discovery_response([
                {"stid": "KNUQ", "distance": 1.0},
                {"stid": "KSJC", "distance": 3.0},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        ids1 = client.discover_stations()
        ids2 = client.discover_stations()
        assert ids1 == ids2
        assert mock_get.call_count == 1

    @patch("src.synoptic_client.requests.get")
    def test_api_error_raises_synoptice_error(self, mock_get):
        """API error on discovery raises SynopticError."""
        mock_get.return_value = _mock_get_fail(500)
        client = SynopticClient(_LAT, _LON, _API_KEY)
        with pytest.raises(SynopticError):
            client.discover_stations()

    @patch("src.synoptic_client.requests.get")
    def test_empty_station_list(self, mock_get):
        """No stations in response → empty list."""
        mock_get.return_value = _mock_get_ok(
            _station_discovery_response([])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        ids = client.discover_stations()
        assert ids == []

    @patch("src.synoptic_client.requests.get")
    def test_response_code_not_1_raises(self, mock_get):
        """RESPONSE_CODE != 1 → SynopticError."""
        mock_get.return_value = _mock_get_ok({
            "SUMMARY": {
                "RESPONSE_CODE": 2,
                "RESPONSE_MESSAGE": "Invalid token",
            },
        })
        client = SynopticClient(_LAT, _LON, _API_KEY)
        with pytest.raises(SynopticError, match="Invalid token"):
            client.discover_stations()

    @patch("src.synoptic_client.requests.get")
    def test_network_error_raises_synoptic_error(self, mock_get):
        """requests.RequestException → SynopticError."""
        import requests as req
        mock_get.side_effect = req.ConnectionError("DNS failure")
        client = SynopticClient(_LAT, _LON, _API_KEY)
        with pytest.raises(SynopticError, match="Network error"):
            client.discover_stations()

    @patch("src.synoptic_client.requests.get")
    def test_station_metadata_stored(self, mock_get):
        """Discovered stations include stid, name, mnet_id, distance_miles."""
        mock_get.return_value = _mock_get_ok(
            _station_discovery_response([
                {"stid": "KNUQ", "name": "MOFFETT FIELD", "distance": 2.3, "mnet_id": "1"},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        client.discover_stations()
        assert len(client._stations) == 1
        stn = client._stations[0]
        assert stn["stid"] == "KNUQ"
        assert stn["name"] == "MOFFETT FIELD"
        assert stn["mnet_id"] == "1"
        assert stn["distance_miles"] == pytest.approx(2.3)


# ------------------------------------------------------------------
# Haversine Distance
# ------------------------------------------------------------------


class TestHaversineDistance:
    """Great-circle distance calculation."""

    def test_same_point_is_zero(self):
        assert SynopticClient._haversine_miles(37.0, -122.0, 37.0, -122.0) == 0.0

    def test_known_distance(self):
        # San Francisco to Los Angeles ≈ 347 miles
        dist = SynopticClient._haversine_miles(37.7749, -122.4194, 34.0522, -118.2437)
        assert 340 < dist < 355

    def test_symmetric(self):
        d1 = SynopticClient._haversine_miles(37.0, -122.0, 38.0, -121.0)
        d2 = SynopticClient._haversine_miles(38.0, -121.0, 37.0, -122.0)
        assert d1 == pytest.approx(d2)


# ------------------------------------------------------------------
# Station Classification
# ------------------------------------------------------------------


class TestStationClassification:
    """MNET_ID 1 or 2 = official; everything else = mesonet."""

    def test_mnet_1_is_official(self):
        assert SynopticClient._is_official({"mnet_id": "1"}) is True

    def test_mnet_2_is_official(self):
        assert SynopticClient._is_official({"mnet_id": "2"}) is True

    def test_mnet_42_is_not_official(self):
        assert SynopticClient._is_official({"mnet_id": "42"}) is False

    def test_missing_mnet_id_is_not_official(self):
        assert SynopticClient._is_official({}) is False


# ------------------------------------------------------------------
# Batch Observation Fetch (Single API Call)
# ------------------------------------------------------------------


class TestBatchObservationFetch:
    """_fetch_batch_observations — single API call for all stations."""

    @patch("src.synoptic_client.requests.get")
    def test_all_stations_return_valid_data(self, mock_get):
        """3 stations with valid temp, humidity, wind → 3 results."""
        mock_get.return_value = _mock_get_ok(
            _observations_response([
                {"stid": "S1", "temp_f": 68.5, "humidity": 55.0, "wind_mph": 7.2},
                {"stid": "S2", "temp_f": 70.0, "humidity": 50.0, "wind_mph": 5.0},
                {"stid": "S3", "temp_f": 72.3, "humidity": 60.0, "wind_mph": 10.0},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1", "S2", "S3"], now)

        assert len(results) == 3
        assert "S1" in results
        assert results["S1"]["temperature_f"] == pytest.approx(68.5, abs=0.1)
        assert results["S2"]["humidity"] == pytest.approx(50.0, abs=0.1)
        assert results["S3"]["wind_speed_mph"] == pytest.approx(10.0, abs=0.1)

    @patch("src.synoptic_client.requests.get")
    def test_station_missing_temp_skipped(self, mock_get):
        """Station with no air_temp_value_1 → excluded from results."""
        mock_get.return_value = _mock_get_ok({
            "SUMMARY": {"RESPONSE_CODE": 1, "NUMBER_OF_OBJECTS": 2},
            "STATION": [
                {
                    "STID": "S1",
                    "OBSERVATIONS": {
                        "air_temp_value_1": {
                            "value": 68.5,
                            "date_time": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                },
                {
                    "STID": "S2",
                    "OBSERVATIONS": {},  # no temperature
                },
            ],
        })
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1", "S2"], now)

        assert "S1" in results
        assert "S2" not in results

    @patch("src.synoptic_client.requests.get")
    def test_station_null_temp_value_skipped(self, mock_get):
        """Station with temp value=None → excluded."""
        ts = datetime.now(timezone.utc).isoformat()
        mock_get.return_value = _mock_get_ok({
            "SUMMARY": {"RESPONSE_CODE": 1, "NUMBER_OF_OBJECTS": 1},
            "STATION": [
                {
                    "STID": "S1",
                    "OBSERVATIONS": {
                        "air_temp_value_1": {"value": None, "date_time": ts},
                    },
                },
            ],
        })
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1"], now)
        assert len(results) == 0

    @patch("src.synoptic_client.requests.get")
    def test_stale_observation_rejected(self, mock_get):
        """Observation >60 min old → rejected."""
        mock_get.return_value = _mock_get_ok(
            _observations_response([
                {"stid": "S1", "temp_f": 68.5, "age_minutes": 90},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1"], now)
        assert len(results) == 0

    @patch("src.synoptic_client.requests.get")
    def test_fresh_observation_at_boundary(self, mock_get):
        """Observation at 59 min old → accepted."""
        mock_get.return_value = _mock_get_ok(
            _observations_response([
                {"stid": "S1", "temp_f": 68.5, "age_minutes": 59},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1"], now)
        assert "S1" in results

    @patch("src.synoptic_client.requests.get")
    def test_missing_humidity_accepted(self, mock_get):
        """Station with no humidity → observation accepted, humidity=None."""
        mock_get.return_value = _mock_get_ok(
            _observations_response([
                {"stid": "S1", "temp_f": 68.5, "humidity": None, "wind_mph": 5.0},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1"], now)
        assert "S1" in results
        assert results["S1"]["humidity"] is None

    @patch("src.synoptic_client.requests.get")
    def test_missing_wind_accepted(self, mock_get):
        """Station with no wind → observation accepted, wind_speed_mph=None."""
        mock_get.return_value = _mock_get_ok(
            _observations_response([
                {"stid": "S1", "temp_f": 68.5, "humidity": 55.0, "wind_mph": None},
            ])
        )
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1"], now)
        assert "S1" in results
        assert results["S1"]["wind_speed_mph"] is None

    @patch("src.synoptic_client.requests.get")
    def test_empty_stids_returns_empty(self, mock_get):
        """Empty station list → empty result, no API call."""
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations([], now)
        assert results == {}
        mock_get.assert_not_called()

    @patch("src.synoptic_client.requests.get")
    def test_missing_date_time_skipped(self, mock_get):
        """Observation with no date_time → excluded."""
        mock_get.return_value = _mock_get_ok({
            "SUMMARY": {"RESPONSE_CODE": 1, "NUMBER_OF_OBJECTS": 1},
            "STATION": [
                {
                    "STID": "S1",
                    "OBSERVATIONS": {
                        "air_temp_value_1": {"value": 68.5},  # no date_time
                    },
                },
            ],
        })
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        results = client._fetch_batch_observations(["S1"], now)
        assert len(results) == 0

    @patch("src.synoptic_client.requests.get")
    def test_api_error_raises_synoptic_error(self, mock_get):
        """API failure during batch fetch → SynopticError."""
        mock_get.return_value = _mock_get_fail(500)
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        with pytest.raises(SynopticError):
            client._fetch_batch_observations(["S1"], now)


# ------------------------------------------------------------------
# _fetch_and_walk — Batch + LKG Cache Walk
# ------------------------------------------------------------------


class TestFetchAndWalk:
    """_fetch_and_walk collects target valid readings from batch + cache."""

    def _make_client_with_stations(self, stations: list[dict]) -> SynopticClient:
        """Create a client with pre-loaded station metadata."""
        client = SynopticClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {
                "stid": s["stid"],
                "name": s.get("name", f"Station {s['stid']}"),
                "mnet_id": s.get("mnet_id", "1"),
                "distance_miles": s.get("distance", 1.0),
            }
            for s in stations
        ]
        return client

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_three_fresh_readings(self, mock_batch):
        """3 stations all return fresh data → 3 results, no cache used."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
            {"stid": "S2", "distance": 2.0},
            {"stid": "S3", "distance": 3.0},
        ])

        mock_batch.return_value = {
            "S1": {"station_id": "S1", "temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
            "S2": {"station_id": "S2", "temperature_f": 72.0, "humidity": 55.0, "wind_speed_mph": 7.0, "timestamp": now},
            "S3": {"station_id": "S3", "temperature_f": 74.0, "humidity": 60.0, "wind_speed_mph": 9.0, "timestamp": now},
        }

        results = client._fetch_and_walk(client._stations, now, target=3)
        assert len(results) == 3
        assert not any(r.get("is_cached") for r in results)

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_one_station_missing_temp_skipped(self, mock_batch):
        """One station missing from batch (bad temp) → 2 valid results."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
            {"stid": "S2", "distance": 2.0},
            {"stid": "S3", "distance": 3.0},
        ])

        mock_batch.return_value = {
            "S1": {"station_id": "S1", "temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
            # S2 missing (bad temp or stale)
            "S3": {"station_id": "S3", "temperature_f": 74.0, "humidity": 60.0, "wind_speed_mph": 9.0, "timestamp": now},
        }

        results = client._fetch_and_walk(client._stations, now, target=3)
        assert len(results) == 2
        stids = [r["station_id"] for r in results]
        assert "S1" in stids
        assert "S3" in stids

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_stops_at_target(self, mock_batch):
        """If target=2 and 5 valid stations → collects only 2."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": f"S{i}", "distance": float(i)} for i in range(1, 6)
        ])

        mock_batch.return_value = {
            f"S{i}": {
                "station_id": f"S{i}", "temperature_f": 70.0 + i,
                "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now,
            }
            for i in range(1, 6)
        }

        results = client._fetch_and_walk(client._stations, now, target=2)
        assert len(results) == 2

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_lkg_cache_used_on_failure(self, mock_batch):
        """Batch returns nothing for S1, but S1 has valid cache → uses cache."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
        ])

        # Pre-populate LKG cache
        client._station_cache["S1"] = {
            "station_id": "S1",
            "temperature_f": 72.0,
            "humidity": 50.0,
            "wind_speed_mph": 5.0,
            "timestamp": now - timedelta(minutes=30),
        }

        mock_batch.return_value = {}  # S1 not in batch results

        results = client._fetch_and_walk(client._stations, now, target=3)
        assert len(results) == 1
        assert results[0]["temperature_f"] == 72.0
        assert results[0].get("is_cached") is True

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_lkg_cache_expired_not_used(self, mock_batch):
        """Cache older than 2 hours → not used."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
        ])

        # Expired cache (3 hours old)
        client._station_cache["S1"] = {
            "station_id": "S1",
            "temperature_f": 72.0,
            "humidity": 50.0,
            "wind_speed_mph": 5.0,
            "timestamp": now - timedelta(hours=3),
        }

        mock_batch.return_value = {}

        results = client._fetch_and_walk(client._stations, now, target=3)
        assert len(results) == 0

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_mixed_fresh_and_cached(self, mock_batch):
        """S1 fresh from batch, S2 from cache → both used."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
            {"stid": "S2", "distance": 2.0},
        ])

        # S2 has LKG cache
        client._station_cache["S2"] = {
            "station_id": "S2",
            "temperature_f": 74.0,
            "humidity": 55.0,
            "wind_speed_mph": 8.0,
            "timestamp": now - timedelta(minutes=15),
        }

        mock_batch.return_value = {
            "S1": {"station_id": "S1", "temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
        }

        results = client._fetch_and_walk(client._stations, now, target=3)
        assert len(results) == 2
        fresh = [r for r in results if not r.get("is_cached")]
        cached = [r for r in results if r.get("is_cached")]
        assert len(fresh) == 1
        assert len(cached) == 1

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_fresh_reading_stored_in_cache(self, mock_batch):
        """Fresh reading from batch → stored in _station_cache for future LKG."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
        ])

        mock_batch.return_value = {
            "S1": {"station_id": "S1", "temperature_f": 72.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
        }

        client._fetch_and_walk(client._stations, now, target=3)
        assert "S1" in client._station_cache
        assert client._station_cache["S1"]["temperature_f"] == 72.0

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_batch_api_failure_falls_back_to_cache(self, mock_batch):
        """Batch fetch raises SynopticError → falls back to LKG cache."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
        ])

        client._station_cache["S1"] = {
            "station_id": "S1",
            "temperature_f": 71.0,
            "humidity": 48.0,
            "wind_speed_mph": 6.0,
            "timestamp": now - timedelta(minutes=20),
        }

        mock_batch.side_effect = SynopticError("API down")

        results = client._fetch_and_walk(client._stations, now, target=3)
        assert len(results) == 1
        assert results[0].get("is_cached") is True

    @patch.object(SynopticClient, "_fetch_batch_observations")
    def test_all_stations_fail_empty_list(self, mock_batch):
        """No batch results, no cache → empty list."""
        now = datetime.now(timezone.utc)
        client = self._make_client_with_stations([
            {"stid": "S1", "distance": 1.0},
            {"stid": "S2", "distance": 2.0},
        ])
        mock_batch.return_value = {}
        results = client._fetch_and_walk(client._stations, now, target=3)
        assert results == []


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
        result = SynopticClient._aggregate(obs)
        assert result["temperature_f"] == 75.0
        assert result["humidity"] == 50.0
        assert result["wind_speed_mph"] == 10.0
        assert result["station_count"] == 3

    def test_median_is_not_mean(self):
        """With outlier, median differs from mean."""
        obs = [
            {"temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0},
            {"temperature_f": 72.0, "humidity": 50.0, "wind_speed_mph": 5.0},
            {"temperature_f": 120.0, "humidity": 50.0, "wind_speed_mph": 5.0},
        ]
        result = SynopticClient._aggregate(obs)
        assert result["temperature_f"] == 72.0

    def test_median_of_two(self):
        """Two observations → average (median of two)."""
        obs = [
            {"temperature_f": 70.0, "humidity": 40.0, "wind_speed_mph": 5.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = SynopticClient._aggregate(obs)
        assert result["temperature_f"] == 75.0

    def test_single_observation(self):
        obs = [{"temperature_f": 72.5, "humidity": 55.0, "wind_speed_mph": 8.0}]
        result = SynopticClient._aggregate(obs)
        assert result["temperature_f"] == 72.5
        assert result["station_count"] == 1

    def test_missing_humidity_excluded(self):
        """Observations with None humidity excluded from humidity median."""
        obs = [
            {"temperature_f": 70.0, "humidity": None, "wind_speed_mph": 5.0},
            {"temperature_f": 75.0, "humidity": 50.0, "wind_speed_mph": 10.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = SynopticClient._aggregate(obs)
        assert result["humidity"] == 55.0

    def test_all_humidity_none(self):
        obs = [{"temperature_f": 70.0, "humidity": None, "wind_speed_mph": 5.0}]
        result = SynopticClient._aggregate(obs)
        assert result["humidity"] is None

    def test_missing_wind_excluded(self):
        obs = [
            {"temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": None},
            {"temperature_f": 75.0, "humidity": 50.0, "wind_speed_mph": 10.0},
        ]
        result = SynopticClient._aggregate(obs)
        assert result["wind_speed_mph"] == 10.0

    def test_is_fallback_always_false(self):
        """Synoptic aggregation always returns is_fallback=False."""
        obs = [{"temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0}]
        result = SynopticClient._aggregate(obs)
        assert result["is_fallback"] is False


# ------------------------------------------------------------------
# get_outdoor_conditions() — Full Integration
# ------------------------------------------------------------------


class TestGetOutdoorConditions:
    """End-to-end: discover → batch fetch → aggregate."""

    @patch.object(SynopticClient, "_fetch_batch_observations")
    @patch.object(SynopticClient, "discover_stations")
    def test_happy_path(self, mock_discover, mock_batch):
        """3 nearby stations → median returned with source='synoptic'."""
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        client._stations = [
            {"stid": "S1", "mnet_id": "1", "distance_miles": 1.0, "name": "S1"},
            {"stid": "S2", "mnet_id": "1", "distance_miles": 2.0, "name": "S2"},
            {"stid": "S3", "mnet_id": "1", "distance_miles": 3.0, "name": "S3"},
        ]

        mock_batch.return_value = {
            "S1": {"station_id": "S1", "temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
            "S2": {"station_id": "S2", "temperature_f": 75.0, "humidity": 55.0, "wind_speed_mph": 8.0, "timestamp": now},
            "S3": {"station_id": "S3", "temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 12.0, "timestamp": now},
        }

        result = client.get_outdoor_conditions()

        assert result["temperature_f"] == 75.0  # median
        assert result["source"] == "synoptic"
        assert result["is_fallback"] is False
        assert result["station_count"] == 3
        assert result["used_cache"] is False

    @patch.object(SynopticClient, "_fetch_batch_observations")
    @patch.object(SynopticClient, "discover_stations")
    def test_used_cache_flag_true(self, mock_discover, mock_batch):
        """used_cache=True when a cached reading contributes."""
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        client._stations = [
            {"stid": "S1", "mnet_id": "1", "distance_miles": 1.0, "name": "S1"},
            {"stid": "S2", "mnet_id": "1", "distance_miles": 2.0, "name": "S2"},
        ]

        # S2 has LKG cache
        client._station_cache["S2"] = {
            "station_id": "S2",
            "temperature_f": 74.0,
            "humidity": 55.0,
            "wind_speed_mph": 8.0,
            "timestamp": now - timedelta(minutes=10),
        }

        mock_batch.return_value = {
            "S1": {"station_id": "S1", "temperature_f": 72.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
        }

        result = client.get_outdoor_conditions()
        assert result["used_cache"] is True

    @patch.object(SynopticClient, "_fetch_batch_observations")
    @patch.object(SynopticClient, "discover_stations")
    def test_no_observations_raises(self, mock_discover, mock_batch):
        """Zero valid observations → SynopticError."""
        client = SynopticClient(_LAT, _LON, _API_KEY)
        client._stations = [
            {"stid": "S1", "mnet_id": "1", "distance_miles": 1.0, "name": "S1"},
        ]
        mock_batch.return_value = {}

        with pytest.raises(SynopticError, match="No valid weather observations"):
            client.get_outdoor_conditions()

    @patch.object(SynopticClient, "_fetch_batch_observations")
    @patch.object(SynopticClient, "discover_stations")
    def test_distance_cap_filters_far_stations(self, mock_discover, mock_batch):
        """Stations beyond 10 miles are excluded from candidate pool."""
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        client._stations = [
            {"stid": "NEAR", "mnet_id": "1", "distance_miles": 2.0, "name": "NEAR"},
            {"stid": "FAR", "mnet_id": "1", "distance_miles": 15.0, "name": "FAR"},
        ]

        mock_batch.return_value = {
            "NEAR": {"station_id": "NEAR", "temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
        }

        result = client.get_outdoor_conditions()
        assert result["station_count"] == 1
        # Verify batch was called only with the near station
        call_stids = mock_batch.call_args[0][0]
        assert "NEAR" in call_stids
        assert "FAR" not in call_stids

    @patch.object(SynopticClient, "_fetch_batch_observations")
    @patch.object(SynopticClient, "discover_stations")
    def test_all_beyond_distance_uses_all(self, mock_discover, mock_batch):
        """When ALL stations are beyond 10 mi, falls back to using all of them."""
        client = SynopticClient(_LAT, _LON, _API_KEY)
        now = datetime.now(timezone.utc)
        client._stations = [
            {"stid": "FAR1", "mnet_id": "1", "distance_miles": 12.0, "name": "FAR1"},
            {"stid": "FAR2", "mnet_id": "1", "distance_miles": 15.0, "name": "FAR2"},
        ]

        mock_batch.return_value = {
            "FAR1": {"station_id": "FAR1", "temperature_f": 70.0, "humidity": 50.0, "wind_speed_mph": 5.0, "timestamp": now},
            "FAR2": {"station_id": "FAR2", "temperature_f": 74.0, "humidity": 55.0, "wind_speed_mph": 8.0, "timestamp": now},
        }

        result = client.get_outdoor_conditions()
        assert result["station_count"] == 2
        # Both stations used despite being > 10mi
        call_stids = mock_batch.call_args[0][0]
        assert "FAR1" in call_stids
        assert "FAR2" in call_stids


# ------------------------------------------------------------------
# Format Age Helper
# ------------------------------------------------------------------


class TestFormatAge:
    """_format_age helper for log messages."""

    def test_minutes_only(self):
        assert SynopticClient._format_age(timedelta(minutes=15)) == "15m ago"

    def test_hours_and_minutes(self):
        assert SynopticClient._format_age(timedelta(hours=1, minutes=30)) == "1h 30m ago"

    def test_zero_minutes(self):
        assert SynopticClient._format_age(timedelta(seconds=0)) == "0m ago"
