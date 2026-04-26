"""Tests for the NWS weather client.

Validates design decisions:
- Station classification heuristic: 4-char K-prefix = official ASOS/AWOS,
  anything else = personal/cooperative.
- Median aggregation of temperature, humidity, wind speed.
- Fallback logic: <2 valid personal station readings → fall back to
  nearest official station(s).
- Celsius→Fahrenheit and km/h→mph conversions.
- Stale observation rejection (>30 min old).
- NWSError on network / API failures.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.nws_client import NWSClient, NWSError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _obs(station_id, temp_c, humidity=50.0, wind_kmh=10.0, age_minutes=5):
    """Build a mock NWS observation response for _fetch_single_observation."""
    ts = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    return {
        "properties": {
            "timestamp": ts,
            "temperature": {"value": temp_c, "unitCode": "wmoUnit:degC"},
            "relativeHumidity": {"value": humidity},
            "windSpeed": {"value": wind_kmh, "unitCode": "wmoUnit:km_h-1"},
        }
    }


def _stations_response(station_ids):
    """Build a mock NWS stations list response."""
    features = []
    for sid in station_ids:
        features.append({
            "properties": {"stationIdentifier": sid, "name": f"Station {sid}"}
        })
    return {"features": features}


# ------------------------------------------------------------------
# Station Classification Heuristic
# ------------------------------------------------------------------


class TestStationClassification:
    """K-prefix 4-char = official. Everything else = personal."""

    @pytest.mark.parametrize(
        "station_id, expected_personal",
        [
            ("KORD", False),     # 4-char K-prefix → official (Chicago O'Hare)
            ("KJFK", False),     # Official airport
            ("KATL", False),     # Official airport
            ("CW1234", True),    # CWOP personal
            ("AW5678", True),    # Personal
            ("EW9012", True),    # Personal
            ("COOP123", True),   # Cooperative observer
            ("CRS456", True),    # Cooperative remote sensing
            ("AS789", True),     # Other non-K non-4-char
        ],
    )
    def test_is_personal_station(self, station_id, expected_personal):
        """Heuristic: 4-char K-prefix = official, all else = personal."""
        result = NWSClient._is_personal_station({"id": station_id})
        assert result is expected_personal

    def test_k_prefix_non_4_char_is_personal(self):
        """K-prefix but 5+ chars → personal (not standard ICAO)."""
        assert NWSClient._is_personal_station({"id": "KTEST5"}) is True

    def test_empty_id_is_personal(self):
        """Empty string → personal (not K-prefix 4-char)."""
        assert NWSClient._is_personal_station({"id": ""}) is True


# ------------------------------------------------------------------
# Unit Conversions
# ------------------------------------------------------------------


class TestConversions:
    """Celsius→Fahrenheit and km/h→mph."""

    @pytest.mark.parametrize(
        "celsius, fahrenheit",
        [
            (0.0, 32.0),
            (100.0, 212.0),
            (-40.0, -40.0),
            (20.0, 68.0),
        ],
    )
    def test_c_to_f(self, celsius, fahrenheit):
        assert NWSClient._c_to_f(celsius) == pytest.approx(fahrenheit)

    @pytest.mark.parametrize(
        "kmh, mph",
        [
            (0.0, 0.0),
            (100.0, 62.1371),
            (16.0934, 10.0),  # ~10 mph
        ],
    )
    def test_kmh_to_mph(self, kmh, mph):
        assert NWSClient._kmh_to_mph(kmh) == pytest.approx(mph, rel=1e-3)


# ------------------------------------------------------------------
# Median Aggregation
# ------------------------------------------------------------------


class TestMedianAggregation:
    """Median of temperature, humidity, wind from multiple stations."""

    def test_median_of_three_temps(self):
        """Three observations → median temperature."""
        obs = [
            {"temperature_f": 70.0, "humidity": 40.0, "wind_speed_mph": 5.0},
            {"temperature_f": 75.0, "humidity": 50.0, "wind_speed_mph": 10.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = NWSClient._aggregate(obs, is_fallback=False)
        assert result["temperature_f"] == 75.0
        assert result["humidity"] == 50.0
        assert result["wind_speed_mph"] == 10.0
        assert result["station_count"] == 3
        assert result["is_fallback"] is False

    def test_median_of_two_temps(self):
        """Two observations → median (average of two)."""
        obs = [
            {"temperature_f": 70.0, "humidity": 40.0, "wind_speed_mph": 5.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = NWSClient._aggregate(obs, is_fallback=False)
        assert result["temperature_f"] == 75.0

    def test_single_observation(self):
        """One observation → that value is the median."""
        obs = [{"temperature_f": 72.5, "humidity": 55.0, "wind_speed_mph": 8.0}]
        result = NWSClient._aggregate(obs, is_fallback=True)
        assert result["temperature_f"] == 72.5
        assert result["is_fallback"] is True

    def test_missing_humidity_excluded(self):
        """Observations with None humidity are excluded from humidity median."""
        obs = [
            {"temperature_f": 70.0, "humidity": None, "wind_speed_mph": 5.0},
            {"temperature_f": 75.0, "humidity": 50.0, "wind_speed_mph": 10.0},
            {"temperature_f": 80.0, "humidity": 60.0, "wind_speed_mph": 15.0},
        ]
        result = NWSClient._aggregate(obs, is_fallback=False)
        assert result["humidity"] == 55.0  # median of [50, 60]

    def test_all_humidity_none(self):
        """All humidity None → result humidity is None."""
        obs = [
            {"temperature_f": 70.0, "humidity": None, "wind_speed_mph": 5.0},
        ]
        result = NWSClient._aggregate(obs, is_fallback=False)
        assert result["humidity"] is None


# ------------------------------------------------------------------
# Fallback Logic
# ------------------------------------------------------------------


class TestFallbackLogic:
    """<2 valid personal readings → fall back to official stations."""

    @patch.object(NWSClient, "_fetch_single_observation")
    @patch.object(NWSClient, "discover_stations")
    def test_fallback_when_no_personal_readings(self, mock_discover, mock_fetch):
        """0 personal station readings → falls back to official."""
        client = NWSClient(40.0, -74.0)
        # Set up internal station list with 2 personal (returning None) + 2 official
        client._stations = [
            {"id": "CW001", "name": "P1", "is_personal": True},
            {"id": "CW002", "name": "P2", "is_personal": True},
            {"id": "KORD", "name": "O1", "is_personal": False},
            {"id": "KJFK", "name": "O2", "is_personal": False},
        ]

        now = datetime.now(timezone.utc)
        official_obs = {
            "station_id": "KORD",
            "temperature_f": 72.0,
            "humidity": 50.0,
            "wind_speed_mph": 10.0,
            "timestamp": now,
        }

        def fetch_side_effect(sid, _now):
            if sid.startswith("CW"):
                return None  # personal stations fail
            return official_obs

        mock_fetch.side_effect = fetch_side_effect

        result = client.get_outdoor_conditions()
        assert result["is_fallback"] is True
        assert result["temperature_f"] == 72.0

    @patch.object(NWSClient, "_fetch_single_observation")
    @patch.object(NWSClient, "discover_stations")
    def test_no_fallback_when_enough_personal(self, mock_discover, mock_fetch):
        """3 valid personal readings → no fallback needed."""
        client = NWSClient(40.0, -74.0)
        client._stations = [
            {"id": "CW001", "name": "P1", "is_personal": True},
            {"id": "CW002", "name": "P2", "is_personal": True},
            {"id": "CW003", "name": "P3", "is_personal": True},
            {"id": "KORD", "name": "O1", "is_personal": False},
        ]

        now = datetime.now(timezone.utc)
        personal_obs = [
            {"station_id": "CW001", "temperature_f": 70.0, "humidity": 45.0, "wind_speed_mph": 5.0, "timestamp": now},
            {"station_id": "CW002", "temperature_f": 72.0, "humidity": 50.0, "wind_speed_mph": 8.0, "timestamp": now},
            {"station_id": "CW003", "temperature_f": 74.0, "humidity": 55.0, "wind_speed_mph": 10.0, "timestamp": now},
        ]

        call_count = [0]

        def fetch_side_effect(sid, _now):
            if sid.startswith("CW"):
                obs = personal_obs[call_count[0]]
                call_count[0] += 1
                return obs
            return None

        mock_fetch.side_effect = fetch_side_effect

        result = client.get_outdoor_conditions()
        assert result["is_fallback"] is False
        assert result["temperature_f"] == 72.0  # median of [70, 72, 74]

    @patch.object(NWSClient, "_fetch_single_observation")
    @patch.object(NWSClient, "discover_stations")
    def test_fallback_with_one_personal_reading(self, mock_discover, mock_fetch):
        """Only 1 personal reading (<2) → triggers fallback, combines both."""
        client = NWSClient(40.0, -74.0)
        client._stations = [
            {"id": "CW001", "name": "P1", "is_personal": True},
            {"id": "CW002", "name": "P2", "is_personal": True},
            {"id": "KORD", "name": "O1", "is_personal": False},
        ]

        now = datetime.now(timezone.utc)

        def fetch_side_effect(sid, _now):
            if sid == "CW001":
                return {"station_id": "CW001", "temperature_f": 70.0, "humidity": 45.0, "wind_speed_mph": 5.0, "timestamp": now}
            if sid == "KORD":
                return {"station_id": "KORD", "temperature_f": 74.0, "humidity": 55.0, "wind_speed_mph": 10.0, "timestamp": now}
            return None

        mock_fetch.side_effect = fetch_side_effect

        result = client.get_outdoor_conditions()
        assert result["is_fallback"] is True
        # Combined observations: CW001(70) + KORD(74) → median = 72.0
        assert result["temperature_f"] == 72.0

    @patch.object(NWSClient, "_fetch_single_observation")
    @patch.object(NWSClient, "discover_stations")
    def test_no_observations_at_all_raises(self, mock_discover, mock_fetch):
        """Zero valid observations from any station → NWSError."""
        client = NWSClient(40.0, -74.0)
        client._stations = [
            {"id": "CW001", "name": "P1", "is_personal": True},
            {"id": "KORD", "name": "O1", "is_personal": False},
        ]
        mock_fetch.return_value = None

        with pytest.raises(NWSError, match="No valid weather observations"):
            client.get_outdoor_conditions()


# ------------------------------------------------------------------
# Observation Staleness
# ------------------------------------------------------------------


class TestObservationStaleness:
    """Observations older than 30 minutes are rejected."""

    @patch("src.nws_client.requests.get")
    def test_stale_observation_rejected(self, mock_get):
        """41-minute-old observation → returns None."""
        client = NWSClient(40.0, -74.0)
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _obs("KORD", temp_c=20.0, age_minutes=41),
        )
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KORD", now)
        assert result is None

    @patch("src.nws_client.requests.get")
    def test_fresh_observation_accepted(self, mock_get):
        """5-minute-old observation → valid dict returned."""
        client = NWSClient(40.0, -74.0)
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _obs("KORD", temp_c=20.0, age_minutes=5),
        )
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KORD", now)
        assert result is not None
        assert result["temperature_f"] == pytest.approx(68.0)

    @patch("src.nws_client.requests.get")
    def test_missing_timestamp_rejected(self, mock_get):
        """Observation with no timestamp → None."""
        client = NWSClient(40.0, -74.0)
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {"properties": {"temperature": {"value": 20.0}}},
        )
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KORD", now)
        assert result is None

    @patch("src.nws_client.requests.get")
    def test_missing_temperature_rejected(self, mock_get):
        """Observation with null temperature → None."""
        ts = datetime.now(timezone.utc).isoformat()
        client = NWSClient(40.0, -74.0)
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {
                "properties": {
                    "timestamp": ts,
                    "temperature": {"value": None},
                    "relativeHumidity": {"value": 50.0},
                }
            },
        )
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KORD", now)
        assert result is None


# ------------------------------------------------------------------
# Station Discovery
# ------------------------------------------------------------------


class TestStationDiscovery:
    """Station discovery via NWS points → stations chain."""

    @patch("src.nws_client.requests.get")
    def test_discover_stations_caches(self, mock_get):
        """Second call returns cached results without extra API calls."""
        points_resp = MagicMock(
            ok=True,
            json=lambda: {
                "properties": {
                    "observationStations": "https://api.weather.gov/gridpoints/X/1,2/stations",
                }
            },
        )
        stations_resp = MagicMock(
            ok=True,
            json=lambda: _stations_response(["KORD", "CW001"]),
        )
        mock_get.side_effect = [points_resp, stations_resp]

        client = NWSClient(40.0, -74.0)
        ids1 = client.discover_stations()
        ids2 = client.discover_stations()

        assert ids1 == ids2
        assert mock_get.call_count == 2  # only initial discovery calls

    @patch("src.nws_client.requests.get")
    def test_discover_classifies_stations(self, mock_get):
        """Discovered stations are classified as personal or official."""
        points_resp = MagicMock(
            ok=True,
            json=lambda: {"properties": {"observationStations": "https://api.weather.gov/test"}},
        )
        stations_resp = MagicMock(
            ok=True,
            json=lambda: _stations_response(["KORD", "CW001", "KJFK", "EW002"]),
        )
        mock_get.side_effect = [points_resp, stations_resp]

        client = NWSClient(40.0, -74.0)
        client.discover_stations()

        personal_count = sum(1 for s in client._stations if s["is_personal"])
        official_count = sum(1 for s in client._stations if not s["is_personal"])
        assert personal_count == 2  # CW001, EW002
        assert official_count == 2  # KORD, KJFK
