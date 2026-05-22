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


# ------------------------------------------------------------------
# Outdoor Freshness Boundary (20-min cutoff — Jacob's audit)
# ------------------------------------------------------------------


class TestOutdoorFreshnessBoundary:
    """Pin the 20-minute outdoor staleness cutoff with literal-minute boundary
    tests so a regression on ``_MAX_OBS_AGE`` (e.g., reverting to 30 min) is
    caught immediately. These tests deliberately do NOT import the constant
    — they assert against a literal value to detect drift from the agreed
    contract.
    """

    @patch("src.nws_client.requests.get")
    def test_fetch_single_observation_rejects_at_21_min(self, mock_get):
        """21-minute-old observation → None, skip reason includes 'stale'."""
        client = NWSClient(40.0, -74.0)
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _obs("KORD", temp_c=20.0, age_minutes=21),
        )
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KORD", now)
        assert result is None
        assert client._last_skip_reason is not None
        assert "stale" in client._last_skip_reason

    @patch("src.nws_client.requests.get")
    def test_fetch_single_observation_accepts_at_19_min(self, mock_get):
        """19-minute-old observation → valid dict (just inside the 20-min cap)."""
        client = NWSClient(40.0, -74.0)
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: _obs("KORD", temp_c=20.0, age_minutes=19),
        )
        now = datetime.now(timezone.utc)
        result = client._fetch_single_observation("KORD", now)
        assert result is not None
        assert result["temperature_f"] == pytest.approx(68.0)

    @patch("src.nws_client.requests.get")
    def test_fetch_single_observation_walks_past_no_temp_newest(self, mock_get):
        """Newest two features have null temp; the 15-min fresh-with-temp
        feature is picked (regression guard for the no-temp ``continue`` walk).

        The 25-min stale-with-temp feature must NOT be reached.
        """
        now = datetime.now(timezone.utc)
        ts5 = (now - timedelta(minutes=5)).isoformat()
        ts8 = (now - timedelta(minutes=8)).isoformat()
        ts15 = (now - timedelta(minutes=15)).isoformat()
        ts25 = (now - timedelta(minutes=25)).isoformat()
        client = NWSClient(40.0, -74.0)
        mock_get.return_value = MagicMock(
            ok=True,
            json=lambda: {
                "features": [
                    {"properties": {
                        "timestamp": ts5,
                        "temperature": {"value": None},
                        "relativeHumidity": {"value": 50.0},
                        "windSpeed": {"value": 10.0},
                    }},
                    {"properties": {
                        "timestamp": ts8,
                        "temperature": {"value": None},
                        "relativeHumidity": {"value": 50.0},
                        "windSpeed": {"value": 10.0},
                    }},
                    {"properties": {
                        "timestamp": ts15,
                        "temperature": {"value": 20.0},  # 68°F → picked
                        "relativeHumidity": {"value": 50.0},
                        "windSpeed": {"value": 10.0},
                    }},
                    {"properties": {
                        "timestamp": ts25,
                        "temperature": {"value": 38.0},  # 100°F → must NOT be picked
                        "relativeHumidity": {"value": 50.0},
                        "windSpeed": {"value": 10.0},
                    }},
                ]
            },
        )
        result = client._fetch_single_observation("KORD", now)
        assert result is not None
        assert result["temperature_f"] == pytest.approx(68.0)
        # Timestamp matches the 15-min candidate, not the 25-min one.
        assert result["timestamp"].isoformat() == ts15

    def test_aggregate_filters_stale_defence_in_depth(self):
        """``_aggregate`` re-applies the 20-min cutoff even if upstream forwards
        a mix of fresh and stale observations. The 25-min stale sample must
        not contribute to any returned field.
        """
        now = datetime.now(timezone.utc)
        ts5 = now - timedelta(minutes=5)
        ts7 = now - timedelta(minutes=7)
        ts25 = now - timedelta(minutes=25)
        obs = [
            {"station_id": "A", "temperature_f": 60.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts5},
            {"station_id": "B", "temperature_f": 62.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts7},
            {"station_id": "C", "temperature_f": 100.0, "humidity": 80.0,
             "wind_speed_mph": 25.0, "timestamp": ts25},
        ]
        result = NWSClient._aggregate(obs, is_fallback=False, now=now)
        assert result["temperature_f"] == pytest.approx(61.0)  # median(60, 62)
        assert result["contributor_count"] == 2
        assert result["station_count"] == 2
        # Oldest fresh contributor drives observation_time (worst-case bucket).
        assert result["observation_time"] == ts7.isoformat()
        # 100°F sample is entirely excluded.
        assert result["temperature_f"] != pytest.approx(62.0)

    def test_aggregate_falls_back_when_refilter_empties(self, caplog):
        """When EVERY observation is stale (LKG cache path), ``_aggregate`` must
        not raise — it keeps the original pool and logs a WARNING.
        """
        import logging
        caplog.set_level(logging.WARNING, logger="windowbot.nws")
        now = datetime.now(timezone.utc)
        ts30 = now - timedelta(minutes=30)
        ts35 = now - timedelta(minutes=35)
        ts40 = now - timedelta(minutes=40)
        obs = [
            {"station_id": "A", "temperature_f": 60.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts30},
            {"station_id": "B", "temperature_f": 62.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts35},
            {"station_id": "C", "temperature_f": 64.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts40},
        ]
        result = NWSClient._aggregate(obs, is_fallback=False, now=now)
        assert result["temperature_f"] == pytest.approx(62.0)  # median(60, 62, 64)
        assert result["contributor_count"] == 3
        assert result["station_count"] == 3
        assert any(
            "re-filter removed all readings" in rec.getMessage()
            for rec in caplog.records
        ), f"Expected warning about re-filter; got: {[r.getMessage() for r in caplog.records]}"

    def test_aggregate_newest_observation_time_distinct_from_oldest(self):
        """``newest_observation_time`` is the freshest contributor; the legacy
        ``observation_time`` remains the oldest. They must differ when the
        pool spans a range of ages.
        """
        now = datetime.now(timezone.utc)
        ts3 = now - timedelta(minutes=3)
        ts8 = now - timedelta(minutes=8)
        ts18 = now - timedelta(minutes=18)
        obs = [
            {"station_id": "A", "temperature_f": 60.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts3},
            {"station_id": "B", "temperature_f": 62.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts8},
            {"station_id": "C", "temperature_f": 64.0, "humidity": 50.0,
             "wind_speed_mph": 5.0, "timestamp": ts18},
        ]
        result = NWSClient._aggregate(obs, is_fallback=False, now=now)
        assert result["newest_observation_time"] == ts3.isoformat()
        assert result["observation_time"] == ts18.isoformat()
        assert result["newest_observation_time"] != result["observation_time"]
