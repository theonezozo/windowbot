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

import json
import statistics
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.nws_client import NWSClient, NWSError, _DEFAULT_CONTRIBUTORS_PATH


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


# ==================================================================
# Phase 2 — per-contributor logging + source stickiness
# ==================================================================

# Fixed single-clock poll timestamp shared by the Phase 2 tests so age math is
# deterministic (the literal CURRENT_DATETIME of this task).
_NOW = datetime(2026, 7, 11, 13, 40, 0, tzinfo=timezone.utc)

# Every documented key of a ``_fetch_batch`` attempt dict.
_ATTEMPT_KEYS = {
    "station_id", "source_type", "station_class", "temp_f", "obs_time",
    "age_minutes", "distance_mi", "outcome", "excluded_reason", "is_cached",
}


def _istation(sid, dist, *, personal=True):
    """Build an internal ``self._stations`` entry for _fetch_batch/get_outdoor."""
    return {"id": sid, "name": sid, "is_personal": personal, "distance_miles": dist}


def _iobs(sid, temp_f, *, now=_NOW, age_minutes=5, humidity=50.0, wind=5.0):
    """Build the internal observation dict returned by _fetch_single_observation."""
    return {
        "station_id": sid,
        "temperature_f": temp_f,
        "humidity": humidity,
        "wind_speed_mph": wind,
        "timestamp": now - timedelta(minutes=age_minutes),
    }


class TestFetchBatchContract:
    """``_fetch_batch`` returns ``(results, batch_stats, attempts)`` and emits
    one fully-populated attempt entry per examined station (checklist #3)."""

    def test_returns_three_tuple_with_all_attempt_keys(self):
        """3-tuple; one attempt per station; all documented keys; single clock."""
        client = NWSClient(40.0, -74.0)
        stations = [
            _istation("KAAA", 1.0, personal=False),
            _istation("CW1", 2.0),
            _istation("CW2", 3.0),
        ]

        def fetch(sid, _now):
            client._last_skip_reason = None
            return _iobs(sid, 70.0, age_minutes=5)

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            results, batch_stats, attempts = client._fetch_batch(stations, _NOW, target=3)

        assert len(results) == 3
        assert isinstance(batch_stats, dict)
        assert batch_stats["checked"] == 3
        assert batch_stats["fresh"] == 3
        assert len(attempts) == 3
        for att in attempts:
            assert set(att.keys()) == _ATTEMPT_KEYS
            assert att["source_type"] == "nws_station"
            # age_minutes is measured against the single passed `now`.
            assert att["age_minutes"] == 5.0
            assert att["outcome"] == "included"
            assert att["excluded_reason"] is None

        by_id = {a["station_id"]: a for a in attempts}
        assert by_id["KAAA"]["station_class"] == "official"
        assert by_id["CW1"]["station_class"] == "personal"
        assert by_id["KAAA"]["distance_mi"] == 1.0
        assert by_id["CW2"]["distance_mi"] == 3.0

    def test_age_minutes_uses_passed_now_not_wall_clock(self):
        """All ages derive from the passed ``now`` — never datetime.now()."""
        client = NWSClient(40.0, -74.0)
        stations = [_istation("CW1", 1.0)]

        def fetch(sid, _now):
            client._last_skip_reason = None
            return _iobs(sid, 70.0, age_minutes=12)

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _results, _stats, attempts = client._fetch_batch(stations, _NOW, target=3)

        assert attempts[0]["age_minutes"] == 12.0
        assert attempts[0]["obs_time"] == (_NOW - timedelta(minutes=12)).isoformat()


class TestFetchBatchReasonMapping:
    """Outcome / excluded_reason mapping for every branch (checklist #4)."""

    def test_fresh_reading_included_reason_none(self):
        client = NWSClient(40.0, -74.0)

        def fetch(sid, _now):
            client._last_skip_reason = None
            return _iobs(sid, 71.0)

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch([_istation("CW1", 1.0)], _NOW, target=3)
        assert attempts[0]["outcome"] == "included"
        assert attempts[0]["excluded_reason"] is None

    def test_stale_reading_maps_to_stale(self):
        client = NWSClient(40.0, -74.0)

        def fetch(sid, _now):
            client._last_skip_reason = "stale (25m ago)"
            return None

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch([_istation("CW1", 1.0)], _NOW, target=3)
        assert attempts[0]["outcome"] == "excluded"
        assert attempts[0]["excluded_reason"] == "stale"

    def test_api_error_maps_to_api_error(self):
        client = NWSClient(40.0, -74.0)

        def fetch(sid, _now):
            raise NWSError("boom")

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch([_istation("CW1", 1.0)], _NOW, target=3)
        assert attempts[0]["excluded_reason"] == "api_error"

    def test_no_temperature_maps_to_no_data(self):
        client = NWSClient(40.0, -74.0)

        def fetch(sid, _now):
            client._last_skip_reason = "no temperature"
            return None

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch([_istation("CW1", 1.0)], _NOW, target=3)
        assert attempts[0]["excluded_reason"] == "no_data"

    def test_cache_within_window_with_fresh_present_is_unused(self):
        """Cached reading available but a fresh one exists → not used, logged
        ``cached_available_but_unused`` (never dilutes a fresh pool)."""
        client = NWSClient(40.0, -74.0)
        client._station_cache["CW2"] = _iobs("CW2", 65.0, age_minutes=30)

        def fetch(sid, _now):
            client._last_skip_reason = None
            if sid == "KAAA":
                return _iobs("KAAA", 70.0, age_minutes=4)
            client._last_skip_reason = "stale (40m ago)"
            return None

        stations = [_istation("KAAA", 1.0, personal=False), _istation("CW2", 2.0)]
        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            results, _s, attempts = client._fetch_batch(stations, _NOW, target=3)

        assert len(results) == 1  # only the fresh KAAA
        cw2 = next(a for a in attempts if a["station_id"] == "CW2")
        assert cw2["is_cached"] is True
        assert cw2["outcome"] == "excluded"
        assert cw2["excluded_reason"] == "cached_available_but_unused"
        assert cw2["temp_f"] == 65.0  # raw cached temp still carried

    def test_cache_expired_maps_to_cache_expired(self):
        client = NWSClient(40.0, -74.0)
        client._station_cache["CW1"] = _iobs("CW1", 60.0, age_minutes=130)  # > 2h

        def fetch(sid, _now):
            client._last_skip_reason = "stale (130m ago)"
            return None

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch([_istation("CW1", 1.0)], _NOW, target=3)
        assert attempts[0]["excluded_reason"] == "cache_expired"
        assert attempts[0]["is_cached"] is False

    def test_lkg_fallback_promotes_cached_to_included(self):
        """Zero fresh readings → cached LKG promoted to included/None."""
        client = NWSClient(40.0, -74.0)
        client._station_cache["CW1"] = _iobs("CW1", 63.0, age_minutes=45)

        def fetch(sid, _now):
            client._last_skip_reason = "stale (45m ago)"
            return None

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            results, _s, attempts = client._fetch_batch([_istation("CW1", 1.0)], _NOW, target=3)

        assert len(results) == 1
        assert results[0]["is_cached"] is True
        assert attempts[0]["outcome"] == "included"
        assert attempts[0]["excluded_reason"] is None
        assert attempts[0]["is_cached"] is True


class TestFetchBatchPriorityIds:
    """``priority_ids`` forces sticky stations to be attempted beyond the
    nearest-`target`; empty/None keeps the legacy walk (checklist #5)."""

    @staticmethod
    def _five_fresh_client():
        client = NWSClient(40.0, -74.0)
        stations = [_istation(f"S{i}", float(i)) for i in range(1, 6)]

        def fetch(sid, _now):
            client._last_skip_reason = None
            return _iobs(sid, 70.0)

        return client, stations, fetch

    def test_priority_id_attempted_beyond_target(self):
        client, stations, fetch = self._five_fresh_client()
        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch(
                stations, _NOW, target=3, priority_ids={"S5"},
            )
        attempted = [a["station_id"] for a in attempts]
        assert attempted == ["S1", "S2", "S3", "S5"]  # S4 skipped, S5 forced

    def test_no_priority_stops_at_target(self):
        client, stations, fetch = self._five_fresh_client()
        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch(stations, _NOW, target=3)
        attempted = [a["station_id"] for a in attempts]
        assert attempted == ["S1", "S2", "S3"]  # legacy: stops once target met

    def test_empty_priority_set_stops_at_target(self):
        client, stations, fetch = self._five_fresh_client()
        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            _r, _s, attempts = client._fetch_batch(
                stations, _NOW, target=3, priority_ids=set(),
            )
        assert [a["station_id"] for a in attempts] == ["S1", "S2", "S3"]


class TestSelectMedianPool:
    """``_select_median_pool`` — legacy passthrough & cold-start (checklist
    #10, #11)."""

    def test_disabled_returns_legacy_nearest_first_pool(self):
        """stickiness_enabled=False → selection bit-for-bit identical to input."""
        client = NWSClient(40.0, -74.0)
        fresh = [_iobs("A", 70.0), _iobs("B", 71.0), _iobs("C", 72.0)]
        selected, active, sticky_id, excluded = client._select_median_pool(
            fresh, ["A", "B"], enabled=False, target=3,
        )
        assert selected == fresh
        assert active is False
        assert sticky_id is None
        assert excluded == set()

    def test_cold_start_no_sticky_ids_passthrough_inactive(self):
        """No prior ids (None or empty) → passthrough, stickiness inactive."""
        client = NWSClient(40.0, -74.0)
        fresh = [_iobs("A", 70.0), _iobs("B", 71.0)]
        for sticky in (None, []):
            selected, active, sticky_id, excluded = client._select_median_pool(
                fresh, sticky, enabled=True, target=3,
            )
            assert selected == fresh
            assert active is False
            assert sticky_id is None
            assert excluded == set()


class TestStickinessGetOutdoorConditions:
    """End-to-end stickiness selection + contributor_log (checklist #8, #9,
    #12). ``_fetch_single_observation`` is mocked; no network."""

    def _client_with_stations(self, stations):
        client = NWSClient(40.0, -74.0)
        client._stations = stations
        return client

    @staticmethod
    def _entry(clog, sid):
        return next(c for c in clog["contributors"] if c["station_id"] == sid)

    def test_fresh_sticky_retained_over_closer_newcomer(self):
        """Sticky sources stay in the median even when a nearer newcomer is
        fresh; the excluded newcomer is logged with its RAW temp + the
        ``stickiness_not_selected`` reason (checklist #8, #12)."""
        stations = [
            _istation("D", 1.0),   # closest newcomer
            _istation("A", 2.0),
            _istation("B", 3.0),
            _istation("C", 4.0),
        ]
        client = self._client_with_stations(stations)
        temps = {"A": 70.0, "B": 71.0, "C": 72.0, "D": 60.0}

        def fetch(sid, _now):
            client._last_skip_reason = None
            return _iobs(sid, temps[sid])

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            result = client.get_outdoor_conditions(
                sticky_source_ids=["A", "B", "C"], stickiness_enabled=True,
            )

        # Median is over the retained sticky pool {A,B,C} = 71, NOT the
        # nearest-first {D,A,B} = 70 — stickiness damps the newcomer's pull.
        assert result["temperature_f"] == 71.0

        clog = result["contributor_log"]
        assert clog["selected_source_ids"] == ["A", "B", "C"]
        assert clog["stickiness_active"] is True
        assert clog["sticky_source_id"] == "A"

        d = self._entry(clog, "D")
        assert d["included_in_median"] is False
        assert d["excluded_reason"] == "stickiness_not_selected"
        # The critical measurability property: an excluded-yet-fresh source
        # still carries its raw reading so its accuracy stays analyzable.
        assert d["temp_f"] == 60.0
        for sid in ("A", "B", "C"):
            assert self._entry(clog, sid)["included_in_median"] is True

    def test_stale_sticky_drops_out_newcomer_backfills(self):
        """A stale sticky source drops; a newcomer backfills to target
        (checklist #9)."""
        stations = [
            _istation("D", 1.0),
            _istation("A", 2.0),   # sticky, but stale this cycle
            _istation("B", 3.0),
            _istation("C", 4.0),
        ]
        client = self._client_with_stations(stations)
        temps = {"B": 71.0, "C": 72.0, "D": 60.0}

        def fetch(sid, _now):
            client._last_skip_reason = None
            if sid == "A":
                client._last_skip_reason = "stale (25m ago)"
                return None
            return _iobs(sid, temps[sid])

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            result = client.get_outdoor_conditions(
                sticky_source_ids=["A", "B", "C"], stickiness_enabled=True,
            )

        clog = result["contributor_log"]
        assert clog["selected_source_ids"] == ["B", "C", "D"]  # A gone, D in
        assert clog["stickiness_active"] is False  # no fresh source held back
        assert self._entry(clog, "D")["included_in_median"] is True
        a = self._entry(clog, "A")
        assert a["included_in_median"] is False
        assert a["excluded_reason"] == "stale"

    def test_openmeteo_prior_id_excluded_from_nws_priority(self):
        """A prior ``OPENMETEO`` id in ``sticky_source_ids`` is never attempted
        as an NWS station — the peer blend is unchanged and OPENMETEO is dropped
        from the NWS priority set (checklist #13, exclusion layer)."""
        stations = [
            _istation("A", 1.0),
            _istation("B", 2.0),
            _istation("C", 3.0),
        ]
        client = self._client_with_stations(stations)
        temps = {"A": 70.0, "B": 71.0, "C": 72.0}
        attempted: list[str] = []

        def fetch(sid, _now):
            client._last_skip_reason = None
            attempted.append(sid)
            return _iobs(sid, temps[sid])

        with patch.object(NWSClient, "_fetch_single_observation", side_effect=fetch):
            result = client.get_outdoor_conditions(
                sticky_source_ids=["OPENMETEO", "A"], stickiness_enabled=True,
            )

        # OPENMETEO is never fetched as an NWS station.
        assert "OPENMETEO" not in attempted
        clog = result["contributor_log"]
        assert "OPENMETEO" not in clog["selected_source_ids"]



class TestRecordContributorLog:
    """``NWSClient._record_contributor_log`` — JSONL schema, best-effort I/O,
    and path resolution (checklist #6, #7, #2)."""

    @staticmethod
    def _sample_record():
        return {
            "type": "outdoor_contributors",
            "schema_version": 1,
            "timestamp": "2026-07-11T13:40:00+00:00",
            "poll_id": "abcdef123456",
            "contributors": [
                {
                    "station_id": "KAAA",
                    "source_type": "nws_station",
                    "station_class": "official",
                    "temp_f": 70.0,
                    "obs_time": "2026-07-11T13:35:00+00:00",
                    "age_minutes": 5.0,
                    "distance_mi": 1.0,
                    "included_in_median": True,
                    "is_cached": False,
                    "excluded_reason": None,
                }
            ],
            "median_temp_f": 70.0,
            "real_station_count": 1,
            "openmeteo_present": False,
            "openmeteo_included": False,
            "used_cache_fallback": False,
            "is_fallback": False,
            "source": "nws",
            "selected_source_ids": ["KAAA"],
            "stickiness_active": False,
            "sticky_source_id": None,
            "validation_reason": "cold_start",
            "suppressed": False,
            "raw_temp_f": 70.0,
            "validated_temp_f": 70.0,
        }

    def test_writes_exactly_one_wellformed_json_line(self, tmp_path, monkeypatch):
        """One JSON line carrying every top-level + contributor schema key."""
        path = tmp_path / "contrib.jsonl"
        monkeypatch.setenv("WINDOWBOT_CONTRIBUTORS_PATH", str(path))

        NWSClient._record_contributor_log(self._sample_record())

        lines = path.read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])

        top_level = {
            "type", "schema_version", "timestamp", "poll_id", "contributors",
            "median_temp_f", "real_station_count", "openmeteo_present",
            "openmeteo_included", "used_cache_fallback", "is_fallback", "source",
            "selected_source_ids", "stickiness_active", "sticky_source_id",
            "validation_reason", "suppressed", "raw_temp_f", "validated_temp_f",
        }
        assert top_level <= set(parsed.keys())

        contrib = parsed["contributors"][0]
        assert set(contrib.keys()) == {
            "station_id", "source_type", "station_class", "temp_f", "obs_time",
            "age_minutes", "distance_mi", "included_in_median", "is_cached",
            "excluded_reason",
        }

    def test_swallows_io_error_without_raising(self, tmp_path, monkeypatch):
        """An un-writable path (missing parent dir) must never raise."""
        bad = tmp_path / "missing_dir" / "contrib.jsonl"
        monkeypatch.setenv("WINDOWBOT_CONTRIBUTORS_PATH", str(bad))

        # Must not raise despite the parent directory not existing.
        NWSClient._record_contributor_log(self._sample_record())
        assert not bad.exists()

    def test_default_path_is_module_constant(self, tmp_path, monkeypatch):
        """Unset env → writes to the default ``outdoor_contributors.jsonl``
        (checklist #2)."""
        assert _DEFAULT_CONTRIBUTORS_PATH == "outdoor_contributors.jsonl"
        monkeypatch.delenv("WINDOWBOT_CONTRIBUTORS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)

        NWSClient._record_contributor_log(self._sample_record())
        assert (tmp_path / "outdoor_contributors.jsonl").exists()

    def test_env_override_path_honored(self, tmp_path, monkeypatch):
        """``WINDOWBOT_CONTRIBUTORS_PATH`` override is honored (checklist #2)."""
        custom = tmp_path / "custom_contributors.jsonl"
        monkeypatch.setenv("WINDOWBOT_CONTRIBUTORS_PATH", str(custom))
        monkeypatch.chdir(tmp_path)

        NWSClient._record_contributor_log(self._sample_record())
        assert custom.exists()
        assert not (tmp_path / "outdoor_contributors.jsonl").exists()
