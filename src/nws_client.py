"""National Weather Service API client for WindowBot.

Fetches outdoor temperature, humidity, and wind speed from nearby NWS
weather stations.  Blends personal/cooperative and official stations into
a single list sorted by distance, then takes the **median** of the 3
closest stations for robustness.

Reference: https://www.weather.gov/documentation/services-web-api
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger("windowbot.nws")

NWS_API_BASE = "https://api.weather.gov"
REQUIRED_HEADERS = {
    "User-Agent": "(WindowBot, contact@example.com)",
    "Accept": "application/geo+json",
}

# Observations older than this are rejected as stale.
_MAX_OBS_AGE = timedelta(minutes=30)
_REQUEST_TIMEOUT = 15
# Cached readings older than this are not used (hard eviction threshold).
_CACHE_MAX_AGE = timedelta(hours=2)
# Only consider stations within this radius; prevents distant outliers from
# bloating the search and keeps results hyperlocal.
_MAX_STATION_DISTANCE_MILES = 10.0

# Station identifier prefixes that indicate personal/cooperative networks.
# CRS = Cooperative Remote Sensing; COOP = Cooperative Observer Program.
_PERSONAL_STATION_PREFIXES = ("CRS", "COOP")


class NWSError(Exception):
    """Raised on unrecoverable NWS API errors."""


class NWSClient:
    """Fetches outdoor weather data from NWS stations.

    Args:
        latitude: User's latitude in decimal degrees.
        longitude: User's longitude in decimal degrees.
    """

    def __init__(self, latitude: float, longitude: float) -> None:
        self._lat = latitude
        self._lon = longitude
        self._stations: list[dict] = []  # cached station metadata
        self._last_skip_reason: str | None = None  # set by _fetch_single_observation
        self._station_cache: dict[str, dict] = {}  # last-known-good readings, keyed by station ID
        self._grid_id: str | None = None
        self._grid_x: int | None = None
        self._grid_y: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get(url: str) -> dict:
        """GET a URL with required NWS headers."""
        try:
            resp = requests.get(url, headers=REQUIRED_HEADERS, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise NWSError(f"Network error fetching {url}: {exc}") from exc

        if not resp.ok:
            raise NWSError(f"NWS API error ({resp.status_code}) for {url}: {resp.text[:300]}")

        return resp.json()

    @staticmethod
    def _c_to_f(celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return celsius * 9.0 / 5.0 + 32.0

    @staticmethod
    def _kmh_to_mph(kmh: float) -> float:
        """Convert km/h to mph."""
        return kmh * 0.621371

    @staticmethod
    def _format_age(delta: timedelta) -> str:
        """Format a timedelta as 'Xh Ym ago' or 'Ym ago'."""
        total_secs = int(delta.total_seconds())
        hours, remainder = divmod(total_secs, 3600)
        minutes = remainder // 60
        if hours > 0:
            return f"{hours}h {minutes:02d}m ago"
        return f"{minutes}m ago"

    @staticmethod
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two points in miles."""
        R = 3958.8  # Earth radius in miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _is_personal_station(station: dict) -> bool:
        """Heuristic: a station is "personal/cooperative" if its identifier
        does NOT start with K (ASOS/AWOS airport) and is not a 4-char ICAO code."""
        sid = station.get("id", "")
        # Official US stations are typically 4-char ICAO codes starting with K
        # Personal/cooperative stations tend to have longer alphanumeric IDs.
        if len(sid) == 4 and sid.startswith("K"):
            return False
        return True

    # ------------------------------------------------------------------
    # Station discovery
    # ------------------------------------------------------------------

    def discover_stations(self) -> list[str]:
        """Discover nearby weather stations via the NWS points → gridpoints
        → stations chain.

        Returns:
            List of station identifiers sorted by distance (closest first).
            Results are cached for subsequent calls.
        """
        if self._stations:
            return [s["id"] for s in self._stations]

        if self._grid_id is not None:
            # Gridpoint already cached — skip the points API entirely.
            stations_url = f"{NWS_API_BASE}/gridpoints/{self._grid_id}/{self._grid_x},{self._grid_y}/stations"
        else:
            # Step 1: Resolve grid coordinates from lat/lon.
            points_url = f"{NWS_API_BASE}/points/{self._lat},{self._lon}"
            points_data = self._get(points_url)
            props = points_data.get("properties", {})

            stations_url = props.get("observationStations")
            grid_id = props.get("gridId")
            grid_x = props.get("gridX")
            grid_y = props.get("gridY")
            if all([grid_id, grid_x is not None, grid_y is not None]):
                self._grid_id = grid_id
                self._grid_x = grid_x
                self._grid_y = grid_y
                logger.debug("Resolved NWS gridpoint: %s/%s,%s", self._grid_id, self._grid_x, self._grid_y)
            if not stations_url:
                if self._grid_id is None:
                    raise NWSError("Could not determine grid coordinates from NWS points API.")
                stations_url = f"{NWS_API_BASE}/gridpoints/{self._grid_id}/{self._grid_x},{self._grid_y}/stations"

        # Step 2: Fetch stations (already sorted by distance from the grid point).
        stations_data = self._get(stations_url)
        features = stations_data.get("features", [])

        self._stations = []
        for feat in features:
            sprops = feat.get("properties", {})
            sid = sprops.get("stationIdentifier", "")
            name = sprops.get("name", "")
            # GeoJSON coordinates: [longitude, latitude]
            coords = feat.get("geometry", {}).get("coordinates", [])
            stn_lon = coords[0] if len(coords) >= 2 else None
            stn_lat = coords[1] if len(coords) >= 2 else None
            dist = (
                self._haversine_miles(self._lat, self._lon, stn_lat, stn_lon)
                if (stn_lat is not None and stn_lon is not None)
                else float("inf")
            )
            self._stations.append(
                {
                    "id": sid,
                    "name": name,
                    "is_personal": self._is_personal_station({"id": sid}),
                    "lat": stn_lat,
                    "lon": stn_lon,
                    "distance_miles": dist,
                }
            )

        ids = [s["id"] for s in self._stations]
        logger.info(
            "Discovered %d stations (%d personal).",
            len(ids),
            sum(1 for s in self._stations if s["is_personal"]),
        )
        return ids

    # ------------------------------------------------------------------
    # Observation fetching
    # ------------------------------------------------------------------

    def get_observations(self, max_stations: int = 3) -> list[dict]:
        """Fetch the latest observation from each of the nearest stations.

        Observations older than 30 minutes are rejected.

        Args:
            max_stations: Maximum number of stations to query.

        Returns:
            List of valid observation dicts, each containing:
            - ``station_id`` (str)
            - ``temperature_f`` (float)
            - ``humidity`` (float)
            - ``wind_speed_mph`` (float)
            - ``timestamp`` (datetime)
        """
        self.discover_stations()

        station_ids = [s["id"] for s in self._stations[:max_stations]]
        now = datetime.now(timezone.utc)
        observations: list[dict] = []

        for sid in station_ids:
            try:
                obs = self._fetch_single_observation(sid, now)
                if obs is not None:
                    observations.append(obs)
            except NWSError:
                logger.warning("Failed to fetch observation for station %s.", sid, exc_info=True)

        logger.info("Got %d valid observations from %d queried stations.", len(observations), len(station_ids))
        return observations

    def _fetch_single_observation(self, station_id: str, now: datetime) -> dict | None:
        """Fetch and validate a single station's most recent fresh observation.

        Queries ``/observations?limit=5`` rather than ``/observations/latest``
        because the ``latest`` endpoint is aggressively cached and frequently
        lags the underlying observation list by 30+ minutes.  We walk the
        returned features newest-first and accept the first one that is both
        fresh (within ``_MAX_OBS_AGE``) and has a temperature value.

        Sets ``self._last_skip_reason`` to a human-readable string whenever
        returning ``None`` so callers can log the specific rejection cause.
        """
        self._last_skip_reason = None
        url = f"{NWS_API_BASE}/stations/{station_id}/observations?limit=5"
        data = self._get(url)

        # The list endpoint returns a FeatureCollection.  Tolerate the legacy
        # single-properties shape too so test fixtures and any future fallback
        # to /observations/latest keep working.
        features = data.get("features")
        if features is None and "properties" in data:
            features = [{"properties": data["properties"]}]
        if not features:
            self._last_skip_reason = "no observations returned"
            return None

        # Parse + sort newest-first; the API usually returns this order already
        # but we don't want to rely on it.
        candidates: list[tuple[datetime, dict]] = []
        for feat in features:
            props = feat.get("properties", {})
            ts_str = props.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            candidates.append((ts, props))

        if not candidates:
            self._last_skip_reason = "no timestamp"
            return None

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        newest_ts = candidates[0][0]

        # Walk newest-first; prefer a fresh reading that actually has a temp.
        props: dict | None = None
        ts: datetime | None = None
        for cand_ts, cand_props in candidates:
            age = now - cand_ts
            if age > _MAX_OBS_AGE:
                break  # remaining candidates are older still
            if cand_props.get("temperature", {}).get("value") is None:
                continue
            ts, props = cand_ts, cand_props
            break

        if props is None:
            newest_age = now - newest_ts
            if newest_age > _MAX_OBS_AGE:
                self._last_skip_reason = f"stale ({self._format_age(newest_age)})"
            else:
                self._last_skip_reason = "no temperature"
            return None

        # Parse temperature (Celsius → Fahrenheit).
        temp_c = props.get("temperature", {}).get("value")
        temperature_f = self._c_to_f(float(temp_c))

        # Parse humidity.
        hum_raw = props.get("relativeHumidity", {})
        humidity = hum_raw.get("value")
        if humidity is not None:
            humidity = float(humidity)

        # Parse wind speed (km/h → mph).
        wind_raw = props.get("windSpeed", {})
        wind_kmh = wind_raw.get("value")
        wind_mph: float | None = None
        if wind_kmh is not None:
            wind_mph = self._kmh_to_mph(float(wind_kmh))

        return {
            "station_id": station_id,
            "temperature_f": round(temperature_f, 1),
            "humidity": round(humidity, 1) if humidity is not None else None,
            "wind_speed_mph": round(wind_mph, 1) if wind_mph is not None else None,
            "timestamp": ts,
        }

    # ------------------------------------------------------------------
    # Aggregated outdoor conditions
    # ------------------------------------------------------------------

    def _station_distance_key(self, station: dict) -> float:
        """Return the pre-computed distance (miles) for sorting; inf if unknown."""
        return station.get("distance_miles", float("inf"))

    def get_outdoor_conditions(
        self, peer_observations: list[dict] | None = None
    ) -> dict:
        """Compute aggregated outdoor conditions using the MEDIAN of the
        closest weather stations, blending personal, official, and optional
        peer sources (e.g., Open-Meteo) together.

        Strategy:
            1. Sort all discovered stations by distance from target coordinates.
            2. Query the 3 closest stations (regardless of type).
            3. Append any fresh ``peer_observations`` to the pool.
            4. Return the **median** temperature, humidity, and wind speed.

        Args:
            peer_observations: Optional list of pre-validated NWS-compatible
                observation dicts (e.g., from Open-Meteo) to blend into the
                median.  Each must have ``station_id``, ``temperature_f``,
                ``humidity``, ``wind_speed_mph``, and ``timestamp``.

        Returns:
            Dict with keys:
            - ``temperature_f`` (float): Median outdoor temperature in °F.
            - ``humidity`` (float | None): Median outdoor humidity %.
            - ``wind_speed_mph`` (float | None): Median wind speed.
            - ``station_count`` (int): Number of stations contributing.
            - ``is_fallback`` (bool): True if any official station contributed,
              or if no NWS stations contributed (peer-only).
        """
        self.discover_stations()

        # Blend all stations, sort by distance, then cap to those within the
        # configured radius so distant outliers don't bloat the search.
        sorted_stations = sorted(self._stations, key=self._station_distance_key)
        nearby = [
            s for s in sorted_stations
            if self._station_distance_key(s) <= _MAX_STATION_DISTANCE_MILES
        ]
        if not nearby:
            logger.warning(
                "No stations within %.0f mi — falling back to all %d stations.",
                _MAX_STATION_DISTANCE_MILES, len(sorted_stations),
            )
            nearby = sorted_stations
        else:
            logger.info(
                "Station pool: %d within %.0f mi (of %d total).",
                len(nearby), _MAX_STATION_DISTANCE_MILES, len(sorted_stations),
            )
        stations_by_id = {s["id"]: s for s in self._stations}

        now = datetime.now(timezone.utc)
        nws_observations, batch_stats = self._fetch_batch(nearby, now, target=3)

        # Blend in any fresh peer observations (e.g., Open-Meteo grid point).
        all_observations = list(nws_observations)
        if peer_observations:
            for peer_obs in peer_observations:
                all_observations.append(peer_obs)
                logger.info(
                    "  peer  %-8s                             → ✓ %.1f°F",
                    peer_obs.get("station_id", "PEER"), peer_obs["temperature_f"],
                )

        if not all_observations:
            raise NWSError("No valid weather observations available from any station.")

        # is_fallback: True if any official (non-personal) NWS station contributed,
        # or if no NWS stations contributed at all (peer-only scenario).
        peer_ids = {o.get("station_id") for o in (peer_observations or [])}
        if not nws_observations:
            is_fallback = True
        else:
            is_fallback = any(
                not stations_by_id.get(o["station_id"], {}).get("is_personal", True)
                for o in nws_observations
            )
        used_cache = any(o.get("is_cached", False) for o in nws_observations)

        result = self._aggregate(all_observations, is_fallback)
        result["used_cache"] = used_cache
        result["source"] = "nws"
        logger.info(
            "Outdoor temperature: %.1f°F (median of %d readings%s)",
            result["temperature_f"],
            len(all_observations),
            ", some from cache" if used_cache else "",
        )
        self._record_freshness_metric(now, batch_stats, result["temperature_f"])
        return result

    def _fetch_batch(
        self, stations: list[dict], now: datetime, target: int
    ) -> list[dict]:
        """Walk *stations* in order, fetching until *target* valid readings
        are collected or the list is exhausted.

        Logs one INFO line per station examined:
            ``#N  SID  (type, dist) → ✓ XX.X°F``  or  ``→ ✗ reason``
            ``#N  SID  (type, dist) → ⚠ cached (last good: Xh Ym ago)``
        followed by a summary line.

        Valid observations are stored in ``self._station_cache`` for use as
        last-known-good (LKG) fallbacks on future calls within the same run.
        """
        results: list[dict] = []
        cached_results: list[dict] = []
        checked = 0
        fresh_count = 0
        cached_available = 0

        for stn in stations:
            if len(results) >= target:
                break

            checked += 1
            sid = stn["id"]
            stype = "personal" if stn.get("is_personal", True) else "official"
            dist = stn.get("distance_miles", float("inf"))
            dist_str = f"{dist:4.1f} mi" if dist != float("inf") else " ??? mi"

            self._last_skip_reason = None
            api_error: NWSError | None = None
            obs: dict | None = None
            try:
                obs = self._fetch_single_observation(sid, now)
            except NWSError as exc:
                api_error = exc

            if obs is not None:
                age = now - obs["timestamp"]
                age_str = self._format_age(age)
                logger.info(
                    "  #%-2d  %-8s  (%-8s %s) → ✓ %.1f°F  (%s)",
                    checked, sid, stype + ",", dist_str, obs["temperature_f"], age_str,
                )
                self._station_cache[sid] = obs
                fresh_count += 1
                results.append(obs)
                continue

            # Station was skipped — try the LKG cache.
            cached = self._station_cache.get(sid)
            if cached is not None:
                cache_age = now - cached["timestamp"]
                if cache_age <= _CACHE_MAX_AGE:
                    cached_obs = {**cached, "is_cached": True}
                    age_str = self._format_age(cache_age)
                    logger.info(
                        "  #%-2d  %-8s  (%-8s %s) → ⚠ %.1f°F  cached (last good: %s)",
                        checked, sid, stype + ",", dist_str, cached["temperature_f"], age_str,
                    )
                    cached_available += 1
                    cached_results.append(cached_obs)
                    continue
                else:
                    logger.info(
                        "  #%-2d  %-8s  (%-8s %s) → ✗ no recent data (cache expired)",
                        checked, sid, stype + ",", dist_str,
                    )
                    continue

            # No cache entry — log original rejection reason.
            if api_error is not None:
                logger.info(
                    "  #%-2d  %-8s  (%-8s %s) → ✗ API error: %s",
                    checked, sid, stype + ",", dist_str, api_error,
                )
            else:
                reason = self._last_skip_reason or "unknown"
                logger.info(
                    "  #%-2d  %-8s  (%-8s %s) → ✗ %s",
                    checked, sid, stype + ",", dist_str, reason,
                )

        # Stale/cached readings must NEVER dilute the median when fresh readings
        # exist. Only fall back to LKG cache if zero fresh readings were found.
        used_cache_fallback = False
        if not results and cached_results:
            results = cached_results
            used_cache_fallback = True
            logger.info(
                "No fresh readings found; falling back to %d cached LKG reading%s.",
                len(cached_results), "s" if len(cached_results) != 1 else "",
            )

        cached_used = len(cached_results) if used_cache_fallback else 0
        if cached_available and not used_cache_fallback:
            logger.info(
                "Station search complete: %d valid reading%s "
                "(%d fresh, %d cached available but unused — fresh readings preferred) "
                "from %d station%s checked.",
                len(results), "s" if len(results) != 1 else "",
                fresh_count, cached_available,
                checked, "s" if checked != 1 else "",
            )
        else:
            logger.info(
                "Station search complete: %d valid reading%s (%d fresh, %d cached) from %d station%s checked.",
                len(results), "s" if len(results) != 1 else "",
                fresh_count, cached_used,
                checked, "s" if checked != 1 else "",
            )
        batch_stats = {
            "checked": checked,
            "fresh": fresh_count,
            "cached": cached_used,
            "cached_available": cached_available,
            "valid": len(results),
        }
        return results, batch_stats

    @staticmethod
    def _aggregate(observations: list[dict], is_fallback: bool) -> dict:
        """Compute MEDIAN values from a list of observations."""
        temps = [o["temperature_f"] for o in observations]
        humidities = [o["humidity"] for o in observations if o["humidity"] is not None]
        winds = [o["wind_speed_mph"] for o in observations if o["wind_speed_mph"] is not None]

        return {
            "temperature_f": round(statistics.median(temps), 1),
            "humidity": round(statistics.median(humidities), 1) if humidities else None,
            "wind_speed_mph": round(statistics.median(winds), 1) if winds else None,
            "station_count": len(observations),
            "is_fallback": is_fallback,
        }

    def _record_freshness_metric(
        self, now: datetime, batch_stats: dict, median_temp_f: float
    ) -> None:
        """Record per-cycle freshness metrics to a JSONL file.

        Metrics are appended to the file specified by WINDOWBOT_METRICS_PATH
        (default: nws_freshness_metrics.jsonl in cwd). Never raises — metric
        writes are best-effort and silent on failure.
        """
        try:
            import json
            import os
            metrics_path = os.environ.get("WINDOWBOT_METRICS_PATH", "nws_freshness_metrics.jsonl")
            checked = batch_stats["checked"]
            record = {
                "timestamp": now.isoformat(),
                "checked": checked,
                "fresh": batch_stats["fresh"],
                "cached": batch_stats["cached"],
                "valid": batch_stats["valid"],
                "fresh_pct": round(100 * batch_stats["fresh"] / checked, 1) if checked else 0,
                "median_temp_f": median_temp_f,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Never let metrics break the main flow
            logger.debug("Could not write freshness metric", exc_info=True)
