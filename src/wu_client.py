"""Weather Underground PWS client for WindowBot.

Fetches outdoor temperature, humidity, and wind speed from nearby
Weather Underground Personal Weather Stations.  Uses a median-of-3
strategy identical to NWSClient for robustness.

Reference: https://docs.google.com/document/d/1eKCnKXI9xnoMGRRzOL1xPCBihNV2rOet08qpE_gArAY
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger("windowbot.wu")

_WU_BASE = "https://api.weather.com"

# Observations older than this are rejected as stale.
_MAX_OBS_AGE = timedelta(minutes=30)
_REQUEST_TIMEOUT = 10
# Cached readings older than this are not used (hard eviction threshold).
_CACHE_MAX_AGE = timedelta(hours=2)
# Only consider stations within this radius.
_MAX_STATION_DISTANCE_MILES = 10.0


class WUError(Exception):
    """Raised on unrecoverable Weather Underground API errors."""


class WUClient:
    """Fetches outdoor weather data from Weather Underground PWS stations.

    Args:
        latitude: User's latitude in decimal degrees.
        longitude: User's longitude in decimal degrees.
        api_key: Weather Underground API key.
    """

    def __init__(self, latitude: float, longitude: float, api_key: str) -> None:
        self._lat = latitude
        self._lon = longitude
        self._api_key = api_key
        self._stations: list[dict] = []
        self._station_cache: dict[str, dict] = {}  # LKG cache keyed by station ID
        self._last_skip_reason: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET a URL with the WU API key."""
        all_params = {"apiKey": self._api_key, "format": "json"}
        if params:
            all_params.update(params)
        try:
            resp = requests.get(url, params=all_params, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise WUError(f"Network error fetching {url}: {exc}") from exc

        if not resp.ok:
            raise WUError(f"WU API error ({resp.status_code}) for {url}: {resp.text[:300]}")

        return resp.json()

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
    def _format_age(delta: timedelta) -> str:
        """Format a timedelta as 'Xh Ym ago' or 'Ym ago'."""
        total_secs = int(delta.total_seconds())
        hours, remainder = divmod(total_secs, 3600)
        minutes = remainder // 60
        if hours > 0:
            return f"{hours}h {minutes:02d}m ago"
        return f"{minutes}m ago"

    # ------------------------------------------------------------------
    # Station discovery
    # ------------------------------------------------------------------

    def discover_stations(self) -> list[str]:
        """Find nearby PWS stations sorted by distance from target.

        Results are cached — only the first call hits the API.

        Returns:
            List of station IDs sorted by distance (closest first).
        """
        if self._stations:
            return [s["id"] for s in self._stations]

        url = f"{_WU_BASE}/v3/location/near"
        params = {
            "geocode": f"{self._lat},{self._lon}",
            "product": "pws",
        }
        data = self._get(url, params)

        # The v3/location/near response contains parallel arrays:
        #   location.stationId[], location.latitude[], location.longitude[],
        #   location.distanceMi[]
        location = data.get("location", {})
        station_ids = location.get("stationId", [])
        latitudes = location.get("latitude", [])
        longitudes = location.get("longitude", [])

        self._stations = []
        for i, sid in enumerate(station_ids):
            stn_lat = latitudes[i] if i < len(latitudes) else None
            stn_lon = longitudes[i] if i < len(longitudes) else None
            dist = (
                self._haversine_miles(self._lat, self._lon, stn_lat, stn_lon)
                if (stn_lat is not None and stn_lon is not None)
                else float("inf")
            )
            self._stations.append({
                "id": sid,
                "lat": stn_lat,
                "lon": stn_lon,
                "distance_miles": dist,
            })

        # Sort by distance (closest first).
        self._stations.sort(key=lambda s: s.get("distance_miles", float("inf")))

        logger.info("Discovered %d WU PWS stations.", len(self._stations))
        return [s["id"] for s in self._stations]

    # ------------------------------------------------------------------
    # Observation fetching
    # ------------------------------------------------------------------

    def _fetch_single_observation(self, station_id: str, now: datetime) -> dict | None:
        """Fetch and validate a single station's current observation.

        WU returns temps in °F already when units=e, so no conversion needed.

        Sets ``self._last_skip_reason`` when returning None.
        """
        self._last_skip_reason = None
        url = f"{_WU_BASE}/v2/pws/observations/current"
        params = {
            "stationId": station_id,
            "numericPrecision": "decimal",
            "units": "e",
        }
        data = self._get(url, params)

        observations = data.get("observations", [])
        if not observations:
            self._last_skip_reason = "no observations in response"
            return None

        obs = observations[0]

        # Validate timestamp.
        ts_str = obs.get("obsTimeUtc")
        if not ts_str:
            self._last_skip_reason = "no timestamp"
            return None

        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            self._last_skip_reason = "invalid timestamp"
            return None

        age = now - ts
        if age > _MAX_OBS_AGE:
            self._last_skip_reason = f"stale ({self._format_age(age)})"
            return None

        # Parse imperial measurements.
        imperial = obs.get("imperial", {})
        temp_f = imperial.get("temp")
        if temp_f is None:
            self._last_skip_reason = "no temperature"
            return None

        humidity = obs.get("humidity")
        if humidity is not None:
            humidity = float(humidity)

        wind_mph = imperial.get("windSpeed")
        if wind_mph is not None:
            wind_mph = float(wind_mph)

        return {
            "station_id": station_id,
            "temperature_f": round(float(temp_f), 1),
            "humidity": round(humidity, 1) if humidity is not None else None,
            "wind_speed_mph": round(wind_mph, 1) if wind_mph is not None else None,
            "timestamp": ts,
        }

    # ------------------------------------------------------------------
    # Aggregated outdoor conditions
    # ------------------------------------------------------------------

    def get_outdoor_conditions(self) -> dict:
        """Compute aggregated outdoor conditions using the MEDIAN of the
        3 closest valid PWS stations.

        Returns:
            Dict with keys matching NWSClient's return format:
            - ``temperature_f`` (float)
            - ``humidity`` (float | None)
            - ``wind_speed_mph`` (float | None)
            - ``station_count`` (int)
            - ``is_fallback`` (bool): Always False for WU (it IS the primary).
            - ``used_cache`` (bool)
            - ``source`` (str): Always ``"wu"``.
        """
        self.discover_stations()

        nearby = [
            s for s in self._stations
            if s.get("distance_miles", float("inf")) <= _MAX_STATION_DISTANCE_MILES
        ]
        if not nearby:
            logger.warning(
                "No WU stations within %.0f mi — using all %d stations.",
                _MAX_STATION_DISTANCE_MILES, len(self._stations),
            )
            nearby = list(self._stations)
        else:
            logger.info(
                "WU station pool: %d within %.0f mi (of %d total).",
                len(nearby), _MAX_STATION_DISTANCE_MILES, len(self._stations),
            )

        now = datetime.now(timezone.utc)
        observations = self._fetch_batch(nearby, now, target=3)

        if not observations:
            raise WUError("No valid weather observations available from any WU station.")

        used_cache = any(o.get("is_cached", False) for o in observations)
        result = self._aggregate(observations)
        result["used_cache"] = used_cache
        result["source"] = "wu"

        logger.info(
            "WU outdoor temperature: %.1f°F (median of %d readings%s)",
            result["temperature_f"],
            len(observations),
            ", some from cache" if used_cache else "",
        )
        return result

    def _fetch_batch(
        self, stations: list[dict], now: datetime, target: int
    ) -> list[dict]:
        """Walk *stations* in order, fetching until *target* valid readings
        are collected or the list is exhausted.

        Mirrors NWSClient._fetch_batch with ✓/✗/⚠ logging and LKG cache.
        """
        results: list[dict] = []
        checked = 0
        fresh_count = 0
        cached_count = 0

        for stn in stations:
            if len(results) >= target:
                break

            checked += 1
            sid = stn["id"]
            dist = stn.get("distance_miles", float("inf"))
            dist_str = f"{dist:4.1f} mi" if dist != float("inf") else " ??? mi"

            self._last_skip_reason = None
            api_error: WUError | None = None
            obs: dict | None = None
            try:
                obs = self._fetch_single_observation(sid, now)
            except WUError as exc:
                api_error = exc

            if obs is not None:
                logger.info(
                    "  #%-2d  %-12s  (%s) → ✓ %.1f°F",
                    checked, sid, dist_str, obs["temperature_f"],
                )
                self._station_cache[sid] = obs
                fresh_count += 1
                results.append(obs)
                continue

            # Station failed — try LKG cache.
            cached = self._station_cache.get(sid)
            if cached is not None:
                cache_age = now - cached["timestamp"]
                if cache_age <= _CACHE_MAX_AGE:
                    cached_obs = {**cached, "is_cached": True}
                    age_str = self._format_age(cache_age)
                    logger.info(
                        "  #%-2d  %-12s  (%s) → ⚠ cached (last good: %s)",
                        checked, sid, dist_str, age_str,
                    )
                    cached_count += 1
                    results.append(cached_obs)
                    continue
                else:
                    logger.info(
                        "  #%-2d  %-12s  (%s) → ✗ no recent data (cache expired)",
                        checked, sid, dist_str,
                    )
                    continue

            if api_error is not None:
                logger.info(
                    "  #%-2d  %-12s  (%s) → ✗ API error: %s",
                    checked, sid, dist_str, api_error,
                )
            else:
                reason = self._last_skip_reason or "unknown"
                logger.info(
                    "  #%-2d  %-12s  (%s) → ✗ %s",
                    checked, sid, dist_str, reason,
                )

        logger.info(
            "WU station search complete: %d valid reading%s (%d fresh, %d cached) from %d station%s checked.",
            len(results), "s" if len(results) != 1 else "",
            fresh_count, cached_count,
            checked, "s" if checked != 1 else "",
        )
        return results

    @staticmethod
    def _aggregate(observations: list[dict]) -> dict:
        """Compute MEDIAN values from a list of observations."""
        temps = [o["temperature_f"] for o in observations]
        humidities = [o["humidity"] for o in observations if o["humidity"] is not None]
        winds = [o["wind_speed_mph"] for o in observations if o["wind_speed_mph"] is not None]

        return {
            "temperature_f": round(statistics.median(temps), 1),
            "humidity": round(statistics.median(humidities), 1) if humidities else None,
            "wind_speed_mph": round(statistics.median(winds), 1) if winds else None,
            "station_count": len(observations),
            "is_fallback": False,
        }
