"""MesoWest / Synoptic Data weather client for WindowBot.

Aggregates ASOS, AWOS, RAWS, and mesonet stations via the Synoptic Data API.
More reliable than NWS's own observation endpoint for the same stations, and
fetches multiple stations in a SINGLE API call (huge for the 5K req/mo free tier).

Reference: https://synopticdata.com/mesonet-api
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger("windowbot.synoptic")

_API_BASE = "https://api.synopticdata.com/v2"
_REQUEST_TIMEOUT = 15

# Observations older than this are rejected as stale.
_MAX_OBS_AGE = timedelta(minutes=60)
# Cached readings older than this are not used (hard eviction threshold).
_CACHE_MAX_AGE = timedelta(hours=2)
# Only consider stations within this radius.
_MAX_STATION_DISTANCE_MILES = 10.0

# Network IDs for station type classification.
# MNET_ID 1 = NWS/ASOS, 2 = RAWS, etc.  IDs ≤ 2 are "official" networks.
_OFFICIAL_NETWORK_IDS = {"1", "2"}


class SynopticError(Exception):
    """Raised on unrecoverable Synoptic API errors."""


class SynopticClient:
    """MesoWest/Synoptic Data weather client.

    Aggregates ASOS, AWOS, RAWS, and mesonet stations via the Synoptic API.
    More reliable than NWS's own observation endpoint for the same stations.

    Args:
        latitude: User's latitude in decimal degrees.
        longitude: User's longitude in decimal degrees.
        api_key: Synoptic Data API token.
    """

    def __init__(self, latitude: float, longitude: float, api_key: str) -> None:
        self._lat = latitude
        self._lon = longitude
        self._api_key = api_key
        self._stations: list[dict] = []
        self._station_cache: dict[str, dict] = {}  # LKG cache keyed by STID

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET a URL with the Synoptic API token."""
        all_params = {"token": self._api_key}
        if params:
            all_params.update(params)
        try:
            resp = requests.get(url, params=all_params, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise SynopticError(f"Network error fetching {url}: {exc}") from exc

        if not resp.ok:
            raise SynopticError(
                f"Synoptic API error ({resp.status_code}) for {url}: {resp.text[:300]}"
            )

        data = resp.json()

        # Synoptic uses SUMMARY.RESPONSE_CODE — 1 = OK, anything else = error.
        summary = data.get("SUMMARY", {})
        resp_code = summary.get("RESPONSE_CODE")
        if resp_code is not None and resp_code != 1:
            msg = summary.get("RESPONSE_MESSAGE", "unknown error")
            raise SynopticError(f"Synoptic API error (code {resp_code}): {msg}")

        return data

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

    @staticmethod
    def _is_official(station: dict) -> bool:
        """True if a station belongs to an official network (NWS/ASOS, RAWS)."""
        return station.get("mnet_id", "") in _OFFICIAL_NETWORK_IDS

    # ------------------------------------------------------------------
    # Station discovery
    # ------------------------------------------------------------------

    def discover_stations(self) -> list[str]:
        """Find nearby stations using /stations/nearest.

        Results are cached — only the first call hits the API.

        Returns:
            List of station IDs (STIDs) sorted by distance (closest first).
        """
        if self._stations:
            return [s["stid"] for s in self._stations]

        url = f"{_API_BASE}/stations/nearest"
        params = {
            "lat": str(self._lat),
            "lon": str(self._lon),
            "radius": "10",
            "limit": "20",
            "status": "active",
            "vars": "air_temp",
        }
        data = self._get(url, params)

        raw_stations = data.get("STATION", [])

        self._stations = []
        for raw in raw_stations:
            stid = raw.get("STID", "")
            name = raw.get("NAME", "")
            mnet_id = str(raw.get("MNET_ID", ""))

            # Distance is provided directly by the API (miles from query point).
            try:
                dist = float(raw.get("DISTANCE", float("inf")))
            except (ValueError, TypeError):
                dist = float("inf")

            self._stations.append({
                "stid": stid,
                "name": name,
                "mnet_id": mnet_id,
                "distance_miles": dist,
            })

        # Already sorted by distance from the API, but sort explicitly to be safe.
        self._stations.sort(key=lambda s: s.get("distance_miles", float("inf")))

        official = sum(1 for s in self._stations if self._is_official(s))
        logger.info(
            "Discovered %d Synoptic stations (%d official, %d mesonet).",
            len(self._stations), official, len(self._stations) - official,
        )
        return [s["stid"] for s in self._stations]

    # ------------------------------------------------------------------
    # Batch observation fetch (single API call!)
    # ------------------------------------------------------------------

    def _fetch_batch_observations(
        self, stids: list[str], now: datetime
    ) -> dict[str, dict]:
        """Fetch observations for multiple stations in ONE API call.

        Uses /stations/latest with comma-separated station IDs.
        This is the key efficiency advantage over WU/NWS (1 request vs N).

        Returns:
            Dict mapping STID → validated observation dict (or absent if invalid).
        """
        if not stids:
            return {}

        url = f"{_API_BASE}/stations/latest"
        params = {
            "stid": ",".join(stids),
            "vars": "air_temp,relative_humidity,wind_speed",
            "units": "english",
            "within": "60",
        }

        data = self._get(url, params)
        raw_stations = data.get("STATION", [])

        results: dict[str, dict] = {}
        for raw in raw_stations:
            stid = raw.get("STID", "")
            obs_block = raw.get("OBSERVATIONS", {})

            # Parse temperature.
            temp_entry = obs_block.get("air_temp_value_1")
            if not temp_entry or temp_entry.get("value") is None:
                continue

            temp_f = float(temp_entry["value"])

            # Validate timestamp.
            ts_str = temp_entry.get("date_time")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            age = now - ts
            if age > _MAX_OBS_AGE:
                continue

            # Parse humidity (optional).
            humidity: float | None = None
            rh_entry = obs_block.get("relative_humidity_value_1")
            if rh_entry and rh_entry.get("value") is not None:
                humidity = float(rh_entry["value"])

            # Parse wind speed (optional).
            wind_mph: float | None = None
            wind_entry = obs_block.get("wind_speed_value_1")
            if wind_entry and wind_entry.get("value") is not None:
                wind_mph = float(wind_entry["value"])

            results[stid] = {
                "station_id": stid,
                "temperature_f": round(temp_f, 1),
                "humidity": round(humidity, 1) if humidity is not None else None,
                "wind_speed_mph": round(wind_mph, 1) if wind_mph is not None else None,
                "timestamp": ts,
            }

        return results

    # ------------------------------------------------------------------
    # Aggregated outdoor conditions
    # ------------------------------------------------------------------

    def get_outdoor_conditions(self) -> dict:
        """Fetch observations from nearest 3 stations.

        KEY OPTIMIZATION: Uses a single /stations/latest call with
        comma-separated station IDs instead of individual calls per station.
        This keeps API usage within the free tier (5K req/mo).

        Returns:
            Dict matching the standard format:
            {
                "temperature_f": float,
                "humidity": float | None,
                "wind_speed_mph": float | None,
                "station_count": int,
                "is_fallback": False,
                "used_cache": bool,
                "source": "synoptic",
            }
        """
        self.discover_stations()

        nearby = [
            s for s in self._stations
            if s.get("distance_miles", float("inf")) <= _MAX_STATION_DISTANCE_MILES
        ]
        if not nearby:
            logger.warning(
                "No Synoptic stations within %.0f mi — using all %d stations.",
                _MAX_STATION_DISTANCE_MILES, len(self._stations),
            )
            nearby = list(self._stations)
        else:
            logger.info(
                "Synoptic station pool: %d within %.0f mi (of %d total).",
                len(nearby), _MAX_STATION_DISTANCE_MILES, len(self._stations),
            )

        now = datetime.now(timezone.utc)
        observations = self._fetch_and_walk(nearby, now, target=3)

        if not observations:
            raise SynopticError(
                "No valid weather observations available from any Synoptic station."
            )

        used_cache = any(o.get("is_cached", False) for o in observations)
        result = self._aggregate(observations)
        result["used_cache"] = used_cache
        result["source"] = "synoptic"

        logger.info(
            "Synoptic outdoor temperature: %.1f°F (median of %d readings%s)",
            result["temperature_f"],
            len(observations),
            ", some from cache" if used_cache else "",
        )
        return result

    def _fetch_and_walk(
        self, stations: list[dict], now: datetime, target: int
    ) -> list[dict]:
        """Fetch observations for candidate stations in a single batch call,
        then walk the sorted list collecting valid readings with LKG fallback.

        Logs one INFO line per station examined, matching the WU/NWS pattern.
        """
        # Collect candidate STIDs for the batch call (up to a reasonable cap
        # so we don't request 20 when we only need 3).
        candidate_stids = [s["stid"] for s in stations]

        # Single batch fetch — the big win.
        try:
            batch_results = self._fetch_batch_observations(candidate_stids, now)
        except SynopticError as exc:
            logger.warning("Synoptic batch fetch failed: %s — falling back to cache.", exc)
            batch_results = {}

        results: list[dict] = []
        checked = 0
        fresh_count = 0
        cached_count = 0

        for stn in stations:
            if len(results) >= target:
                break

            checked += 1
            stid = stn["stid"]
            stype = "official" if self._is_official(stn) else "mesonet"
            dist = stn.get("distance_miles", float("inf"))
            dist_str = f"{dist:4.1f} mi" if dist != float("inf") else " ??? mi"

            obs = batch_results.get(stid)

            if obs is not None:
                logger.info(
                    "  #%-2d  %-8s  (%-8s %s) → ✓ %.1f°F",
                    checked, stid, stype + ",", dist_str, obs["temperature_f"],
                )
                self._station_cache[stid] = obs
                fresh_count += 1
                results.append(obs)
                continue

            # Station missing from batch results — try LKG cache.
            cached = self._station_cache.get(stid)
            if cached is not None:
                cache_age = now - cached["timestamp"]
                if cache_age <= _CACHE_MAX_AGE:
                    cached_obs = {**cached, "is_cached": True}
                    age_str = self._format_age(cache_age)
                    logger.info(
                        "  #%-2d  %-8s  (%-8s %s) → ⚠ cached (last good: %s)",
                        checked, stid, stype + ",", dist_str, age_str,
                    )
                    cached_count += 1
                    results.append(cached_obs)
                    continue
                else:
                    logger.info(
                        "  #%-2d  %-8s  (%-8s %s) → ✗ no recent data (cache expired)",
                        checked, stid, stype + ",", dist_str,
                    )
                    continue

            # No observation and no cache.
            logger.info(
                "  #%-2d  %-8s  (%-8s %s) → ✗ no temperature or stale",
                checked, stid, stype + ",", dist_str,
            )

        logger.info(
            "Synoptic station search complete: %d valid reading%s (%d fresh, %d cached) from %d station%s checked.",
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
