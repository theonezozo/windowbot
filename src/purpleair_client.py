"""PurpleAir API client for WindowBot.

Fetches PM2.5 readings from nearby outdoor PurpleAir sensors and converts
them to an AQI value using the EPA breakpoint table.  Uses the **median**
of up to 3 sensors for robustness against outliers and faulty hardware.

**Point-cost optimization (PurpleAir's read API is metered):**
- The set of nearby sensors barely changes, so we run the expensive
  bounding-box *discovery* query only ONCE and cache the resulting sensor
  IDs for ``sensor_cache_ttl_hours`` (default 12 h). Subsequent reads fetch
  live PM2.5 for just those cached IDs via the cheap ``show_only`` filter.
- Each request asks for the **minimum** field set: discovery needs
  lat/lon (distance) + last_seen (freshness); live reads need only
  pm2.5 + last_seen. Fewer fields = fewer points per call.
On TTL expiry — or if every cached sensor goes stale/offline — discovery
re-runs once and refreshes the cache.

Reference: https://api.purpleair.com/
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger("windowbot.purpleair")

PURPLEAIR_API_BASE = "https://api.purpleair.com/v1"
_REQUEST_TIMEOUT = 15

# Reject sensor readings older than this.
_MAX_SENSOR_AGE = timedelta(minutes=30)

# Minimal field sets — request only what downstream code consumes.
# Discovery needs lat/lon to compute distance and last_seen for the freshness
# gate; live reads reuse the cached distance and need only pm2.5 + last_seen.
_DISCOVERY_FIELDS = "pm2.5,latitude,longitude,last_seen"
_LIVE_READ_FIELDS = "pm2.5,last_seen"

# How many of the closest discovered sensors to cache. A handful (rather than
# just the 3 used for the median) gives resilience: if one goes offline the
# next live read still has candidates before a re-discovery is forced.
_MAX_CACHED_SENSORS = 5

# Default lifetime of the cached nearby-sensor ID set.
_DEFAULT_CACHE_TTL_HOURS = 12.0

# Approximate distance per degree at mid-latitudes (km).
_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LON_APPROX = 85.0  # ~cos(40°) × 111

# EPA PM2.5 AQI breakpoint table: (C_low, C_high, I_low, I_high)
_AQI_BREAKPOINTS: list[tuple[float, float, int, int]] = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


class PurpleAirError(Exception):
    """Raised on unrecoverable PurpleAir API errors."""


def _describe_http_error(status_code: int) -> str:
    """Return an actionable, human-readable hint for a PurpleAir HTTP error.

    PurpleAir's read API is metered: read calls consume prepaid *points*, and a
    depleted (or negative) balance returns HTTP 402. Surfacing these reasons
    plainly is critical — otherwise a payment/auth failure looks identical to a
    generic outage and WindowBot silently falls back to a slower AQI source.
    """
    hints = {
        401: "Unauthorized — check PURPLEAIR_API_KEY (a PurpleAir READ key is required).",
        403: "Forbidden — the PURPLEAIR_API_KEY is invalid or lacks read access.",
        402: (
            "Payment Required — the PurpleAir account is out of API points/credits. "
            "Top up at https://develop.purpleair.com to restore near-real-time AQI; "
            "until then WindowBot falls back to AirNow (slower to react to smoke)."
        ),
        429: "Rate limited — too many PurpleAir requests; back off and retry later.",
    }
    return hints.get(status_code, "")


class PurpleAirClient:
    """Fetches AQI from nearby outdoor PurpleAir sensors.

    Args:
        latitude: User's latitude in decimal degrees.
        longitude: User's longitude in decimal degrees.
        api_key: Optional PurpleAir read API key for higher rate limits.
        sensor_cache_ttl_hours: How long (hours) to reuse a discovered set of
            nearby sensor IDs before re-running the expensive bounding-box
            discovery query. Defaults to 12 h.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        api_key: str | None = None,
        sensor_cache_ttl_hours: float = _DEFAULT_CACHE_TTL_HOURS,
    ) -> None:
        self._lat = latitude
        self._lon = longitude
        self._api_key = api_key
        self._cache_ttl = timedelta(hours=sensor_cache_ttl_hours)
        # Cache of nearby sensor IDs → cached distance_km, plus its expiry.
        self._sensor_cache: dict = {}
        self._cache_expiry: datetime | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometres between two points."""
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _get(self, params: dict) -> dict:
        """Issue a GET /v1/sensors request and return the parsed JSON body.

        Centralizes network + metered-error handling so both the discovery
        (bounding-box) query and the cheap ``show_only`` live read share the
        same actionable 401/402/403 diagnostics.
        """
        url = f"{PURPLEAIR_API_BASE}/sensors"
        try:
            resp = requests.get(
                url, params=params, headers=self._headers(), timeout=_REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            raise PurpleAirError(f"Network error querying PurpleAir: {exc}") from exc

        if not resp.ok:
            hint = _describe_http_error(resp.status_code)
            # Log payment/auth failures loudly — these are actionable config
            # problems, not transient outages, and would otherwise be masked by
            # the silent AirNow fallback in the orchestrator.
            if resp.status_code in (401, 402, 403):
                logger.error("PurpleAir API %d: %s", resp.status_code, hint)
            prefix = f"{resp.status_code}: {hint}" if hint else str(resp.status_code)
            raise PurpleAirError(
                f"PurpleAir API error ({prefix}) body={resp.text[:300]}"
            )

        return resp.json()

    @staticmethod
    def _parse_last_seen(raw) -> datetime | None:
        """Parse a PurpleAir ``last_seen`` epoch value into a datetime."""
        if raw is None:
            return None
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            return None

    # ------------------------------------------------------------------
    # Sensor discovery
    # ------------------------------------------------------------------

    def find_nearby_sensors(self, radius_km: float = 5.0) -> list[dict]:
        """Find outdoor PurpleAir sensors within *radius_km* of the user's
        location.

        Returns:
            Sensor dicts sorted by distance (closest first), each with:
            - ``sensor_index`` (int)
            - ``pm25`` (float): Current PM2.5 reading.
            - ``distance_km`` (float)
            - ``last_seen`` (datetime | None)
        """
        # Build bounding box from radius.
        lat_offset = radius_km / _KM_PER_DEG_LAT
        lon_offset = radius_km / _KM_PER_DEG_LON_APPROX

        nwlat = self._lat + lat_offset
        nwlng = self._lon - lon_offset
        selat = self._lat - lat_offset
        selng = self._lon + lon_offset

        url = f"{PURPLEAIR_API_BASE}/sensors"
        params = {
            "fields": _DISCOVERY_FIELDS,
            "location_type": "0",  # outdoor only
            "nwlng": str(round(nwlng, 6)),
            "nwlat": str(round(nwlat, 6)),
            "selng": str(round(selng, 6)),
            "selat": str(round(selat, 6)),
        }

        body = self._get(params)
        fields = body.get("fields", [])
        data_rows = body.get("data", [])

        if not fields or not data_rows:
            logger.info("No PurpleAir sensors found within %.1f km.", radius_km)
            return []

        # Map field names to column indices.
        col = {name: idx for idx, name in enumerate(fields)}

        now = datetime.now(timezone.utc)
        sensors: list[dict] = []

        for row in data_rows:
            pm25_raw = row[col.get("pm2.5", -1)] if "pm2.5" in col else None
            lat_raw = row[col.get("latitude", -1)] if "latitude" in col else None
            lon_raw = row[col.get("longitude", -1)] if "longitude" in col else None
            last_seen_raw = row[col.get("last_seen", -1)] if "last_seen" in col else None
            sensor_index = row[col.get("sensor_index", 0)] if "sensor_index" in col else row[0]

            if pm25_raw is None or lat_raw is None or lon_raw is None:
                continue

            pm25 = float(pm25_raw)
            if pm25 < 0:
                continue  # discard negative readings

            # Check freshness.
            last_seen = self._parse_last_seen(last_seen_raw)

            if last_seen is not None and (now - last_seen) > _MAX_SENSOR_AGE:
                continue

            distance = self._haversine_km(self._lat, self._lon, float(lat_raw), float(lon_raw))

            sensors.append(
                {
                    "sensor_index": sensor_index,
                    "pm25": pm25,
                    "distance_km": round(distance, 2),
                    "last_seen": last_seen,
                }
            )

        sensors.sort(key=lambda s: s["distance_km"])
        logger.info("Found %d valid outdoor PurpleAir sensors within %.1f km.", len(sensors), radius_km)
        return sensors

    # ------------------------------------------------------------------
    # Cached live reads (metered-cost optimization)
    # ------------------------------------------------------------------

    def _read_sensors_by_id(self, sensor_ids: list) -> list[dict]:
        """Read live PM2.5 for specific *sensor_ids* via the ``show_only`` filter.

        This is the cheap per-cycle path: it requests only the minimal live
        field set (``pm2.5,last_seen``) for a known handful of sensors instead
        of re-running the bounding-box discovery over every nearby sensor.

        Returns fresh, non-negative readings as dicts with ``sensor_index``,
        ``pm25`` and ``last_seen`` (distance is filled in by the caller from
        the cache).
        """
        if not sensor_ids:
            return []

        params = {
            "fields": _LIVE_READ_FIELDS,
            "show_only": ",".join(str(s) for s in sensor_ids),
        }
        body = self._get(params)
        fields = body.get("fields", [])
        data_rows = body.get("data", [])
        if not fields or not data_rows:
            return []

        col = {name: idx for idx, name in enumerate(fields)}
        now = datetime.now(timezone.utc)
        readings: list[dict] = []

        for row in data_rows:
            pm25_raw = row[col["pm2.5"]] if "pm2.5" in col else None
            last_seen_raw = row[col.get("last_seen", -1)] if "last_seen" in col else None
            sensor_index = row[col.get("sensor_index", 0)] if "sensor_index" in col else row[0]

            if pm25_raw is None:
                continue
            pm25 = float(pm25_raw)
            if pm25 < 0:
                continue  # discard negative readings

            last_seen = self._parse_last_seen(last_seen_raw)
            if last_seen is not None and (now - last_seen) > _MAX_SENSOR_AGE:
                continue

            readings.append(
                {"sensor_index": sensor_index, "pm25": pm25, "last_seen": last_seen}
            )

        return readings

    def _refresh_cache(self, sensors: list[dict]) -> None:
        """Cache the closest discovered sensor IDs (+distance) and reset TTL."""
        top = sensors[: _MAX_CACHED_SENSORS]
        self._sensor_cache = {
            s["sensor_index"]: s.get("distance_km", 0.0) for s in top
        }
        self._cache_expiry = datetime.now(timezone.utc) + self._cache_ttl

    def _current_readings(self) -> list[dict]:
        """Return fresh sensor readings, using the cached IDs when possible.

        Fast path: if we have unexpired cached sensor IDs, read just those
        live (cheap ``show_only`` call). If none return fresh data, or the TTL
        has expired, fall back to a one-time bounding-box discovery and refresh
        the cache.
        """
        now = datetime.now(timezone.utc)
        cache_valid = (
            self._sensor_cache
            and self._cache_expiry is not None
            and now < self._cache_expiry
        )
        if cache_valid:
            readings = self._read_sensors_by_id(list(self._sensor_cache.keys()))
            if readings:
                for r in readings:
                    r["distance_km"] = self._sensor_cache.get(r["sensor_index"], 0.0)
                readings.sort(key=lambda s: s["distance_km"])
                return readings
            logger.info(
                "Cached PurpleAir sensors returned no fresh data — re-discovering."
            )

        # (Re)discover: TTL expired, cold start, or cached sensors went stale.
        sensors = self.find_nearby_sensors()
        self._refresh_cache(sensors)
        return sensors

    # ------------------------------------------------------------------
    # AQI computation
    # ------------------------------------------------------------------

    def get_aqi(self) -> dict:
        """Compute AQI from the median PM2.5 of the 3 closest sensors.

        Returns:
            Dict with:
            - ``aqi`` (int): Computed AQI value.
            - ``pm25`` (float): Median PM2.5 value used.
            - ``source`` (str): Always ``"purpleair"``.
            - ``sensor_count`` (int): Number of sensors contributing.

        Raises:
            PurpleAirError: If no sensors are available.
        """
        sensors = self._current_readings()
        if not sensors:
            raise PurpleAirError("No PurpleAir sensors available near the configured location.")

        top = sensors[:3]
        pm25_values = [s["pm25"] for s in top]
        median_pm25 = statistics.median(pm25_values)
        aqi = self.pm25_to_aqi(median_pm25)

        logger.info(
            "PurpleAir AQI: %d (median PM2.5=%.1f from %d sensors).",
            aqi, median_pm25, len(top),
        )

        # Use the oldest last_seen among contributing sensors as the observation time.
        last_seen_times = [s["last_seen"] for s in top if s.get("last_seen") is not None]
        oldest_seen = min(last_seen_times).isoformat() if last_seen_times else None

        return {
            "aqi": aqi,
            "pm25": round(median_pm25, 1),
            "source": "purpleair",
            "sensor_count": len(top),
            "observation_time": oldest_seen,
        }

    @staticmethod
    def pm25_to_aqi(pm25: float) -> int:
        """Convert a PM2.5 concentration (µg/m³) to an AQI value using the
        standard EPA breakpoint table.

        Values above 500.4 µg/m³ are capped at AQI 500.
        Negative inputs return 0.
        """
        if pm25 < 0:
            return 0

        # Truncate to one decimal place per EPA methodology.
        pm25 = math.floor(pm25 * 10) / 10.0

        for c_low, c_high, i_low, i_high in _AQI_BREAKPOINTS:
            if c_low <= pm25 <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
                return round(aqi)

        # Above the highest breakpoint — cap at 500.
        return 500
