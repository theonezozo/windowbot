"""PurpleAir API client for WindowBot.

Fetches PM2.5 readings from nearby outdoor PurpleAir sensors and converts
them to an AQI value using the EPA breakpoint table.  Uses the **median**
of up to 3 sensors for robustness against outliers and faulty hardware.

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


class PurpleAirClient:
    """Fetches AQI from nearby outdoor PurpleAir sensors.

    Args:
        latitude: User's latitude in decimal degrees.
        longitude: User's longitude in decimal degrees.
        api_key: Optional PurpleAir read API key for higher rate limits.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        api_key: str | None = None,
    ) -> None:
        self._lat = latitude
        self._lon = longitude
        self._api_key = api_key

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
            "fields": "pm2.5,latitude,longitude,last_seen",
            "location_type": "0",  # outdoor only
            "nwlng": str(round(nwlng, 6)),
            "nwlat": str(round(nwlat, 6)),
            "selng": str(round(selng, 6)),
            "selat": str(round(selat, 6)),
        }

        try:
            resp = requests.get(
                url, params=params, headers=self._headers(), timeout=_REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            raise PurpleAirError(f"Network error querying PurpleAir: {exc}") from exc

        if not resp.ok:
            raise PurpleAirError(
                f"PurpleAir API error ({resp.status_code}): {resp.text[:300]}"
            )

        body = resp.json()
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
            last_seen: datetime | None = None
            if last_seen_raw is not None:
                try:
                    last_seen = datetime.fromtimestamp(int(last_seen_raw), tz=timezone.utc)
                except (ValueError, TypeError, OSError):
                    pass

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
        sensors = self.find_nearby_sensors()
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

        return {
            "aqi": aqi,
            "pm25": round(median_pm25, 1),
            "source": "purpleair",
            "sensor_count": len(top),
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
