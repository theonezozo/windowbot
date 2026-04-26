"""Open-Meteo free weather peer/fallback for WindowBot.

Fetches current outdoor temperature, humidity, and wind speed from the
Open-Meteo API.  Completely free, requires NO API key, and returns
model-interpolated data for exact coordinates.

In normal operation Open-Meteo acts as a **peer station** alongside NWS:
when its reading is fresh (≤30 min) it is blended into the NWS median pool.
If all NWS stations fail, Open-Meteo serves as the sole fallback source.

Reference: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger("windowbot.openmeteo")

# Observations older than this are not blended into the NWS peer pool.
_MAX_OBS_AGE = timedelta(minutes=30)


class OpenMeteoError(Exception):
    """Raised on Open-Meteo API errors."""


class OpenMeteoClient:
    """Free, zero-auth weather client using Open-Meteo.

    Returns model-interpolated weather data for exact coordinates.
    No station discovery needed — this is grid-based, not station-based.
    """

    _API_BASE = "https://api.open-meteo.com/v1/forecast"
    _REQUEST_TIMEOUT = 10

    def __init__(self, latitude: float, longitude: float) -> None:
        self._lat = latitude
        self._lon = longitude

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self) -> dict:
        """Fetch current weather from the API and return parsed fields.

        Returns:
            Dict with keys: temperature_f, humidity, wind_speed_mph, timestamp.

        Raises:
            OpenMeteoError: On network errors, HTTP errors, or missing fields.
        """
        params = {
            "latitude": self._lat,
            "longitude": self._lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
        }

        try:
            resp = requests.get(
                self._API_BASE, params=params, timeout=self._REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise OpenMeteoError(f"Network error: {exc}") from exc

        if not resp.ok:
            raise OpenMeteoError(
                f"API error ({resp.status_code}): {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise OpenMeteoError(f"Invalid JSON response: {exc}") from exc

        current = data.get("current")
        if not current:
            raise OpenMeteoError("Response missing 'current' block.")

        temp_f = current.get("temperature_2m")
        if temp_f is None:
            raise OpenMeteoError("Response missing temperature_2m.")

        humidity = current.get("relative_humidity_2m")
        wind_mph = current.get("wind_speed_10m")

        # Parse observation timestamp.
        # current["time"] is in the local timezone; utc_offset_seconds converts to UTC.
        # local = UTC + offset  →  UTC = local − offset
        ts_str = current.get("time")
        utc_offset_seconds = data.get("utc_offset_seconds", 0)
        if ts_str:
            try:
                local_naive = datetime.fromisoformat(ts_str)
                ts = (
                    local_naive - timedelta(seconds=utc_offset_seconds)
                ).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        return {
            "temperature_f": round(float(temp_f), 1),
            "humidity": round(float(humidity), 1) if humidity is not None else None,
            "wind_speed_mph": round(float(wind_mph), 1) if wind_mph is not None else None,
            "timestamp": ts,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_observation(self) -> dict:
        """Fetch current weather and return an NWS-compatible observation dict.

        The observation is considered fresh only if its timestamp is within
        30 minutes of now.  Stale readings raise an error so the caller can
        skip this peer without using outdated data.

        Returns:
            Dict matching the NWS single-observation shape:
            ``station_id``, ``temperature_f``, ``humidity``,
            ``wind_speed_mph``, ``timestamp``.

        Raises:
            OpenMeteoError: On network/API errors or stale data (> 30 min).
        """
        data = self._fetch()
        age = datetime.now(timezone.utc) - data["timestamp"]
        if age > _MAX_OBS_AGE:
            raise OpenMeteoError(
                f"Open-Meteo observation stale ({int(age.total_seconds() // 60)}m old)"
            )

        logger.debug(
            "Open-Meteo peer: %.1f°F, %dm old",
            data["temperature_f"],
            int(age.total_seconds() // 60),
        )

        return {
            "station_id": "OPENMETEO",
            "temperature_f": data["temperature_f"],
            "humidity": data["humidity"],
            "wind_speed_mph": data["wind_speed_mph"],
            "timestamp": data["timestamp"],
        }

    def get_outdoor_conditions(self) -> dict:
        """Fetch current weather as a last-resort fallback (no freshness check).

        Unlike ``get_observation()``, this method does NOT enforce the 30-minute
        freshness limit.  Use it only when all NWS stations have failed and no
        fresh OM peer reading is available.

        Returns:
            Dict matching the NWS aggregated-conditions format:
            ``temperature_f``, ``humidity``, ``wind_speed_mph``,
            ``station_count``, ``is_fallback``, ``used_cache``, ``source``.

        Raises:
            OpenMeteoError: On any network or API failure.
        """
        data = self._fetch()

        logger.info(
            "Open-Meteo fallback: %.1f°F, %s%% humidity, %.1f mph wind",
            data["temperature_f"],
            f"{int(data['humidity'])}" if data["humidity"] is not None else "?",
            data["wind_speed_mph"] if data["wind_speed_mph"] is not None else 0.0,
        )

        return {
            "temperature_f": data["temperature_f"],
            "humidity": data["humidity"],
            "wind_speed_mph": data["wind_speed_mph"],
            "station_count": 1,
            "is_fallback": True,
            "used_cache": False,
            "source": "openmeteo",
        }
