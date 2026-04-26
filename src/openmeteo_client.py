"""Open-Meteo free weather fallback for WindowBot.

Fetches current outdoor temperature, humidity, and wind speed from the
Open-Meteo API.  Completely free, requires NO API key, and returns
model-interpolated data for exact coordinates.  This is the last-resort
fallback behind both Weather Underground and NWS.

Reference: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger("windowbot.openmeteo")


class OpenMeteoError(Exception):
    """Raised on Open-Meteo API errors."""


class OpenMeteoClient:
    """Free, zero-auth weather fallback using Open-Meteo.

    Returns model-interpolated weather data for exact coordinates.
    No station discovery needed — this is grid-based, not station-based.
    """

    _API_BASE = "https://api.open-meteo.com/v1/forecast"
    _REQUEST_TIMEOUT = 10

    def __init__(self, latitude: float, longitude: float) -> None:
        self._lat = latitude
        self._lon = longitude

    def get_outdoor_conditions(self) -> dict:
        """Fetch current weather from Open-Meteo.

        Returns dict matching the same format as NWSClient/WUClient:
        {
            "temperature_f": float,
            "humidity": float | None,
            "wind_speed_mph": float | None,
            "station_count": 1,     # always 1 (grid point)
            "is_fallback": True,    # always True (this IS the fallback)
            "used_cache": False,    # no cache needed
            "source": "openmeteo",
        }

        Raises:
            OpenMeteoError: On any network or API failure.
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
        if humidity is not None:
            humidity = float(humidity)

        wind_mph = current.get("wind_speed_10m")
        if wind_mph is not None:
            wind_mph = float(wind_mph)

        temp_f = float(temp_f)

        logger.info(
            "Open-Meteo: %.1f°F, %d%% humidity, %.1f mph wind",
            temp_f,
            int(humidity) if humidity is not None else 0,
            wind_mph if wind_mph is not None else 0.0,
        )

        return {
            "temperature_f": round(temp_f, 1),
            "humidity": round(humidity, 1) if humidity is not None else None,
            "wind_speed_mph": round(wind_mph, 1) if wind_mph is not None else None,
            "station_count": 1,
            "is_fallback": True,
            "used_cache": False,
            "source": "openmeteo",
        }
