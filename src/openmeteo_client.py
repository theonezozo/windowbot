"""Open-Meteo free weather peer/fallback for WindowBot.

Fetches current outdoor temperature, humidity, and wind speed from the
Open-Meteo API.  Completely free, requires NO API key, and returns
model-interpolated data for exact coordinates.

In normal operation Open-Meteo acts as a **peer station** alongside NWS:
when its reading is fresh (≤20 min) it is blended into the NWS median pool.
If all NWS stations fail, Open-Meteo serves as the sole fallback source
(with a generous 60-min hard cap so a stuck timestamp can't display
hours-old data).

Reference: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger("windowbot.openmeteo")

# Observations older than this are not blended into the NWS peer pool.
# Matches ``nws_client._MAX_OBS_AGE`` so the median pool ages are symmetric
# across sources.
_MAX_OBS_AGE = timedelta(minutes=20)
# Hard cap for the last-resort ``get_outdoor_conditions`` path. Generous on
# purpose — only fires when the Open-Meteo API itself returns a stuck
# timestamp; under normal conditions current-weather responses are well
# under this bound.
_LAST_RESORT_MAX_AGE = timedelta(minutes=60)


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
        20 minutes of now.  Stale readings raise an error so the caller can
        skip this peer without using outdated data.

        Returns:
            Dict matching the NWS single-observation shape:
            ``station_id``, ``temperature_f``, ``humidity``,
            ``wind_speed_mph``, ``timestamp``.

        Raises:
            OpenMeteoError: On network/API errors or stale data (> 20 min).
        """
        data = self._fetch()
        age = datetime.now(timezone.utc) - data["timestamp"]
        
        # Format age consistently with NWS
        total_secs = int(age.total_seconds())
        hours, remainder = divmod(total_secs, 3600)
        minutes = remainder // 60
        if hours > 0:
            age_str = f"{hours}h {minutes:02d}m ago"
        else:
            age_str = f"{minutes}m ago"
        
        if age > _MAX_OBS_AGE:
            logger.info(
                "Open-Meteo peer stale: %.1f°F (%s) — skipped",
                data["temperature_f"],
                age_str,
            )
            raise OpenMeteoError(
                f"Open-Meteo observation stale ({int(age.total_seconds() // 60)}m old)"
            )

        logger.info(
            "Open-Meteo peer: %.1f°F (%s)",
            data["temperature_f"],
            age_str,
        )

        return {
            "station_id": "OPENMETEO",
            "temperature_f": data["temperature_f"],
            "humidity": data["humidity"],
            "wind_speed_mph": data["wind_speed_mph"],
            "timestamp": data["timestamp"],
        }

    def get_outdoor_conditions(self) -> dict:
        """Fetch current weather as a last-resort fallback.

        Unlike ``get_observation()``, this method does NOT enforce the 20-minute
        peer-pool freshness limit.  Used only when all NWS stations have failed
        and no fresh OM peer reading is available.  A generous 60-minute hard
        cap still applies so a stuck timestamp from the API cannot propagate
        hours-old data to the status page.

        Returns:
            Dict matching the NWS aggregated-conditions format:
            ``temperature_f``, ``humidity``, ``wind_speed_mph``,
            ``station_count``, ``is_fallback``, ``used_cache``, ``source``.

        Raises:
            OpenMeteoError: On any network or API failure, or if the returned
                observation is older than the 60-minute hard cap.
        """
        data = self._fetch()

        # Hard cap: even on the last-resort path we refuse to surface an
        # observation older than _LAST_RESORT_MAX_AGE. This catches the
        # "OM API returns a stuck timestamp" failure mode.
        age = datetime.now(timezone.utc) - data["timestamp"]
        if age > _LAST_RESORT_MAX_AGE:
            raise OpenMeteoError(
                f"Open-Meteo last-resort observation too stale "
                f"({int(age.total_seconds() // 60)}m old)"
            )

        # Format age consistently with NWS
        total_secs = int(age.total_seconds())
        hours, remainder = divmod(total_secs, 3600)
        minutes = remainder // 60
        if hours > 0:
            age_str = f"{hours}h {minutes:02d}m ago"
        else:
            age_str = f"{minutes}m ago"

        logger.info(
            "Open-Meteo fallback: %.1f°F (%s), %s%% humidity, %.1f mph wind",
            data["temperature_f"],
            age_str,
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
            "observation_time": data["timestamp"].isoformat() if data.get("timestamp") else None,
        }
