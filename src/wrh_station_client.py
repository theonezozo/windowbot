"""Weather.gov Western Region time-series station client."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import requests

_API_KEY_URL = "https://www.weather.gov/source/wrh/apiKey.js"
_TIMESERIES_URL = "https://api.synopticdata.com/v2/stations/timeseries"
_VIEWER_URL = "https://www.weather.gov/wrh/timeseries"
_REQUEST_TIMEOUT = 15
_TOKEN_PATTERN = re.compile(r"\b[a-fA-F0-9]{32}\b")
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WindowBot/1.0)",
    "Origin": "https://www.weather.gov",
    "Referer": f"{_VIEWER_URL}?site=E7138",
}


class WRHStationError(Exception):
    """Raised when the weather.gov time-series station cannot be read."""


class WRHStationClient:
    """Reads the Synoptic JSON feed used by weather.gov's time-series viewer."""

    def _get_public_token(self) -> str:
        try:
            response = requests.get(
                _API_KEY_URL,
                headers=_BROWSER_HEADERS,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise WRHStationError(
                f"Network error fetching weather.gov viewer token: {exc}"
            ) from exc

        if not response.ok:
            raise WRHStationError(
                f"weather.gov viewer token error ({response.status_code})"
            )

        match = _TOKEN_PATTERN.search(response.text)
        if not match:
            raise WRHStationError("weather.gov viewer token was not found")
        return match.group(0)

    def get_station_observation(
        self, station_id: str, max_age_minutes: int = 15
    ) -> dict:
        """Return the newest fresh reading from the viewer's backing JSON API."""
        if not station_id:
            raise ValueError("station_id must not be empty")
        if max_age_minutes <= 0:
            raise ValueError("max_age_minutes must be positive")

        params = {
            "STID": station_id,
            "showemptystations": "1",
            "units": "temp|F,speed|mph,english",
            "recent": str(max(60, max_age_minutes * 2)),
            "complete": "1",
            "obtimezone": "utc",
            "token": self._get_public_token(),
        }
        try:
            response = requests.get(
                _TIMESERIES_URL,
                params=params,
                headers=_BROWSER_HEADERS,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise WRHStationError(
                f"Network error fetching weather.gov station {station_id}: {exc}"
            ) from exc

        if not response.ok:
            raise WRHStationError(
                f"weather.gov station API error ({response.status_code})"
            )

        data = response.json()
        summary = data.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") != 1:
            raise WRHStationError(
                f"weather.gov station API error: "
                f"{summary.get('RESPONSE_MESSAGE', 'unknown error')}"
            )

        stations = data.get("STATION", [])
        if not stations:
            raise WRHStationError(
                f"No observation available for weather.gov station {station_id}"
            )

        observations = stations[0].get("OBSERVATIONS", {})
        dates = observations.get("date_time", [])
        temperatures = observations.get("air_temp_set_1", [])
        if not dates or not temperatures:
            raise WRHStationError(
                f"No temperature available for weather.gov station {station_id}"
            )

        index = min(len(dates), len(temperatures)) - 1
        while index >= 0 and temperatures[index] is None:
            index -= 1
        if index < 0:
            raise WRHStationError(
                f"No temperature available for weather.gov station {station_id}"
            )

        try:
            timestamp = datetime.fromisoformat(dates[index])
        except (TypeError, ValueError) as exc:
            raise WRHStationError(
                f"Invalid timestamp for weather.gov station {station_id}"
            ) from exc
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - timestamp
        if age > timedelta(minutes=max_age_minutes):
            raise WRHStationError(
                f"weather.gov station {station_id} observation is stale "
                f"({int(age.total_seconds() // 60)}m; maximum {max_age_minutes}m)"
            )

        def _value(name: str) -> float | None:
            values = observations.get(name, [])
            if index >= len(values) or values[index] is None:
                return None
            return round(float(values[index]), 1)

        return {
            "station_id": station_id,
            "temperature_f": round(float(temperatures[index]), 1),
            "humidity": _value("relative_humidity_set_1"),
            "wind_speed_mph": _value("wind_speed_set_1"),
            "timestamp": timestamp,
        }
