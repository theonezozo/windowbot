"""AirNow API client for WindowBot.

Provides a fallback AQI source using the EPA's AirNow Current
Observations endpoint when PurpleAir data is unavailable.

Reference: https://docs.airnowapi.org/CurrentObservationsByLatLon/docs
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger("windowbot.airnow")

AIRNOW_API_BASE = "https://www.airnowapi.org/aq/observation/latLong/current/"
_REQUEST_TIMEOUT = 15


class AirNowError(Exception):
    """Raised on unrecoverable AirNow API errors."""


class AirNowClient:
    """Fetches AQI from the EPA's AirNow API.

    Args:
        api_key: AirNow API key (free registration at airnowapi.org).
        latitude: User's latitude in decimal degrees.
        longitude: User's longitude in decimal degrees.
    """

    def __init__(self, api_key: str, latitude: float, longitude: float) -> None:
        self._api_key = api_key
        self._lat = latitude
        self._lon = longitude

    def get_aqi(self) -> dict:
        """Fetch the current AQI for the configured location.

        Uses the AirNow \"Current Observation by Lat/Lon\" endpoint which
        searches within a 25-mile radius.

        Returns:
            Dict with:
            - ``aqi`` (int): The highest reported AQI value (worst pollutant).
            - ``source`` (str): Always ``\"airnow\"``.
            - ``category`` (str): EPA category name (e.g. \"Good\", \"Moderate\").
            - ``parameter`` (str): Dominant pollutant (e.g. \"PM2.5\", \"O3\").

        Raises:
            AirNowError: If the API call fails or returns no data.
        """
        params = {
            "format": "application/json",
            "latitude": str(self._lat),
            "longitude": str(self._lon),
            "distance": "25",
            "API_KEY": self._api_key,
        }

        try:
            resp = requests.get(
                AIRNOW_API_BASE, params=params, timeout=_REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            raise AirNowError(f"Network error querying AirNow: {exc}") from exc

        if not resp.ok:
            raise AirNowError(
                f"AirNow API error ({resp.status_code}): {resp.text[:300]}"
            )

        observations = resp.json()
        if not observations:
            raise AirNowError("AirNow returned no observations for this location.")

        # AirNow may return multiple pollutants (PM2.5, O3, etc.).
        # Pick the one with the highest (worst) AQI.
        worst = max(observations, key=lambda o: o.get("AQI", 0))

        aqi = int(worst.get("AQI", 0))
        category = worst.get("Category", {}).get("Name", "Unknown")
        parameter = worst.get("ParameterName", "Unknown")

        logger.info("AirNow AQI: %d (%s) — dominant pollutant: %s.", aqi, category, parameter)

        return {
            "aqi": aqi,
            "source": "airnow",
            "category": category,
            "parameter": parameter,
        }
