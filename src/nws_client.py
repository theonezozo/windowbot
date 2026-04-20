"""National Weather Service API client for WindowBot.

Fetches outdoor temperature, humidity, and wind speed from nearby NWS
weather stations.  Uses the **median** of up to 3 personal/cooperative
stations for robustness, falling back to the nearest official ASOS/AWOS
station when fewer than 3 personal stations are available.

Reference: https://www.weather.gov/documentation/services-web-api
"""

from __future__ import annotations

import logging
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

        # Step 1: Resolve grid coordinates from lat/lon.
        points_url = f"{NWS_API_BASE}/points/{self._lat},{self._lon}"
        points_data = self._get(points_url)
        props = points_data.get("properties", {})

        stations_url = props.get("observationStations")
        if not stations_url:
            grid_id = props.get("gridId")
            grid_x = props.get("gridX")
            grid_y = props.get("gridY")
            if not all([grid_id, grid_x is not None, grid_y is not None]):
                raise NWSError("Could not determine grid coordinates from NWS points API.")
            stations_url = f"{NWS_API_BASE}/gridpoints/{grid_id}/{grid_x},{grid_y}/stations"

        # Step 2: Fetch stations (already sorted by distance from the grid point).
        stations_data = self._get(stations_url)
        features = stations_data.get("features", [])

        self._stations = []
        for feat in features:
            sprops = feat.get("properties", {})
            sid = sprops.get("stationIdentifier", "")
            name = sprops.get("name", "")
            self._stations.append(
                {
                    "id": sid,
                    "name": name,
                    "is_personal": self._is_personal_station({"id": sid}),
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
        """Fetch and validate a single station's latest observation."""
        url = f"{NWS_API_BASE}/stations/{station_id}/observations/latest"
        data = self._get(url)
        props = data.get("properties", {})

        # Validate timestamp.
        ts_str = props.get("timestamp")
        if not ts_str:
            logger.debug("Station %s: no timestamp — skipping.", station_id)
            return None

        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            logger.debug("Station %s: unparseable timestamp '%s'.", station_id, ts_str)
            return None

        if now - ts > _MAX_OBS_AGE:
            logger.debug("Station %s: observation too old (%s).", station_id, ts_str)
            return None

        # Parse temperature (Celsius → Fahrenheit).
        temp_raw = props.get("temperature", {})
        temp_c = temp_raw.get("value")
        if temp_c is None:
            return None
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

    def get_outdoor_conditions(self) -> dict:
        """Compute aggregated outdoor conditions using the MEDIAN of nearby
        personal weather stations, with an automatic fallback to official
        stations.

        Strategy:
            1. Query the 3 closest personal/cooperative stations.
            2. If fewer than 2 valid readings, fall back to the nearest
               official ASOS/AWOS station(s).
            3. Return the **median** temperature, humidity, and wind speed.

        Returns:
            Dict with keys:
            - ``temperature_f`` (float): Median outdoor temperature in °F.
            - ``humidity`` (float | None): Median outdoor humidity %.
            - ``wind_speed_mph`` (float | None): Median wind speed.
            - ``station_count`` (int): Number of stations contributing.
            - ``is_fallback`` (bool): True if official stations were used.
        """
        self.discover_stations()

        # Separate personal and official stations.
        personal_ids = [s["id"] for s in self._stations if s["is_personal"]]
        official_ids = [s["id"] for s in self._stations if not s["is_personal"]]

        now = datetime.now(timezone.utc)
        is_fallback = False

        # Try personal stations first (up to 3).
        observations = self._fetch_batch(personal_ids[:5], now, target=3)

        if len(observations) < 2:
            # Fallback: use official stations.
            logger.info("Only %d personal station readings — falling back to official.", len(observations))
            fallback_obs = self._fetch_batch(official_ids[:3], now, target=3)
            observations = observations + fallback_obs
            is_fallback = True

        if not observations:
            raise NWSError("No valid weather observations available from any station.")

        return self._aggregate(observations, is_fallback)

    def _fetch_batch(
        self, station_ids: list[str], now: datetime, target: int
    ) -> list[dict]:
        """Fetch observations from *station_ids* until *target* valid
        readings are collected or the list is exhausted."""
        results: list[dict] = []
        for sid in station_ids:
            if len(results) >= target:
                break
            try:
                obs = self._fetch_single_observation(sid, now)
                if obs is not None:
                    results.append(obs)
            except NWSError:
                logger.warning("Skipping station %s due to error.", sid, exc_info=True)
        return results

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
