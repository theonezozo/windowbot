"""Beestat API client for WindowBot.

Fetches indoor sensor data and HVAC state via the Beestat.io private API,
which proxies Ecobee data through Beestat's OAuth connection.

Beestat stores temperatures already divided by 10 (actual °F) and humidity
also divided by 10.  This client reverses the humidity scaling to return
percentages matching the EcobeeClient interface.
"""

from __future__ import annotations

import json
import logging

import requests

logger = logging.getLogger("windowbot.beestat")

BEESTAT_API_URL = "https://app.beestat.io/api/"

_REQUEST_TIMEOUT = 15


class BeestatAuthError(Exception):
    """Raised when the Beestat API key is invalid or revoked."""


class BeestatApiError(Exception):
    """Raised on unexpected Beestat API errors."""


class BeestatClient:
    """Fetches sensor data and HVAC state from the Beestat API.

    Provides the same public interface as :class:`EcobeeClient` so the
    orchestrator can swap providers transparently.

    Args:
        api_key: 40-character hex Beestat API key.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    # ------------------------------------------------------------------
    # Low-level API helpers
    # ------------------------------------------------------------------

    def _post(self, payload: dict) -> dict:
        """Send a single POST to the Beestat API.

        Returns:
            Parsed JSON response body.

        Raises:
            BeestatAuthError: On 401 or explicit auth failure.
            BeestatApiError: On any other HTTP / API error.
        """
        payload["api_key"] = self._api_key

        try:
            resp = requests.post(
                BEESTAT_API_URL,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise BeestatApiError(f"Network error calling Beestat: {exc}") from exc

        if resp.status_code == 401:
            raise BeestatAuthError(
                "Beestat API key rejected (401) — check BEESTAT_API_KEY."
            )

        if not resp.ok:
            raise BeestatApiError(
                f"Beestat API error ({resp.status_code}): {resp.text}"
            )

        body = resp.json()
        if not body.get("success", True):
            error_msg = body.get("data", {}).get("error_message", resp.text)
            if "session" in str(error_msg).lower() or "api_key" in str(error_msg).lower():
                raise BeestatAuthError(f"Beestat auth error: {error_msg}")
            raise BeestatApiError(f"Beestat API error: {error_msg}")

        return body

    def _batch(self, calls: list[dict]) -> dict:
        """Execute a batch request containing multiple API calls.

        Each item in *calls* should be a dict with ``resource``, ``method``,
        ``arguments``, and optionally ``alias`` keys.

        Returns:
            The ``data`` dict from the batch response, keyed by alias
            (or ``resource.method`` if no alias given).
        """
        payload = {
            "batch": json.dumps(calls),
        }
        body = self._post(payload)
        return body.get("data", {})

    # ------------------------------------------------------------------
    # Data fetching (single batch call)
    # ------------------------------------------------------------------

    def _fetch_all(self) -> dict:
        """Fetch sensors and thermostat data in a single batch request.

        Returns:
            Dict with ``"sensors"`` and ``"thermostat"`` keys from the
            batch response.
        """
        calls = [
            {
                "resource": "sensor",
                "method": "read_id",
                "arguments": "{}",
                "alias": "sensors",
            },
            {
                "resource": "ecobee_thermostat",
                "method": "read_id",
                "arguments": "{}",
                "alias": "thermostat",
            },
        ]
        data = self._batch(calls)
        logger.debug("Beestat batch response keys: %s", list(data.keys()))
        return data

    # ------------------------------------------------------------------
    # Public API  (matches EcobeeClient interface)
    # ------------------------------------------------------------------

    def get_sensors(self) -> list[dict]:
        """Parse sensor readings from Beestat.

        Returns:
            List of sensor dicts matching the EcobeeClient format::

                {"name": str, "temperature_f": float|None,
                 "humidity": int|None, "is_online": bool}
        """
        data = self._fetch_all()
        raw_sensors = data.get("sensors", {})

        # Collect all non-inactive sensors. The Ecobee `in_use` flag reflects
        # comfort-profile participation (e.g. a sensor only in the "Home"
        # profile is in_use=False while in "Away" mode) — it is NOT a
        # hardware status indicator. We include every sensor that isn't
        # explicitly decommissioned so all floors are represented regardless
        # of the active comfort setting.
        selected: list[tuple[str, dict]] = []
        for sensor_id, raw in raw_sensors.items():
            if raw.get("inactive", False):
                logger.debug("Skipping inactive sensor %s", sensor_id)
                continue
            selected.append((sensor_id, raw))

        # --- Build sensor dicts from selected items -------------------
        sensors: list[dict] = []
        for sensor_id, raw in selected:
            name = raw.get("name", "Unknown")

            # Temperature: Beestat stores actual °F (already /10 from Ecobee)
            temp_f: float | None = None
            raw_temp = raw.get("temperature")
            if raw_temp is not None:
                try:
                    temp_f = float(raw_temp)
                except (ValueError, TypeError):
                    logger.warning(
                        "Bad temperature value '%s' for sensor '%s'.",
                        raw_temp, name,
                    )

            # Humidity: Beestat stores value/10, so 4.5 means 45%.
            # Multiply by 10 and round to int for the standard interface.
            humidity: int | None = None
            raw_hum = raw.get("humidity")
            if raw_hum is not None:
                try:
                    humidity = round(float(raw_hum) * 10)
                except (ValueError, TypeError):
                    logger.warning(
                        "Bad humidity value '%s' for sensor '%s'.",
                        raw_hum, name,
                    )

            is_online = temp_f is not None

            sensors.append(
                {
                    "name": name,
                    "temperature_f": temp_f,
                    "humidity": humidity,
                    "is_online": is_online,
                }
            )

        logger.info(
            "Parsed %d sensors (%d online) from Beestat.",
            len(sensors),
            sum(1 for s in sensors if s["is_online"]),
        )
        return sensors

    def get_hvac_mode(self) -> str:
        """Return the current HVAC operating mode.

        Reads from ``ecobee_thermostat`` data via the batch call.
        Possible values match Ecobee: ``'heat'``, ``'cool'``,
        ``'heatCool'``, ``'auto'``, ``'off'``, ``'auxHeatOnly'``.
        """
        data = self._fetch_all()
        thermostats = data.get("thermostat", {})

        # Beestat returns thermostats indexed by ID — grab the first one.
        for _tid, tstat in thermostats.items():
            settings = tstat.get("settings", {})
            mode = settings.get("hvacMode", "off")
            logger.info("HVAC mode via Beestat: %s", mode)
            return mode

        logger.warning("No thermostat data found in Beestat response.")
        return "off"
