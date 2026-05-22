"""Ecobee API client for WindowBot.

Fetches indoor sensor data and HVAC state from Ecobee thermostats via
the Ecobee Developer API.  Handles OAuth 2.0 token refresh automatically.

Reference: https://www.ecobee.com/home/developer/api/introduction/index.shtml
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger("windowbot.ecobee")

ECOBEE_API_BASE = "https://api.ecobee.com"
TOKEN_URL = f"{ECOBEE_API_BASE}/token"
THERMOSTAT_URL = f"{ECOBEE_API_BASE}/1/thermostat"

# Ecobee reports temperature as Fahrenheit × 10
_TEMP_SCALE = 10.0

# Timeout for all Ecobee HTTP requests (seconds)
_REQUEST_TIMEOUT = 15


class EcobeeAuthError(Exception):
    """Raised when authentication fails irrecoverably (e.g. token revoked)."""


class EcobeeApiError(Exception):
    """Raised on unexpected API errors."""


class EcobeeClient:
    """Fetches sensor data and HVAC state from the Ecobee API.

    Args:
        client_id: Ecobee developer API key.
        refresh_token: Initial OAuth refresh token (from PIN authorization).
        state_manager: A :class:`src.state.StateManager` used to persist
            and retrieve OAuth tokens across invocations.
    """

    def __init__(
        self,
        client_id: str,
        refresh_token: str,
        state_manager,
    ) -> None:
        self._client_id = client_id
        self._state_manager = state_manager

        # Bootstrap tokens from persisted state; fall back to init value.
        stored = state_manager.get_oauth_tokens()
        self._access_token: str = stored.get("AccessToken", "")
        self._refresh_token: str = stored.get("RefreshToken", "") or refresh_token

    # ------------------------------------------------------------------
    # OAuth helpers
    # ------------------------------------------------------------------

    def _refresh_access_token(self) -> str:
        """Exchange the refresh token for a new access/refresh pair.

        Persists the new tokens via *state_manager* and returns the fresh
        access token.

        Raises:
            EcobeeAuthError: If the refresh token has been revoked or the
                response indicates an unrecoverable auth failure.
        """
        logger.info("Refreshing Ecobee access token.")
        try:
            resp = requests.post(
                TOKEN_URL,
                params={
                    "grant_type": "refresh_token",
                    "code": self._refresh_token,
                    "client_id": self._client_id,
                },
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise EcobeeApiError(f"Network error refreshing token: {exc}") from exc

        if resp.status_code == 401:
            raise EcobeeAuthError(
                "Ecobee refresh token revoked — re-authorization required."
            )

        if not resp.ok:
            raise EcobeeApiError(
                f"Token refresh failed ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        self._access_token = data["access_token"]
        self._refresh_token = data["refresh_token"]

        self._state_manager.update_oauth_tokens(
            {
                "AccessToken": self._access_token,
                "RefreshToken": self._refresh_token,
                "ExpiresAt": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("Ecobee access token refreshed successfully.")
        return self._access_token

    def _authed_get(self, url: str, params: dict | None = None) -> dict:
        """GET with automatic token refresh on 401.

        Returns:
            Parsed JSON response body.
        """
        for attempt in range(2):
            if not self._access_token or attempt == 1:
                self._refresh_access_token()

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            }
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=_REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                raise EcobeeApiError(f"Network error: {exc}") from exc

            if resp.status_code == 401:
                logger.warning("Ecobee returned 401 — refreshing token (attempt %d).", attempt + 1)
                continue

            if not resp.ok:
                raise EcobeeApiError(
                    f"Ecobee API error ({resp.status_code}): {resp.text}"
                )

            body = resp.json()
            status = body.get("status", {})
            if status.get("code", 0) == 14:
                # Code 14 = token expired — retry once.
                logger.warning("Ecobee token expired (code 14) — refreshing.")
                continue
            if status.get("code", 0) != 0:
                raise EcobeeApiError(
                    f"Ecobee API status error: {status}"
                )
            return body

        raise EcobeeAuthError("Failed to authenticate after token refresh.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_thermostat_data(self) -> dict:
        """Fetch full thermostat data including sensors and runtime.

        Returns:
            The first thermostat dict from the API response, with
            temperatures already converted from F×10 to float °F.
        """
        selection = (
            '{"selection":{"selectionType":"registered",'
            '"includeRuntime":true,"includeSensors":true,'
            '"includeEquipmentStatus":true}}'
        )
        body = self._authed_get(THERMOSTAT_URL, params={"json": selection})

        thermostats = body.get("thermostatList", [])
        if not thermostats:
            raise EcobeeApiError("No thermostats found in account.")

        return thermostats[0]

    def get_sensors(self) -> list[dict]:
        """Parse remote sensors from thermostat data.

        Returns:
            List of sensor dicts, each containing:
            - ``name`` (str): Sensor display name.
            - ``temperature_f`` (float | None): Temperature in °F, or None if offline.
            - ``humidity`` (int | None): Relative humidity %, or None if unavailable.
            - ``is_online`` (bool): ``True`` if the sensor appears to have valid data.
        """
        thermostat = self.get_thermostat_data()
        raw_sensors = thermostat.get("remoteSensors", [])
        sensors: list[dict] = []

        for raw in raw_sensors:
            name = raw.get("name", "Unknown")
            caps = {c["type"]: c["value"] for c in raw.get("capability", [])}

            temp_raw = caps.get("temperature")
            humidity_raw = caps.get("humidity")

            temp_f: float | None = None
            if temp_raw is not None and temp_raw != "":
                try:
                    temp_f = int(temp_raw) / _TEMP_SCALE
                except (ValueError, TypeError):
                    logger.warning("Bad temperature value '%s' for sensor '%s'.", temp_raw, name)

            humidity: int | None = None
            if humidity_raw is not None and humidity_raw != "":
                try:
                    humidity = int(humidity_raw)
                except (ValueError, TypeError):
                    pass

            # A sensor is online if it has a valid temperature reading.
            is_online = temp_f is not None

            sensors.append(
                {
                    "name": name,
                    "temperature_f": temp_f,
                    "humidity": humidity,
                    "is_online": is_online,
                    # Provenance tag — keeps Ecobee/Beestat dicts symmetric so the
                    # orchestrator and status page don't need provider branching.
                    # Direct Ecobee has no upstream sync timestamp, so age is unknown.
                    "source": "ecobee:direct",
                    "data_age_seconds": None,
                }
            )

        logger.info(
            "Parsed %d sensors (%d online).",
            len(sensors),
            sum(1 for s in sensors if s["is_online"]),
        )
        return sensors

    def get_hvac_mode(self) -> str:
        """Return the current HVAC operating mode.

        Possible values: ``'heat'``, ``'cool'``, ``'heatCool'``, ``'auto'``,
        ``'off'``, ``'auxHeatOnly'``.
        """
        thermostat = self.get_thermostat_data()
        settings = thermostat.get("settings", {})
        mode = settings.get("hvacMode", "off")
        logger.info("HVAC mode: %s", mode)
        return mode
