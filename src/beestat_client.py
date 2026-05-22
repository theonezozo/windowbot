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
from datetime import datetime, timezone

import requests

logger = logging.getLogger("windowbot.beestat")

BEESTAT_API_URL = "https://app.beestat.io/api/"

_REQUEST_TIMEOUT = 15

# Sync-age threshold above which we WARN. Mirrors the indoor analogue of the
# outdoor 20-min freshness gate; Beestat's own UI used 15 min as its
# "ecobee is down" threshold so 10 min is conservative but actionable.
_SYNC_AGE_WARN_SECONDS = 600


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

        The batch is prefixed with ``thermostat.sync`` and ``sensor.sync`` so
        Beestat pulls fresh data from Ecobee BEFORE the reads execute against
        its MySQL mirror. Both sync methods are cached server-side at 3 min
        (see https://github.com/beestat/app api/thermostat.php sync() /
        api/sensor.php sync()) so calling them every poll is rate-safe.
        ``user.read_id`` is included so we can surface the sync_status
        timestamps and reason about freshness.

        Returns:
            Dict with ``"sensors"``, ``"thermostat"``, and ``"user"`` keys
            from the batch response (plus the two ``*_sync`` aliases, which
            we don't use directly but are required to force the refresh).
        """
        calls = [
            # Force Beestat to pull fresh data from Ecobee BEFORE the reads.
            # Server-side cached at 3 min so safe to call every poll.
            {
                "resource": "thermostat",
                "method": "sync",
                "arguments": "{}",
                "alias": "thermostat_sync",
            },
            {
                "resource": "sensor",
                "method": "sync",
                "arguments": "{}",
                "alias": "sensor_sync",
            },
            # User read — surfaces sync_status timestamps so we know how
            # fresh the freshly-synced data actually is.
            {
                "resource": "user",
                "method": "read_id",
                "arguments": "{}",
                "alias": "user",
            },
            # Existing reads — now operate on freshly-synced data.
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
    # Sync-status helpers
    # ------------------------------------------------------------------

    def _extract_sync_age_seconds(self, data: dict) -> float | None:
        """Compute the age in seconds of the oldest Beestat sync timestamp.

        Reads ``data["user"][<user_id>]["sync_status"]["thermostat"]`` and
        ``[...]["sync_status"]["sensor"]``, both ISO timestamps written by
        Beestat after each successful Ecobee sync. We take the OLDER of the
        two — that's the worst-case staleness for the indoor pool, mirroring
        the outdoor ``_aggregate`` "oldest contributor" semantic.

        Returns the age in seconds, or ``None`` when ``sync_status`` is
        absent or unparseable (the caller treats ``None`` as "unknown" —
        NOT zero, NOT a large default).
        """
        user_dict = data.get("user", {})
        if not isinstance(user_dict, dict) or not user_dict:
            logger.info("Beestat sync_status unavailable — cannot determine data age")
            return None

        # `user` is keyed by user_id like the other resources; grab the
        # first/only entry.
        user_entry = next(iter(user_dict.values()), None)
        if not isinstance(user_entry, dict):
            logger.info("Beestat sync_status unavailable — cannot determine data age")
            return None

        sync_status = user_entry.get("sync_status") or {}
        thermo_iso = sync_status.get("thermostat")
        sensor_iso = sync_status.get("sensor")

        parsed: list[tuple[str, datetime]] = []
        for label, iso in (("thermostat", thermo_iso), ("sensor", sensor_iso)):
            if not iso:
                continue
            try:
                # Beestat returns "YYYY-MM-DD HH:MM:SS" in UTC without a tz
                # suffix in some responses, and ISO 8601 in others. Try both.
                if "T" in iso:
                    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                parsed.append((label, dt))
            except (ValueError, TypeError):
                logger.debug("Could not parse Beestat sync_status.%s = %r", label, iso)

        if not parsed:
            logger.info("Beestat sync_status unavailable — cannot determine data age")
            return None

        # Worst-case staleness = oldest sync timestamp.
        oldest_label, oldest_dt = min(parsed, key=lambda p: p[1])
        age_seconds = (datetime.now(timezone.utc) - oldest_dt).total_seconds()

        # Per-cycle INFO summary — one line covering both timestamps.
        thermo_str = (
            f"thermostat synced at {next((dt.isoformat() for lbl, dt in parsed if lbl == 'thermostat'), 'unknown')}"
        )
        sensor_str = (
            f"sensor synced at {next((dt.isoformat() for lbl, dt in parsed if lbl == 'sensor'), 'unknown')}"
        )
        logger.info(
            "Beestat sync age: %.0fs (%s, %s)", age_seconds, thermo_str, sensor_str
        )

        if age_seconds > _SYNC_AGE_WARN_SECONDS:
            logger.warning(
                "Beestat sync age %.0fs exceeds 10min threshold — "
                "data may be stale despite sync calls",
                age_seconds,
            )

        return age_seconds

    # ------------------------------------------------------------------
    # Public API  (matches EcobeeClient interface)
    # ------------------------------------------------------------------

    def get_sensors(self) -> list[dict]:
        """Parse sensor readings from Beestat.

        Returns:
            List of sensor dicts matching the EcobeeClient format::

                {"name": str, "temperature_f": float|None,
                 "humidity": int|None, "is_online": bool,
                 "source": str, "data_age_seconds": float|None}

            ``source`` is one of ``"beestat:live_temps"`` (temperature came
            from ``ecobee_thermostat.remoteSensors``) or
            ``"beestat:sensor-resource"`` (fell back to the sensor row's
            ``temperature`` column). All sensors in a single call share the
            same ``data_age_seconds`` — it's the upstream Beestat sync age,
            not a per-sensor reading age. ``None`` means "unknown".
        """
        data = self._fetch_all()
        sync_age_seconds = self._extract_sync_age_seconds(data)
        raw_sensors = data.get("sensors", {})

        # Build a lookup from ecobee_sensor_id → live temp from the
        # ecobee_thermostat remoteSensors, so we can cross-check (and
        # override stale sensor-resource values when available).
        # ecobee_thermostat stores capability temps in tenths of °F (e.g. 740
        # = 74.0°F); divide by 10 to get actual °F.
        live_temps: dict[int, float] = {}
        for _tid, tstat in data.get("thermostat", {}).items():
            for rs in tstat.get("remoteSensors", []):
                rs_id = rs.get("id", "")  # e.g. "rs:100123456:1"
                ecobee_sensor_id = rs.get("ecobee_sensor_id")
                for cap in rs.get("capability", []):
                    if cap.get("type") == "temperature":
                        raw_val = cap.get("value")
                        try:
                            raw_int = int(raw_val)
                            if raw_int > 0:
                                live_temps_key = ecobee_sensor_id or rs_id
                                live_temps[live_temps_key] = raw_int / 10.0
                        except (ValueError, TypeError):
                            pass
        if live_temps:
            logger.debug(
                "Live temps from ecobee_thermostat remoteSensors: %s", live_temps
            )

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
        live_temps_count = 0
        fallback_count = 0
        for sensor_id, raw in selected:
            name = raw.get("name", "Unknown")

            # Temperature: prefer live reading from ecobee_thermostat
            # remoteSensors (tenths of °F, divided by 10) when available.
            # Fall back to the sensor-resource value which may lag by a few
            # minutes because Beestat aggregates it for charting.
            #
            # NOTE: After the ``*.sync`` prefix landed in ``_fetch_all`` both
            # paths read from the same per-sync DB snapshot, so the freshness
            # delta between them is zero. The branching is retained so we can
            # observe which path was taken via the ``source`` tag — a future
            # cleanup may collapse this.
            temp_f: float | None = None
            source: str | None = None
            ecobee_sensor_id = raw.get("ecobee_sensor_id")
            if ecobee_sensor_id and ecobee_sensor_id in live_temps:
                temp_f = live_temps[ecobee_sensor_id]
                source = "beestat:live_temps"
                live_temps_count += 1
                logger.debug(
                    "Sensor '%s': using live temp %.1f°F from ecobee_thermostat",
                    name, temp_f,
                )
            else:
                raw_temp = raw.get("temperature")
                logger.debug(
                    "Sensor '%s': raw sensor-resource temperature = %r", name, raw_temp
                )
                source = "beestat:sensor-resource"
                fallback_count += 1
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
                    "source": source,
                    "data_age_seconds": sync_age_seconds,
                }
            )

        logger.info(
            "Beestat get_sensors: %d sensors (%d via live_temps, %d via "
            "sensor-resource fallback)",
            len(sensors), live_temps_count, fallback_count,
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
