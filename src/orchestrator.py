"""Main orchestration logic for WindowBot.

Coordinates the fetch → decide → notify pipeline each time the
timer trigger fires.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import get_config
from src.state import get_state_manager
from src.notifier import send_notification
from src.ecobee_client import EcobeeClient, EcobeeAuthError, EcobeeApiError
from src.beestat_client import BeestatClient, BeestatAuthError, BeestatApiError
from src.nws_client import NWSClient, NWSError
from src.openmeteo_client import OpenMeteoClient, OpenMeteoError
from src.purpleair_client import PurpleAirClient
from src.airnow_client import AirNowClient
from src.decision_engine import DecisionEngine, FloorDecision, InsufficientDataError
from src.quiet_hours import get_quiet_hours, is_active, just_started, just_ended

logger = logging.getLogger("windowbot")

# Notification cooldown (seconds) — 1 hour between non-urgent notifications.
_NOTIFICATION_COOLDOWN = 3600

# Module-level NWSClient — kept alive across run_check() cycles so its LKG
# station cache survives between calls.
_nws_client: "NWSClient | None" = None
_nws_client_key: tuple = ()


def _get_nws_client(config: dict) -> NWSClient:
    """Return or create the NWSClient singleton."""
    global _nws_client, _nws_client_key
    lat, lon = config["user_latitude"], config["user_longitude"]
    key = (lat, lon, NWSClient)
    if _nws_client is None or _nws_client_key != key:
        _nws_client = NWSClient(lat, lon)
        _nws_client_key = key
    return _nws_client


def run_check() -> None:
    """Top-level orchestration called by the timer trigger.

    Steps:
        1. Load configuration
        2. Fetch data from all sources (Ecobee, NWS, AQI)
        3. For each floor, run the decision engine
        4. If state changed, persist new state and send notification
    """
    try:
        logger.info("WindowBot check starting at %s", datetime.now(timezone.utc).isoformat())

        config = get_config()
        state_mgr = get_state_manager()

        # ------------------------------------------------------------------
        # Step 1: Fetch data from external APIs
        # ------------------------------------------------------------------
        try:
            sensors, hvac_mode = _fetch_indoor_data(config, state_mgr)
        except (EcobeeAuthError, BeestatAuthError) as exc:
            provider = "Beestat" if isinstance(exc, BeestatAuthError) else "Ecobee"
            logger.error("%s auth failed: %s", provider, exc)
            send_notification(
                title=f"⚠️ WindowBot — {provider} Auth Failed",
                message=(
                    f"Your {provider} credentials are invalid or expired. "
                    "WindowBot cannot read sensor data until this is fixed."
                ),
                priority="urgent",
                urgent=True,
            )
            return
        except (EcobeeApiError, BeestatApiError) as exc:
            logger.error("Indoor sensor API error: %s", exc)
            return

        # Fetch outdoor conditions.
        # Open-Meteo is always attempted as a fresh peer alongside NWS —
        # if its reading is ≤30 min old it is blended into the NWS median.
        # If NWS station discovery fails entirely, Open-Meteo is the sole source.
        lat, lon = config["user_latitude"], config["user_longitude"]
        om = OpenMeteoClient(lat, lon)
        om_obs: "dict | None" = None
        try:
            om_obs = om.get_observation()
            # OpenMeteoClient already logs peer reading with freshness in get_observation()
        except OpenMeteoError as exc:
            logger.warning("Open-Meteo peer unavailable (OpenMeteoError): %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Open-Meteo peer unavailable (unexpected %s): %s", type(exc).__name__, exc)

        outdoor = None
        try:
            outdoor = _get_nws_client(config).get_outdoor_conditions(
                peer_observations=[om_obs] if om_obs else None
            )
        except NWSError as exc:
            logger.warning("NWS failed (%s).", exc)
            if om_obs:
                logger.info("Using Open-Meteo peer as sole outdoor source.")
                outdoor = {
                    "temperature_f": om_obs["temperature_f"],
                    "humidity": om_obs["humidity"],
                    "wind_speed_mph": om_obs["wind_speed_mph"],
                    "station_count": 1,
                    "is_fallback": True,
                    "used_cache": False,
                    "source": "openmeteo",
                }

        if outdoor is None:
            # NWS failed and no fresh OM peer — try OM without freshness check.
            try:
                outdoor = om.get_outdoor_conditions()
                logger.info("Outdoor data from Open-Meteo (last-resort fallback).")
            except OpenMeteoError as exc:
                logger.error("All outdoor sources failed. Open-Meteo: %s", exc)
                return

        # ------------------------------------------------------------------
        # Step 2: Evaluate each floor independently (lazy AQI fetching)
        # ------------------------------------------------------------------
        engine = DecisionEngine(config)
        aqi_cache: dict = {}  # shared across floors within this cycle

        for floor_name, sensor_names in [
            ("upstairs", config["upstairs_sensors"]),
            ("downstairs", config["downstairs_sensors"]),
        ]:
            if not sensor_names:
                continue

            # Pre-check: does this floor actually need AQI data?
            previous = state_mgr.get_floor_state(floor_name)
            last_state = previous.get("CurrentState", "CLOSED")

            needs, skip_reason = engine.needs_aqi(
                floor_sensors=sensors,
                outdoor=outdoor,
                hvac_mode=hvac_mode,
                last_state=last_state,
                floor_group=sensor_names,
            )

            if needs:
                if "result" not in aqi_cache:
                    aqi_cache["result"] = _fetch_aqi(config)
                aqi_data = aqi_cache["result"]
            else:
                logger.info(
                    "Skipping AQI fetch for %s — %s", floor_name, skip_reason,
                )
                aqi_data = {"aqi": 0, "source": "skipped"}

            _evaluate_floor(
                floor_name, sensor_names, sensors, outdoor, aqi_data,
                hvac_mode, engine, state_mgr, config,
            )

        logger.info("WindowBot check complete.")

    except Exception:
        logger.exception("WindowBot check failed — will retry next cycle.")


def _fetch_indoor_data(config: dict, state_mgr: StateManager) -> tuple[list[dict], str]:
    """Fetch indoor sensors and HVAC mode from the configured provider.

    Uses ``config["indoor_provider"]`` to select between Beestat (default)
    and Ecobee.

    Returns:
        Tuple of (sensors list, hvac_mode string).

    Raises:
        BeestatAuthError / EcobeeAuthError: On authentication failure.
        BeestatApiError / EcobeeApiError: On other API errors.
    """
    provider = config.get("indoor_provider", "beestat")

    if provider == "beestat" and config.get("beestat_api_key"):
        logger.info("Using Beestat for indoor sensor data.")
        client = BeestatClient(api_key=config["beestat_api_key"])
    else:
        if provider == "beestat" and not config.get("beestat_api_key"):
            logger.info("Beestat selected but no API key set — falling back to Ecobee.")
        else:
            logger.info("Using Ecobee for indoor sensor data.")
        client = EcobeeClient(
            client_id=config["ecobee_client_id"],
            refresh_token=config["ecobee_refresh_token"],
            state_manager=state_mgr,
        )

    sensors = client.get_sensors()
    hvac_mode = client.get_hvac_mode()
    return sensors, hvac_mode


def _fetch_aqi(config: dict) -> dict:
    """Fetch AQI: PurpleAir (median of 3) → AirNow fallback.

    Returns a dict compatible with DecisionEngine.decide(aqi=...) parameter.
    Always returns a dict with at least {"aqi": <value>}.
    """
    # Try PurpleAir first
    if config.get("purpleair_api_key"):
        try:
            pa = PurpleAirClient(
                config["user_latitude"], config["user_longitude"],
                api_key=config["purpleair_api_key"],
            )
            result = pa.get_aqi()
            if result and result.get("aqi") is not None:
                logger.info("AQI from PurpleAir: %d (sensors: %d)", result["aqi"], result.get("sensor_count", 0))
                return result
        except Exception:
            logger.warning("PurpleAir failed, falling back to AirNow.", exc_info=True)

    # Fallback to AirNow
    if config.get("airnow_api_key"):
        try:
            airnow = AirNowClient(
                config["airnow_api_key"], config["user_latitude"], config["user_longitude"],
            )
            result = airnow.get_aqi()
            if result and result.get("aqi") is not None:
                logger.info("AQI from AirNow: %d", result["aqi"])
                return result
        except Exception:
            logger.warning("AirNow also failed.", exc_info=True)

    logger.warning("No AQI data available — proceeding without AQI gate.")
    return {"aqi": 0, "source": "none"}


def _log_floor_sensor_range(
    floor_name: str, all_sensors: list[dict], sensor_names: list[str]
) -> None:
    """Log the warmest and coolest sensors for a floor (regardless of online status)."""
    valid = [
        s for s in all_sensors
        if s["name"] in sensor_names
        and s.get("temperature_f") is not None
    ]
    if not valid:
        return
    warmest = max(valid, key=lambda s: s["temperature_f"])
    coolest = min(valid, key=lambda s: s["temperature_f"])
    warmest_status = "[online]" if warmest.get("is_online", False) else "[offline]"
    coolest_status = "[online]" if coolest.get("is_online", False) else "[offline]"
    if warmest["name"] == coolest["name"]:
        logger.info(
            "Floor %s indoor: %s → %.1f°F %s (only sensor)",
            floor_name, warmest["name"], warmest["temperature_f"], warmest_status,
        )
    else:
        logger.info(
            "Floor %s indoor — warmest: %s %.1f°F %s  coolest: %s %.1f°F %s",
            floor_name,
            warmest["name"], warmest["temperature_f"], warmest_status,
            coolest["name"], coolest["temperature_f"], coolest_status,
        )


def _log_coolest_sensor(
    floor_name: str, all_sensors: list[dict], sensor_names: list[str]
) -> None:
    """Log the coolest indoor sensor for a floor before deciding (regardless of online status)."""
    valid = [
        s for s in all_sensors
        if s["name"] in sensor_names
        and s.get("temperature_f") is not None
    ]
    if not valid:
        logger.warning(
            "Coolest indoor (%s): no valid sensor data — no temperature readings",
            floor_name,
        )
        return
    coolest = min(valid, key=lambda s: s["temperature_f"])
    status = "[online]" if coolest.get("is_online", False) else "[offline]"
    
    # Build sensor list summary: "sensor1 71.2°F [offline], sensor2 72.5°F [online], ..."
    sensor_details = ", ".join(
        f"{s['name']} {s['temperature_f']:.1f}°F [{'online' if s.get('is_online', False) else 'offline'}]"
        for s in sorted(valid, key=lambda x: x["temperature_f"])
    )
    
    logger.info(
        "Coolest indoor (%s): %s → %.1f°F %s (of %d sensor%s: %s)",
        floor_name, coolest["name"], coolest["temperature_f"], status,
        len(valid), "s" if len(valid) > 1 else "", sensor_details,
    )


def _try_precool_notification(
    floor_name: str,
    outdoor: dict,
    aqi_data: dict,
    config: dict,
    now: datetime,
    new_state_record: dict,
    engine: DecisionEngine,
    all_sensors: list[dict],
    sensor_names: list[str],
) -> None:
    """Send a precool opportunity notification if temperature and AQI conditions are met.

    Sets ``new_state_record["LastNotificationTime"]`` on success. Does nothing
    (silent no-op) when conditions are not met — no "no precool today" notice.
    """
    # Guard: we need real AQI data, not a skipped or unavailable reading.
    if aqi_data.get("source") in ("skipped", "none"):
        logger.info("Morning precool skipped (%s): AQI unavailable.", floor_name)
        return

    outdoor_temp = outdoor.get("temperature_f")
    if outdoor_temp is None:
        logger.info("Morning precool skipped (%s): no outdoor temperature.", floor_name)
        return

    try:
        _, indoor_coolest = engine.get_floor_temps(all_sensors, sensor_names)
    except InsufficientDataError:
        logger.info("Morning precool skipped (%s): no valid indoor sensor data.", floor_name)
        return

    hysteresis: float = float(config.get("hysteresis_open_diff", 1.0))
    aqi_threshold: int = int(config.get("min_aqi_for_opening", 50))
    aqi_value: int = aqi_data.get("aqi", 999)

    if outdoor_temp >= indoor_coolest - hysteresis:
        logger.info(
            "Morning precool skipped (%s): outdoor %.1f°F not cool enough vs indoor %.1f°F.",
            floor_name, outdoor_temp, indoor_coolest,
        )
        return

    if aqi_value >= aqi_threshold:
        logger.info(
            "Morning precool skipped (%s): AQI %d not good enough (threshold %d).",
            floor_name, aqi_value, aqi_threshold,
        )
        return

    delta = indoor_coolest - outdoor_temp
    title = f"🌅 WindowBot — Precool Opportunity ({floor_name.title()})"
    body = (
        f"Quiet hours ended. Outdoor {outdoor_temp:.1f}°F is {delta:.1f}°F cooler than "
        f"indoor {indoor_coolest:.1f}°F, AQI {aqi_value} (good). "
        f"Open windows now to precool before it warms up."
    )
    send_notification(title=title, message=body, priority="default", urgent=False)
    new_state_record["LastNotificationTime"] = now.isoformat()
    logger.info("Precool opportunity notification sent for %s.", floor_name)


def _evaluate_floor(
    floor_name: str,
    sensor_names: list[str],
    all_sensors: list[dict],
    outdoor: dict,
    aqi_data: dict,
    hvac_mode: str,
    engine: DecisionEngine,
    state_mgr: StateManager,
    config: dict | None = None,
) -> None:
    """Run the decision pipeline for a single floor.

    Args:
        floor_name: "upstairs" or "downstairs"
        sensor_names: Names of sensors assigned to this floor
        all_sensors: All parsed Ecobee sensor readings
        outdoor: Aggregated outdoor conditions dict
        aqi_data: AQI result dict (from PurpleAir or AirNow)
        hvac_mode: Current Ecobee HVAC mode string
        engine: DecisionEngine instance
        state_mgr: StateManager for reading/writing state
        config: Full config dict from get_config() (optional; quiet-hours
            logic is skipped when None)
    """
    try:
        previous = state_mgr.get_floor_state(floor_name)
        last_state = previous.get("CurrentState", "CLOSED")
        last_notify_time = previous.get("LastNotificationTime")
        now = datetime.now(timezone.utc)
        new_state_record: dict = {}

        # -- Quiet hours logic --
        qh = get_quiet_hours(config) if config is not None else None
        if qh is not None:
            last_check_str = previous.get("LastDecisionTime")
            last_check_utc = (
                datetime.fromisoformat(last_check_str) if last_check_str else None
            )

            if last_check_utc and just_started(now, last_check_utc, qh):
                # Entering quiet hours — presume CLOSED, no notification.
                if last_state == "OPEN":
                    new_state_record["OpenedBeforeQuietHours"] = True
                new_state_record["QuietHoursActive"] = True
                new_state_record["LastQuietHoursStart"] = now.isoformat()
                state_mgr.update_floor_state(
                    floor_name,
                    {
                        **previous,
                        **new_state_record,
                        "CurrentState": "CLOSED",
                        "DecisionReason": "Quiet hours started",
                        "LastDecisionTime": now.isoformat(),
                    },
                )
                logger.info(
                    "Quiet hours started (%s) — presume CLOSED, no notification.",
                    floor_name,
                )
                return

            if is_active(now, qh):
                # Mid-quiet hours — skip decision cycle entirely.
                new_state_record["QuietHoursActive"] = True
                state_mgr.update_floor_state(
                    floor_name,
                    {
                        **previous,
                        **new_state_record,
                        "LastDecisionTime": now.isoformat(),
                    },
                )
                logger.info("Quiet hours active (%s) — skipping decision cycle.", floor_name)
                return

            if last_check_utc and just_ended(now, last_check_utc, qh):
                # Exiting quiet hours — try morning precool notification, then
                # fall through to normal decide() below.
                new_state_record["QuietHoursActive"] = False
                new_state_record["LastQuietHoursEnd"] = now.isoformat()
                opened_before = previous.get("OpenedBeforeQuietHours", False)
                if opened_before:
                    _try_precool_notification(
                        floor_name, outdoor, aqi_data, config, now,
                        new_state_record, engine, all_sensors, sensor_names,
                    )
                new_state_record["OpenedBeforeQuietHours"] = False
                # Refresh last_notify_time in case precool sent a notification so
                # the cooldown check below correctly sees it.
                last_notify_time = new_state_record.get(
                    "LastNotificationTime", last_notify_time
                )
                # Fall through to normal decide() after quiet hours.

        # Log the coolest indoor sensor before deciding so we can see inputs.
        _log_coolest_sensor(floor_name, all_sensors, sensor_names)

        decision: FloorDecision = engine.decide(
            floor=floor_name,
            floor_sensors=all_sensors,
            outdoor=outdoor,
            aqi=aqi_data,
            hvac_mode=hvac_mode,
            last_state=last_state,
            floor_group=sensor_names,
        )

        # Log warmest/coolest indoor sensor for this floor (name + temperature).
        # Sensor dicts have no per-sensor timestamp — readings are always from
        # this cycle (< polling interval old).
        _log_floor_sensor_range(floor_name, all_sensors, sensor_names)

        # Merge decision results into new_state_record (preserves any quiet-hours
        # fields already written above, e.g. from just_ended path).
        new_state_record.update({
            "CurrentState": decision.new_state,
            "LastDecisionTime": now.isoformat(),
            "LastOutdoorTemp": outdoor.get("temperature_f"),
            "LastAQI": aqi_data.get("aqi"),
            "DecisionReason": decision.reason,
        })

        if decision.changed:
            # Check notification cooldown (unless urgent)
            should_notify = decision.urgent  # always notify if urgent
            if not should_notify and last_notify_time:
                elapsed = (now - datetime.fromisoformat(last_notify_time)).total_seconds()
                should_notify = elapsed >= _NOTIFICATION_COOLDOWN
            elif not should_notify:
                should_notify = True  # first notification ever

            if should_notify:
                action = "🪟 Open" if decision.new_state == "OPEN" else "🚪 Close"
                title = f"WindowBot — {action} windows ({floor_name.title()})"
                send_notification(
                    title=title,
                    message=decision.reason,
                    priority="urgent" if decision.urgent else "default",
                    urgent=decision.urgent,
                )
                new_state_record["LastNotificationTime"] = now.isoformat()
                logger.info("Notified: %s → %s (%s)", last_state, decision.new_state, floor_name)
            else:
                logger.info(
                    "State changed %s → %s (%s) but cooldown active.",
                    last_state, decision.new_state, floor_name,
                )

        state_mgr.update_floor_state(floor_name, new_state_record)
        logger.info("Floor %s: %s (reason: %s)", floor_name, decision.new_state, decision.reason)

    except InsufficientDataError as exc:
        logger.warning("Floor '%s': insufficient sensor data — %s", floor_name, exc)
    except Exception:
        logger.exception("Error evaluating floor '%s'.", floor_name)
