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

        # Fetch outdoor conditions — NWS primary, Open-Meteo free fallback.
        # WU and Synoptic clients exist in src/ but are not active without
        # their respective API keys (WU_API_KEY, SYNOPTIC_API_KEY).
        outdoor = None
        try:
            outdoor = _get_nws_client(config).get_outdoor_conditions()
            logger.info("Outdoor data from NWS.")
        except NWSError as exc:
            logger.warning("NWS failed (%s), trying Open-Meteo.", exc)

        if outdoor is None:
            try:
                om = OpenMeteoClient(config["user_latitude"], config["user_longitude"])
                outdoor = om.get_outdoor_conditions()
                logger.info("Outdoor data from Open-Meteo (free fallback).")
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
                hvac_mode, engine, state_mgr,
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


def _evaluate_floor(
    floor_name: str,
    sensor_names: list[str],
    all_sensors: list[dict],
    outdoor: dict,
    aqi_data: dict,
    hvac_mode: str,
    engine: DecisionEngine,
    state_mgr: StateManager,
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
    """
    try:
        previous = state_mgr.get_floor_state(floor_name)
        last_state = previous.get("CurrentState", "CLOSED")
        last_notify_time = previous.get("LastNotificationTime")

        decision: FloorDecision = engine.decide(
            floor=floor_name,
            floor_sensors=all_sensors,
            outdoor=outdoor,
            aqi=aqi_data,
            hvac_mode=hvac_mode,
            last_state=last_state,
            floor_group=sensor_names,
        )

        # Persist new state regardless of notification
        now = datetime.now(timezone.utc)
        new_state_record = {
            "CurrentState": decision.new_state,
            "LastDecisionTime": now.isoformat(),
            "LastOutdoorTemp": outdoor.get("temperature_f"),
            "LastAQI": aqi_data.get("aqi"),
            "DecisionReason": decision.reason,
        }

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
