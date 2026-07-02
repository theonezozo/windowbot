"""Main orchestration logic for WindowBot.

Coordinates the fetch → decide → notify pipeline each time the
timer trigger fires.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from src.config import get_config
from src.state import get_state_manager
from src.notifier import send_notification
from src.ecobee_client import EcobeeClient, EcobeeAuthError, EcobeeApiError
from src.beestat_client import BeestatClient, BeestatAuthError, BeestatApiError
from src.nws_client import NWSClient, NWSError
from src.openmeteo_client import OpenMeteoClient, OpenMeteoError
from src.outdoor_validator import validate_outdoor_temperature, OutdoorValidationResult
from src.purpleair_client import PurpleAirClient
from src.airnow_client import AirNowClient
from src.decision_engine import DecisionEngine, FloorDecision, InsufficientDataError
from src.quiet_hours import get_quiet_hours, is_active, just_started, just_ended
from src.diagnostic import (
    SnapshotManager,
    FloorSnapshot,
    GlobalSnapshot,
    SensorReading,
    OutdoorStation,
    AQIStation,
    GateEvaluation,
    TemperatureHistoryEntry,
)

logger = logging.getLogger("windowbot")

# Notification cooldown (seconds) — 1 hour between non-urgent notifications.
# Retained for reference; transitions now notify via type-dedup (Option A).
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


# Module-level PurpleAirClient — kept alive across run_check() cycles so its
# discovered nearby-sensor ID cache survives between calls. Without this, the
# client (and its cache) was rebuilt every 10-minute run, forcing an expensive
# bounding-box discovery query on every cycle and burning PurpleAir points.
_purpleair_client: "PurpleAirClient | None" = None
_purpleair_client_key: tuple = ()


def _get_purpleair_client(config: dict) -> PurpleAirClient:
    """Return or create the PurpleAirClient singleton.

    Persisting the instance persists its sensor-ID cache across cycles.
    """
    global _purpleair_client, _purpleair_client_key
    lat, lon = config["user_latitude"], config["user_longitude"]
    api_key = config.get("purpleair_api_key")
    ttl_hours = config.get("purpleair_sensor_cache_hours", 12.0)
    key = (lat, lon, api_key, ttl_hours)
    if _purpleair_client is None or _purpleair_client_key != key:
        _purpleair_client = PurpleAirClient(
            lat, lon, api_key=api_key, sensor_cache_ttl_hours=ttl_hours,
        )
        _purpleair_client_key = key
    return _purpleair_client


def _record_validation_metric(
    now: datetime,
    raw_temp: float,
    result: "OutdoorValidationResult",
) -> None:
    """Append the outdoor jitter-validation outcome to the metrics JSONL.

    Best-effort: never raises. Reuses WINDOWBOT_METRICS_PATH so validation
    outcomes sit alongside the NWS freshness metrics.
    """
    try:
        import json
        import os
        metrics_path = os.environ.get(
            "WINDOWBOT_METRICS_PATH", "nws_freshness_metrics.jsonl"
        )
        record = {
            "type": "outdoor_validation",
            "timestamp": now.isoformat(),
            "reason": result.reason,
            "suppressed": result.suppressed,
            "raw_temp_f": round(raw_temp, 1),
            "validated_temp_f": round(result.temperature_f, 1),
            "delta_f": round(result.temperature_f - raw_temp, 2),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.debug("Could not write validation metric", exc_info=True)


def run_check() -> None:
    """Top-level orchestration called by the timer trigger.

    Steps:
        1. Load configuration
        2. Fetch data from all sources (Ecobee, NWS, AQI)
        3. For each floor, run the decision engine
        4. If state changed, persist new state and send notification
        5. Persist diagnostic snapshots for status page
    """
    poll_start = datetime.now(timezone.utc)
    errors: list[str] = []
    floor_snapshots: dict[str, FloorSnapshot] = {}
    
    try:
        logger.info("WindowBot check starting at %s", poll_start.isoformat())

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
            errors.append(f"{provider} auth failed: {exc}")
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
            errors.append(f"Indoor sensor API error: {exc}")
            return

        # Fetch outdoor conditions.
        # Open-Meteo is always attempted as a fresh peer alongside NWS —
        # if its reading is ≤20 min old it is blended into the NWS median.
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
            errors.append(f"NWS error: {exc}")
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
                    "observation_time": om_obs["timestamp"].isoformat() if om_obs.get("timestamp") else None,
                }

        if outdoor is None:
            # NWS failed and no fresh OM peer — try OM without freshness check.
            try:
                outdoor = om.get_outdoor_conditions()
                logger.info("Outdoor data from Open-Meteo (last-resort fallback).")
            except OpenMeteoError as exc:
                logger.error("All outdoor sources failed. Open-Meteo: %s", exc)
                errors.append(f"All outdoor sources failed: {exc}")
                return

        # Suppress availability-driven jitter in the fused outdoor temperature
        # without lagging genuine movement (see outdoor_validator).
        try:
            prev_outdoor_state = state_mgr.get_floor_state("__global__")
            _raw_outdoor_temp = outdoor["temperature_f"]
            _val = validate_outdoor_temperature(
                fused_temp=_raw_outdoor_temp,
                contributors=outdoor.get("contributors", []),
                prev_state=prev_outdoor_state,
                jitter_threshold_f=config.get("outdoor_jitter_threshold_f", 0.5),
                trend_window=config.get("outdoor_jitter_trend_window", 6),
                spike_max_rate_f=config.get("outdoor_spike_max_rate_f", 2.0),
            )
            if _val.suppressed:
                logger.info(
                    "Outdoor temp jitter suppressed: %.1f°F → %.1f°F (%s)",
                    outdoor["temperature_f"], _val.temperature_f, _val.reason,
                )
            outdoor["temperature_f"] = _val.temperature_f
            outdoor["validation_reason"] = _val.reason
            state_mgr.update_floor_state("__global__", _val.state_fields)
            _record_validation_metric(
                datetime.now(timezone.utc), _raw_outdoor_temp, _val
            )
        except Exception:
            logger.exception("Outdoor temp validation failed — using raw fused temp.")

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
            
            # Build snapshot for this floor
            try:
                decision = engine.decide(
                    floor=floor_name,
                    floor_sensors=sensors,
                    outdoor=outdoor,
                    aqi=aqi_data,
                    hvac_mode=hvac_mode,
                    last_state=last_state,
                    floor_group=sensor_names,
                )
                snapshot = _build_floor_snapshot(
                    floor_name, sensor_names, sensors, decision, outdoor,
                    aqi_data, engine, previous, datetime.now(timezone.utc),
                )
                floor_snapshots[floor_name] = snapshot
            except Exception as exc:
                logger.exception("Failed to build snapshot for %s.", floor_name)
                errors.append(f"Snapshot build failed for {floor_name}: {exc}")

        # ------------------------------------------------------------------
        # Step 3: Persist diagnostic snapshots
        # ------------------------------------------------------------------
        poll_end = datetime.now(timezone.utc)
        poll_duration = (poll_end - poll_start).total_seconds()
        
        qh = get_quiet_hours(config)
        quiet_active = is_active(poll_end, qh) if qh else False
        next_transition = None
        if qh:
            # Calculate next transition time
            if quiet_active:
                # Currently in quiet hours — next transition is end time
                next_transition = datetime.combine(poll_end.date(), qh.end)
                if next_transition.replace(tzinfo=None) <= poll_end.replace(tzinfo=None):
                    next_transition += timedelta(days=1)
            else:
                # Not in quiet hours — next transition is start time
                next_transition = datetime.combine(poll_end.date(), qh.start)
                if next_transition.replace(tzinfo=None) <= poll_end.replace(tzinfo=None):
                    next_transition += timedelta(days=1)
        
        next_poll = poll_end + timedelta(minutes=10)
        
        global_snapshot = GlobalSnapshot(
            poll_start=poll_start.isoformat(),
            poll_duration_seconds=round(poll_duration, 2),
            hvac_mode=hvac_mode,
            quiet_hours_active=quiet_active,
            quiet_hours_next_transition=next_transition.isoformat() if next_transition else None,
            next_poll_eta=next_poll.isoformat(),
            errors=errors,
        )

        # Build temperature history entry for this cycle.  Coolest valid
        # indoor reading per floor; outdoor median temp (or None).
        indoor_temps: dict[str, float | None] = {}
        for floor_name, snap in floor_snapshots.items():
            valid = [
                s.temperature_f for s in snap.indoor_sensors
                if s.temperature_f is not None
            ]
            indoor_temps[floor_name] = min(valid) if valid else None
        history_entry = TemperatureHistoryEntry(
            timestamp=poll_end.isoformat(),
            outdoor_temp_f=outdoor.get("temperature_f"),
            indoor_temps=indoor_temps,
        )

        _persist_snapshots(state_mgr, floor_snapshots, global_snapshot, history_entry)

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
            pa = _get_purpleair_client(config)
            result = pa.get_aqi()
            if result and result.get("aqi") is not None:
                return result
        except Exception as exc:
            # Surface the real reason inline (e.g. "402: Payment Required — out of
            # points") so a config/account problem is visible in logs instead of
            # being masked by the silent AirNow fallback below.
            logger.warning(
                "PurpleAir AQI unavailable — falling back to AirNow. Reason: %s",
                exc,
            )

    # Fallback to AirNow
    if config.get("airnow_api_key"):
        try:
            airnow = AirNowClient(
                config["airnow_api_key"], config["user_latitude"], config["user_longitude"],
            )
            result = airnow.get_aqi()
            if result and result.get("aqi") is not None:
                return result
        except Exception:
            logger.warning("AirNow also failed.", exc_info=True)

    logger.warning("No AQI data available — proceeding without AQI gate.")
    return {"aqi": 0, "source": "none"}


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
        last_notify_type = previous.get("LastNotificationType")
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
            candidate_type = "open" if decision.new_state == "OPEN" else "close"

            # Type dedup: never repeat a notification of the same type the user
            # last received (e.g. a second "close" while windows are already
            # shut). Urgent alerts bypass dedup.
            duplicate_type = (
                not decision.urgent and last_notify_type == candidate_type
            )

            # Option A: a genuine open<->close TRANSITION always notifies, even
            # within the legacy time cooldown. Dedup (above) handles same-type
            # repeats, so the time cooldown is no longer needed to gate
            # transitions — and keeping it active would silently drop a real
            # transition (e.g. an OPEN that flipped back from a humidity CLOSE
            # within the hour), which is bug #10. Only same-type duplicates are
            # suppressed now.
            should_notify = not duplicate_type

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
                new_state_record["LastNotificationType"] = candidate_type
                logger.info("Notified: %s → %s (%s)", last_state, decision.new_state, floor_name)
            else:
                logger.info(
                    "State changed %s → %s (%s) but suppressed duplicate '%s' notification.",
                    last_state, decision.new_state, floor_name, candidate_type,
                )

        state_mgr.update_floor_state(floor_name, new_state_record)
        logger.info("Floor %s: %s (reason: %s)", floor_name, decision.new_state, decision.reason)

    except InsufficientDataError as exc:
        logger.warning("Floor '%s': insufficient sensor data — %s", floor_name, exc)
    except Exception:
        logger.exception("Error evaluating floor '%s'.", floor_name)


def _build_floor_snapshot(
    floor_name: str,
    sensor_names: list[str],
    all_sensors: list[dict],
    decision: FloorDecision,
    outdoor: dict,
    aqi_data: dict,
    engine: DecisionEngine,
    previous_state: dict,
    now: datetime,
) -> FloorSnapshot:
    """Build a diagnostic snapshot for one floor."""
    # Indoor sensors
    floor_sensors = [s for s in all_sensors if s["name"] in sensor_names and s.get("temperature_f") is not None]
    coolest_temp = min((s["temperature_f"] for s in floor_sensors), default=None)
    
    sensor_readings = []
    for s in all_sensors:
        if s["name"] in sensor_names:
            sensor_readings.append(SensorReading(
                name=s["name"],
                temperature_f=s.get("temperature_f"),
                is_online=s.get("is_online", False),
                is_coolest=(s.get("temperature_f") == coolest_temp if coolest_temp else False),
                source=s.get("source"),
                data_age_seconds=s.get("data_age_seconds"),
            ))
    
    # Outdoor stations — we don't have per-station detail from NWS's aggregated result,
    # but we can infer based on source and count
    outdoor_stations = []
    outdoor_source = outdoor.get("source", "unknown")
    station_count = outdoor.get("station_count", 0)
    if station_count > 0:
        # Placeholder — we'd need NWS client to expose raw observations
        outdoor_stations.append(OutdoorStation(
            station_id=f"{outdoor_source}_aggregate",
            distance_mi=None,
            temperature_f=outdoor["temperature_f"],
            age_minutes=None,
        ))
    
    # AQI stations — similarly, we only have aggregated data
    aqi_stations = []
    aqi_source = aqi_data.get("source", "unknown")
    aqi_sensor_count = aqi_data.get("sensor_count", 0)
    if aqi_sensor_count > 0:
        aqi_stations.append(AQIStation(
            sensor_id=f"{aqi_source}_aggregate",
            distance_mi=None,
            aqi=aqi_data.get("aqi", 0),
            pm25=aqi_data.get("pm25"),
        ))
    
    # Gates evaluation — reconstruct based on decision reason
    gates = _evaluate_gates(engine, decision, outdoor, aqi_data, floor_sensors, sensor_names)
    
    # Last notification
    last_notif_time = previous_state.get("LastNotificationTime")
    last_notif_type = previous_state.get("LastNotificationType")
    if last_notif_type is None and last_notif_time:
        # Backward-compat: infer from current state for pre-Fix-1 snapshots.
        current_state = previous_state.get("CurrentState", "UNKNOWN")
        if current_state == "OPEN":
            last_notif_type = "open"
        elif current_state == "CLOSED":
            last_notif_type = "close"
    
    return FloorSnapshot(
        floor=floor_name,
        decision=decision.new_state,
        reason=decision.reason,
        indoor_sensors=sensor_readings,
        outdoor_temp_f=outdoor["temperature_f"],
        outdoor_source=outdoor_source,
        outdoor_stations=outdoor_stations,
        outdoor_humidity=outdoor.get("humidity"),
        aqi_value=aqi_data.get("aqi", 0),
        aqi_source=aqi_source,
        aqi_stations=aqi_stations,
        gates=gates,
        last_notification_type=last_notif_type,
        last_notification_time=last_notif_time,
        timestamp=now.isoformat(),
        outdoor_observation_time=outdoor.get("observation_time"),
        aqi_observation_time=aqi_data.get("observation_time"),
        outdoor_newest_observation_time=outdoor.get("newest_observation_time"),
        outdoor_contributor_count=(
            outdoor.get("contributor_count") or outdoor.get("station_count")
        ),
        outdoor_validation_reason=outdoor.get("validation_reason"),
    )


def _evaluate_gates(
    engine: DecisionEngine,
    decision: FloorDecision,
    outdoor: dict,
    aqi_data: dict,
    floor_sensors: list[dict],
    sensor_names: list[str],
) -> list[GateEvaluation]:
    """Build gate evaluation list based on engine config and decision."""
    gates = []
    
    # Humidity gate
    if engine.enable_humidity_gate and outdoor.get("humidity") is not None:
        humidity = outdoor["humidity"]
        passed = humidity <= engine.max_humidity
        gates.append(GateEvaluation(
            name="Humidity",
            passed=passed,
            threshold=f"≤ {engine.max_humidity}%",
            actual=f"{humidity:.0f}%",
        ))
    
    # AQI gate
    if engine.enable_aqi_gate:
        aqi = aqi_data.get("aqi", 0)
        # AQI gate is bidirectional: fail if >= 100 (close), neutral 50-99, pass if < 50 (can open)
        if aqi >= engine.aqi_close_threshold:
            passed = False  # urgent close
        elif aqi >= engine.aqi_open_threshold:
            passed = None  # neutral — don't block, but don't encourage
        else:
            passed = True  # good for opening
        gates.append(GateEvaluation(
            name="AQI",
            passed=passed if passed is not None else True,  # for display, treat neutral as pass
            threshold=f"< {engine.aqi_open_threshold} (good), ≥ {engine.aqi_close_threshold} (close)",
            actual=f"{aqi}",
        ))
    
    # Wind gate — not currently implemented in engine, but placeholder
    if outdoor.get("wind_speed_mph") is not None:
        gates.append(GateEvaluation(
            name="Wind",
            passed=True,  # no wind threshold currently enforced
            threshold="No limit",
            actual=f"{outdoor['wind_speed_mph']:.1f} mph",
        ))
    
    return gates


def _persist_snapshots(
    state_mgr: StateManager,
    floor_snapshots: dict[str, FloorSnapshot],
    global_snapshot: GlobalSnapshot,
    history_entry: TemperatureHistoryEntry | None = None,
) -> None:
    """Persist all diagnostic snapshots to Table Storage."""
    try:
        if hasattr(state_mgr, 'get_snapshot_table'):
            snapshot_table = state_mgr.get_snapshot_table()
            mgr = SnapshotManager(snapshot_table)
            
            for snapshot in floor_snapshots.values():
                mgr.save_floor_snapshot(snapshot)
            
            mgr.save_global_snapshot(global_snapshot)

            if history_entry is not None:
                try:
                    mgr.record_temperature_history(history_entry)
                except Exception:
                    # Defensive — record_temperature_history is already
                    # best-effort, but a history-write failure must never
                    # break the poll cycle.
                    logger.exception("Temperature history write failed.")

            logger.info("Diagnostic snapshots persisted successfully.")
        else:
            logger.warning("Snapshot table not available (local state mode).")
    except Exception:
        logger.exception("Failed to persist diagnostic snapshots.")
