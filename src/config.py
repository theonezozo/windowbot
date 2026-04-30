"""Configuration loader for WindowBot.

Reads all settings from environment variables and provides typed defaults
matching the architecture spec (Section 5.1).
"""

import os
from typing import Any


def _env(key: str, default: Any = None) -> str | None:
    """Read an environment variable, returning *default* if unset or empty."""
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return value


def _env_float(key: str, default: float) -> float:
    val = _env(key)
    return float(val) if val is not None else default


def _env_int(key: str, default: int) -> int:
    val = _env(key)
    return int(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_list(key: str, default: list[str] | None = None) -> list[str]:
    val = _env(key)
    if val is None:
        return default or []
    return [item.strip() for item in val.split(",") if item.strip()]


def get_config() -> dict:
    """Load all WindowBot configuration from environment variables.

    Returns a dict with typed values ready for use by the orchestrator
    and decision engine.
    """
    return {
        # --- API credentials ---
        "ecobee_client_id": _env("ECOBEE_CLIENT_ID", ""),
        "ecobee_refresh_token": _env("ECOBEE_REFRESH_TOKEN", ""),
        "beestat_api_key": _env("BEESTAT_API_KEY", ""),

        # --- Indoor sensor provider ---
        "indoor_provider": _env("INDOOR_PROVIDER", "beestat"),
        "airnow_api_key": _env("AIRNOW_API_KEY", ""),
        "purpleair_api_key": _env("PURPLEAIR_API_KEY", ""),

        # --- Outdoor weather provider ---
        "synoptic_api_key": _env("SYNOPTIC_API_KEY", ""),
        "wu_api_key": _env("WU_API_KEY", ""),
        "outdoor_provider": _env("OUTDOOR_PROVIDER", "synoptic"),

        # --- Location ---
        "user_latitude": _env_float("USER_LATITUDE", 0.0),
        "user_longitude": _env_float("USER_LONGITUDE", 0.0),

        # --- Notifications ---
        "ntfy_topic": _env("NTFY_TOPIC", ""),

        # --- Sensor grouping ---
        "upstairs_sensors": _env_list("UPSTAIRS_SENSORS"),
        "downstairs_sensors": _env_list("DOWNSTAIRS_SENSORS"),

        # --- Decision thresholds ---
        "hysteresis_open_diff": _env_float("HYSTERESIS_OPEN_DIFF", 1.0),
        "hysteresis_close_diff": _env_float("HYSTERESIS_CLOSE_DIFF", 1.0),
        "max_outdoor_humidity": _env_int("MAX_OUTDOOR_HUMIDITY", 80),
        "max_aqi_threshold": _env_int("MAX_AQI_THRESHOLD", 100),
        "min_aqi_for_opening": _env_int("MIN_AQI_FOR_OPENING", 50),
        "comfort_temp_max": _env_float("COMFORT_TEMP_MAX", 72.0),

        # --- Polling & runtime ---
        "polling_interval_minutes": _env_int("POLLING_INTERVAL_MINUTES", 10),
        "max_observation_age_minutes": _env_int("MAX_OBSERVATION_AGE_MINUTES", 30),
        "notification_cooldown_hours": _env_int("NOTIFICATION_COOLDOWN_HOURS", 1),

        # --- HVAC ---
        "allowed_hvac_modes": _env_list("ALLOWED_HVAC_MODES", ["cool", "heatCool", "auto"]),

        # --- Air quality ---
        "aq_provider": _env("AQ_PROVIDER", "purpleair"),

        # --- Feature flags ---
        "enable_humidity_gate": _env_bool("ENABLE_HUMIDITY_GATE", True),
        "enable_aqi_gate": _env_bool("ENABLE_AQI_GATE", True),
        "enable_wind_check": _env_bool("ENABLE_WIND_CHECK", False),

        # --- Quiet hours (all three required to enable; feature disabled if any is absent) ---
        "quiet_hours_start":    _env("QUIET_HOURS_START", None),    # "HH:MM" 24-hour local time
        "quiet_hours_end":      _env("QUIET_HOURS_END", None),      # "HH:MM" 24-hour local time
        "quiet_hours_timezone": _env("QUIET_HOURS_TIMEZONE", None), # IANA, e.g. "America/Los_Angeles"
    }
