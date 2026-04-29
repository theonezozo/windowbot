"""Shared fixtures for WindowBot tests."""

from __future__ import annotations

import pytest

from src.decision_engine import DecisionEngine


@pytest.fixture(autouse=True)
def _isolate_freshness_metrics(tmp_path, monkeypatch):
    """Redirect NWS freshness metrics to a per-test tmp file.

    Without this, any test that exercises NWSClient.get_outdoor_conditions
    end-to-end would append to the repo-root nws_freshness_metrics.jsonl
    and pollute the live runtime metrics file with fixture temperatures.
    """
    monkeypatch.setenv(
        "WINDOWBOT_METRICS_PATH", str(tmp_path / "test_metrics.jsonl")
    )


# ------------------------------------------------------------------
# Default config
# ------------------------------------------------------------------

@pytest.fixture
def default_config() -> dict:
    """Standard config matching architecture spec defaults."""
    return {
        "hysteresis_open_diff": 1.0,
        "hysteresis_close_diff": 1.0,
        "max_outdoor_humidity": 80,
        "max_aqi_threshold": 100,
        "min_aqi_for_opening": 50,
        "allowed_hvac_modes": ["cool", "heatCool", "auto"],
        "comfort_temp_max": 72.0,
        "enable_humidity_gate": True,
        "enable_aqi_gate": True,
    }


@pytest.fixture
def engine(default_config: dict) -> DecisionEngine:
    """A DecisionEngine wired with default config."""
    return DecisionEngine(default_config)


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def make_sensors(
    temps: list[float | None],
    names: list[str] | None = None,
    *,
    online: list[bool] | None = None,
) -> list[dict]:
    """Build a list of sensor dicts.

    Args:
        temps: Temperature readings (``None`` → missing reading).
        names: Sensor names; defaults to ``sensor_0``, ``sensor_1``, etc.
        online: Per-sensor online flag; defaults to all ``True``.
    """
    if names is None:
        names = [f"sensor_{i}" for i in range(len(temps))]
    if online is None:
        online = [True] * len(temps)
    sensors = []
    for name, temp, is_on in zip(names, temps, online):
        d: dict = {"name": name, "is_online": is_on}
        if temp is not None:
            d["temperature_f"] = temp
        sensors.append(d)
    return sensors


def outdoor_conditions(
    temp: float,
    humidity: float = 50.0,
    wind: float = 5.0,
) -> dict:
    """Build an outdoor-conditions dict."""
    return {
        "temperature_f": temp,
        "humidity": humidity,
        "wind_speed_mph": wind,
    }


def aqi_reading(value: int) -> dict:
    """Build an AQI result dict."""
    return {"aqi": value}


# Default floor group matching make_sensors defaults
DEFAULT_FLOOR = "upstairs"
DEFAULT_GROUP = ["sensor_0", "sensor_1", "sensor_2"]
