"""Comprehensive tests for the WindowBot decision engine.

Covers all gate layers (HVAC → AQI → Humidity → Comfort → Temperature) with
hysteresis, boundary conditions, and combined-condition scenarios.
"""

from __future__ import annotations

import pytest

from src.decision_engine import DecisionEngine, FloorDecision, InsufficientDataError
from tests.conftest import (
    DEFAULT_FLOOR,
    DEFAULT_GROUP,
    aqi_reading,
    make_sensors,
    outdoor_conditions,
)


# ==================================================================
# Helpers
# ==================================================================

def _decide(
    engine: DecisionEngine,
    *,
    indoor_temps: list[float] | None = None,
    outdoor_temp: float = 68.0,
    humidity: float = 50.0,
    aqi: int = 30,
    hvac_mode: str = "cool",
    last_state: str = "CLOSED",
    sensors: list[dict] | None = None,
    sensor_names: list[str] | None = None,
    sensor_online: list[bool] | None = None,
    floor: str = DEFAULT_FLOOR,
    floor_group: list[str] | None = None,
) -> FloorDecision:
    """Convenience wrapper around ``engine.decide`` with sane defaults."""
    if sensors is None:
        indoor_temps = indoor_temps or [74.0]
        sensors = make_sensors(indoor_temps, names=sensor_names, online=sensor_online)
    if floor_group is None:
        floor_group = [s["name"] for s in sensors]
    return engine.decide(
        floor=floor,
        floor_sensors=sensors,
        outdoor=outdoor_conditions(outdoor_temp, humidity),
        aqi=aqi_reading(aqi),
        hvac_mode=hvac_mode,
        last_state=last_state,
        floor_group=floor_group,
    )


# ==================================================================
# 1. Basic Temperature Logic (1°F symmetric hysteresis)
# ==================================================================


class TestTemperatureLogic:
    """Temperature comparison with 1°F hysteresis on both sides."""

    def test_open_when_outdoor_cooler_than_warmest_minus_1(self, engine):
        """indoor 74°F, outdoor 72°F → diff=2 > 1 → OPEN."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=72.0, aqi=30)
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_no_open_when_diff_exactly_1(self, engine):
        """indoor 74°F, outdoor 73°F → diff=1 → NOT open (need strictly less)."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=73.0, aqi=30)
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_no_open_when_outdoor_warmer(self, engine):
        """outdoor 76°F, indoor 74°F → outdoor warmer → stay CLOSED."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=76.0, aqi=30)
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_close_when_outdoor_warmer_than_coolest_plus_1(self, engine):
        """OPEN, coolest 70°F, outdoor 72°F → diff=2 > 1 → CLOSE."""
        result = _decide(
            engine,
            indoor_temps=[70.0],
            outdoor_temp=72.0,
            aqi=30,
            last_state="OPEN",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is True

    def test_stay_open_when_outdoor_equals_coolest(self, engine):
        """OPEN, coolest 70°F, outdoor 70°F → diff=0 ≤ 1 → stay OPEN."""
        result = _decide(
            engine,
            indoor_temps=[70.0],
            outdoor_temp=70.0,
            aqi=30,
            last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert result.changed is False

    def test_stay_open_at_exact_close_threshold(self, engine):
        """OPEN, coolest 70°F, outdoor 71°F → diff=1 = threshold → stay OPEN (need >)."""
        result = _decide(
            engine,
            indoor_temps=[70.0],
            outdoor_temp=71.0,
            aqi=30,
            last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert result.changed is False

    def test_hysteresis_prevents_oscillation(self, engine):
        """Just barely crossed open threshold, then bounced back."""
        # First: open (outdoor 72, indoor 74 → diff 2 > 1 → OPEN)
        r1 = _decide(engine, indoor_temps=[74.0], outdoor_temp=72.0, aqi=30)
        assert r1.new_state == "OPEN"

        # Now outdoor warms to 73.5 — close threshold is coolest+1 = 75.
        # 73.5 < 75 → stay open.
        r2 = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=73.5,
            aqi=30,
            last_state="OPEN",
        )
        assert r2.new_state == "OPEN"

    @pytest.mark.parametrize(
        "outdoor, expected_state",
        [
            (72.9, "OPEN"),     # diff = 1.1, 72.9 < 73 → OPEN
            (72.0, "OPEN"),     # diff = 2 → OPEN
            (73.0, "CLOSED"),   # diff = 1 (exactly) → not open
            (73.1, "CLOSED"),   # diff = 0.9 < 1 → not open
            (60.0, "OPEN"),     # large diff → OPEN
        ],
    )
    def test_open_boundary_parametrized(self, engine, outdoor, expected_state):
        """Parametrized boundary tests for CLOSED→OPEN transition.

        Indoor = 74°F. Open threshold = 74 - 1 = 73°F. Need outdoor < 73.
        """
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=outdoor, aqi=30)
        assert result.new_state == expected_state

    @pytest.mark.parametrize(
        "outdoor, expected_state",
        [
            (71.0, "OPEN"),     # below threshold → stay open
            (71.1, "CLOSED"),   # 71.1 > 70+1=71 → CLOSE
            (71.01, "CLOSED"),  # just above → CLOSE
            (70.0, "OPEN"),     # equal to coolest → stay open
            (80.0, "CLOSED"),   # way above → CLOSE
        ],
    )
    def test_close_boundary_parametrized(self, engine, outdoor, expected_state):
        """Parametrized boundary tests for OPEN→CLOSED transition.

        Indoor = 70°F (single sensor = coolest). Close threshold = 70+1 = 71.
        Need outdoor > 71.
        """
        result = _decide(
            engine,
            indoor_temps=[70.0],
            outdoor_temp=outdoor,
            aqi=30,
            last_state="OPEN",
        )
        assert result.new_state == expected_state

    def test_very_large_differential_opens(self, engine):
        """outdoor 50°F, indoor 80°F → OPEN."""
        result = _decide(engine, indoor_temps=[80.0], outdoor_temp=50.0, aqi=30)
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_negative_temperatures(self, engine):
        """Winter edge: indoor 30°F, outdoor -5°F → comfort gate keeps CLOSED (30 ≤ 72)."""
        result = _decide(engine, indoor_temps=[30.0], outdoor_temp=-5.0, aqi=30)
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_indoor_outdoor_equal_stays_closed(self, engine):
        """Indoor and outdoor both 74°F → diff=0 → stay CLOSED."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=74.0, aqi=30)
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_indoor_outdoor_equal_stays_open(self, engine):
        """OPEN, indoor=outdoor=74°F → close threshold 74+1=75, outdoor 74 < 75 → stay OPEN."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=74.0,
            aqi=30,
            last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert result.changed is False


# ==================================================================
# 2. AQI Bidirectional Logic
# ==================================================================


class TestAQILogic:
    """Bidirectional AQI gating per user refinement."""

    def test_aqi_100_forces_closed_when_open(self, engine):
        """AQI >= 100, currently OPEN → urgent CLOSE."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=120,
            last_state="OPEN",
        )
        assert result.new_state == "CLOSED"
        assert result.urgent is True
        assert result.changed is True

    def test_aqi_100_stays_closed_no_notification(self, engine):
        """AQI >= 100, already CLOSED → CLOSED, not urgent, not changed."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=120,
            last_state="CLOSED",
        )
        assert result.new_state == "CLOSED"
        assert result.urgent is False
        assert result.changed is False

    def test_aqi_exactly_100_closes(self, engine):
        """AQI boundary: exactly 100 → CLOSE (>= threshold)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=100,
            last_state="OPEN",
        )
        assert result.new_state == "CLOSED"
        assert result.urgent is True
        assert result.changed is True

    def test_aqi_neutral_zone_maintains_closed(self, engine):
        """AQI 50-99: if CLOSED, stays CLOSED (blocks opening)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=75,
            last_state="CLOSED",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_aqi_neutral_zone_open_falls_through_to_temp(self, engine):
        """AQI 50-99 when OPEN: temp logic decides. Outdoor still cool → stay open."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=75,
            last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert result.changed is False

    def test_aqi_below_50_allows_opening(self, engine):
        """AQI < 50: allows temperature-driven OPEN."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            last_state="CLOSED",
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_aqi_recovery_opens_when_temp_is_good(self, engine):
        """AQI drops from 120 to 45, temp favourable → OPEN."""
        # Simulate: was closed due to high AQI, now AQI dropped.
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=45,
            last_state="CLOSED",
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_aqi_exactly_50_is_neutral_zone(self, engine):
        """AQI 50 → neutral zone (50 <= aqi < 100) → blocks opening if CLOSED."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=50,
            last_state="CLOSED",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_aqi_exactly_49_allows_opening(self, engine):
        """AQI 49 → below neutral zone → allow opening."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=49,
            last_state="CLOSED",
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_aqi_99_neutral_zone_upper_bound(self, engine):
        """AQI 99 → neutral zone (blocks opening when closed)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=99,
            last_state="CLOSED",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is False

    @pytest.mark.parametrize(
        "aqi_val, last, expected_urgent",
        [
            (100, "OPEN", True),
            (150, "OPEN", True),
            (100, "CLOSED", False),
            (99, "OPEN", False),
            (30, "CLOSED", False),
        ],
    )
    def test_urgency_flag(self, engine, aqi_val, last, expected_urgent):
        """urgent=True ONLY when AQI forces close on open windows."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=aqi_val,
            last_state=last,
        )
        assert result.urgent is expected_urgent


# ==================================================================
# 3. Humidity Gate
# ==================================================================


class TestHumidityGate:
    """Outdoor humidity > 80% blocks opening and triggers close."""

    def test_high_humidity_blocks_opening(self, engine):
        """Humidity 85% → stay CLOSED even if temp favours opening."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=85.0,
        )
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_high_humidity_closes_open_windows(self, engine):
        """Humidity 85% when OPEN → CLOSE."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=85.0,
            last_state="OPEN",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is True

    def test_humidity_exactly_80_allows(self, engine):
        """Humidity exactly 80% → allowed (gate is strictly >80)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=80.0,
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_humidity_79_allows(self, engine):
        """Humidity 79% → allowed."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=79.0,
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_humidity_close_is_not_urgent(self, engine):
        """Humidity-triggered close is NOT urgent."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=90.0,
            last_state="OPEN",
        )
        assert result.urgent is False


# ==================================================================
# 4. HVAC Mode Gate
# ==================================================================


class TestHVACModeGate:
    """Only cooling/auto modes allow decisions."""

    @pytest.mark.parametrize("mode", ["cool", "heatCool", "auto"])
    def test_allowed_modes_permit_decisions(self, engine, mode):
        """cool/heatCool/auto → engine processes normally."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            hvac_mode=mode,
        )
        # Should reach temp logic and open
        assert result.new_state == "OPEN"
        assert result.changed is True

    @pytest.mark.parametrize("mode", ["heat", "off"])
    def test_disallowed_modes_maintain_last_state(self, engine, mode):
        """heat/off → maintain last state, no decision."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            hvac_mode=mode,
            last_state="CLOSED",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_heat_mode_maintains_open_state(self, engine):
        """heat mode with last_state=OPEN → stays OPEN (no decision made)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            hvac_mode="heat",
            last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert result.changed is False

    def test_off_mode_maintains_open_state(self, engine):
        """off mode with last_state=OPEN → stays OPEN."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            hvac_mode="off",
            last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert result.changed is False


# ==================================================================
# 5. Per-Floor Sensor Logic
# ==================================================================


class TestSensorLogic:
    """Sensor grouping, offline handling, warmest/coolest selection."""

    def test_multiple_sensors_uses_warmest_for_open(self, engine):
        """warmest=76, coolest=70. Open threshold = 76-1=75. outdoor 74 < 75 → OPEN."""
        result = _decide(
            engine,
            indoor_temps=[76.0, 72.0, 70.0],
            outdoor_temp=74.0,
            aqi=30,
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_multiple_sensors_uses_coolest_for_close(self, engine):
        """OPEN. coolest=70, close threshold=71. outdoor 72 > 71 → CLOSE."""
        result = _decide(
            engine,
            indoor_temps=[76.0, 72.0, 70.0],
            outdoor_temp=72.0,
            aqi=30,
            last_state="OPEN",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is True

    def test_single_sensor_warmest_equals_coolest(self, engine):
        """Single sensor: warmest = coolest = that reading."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
        )
        assert result.new_state == "OPEN"

    def test_offline_sensor_excluded(self, engine):
        """Offline sensor not used. Two online sensors: 76, 70."""
        sensors = make_sensors(
            [76.0, 70.0, 80.0],
            names=["s0", "s1", "s_offline"],
            online=[True, True, False],
        )
        result = _decide(
            engine,
            sensors=sensors,
            floor_group=["s0", "s1", "s_offline"],
            outdoor_temp=74.0,
            aqi=30,
        )
        # warmest = 76, threshold = 75, outdoor 74 < 75 → OPEN
        assert result.new_state == "OPEN"

    def test_all_sensors_offline_raises_error(self, engine):
        """All sensors offline → InsufficientDataError → keep last state."""
        sensors = make_sensors(
            [74.0, 72.0],
            names=["s0", "s1"],
            online=[False, False],
        )
        result = _decide(
            engine,
            sensors=sensors,
            floor_group=["s0", "s1"],
            outdoor_temp=68.0,
            aqi=30,
            last_state="CLOSED",
        )
        # Engine catches InsufficientDataError and calls _keep
        assert result.new_state == "CLOSED"
        assert result.changed is False
        assert "Insufficient" in result.reason

    def test_all_sensors_offline_keeps_open(self, engine):
        """All offline when OPEN → stays OPEN (safe keep)."""
        sensors = make_sensors(
            [74.0],
            names=["s0"],
            online=[False],
        )
        result = _decide(
            engine,
            sensors=sensors,
            floor_group=["s0"],
            outdoor_temp=68.0,
            aqi=30,
            last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert result.changed is False

    def test_mixed_online_offline_uses_only_online(self, engine):
        """Three sensors, one offline. Uses only the two online ones."""
        sensors = make_sensors(
            [74.0, None, 70.0],
            names=["s0", "s_offline", "s2"],
            online=[True, False, True],
        )
        result = _decide(
            engine,
            sensors=sensors,
            floor_group=["s0", "s_offline", "s2"],
            outdoor_temp=68.0,
            aqi=30,
        )
        # warmest=74, threshold=73, outdoor 68 < 73 → OPEN
        assert result.new_state == "OPEN"

    def test_sensor_with_none_temp_excluded(self, engine):
        """Online sensor with temperature_f=None is excluded."""
        sensors = [
            {"name": "s0", "is_online": True, "temperature_f": 74.0},
            {"name": "s1", "is_online": True},  # no temperature_f key
        ]
        result = _decide(
            engine,
            sensors=sensors,
            floor_group=["s0", "s1"],
            outdoor_temp=68.0,
            aqi=30,
        )
        # Only s0 (74°F), threshold = 73, outdoor 68 < 73 → OPEN
        assert result.new_state == "OPEN"

    def test_get_floor_temps_raises_directly(self, engine):
        """get_floor_temps raises InsufficientDataError with no valid sensors."""
        sensors = make_sensors([74.0], names=["other"], online=[True])
        with pytest.raises(InsufficientDataError):
            DecisionEngine.get_floor_temps(sensors, ["not_matching"])

    def test_floor_group_filters_sensors(self, engine):
        """Only sensors matching floor_group are considered."""
        sensors = make_sensors(
            [80.0, 70.0],
            names=["upstairs_0", "downstairs_0"],
        )
        result = _decide(
            engine,
            sensors=sensors,
            floor_group=["upstairs_0"],
            outdoor_temp=68.0,
            aqi=30,
        )
        # Only upstairs_0 (80°F), threshold = 79, outdoor 68 < 79 → OPEN
        assert result.new_state == "OPEN"


# ==================================================================
# 6. Edge Cases
# ==================================================================


class TestEdgeCases:
    """Boundary, extreme, and unusual input scenarios."""

    def test_rapid_oscillation_half_degree(self, engine):
        """Temp bounces ±0.5°F around threshold — hysteresis prevents flip-flop.

        Indoor 74, open threshold 73. Outdoor oscillates 72.8 ↔ 73.2.
        When CLOSED: 72.8 < 73 → OPEN. Then 73.2 — close threshold is 75.
        73.2 < 75 → stays OPEN. Hysteresis working.
        """
        r1 = _decide(engine, indoor_temps=[74.0], outdoor_temp=72.8, aqi=30)
        assert r1.new_state == "OPEN"

        r2 = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=73.2,
            aqi=30,
            last_state="OPEN",
        )
        assert r2.new_state == "OPEN"  # hysteresis prevents close

    def test_unknown_last_state_defaults_to_closed(self, engine):
        """Unknown last_state normalised to CLOSED."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            last_state="UNKNOWN",
        )
        # Treated as CLOSED, outdoor 68 < 73 → OPEN
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_empty_string_last_state_defaults_to_closed(self, engine):
        """Empty last_state → treated as CLOSED."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            last_state="",
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_aqi_zero(self, engine):
        """AQI 0 (excellent) → allows opening."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=0,
        )
        assert result.new_state == "OPEN"

    def test_aqi_missing_from_dict_defaults_to_zero(self, engine):
        """AQI dict without 'aqi' key → defaults to 0 (safe fallback)."""
        sensors = make_sensors([74.0], names=["s0"])
        result = engine.decide(
            floor="upstairs",
            floor_sensors=sensors,
            outdoor=outdoor_conditions(68.0),
            aqi={},  # no 'aqi' key
            hvac_mode="cool",
            last_state="CLOSED",
            floor_group=["s0"],
        )
        assert result.new_state == "OPEN"

    def test_floor_name_preserved_in_result(self, engine):
        """FloorDecision.floor matches the input floor name."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=68.0, aqi=30, floor="downstairs")
        assert result.floor == "downstairs"


# ==================================================================
# 7. Notification Urgency
# ==================================================================


class TestUrgency:
    """urgent flag only set for AQI-triggered close on open windows."""

    def test_normal_temp_open_not_urgent(self, engine):
        """Temperature-driven OPEN → not urgent."""
        result = _decide(
            engine, indoor_temps=[74.0], outdoor_temp=68.0, aqi=30
        )
        assert result.urgent is False

    def test_temp_close_not_urgent(self, engine):
        """Temperature-driven CLOSE → not urgent."""
        result = _decide(
            engine,
            indoor_temps=[70.0],
            outdoor_temp=72.0,
            aqi=30,
            last_state="OPEN",
        )
        assert result.urgent is False

    def test_aqi_close_is_urgent(self, engine):
        """AQI-triggered close → urgent."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=150,
            last_state="OPEN",
        )
        assert result.urgent is True

    def test_humidity_close_not_urgent(self, engine):
        """Humidity-triggered close → not urgent."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=90.0,
            last_state="OPEN",
        )
        assert result.urgent is False

    def test_hvac_gate_not_urgent(self, engine):
        """HVAC gate maintain → not urgent."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            hvac_mode="heat",
        )
        assert result.urgent is False


# ==================================================================
# 8. Combined Conditions
# ==================================================================


class TestCombinedConditions:
    """Multiple gates interacting — priority order matters."""

    def test_good_temp_bad_aqi_closes(self, engine):
        """Good temp conditions + AQI >= 100 → CLOSED (AQI wins)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=120,
            humidity=50.0,
            last_state="OPEN",
        )
        assert result.new_state == "CLOSED"
        assert result.urgent is True

    def test_good_temp_good_aqi_bad_humidity_closes(self, engine):
        """Good temp + good AQI + humidity > 80 → CLOSED (humidity gate)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=85.0,
        )
        assert result.new_state == "CLOSED"

    def test_bad_temp_good_aqi_good_humidity_closes(self, engine):
        """Outdoor warmer than indoor + everything else good → CLOSED (temp logic)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=80.0,
            aqi=30,
            humidity=50.0,
        )
        assert result.new_state == "CLOSED"

    def test_hvac_heat_overrides_everything(self, engine):
        """HVAC in heat mode → no change regardless of other conditions."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=30,
            humidity=50.0,
            hvac_mode="heat",
            last_state="CLOSED",
        )
        assert result.new_state == "CLOSED"
        assert result.changed is False

    def test_aqi_checked_before_humidity(self, engine):
        """AQI gate runs before humidity gate (priority order)."""
        result = _decide(
            engine,
            indoor_temps=[74.0],
            outdoor_temp=68.0,
            aqi=120,
            humidity=90.0,
            last_state="OPEN",
        )
        # AQI catches it first → urgent
        assert result.urgent is True
        assert "AQI" in result.reason

    def test_humidity_checked_before_temperature(self, engine):
        """Humidity blocks opening even when temp is great."""
        result = _decide(
            engine,
            indoor_temps=[80.0],
            outdoor_temp=60.0,
            aqi=30,
            humidity=85.0,
            last_state="CLOSED",
        )
        assert result.new_state == "CLOSED"
        assert "umidity" in result.reason

    def test_all_conditions_perfect_opens(self, engine):
        """All conditions good → OPEN."""
        result = _decide(
            engine,
            indoor_temps=[76.0],
            outdoor_temp=68.0,
            aqi=25,
            humidity=45.0,
            hvac_mode="cool",
        )
        assert result.new_state == "OPEN"
        assert result.changed is True

    def test_neutral_aqi_open_state_temp_close(self, engine):
        """AQI 75 (neutral), OPEN, outdoor warming → temp closes."""
        result = _decide(
            engine,
            indoor_temps=[70.0],
            outdoor_temp=72.0,
            aqi=75,
            last_state="OPEN",
        )
        # AQI neutral returns None for OPEN → falls to temp logic
        # close threshold = 70+1 = 71, outdoor 72 > 71 → CLOSE
        assert result.new_state == "CLOSED"
        assert result.changed is True
        assert result.urgent is False


# ==================================================================
# 9. Config Customisation
# ==================================================================


class TestConfigCustomisation:
    """Engine respects config overrides."""

    def test_custom_hysteresis_wider(self):
        """Wider hysteresis (3°F) requires bigger diff to open."""
        config = {
            "hysteresis_open_diff": 3.0,
            "hysteresis_close_diff": 3.0,
            "max_outdoor_humidity": 80,
            "max_aqi_threshold": 100,
            "min_aqi_for_opening": 50,
            "allowed_hvac_modes": ["cool", "heatCool", "auto"],
        }
        eng = DecisionEngine(config)
        # Indoor 74, threshold = 74-3 = 71. Outdoor 72 ≥ 71 → stay CLOSED.
        result = _decide(eng, indoor_temps=[74.0], outdoor_temp=72.0, aqi=30)
        assert result.new_state == "CLOSED"

    def test_aqi_gate_disabled_no_urgent_close(self):
        """With enable_aqi_gate=False, high AQI doesn't force urgent close."""
        config = {
            "hysteresis_open_diff": 1.0,
            "hysteresis_close_diff": 1.0,
            "max_outdoor_humidity": 80,
            "max_aqi_threshold": 100,
            "min_aqi_for_opening": 50,
            "allowed_hvac_modes": ["cool", "heatCool", "auto"],
            "enable_aqi_gate": False,
            "enable_humidity_gate": True,
        }
        eng = DecisionEngine(config)
        # When OPEN with AQI gate disabled, AQI 150 doesn't trigger urgent close.
        # Temperature logic for OPEN→CLOSED only checks outdoor vs coolest.
        result = _decide(eng, indoor_temps=[74.0], outdoor_temp=68.0, aqi=150, last_state="OPEN")
        assert result.new_state == "OPEN"
        assert result.urgent is False

    def test_humidity_gate_disabled(self):
        """With enable_humidity_gate=False, high humidity doesn't block."""
        config = {
            "hysteresis_open_diff": 1.0,
            "hysteresis_close_diff": 1.0,
            "max_outdoor_humidity": 80,
            "max_aqi_threshold": 100,
            "min_aqi_for_opening": 50,
            "allowed_hvac_modes": ["cool", "heatCool", "auto"],
            "enable_aqi_gate": True,
            "enable_humidity_gate": False,
        }
        eng = DecisionEngine(config)
        result = _decide(eng, indoor_temps=[74.0], outdoor_temp=68.0, aqi=30, humidity=95.0)
        assert result.new_state == "OPEN"


# ==================================================================
# 10. FloorDecision Dataclass
# ==================================================================


# ==================================================================
# Comfort Threshold Gate
# ==================================================================

class TestComfortThreshold:
    """Gate 4 — don't open windows when indoor temps are already comfortable."""

    def test_no_open_when_indoor_at_72(self, engine):
        """At exactly the comfort boundary, windows stay closed."""
        result = _decide(engine, indoor_temps=[72.0], outdoor_temp=60.0, aqi=30)
        assert result.new_state == "CLOSED"
        assert not result.changed
        assert "comfortable" in result.reason.lower()

    def test_no_open_when_indoor_below_72(self, engine):
        """Below the comfort max, windows stay closed."""
        result = _decide(engine, indoor_temps=[70.0], outdoor_temp=60.0, aqi=30)
        assert result.new_state == "CLOSED"
        assert not result.changed
        assert "comfortable" in result.reason.lower()

    def test_open_allowed_when_indoor_above_72(self, engine):
        """Above the comfort max, normal temperature logic can open windows."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=60.0, aqi=30)
        assert result.new_state == "OPEN"
        assert result.changed

    def test_comfort_gate_does_not_force_close_open_windows(self, engine):
        """When windows are already open, the comfort gate is skipped."""
        result = _decide(
            engine, indoor_temps=[71.0], outdoor_temp=70.0, aqi=30, last_state="OPEN",
        )
        assert result.new_state == "OPEN"
        assert not result.changed

    def test_comfort_threshold_custom_config(self, default_config):
        """A custom comfort_temp_max of 70 allows opening at 71°F."""
        custom = {**default_config, "comfort_temp_max": 70.0}
        eng = DecisionEngine(custom)
        result = _decide(eng, indoor_temps=[71.0], outdoor_temp=60.0, aqi=30)
        assert result.new_state == "OPEN"
        assert result.changed

    def test_comfort_at_boundary_with_multiple_sensors(self, engine):
        """Warmest sensor determines the comfort check."""
        result = _decide(engine, indoor_temps=[72.0, 71.0, 70.0], outdoor_temp=60.0, aqi=30)
        assert result.new_state == "CLOSED"
        assert not result.changed
        assert "72.0" in result.reason


class TestFloorDecision:
    """FloorDecision is frozen and has expected fields."""

    def test_decision_is_frozen(self, engine):
        """Cannot mutate a FloorDecision."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=68.0, aqi=30)
        with pytest.raises(AttributeError):
            result.new_state = "CLOSED"  # type: ignore[misc]

    def test_reason_is_human_readable(self, engine):
        """Reason field is a non-empty string."""
        result = _decide(engine, indoor_temps=[74.0], outdoor_temp=68.0, aqi=30)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
