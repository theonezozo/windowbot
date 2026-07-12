"""Tests for outdoor-temperature jitter suppression (src/outdoor_validator.py).

Validates the jitter-suppression feature: a jump from the previous validated
outdoor temperature is held back ONLY when ALL discriminators agree it is
availability-driven noise (set rotation + no corroborating survivor +
against-trend), while genuine movement (stable set, corroborated, or
trend-aligned) always passes through without lag.

state_fields are ALWAYS returned and persist:
  - LastOutdoorTemp        = validated temp
  - LastOutdoorContributors = json.dumps({station_id: temp}) of CURRENT readings
  - OutdoorTempHistory      = json.dumps of validated temps, oldest->newest, capped
"""

from __future__ import annotations

import json

import pytest

from src.outdoor_validator import (
    OutdoorValidationResult,
    validate_outdoor_temperature,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _contributors(mapping: dict) -> list[dict]:
    """Build the contributor list shape from a ``{station_id: temp}`` map."""
    return [
        {"station_id": sid, "temperature_f": temp}
        for sid, temp in mapping.items()
    ]


def _prev_state(
    *,
    last_temp,
    contributors: dict | None = None,
    history: list | None = None,
) -> dict:
    """Build a ``__global__`` state dict exactly as the module persists it."""
    return {
        "LastOutdoorTemp": last_temp,
        "LastOutdoorContributors": (
            json.dumps(contributors) if contributors is not None else ""
        ),
        "OutdoorTempHistory": (
            json.dumps(history) if history is not None else ""
        ),
    }


# ==================================================================
# A. SUPPRESSION (the bug being fixed)
# ==================================================================


def test_false_upward_spike_on_sensor_rotation_is_suppressed():
    """Up-jump on rotation, no corroborating survivor, against cooling trend → held."""
    # Arrange
    prev = _prev_state(
        last_temp=68.8,
        contributors={"KAAA": 68.8, "KBBB": 68.8},
        history=[70.0, 69.6, 69.2, 68.8],
    )
    contributors = _contributors({"KAAA": 68.8, "KCCC": 71.0})

    # Act
    result = validate_outdoor_temperature(69.9, contributors, prev)

    # Assert
    assert result.suppressed is True
    assert result.reason == "held_uncorroborated_churn"
    assert result.temperature_f == 68.8
    assert result.state_fields["LastOutdoorTemp"] == 68.8
    assert json.loads(result.state_fields["LastOutdoorContributors"]) == {
        "KAAA": 68.8,
        "KCCC": 71.0,
    }


def test_false_downward_dip_on_rotation_against_warming_trend_is_suppressed():
    """Down-jump on rotation, no corroboration, against warming trend → held."""
    # Arrange
    prev = _prev_state(
        last_temp=67.2,
        contributors={"KAAA": 67.2, "KBBB": 67.2},
        history=[66.0, 66.4, 66.8, 67.2],
    )
    contributors = _contributors({"KAAA": 67.2, "KCCC": 65.0})

    # Act
    result = validate_outdoor_temperature(66.1, contributors, prev)

    # Assert
    assert result.suppressed is True
    assert result.reason == "held_uncorroborated_churn"
    assert result.temperature_f == 67.2


# ==================================================================
# B. GENUINE PASSTHROUGH (must NOT lag)
# ==================================================================


def test_same_set_large_move_passes():
    """Identical contributor set → genuine movement, never suppressed."""
    # Arrange
    prev = _prev_state(
        last_temp=68.0,
        contributors={"KAAA": 68.0, "KBBB": 68.0},
        history=[68.0],
    )
    contributors = _contributors({"KAAA": 69.2, "KBBB": 69.2})

    # Act
    result = validate_outdoor_temperature(69.2, contributors, prev)

    # Assert
    assert result.suppressed is False
    assert result.reason == "confident_stable_set"
    assert result.temperature_f == 69.2


def test_corroborated_warming_passes_despite_rotation():
    """Surviving sensor moved same direction by >= epsilon → corroborated pass.

    With the corroboration bar set to 1 (``min_corroborating_sources=1``) a
    single surviving sensor moving the same direction as the median jump is
    enough independent backing to authorize the churn-coincident move, so it
    passes as ``confident_corroborated`` on the first poll (no lag).
    """
    # Arrange
    prev = _prev_state(
        last_temp=68.0,
        contributors={"KAAA": 68.0, "KBBB": 68.0},
        history=[68.0],
    )
    contributors = _contributors({"KAAA": 68.9, "KCCC": 68.8})

    # Act
    result = validate_outdoor_temperature(
        68.85, contributors, prev, min_corroborating_sources=1
    )

    # Assert
    assert result.suppressed is False
    assert result.reason == "confident_corroborated"
    assert result.temperature_f == pytest.approx(68.85)


def test_trend_aligned_churn_jump_is_held():
    """Trend alone no longer authorizes a churn-coincident jump → HELD.

    A jump aligned with the validated-history trend used to pass
    (``genuine_trend_aligned``). Under the confidence gate, trend is NOT a
    pass authorizer: on a contributor-set rotation with no surviving
    corroborator, the move is held at the last confident value as
    ``held_uncorroborated_churn`` (signal-trust, not a close-side margin).
    """
    # Arrange
    prev = _prev_state(
        last_temp=67.5,
        contributors={"KAAA": 67.5, "KBBB": 67.5},
        history=[66.0, 66.5, 67.0, 67.5],
    )
    contributors = _contributors({"KAAA": 67.5, "KCCC": 69.0})

    # Act
    result = validate_outdoor_temperature(68.7, contributors, prev)

    # Assert
    assert result.suppressed is True
    assert result.reason == "held_uncorroborated_churn"
    assert result.temperature_f == pytest.approx(67.5)


# ==================================================================
# C. EDGES
# ==================================================================


def test_cold_start_passes():
    """No prior validated temp → cold start, history seeded with fused value."""
    # Arrange
    prev = {
        "LastOutdoorTemp": None,
        "LastOutdoorContributors": "",
        "OutdoorTempHistory": "",
    }
    contributors = _contributors({"KAAA": 70.0, "KBBB": 70.0})

    # Act
    result = validate_outdoor_temperature(70.0, contributors, prev)

    # Assert
    assert result.reason == "cold_start"
    assert result.suppressed is False
    assert result.temperature_f == 70.0
    assert result.state_fields["OutdoorTempHistory"] == json.dumps([70.0])


def test_within_threshold_passes():
    """Delta within the jitter band passes even with a changed set."""
    # Arrange
    prev = _prev_state(
        last_temp=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=[70.0],
    )
    contributors = _contributors({"KAAA": 70.3, "KCCC": 70.3})

    # Act
    result = validate_outdoor_temperature(70.3, contributors, prev)

    # Assert
    assert result.reason == "within_threshold"
    assert result.suppressed is False


def test_no_contributors_passthrough():
    """Single-source fallback (no contributor detail) always passes through."""
    # Arrange
    prev = _prev_state(
        last_temp=60.0,
        contributors={"KAAA": 60.0, "KBBB": 60.0},
        history=[60.0],
    )

    # Act
    result = validate_outdoor_temperature(72.0, [], prev)

    # Assert
    assert result.reason == "no_contributors_passthrough"
    assert result.suppressed is False
    assert result.temperature_f == 72.0
    assert json.loads(result.state_fields["LastOutdoorContributors"]) == {}


def test_history_capped_at_max_history():
    """Persisted history is capped at max_history after appending."""
    # Arrange
    full_history = [60.0 + i * 0.1 for i in range(12)]
    prev = _prev_state(
        last_temp=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=full_history,
    )
    contributors = _contributors({"KAAA": 70.3, "KBBB": 70.3})

    # Act
    result = validate_outdoor_temperature(70.3, contributors, prev)

    # Assert
    stored = json.loads(result.state_fields["OutdoorTempHistory"])
    assert len(stored) == 12
    assert stored[-1] == pytest.approx(70.3)


def test_malformed_prev_state_is_robust():
    """Corrupt stored fields must not raise; jump with changed set is held."""
    # Arrange
    prev = {
        "LastOutdoorTemp": 68.0,
        "LastOutdoorContributors": "not json",
        "OutdoorTempHistory": "{bad",
    }
    contributors = _contributors({"KAAA": 71.0, "KCCC": 71.0})

    # Act
    result = validate_outdoor_temperature(71.0, contributors, prev)

    # Assert
    assert isinstance(result, OutdoorValidationResult)
    assert result.suppressed is True
    assert result.temperature_f == 68.0


# ==================================================================
# D. INTEGRATION (light)
# ==================================================================


def test_orchestrator_exposes_validator():
    """The orchestrator namespace re-exports the validator entry point."""
    # Arrange / Act
    import src.orchestrator as orchestrator

    # Assert
    assert hasattr(orchestrator, "validate_outdoor_temperature")
    assert orchestrator.validate_outdoor_temperature is validate_outdoor_temperature


def test_config_exposes_jitter_keys():
    """get_config() exposes the jitter tuning keys with spec defaults."""
    # Arrange
    from src.config import get_config

    # Act
    try:
        config = get_config()
    except Exception:  # pragma: no cover - defensive; config tolerates defaults
        pytest.skip("get_config() requires environment that is unavailable")

    # Assert
    assert config["outdoor_jitter_threshold_f"] == 0.5
    assert config["outdoor_jitter_trend_window"] == 6


# ==================================================================
# E. SPIKE GATE (Fix 3: single-station / cross-cycle spike suppression)
# ==================================================================
#
# Fix 3 adds a SPIKE GATE that runs BEFORE the genuine_stable_set return so
# that an implausibly large one-cycle jump (abs(delta) > spike_max_rate_f) is
# held unless something supports it: a surviving sensor corroborates it, the
# validated-history trend aligns with it, or the previous RAW observation was
# already on the same side of last_temp (sustained). It also tracks a new
# OutdoorRawHistory state field: the RAW fused temp is appended every cycle,
# even when a value is suppressed/held.
#
# NOTE ON SINGLE-STATION CORROBORATION (verified against src/outdoor_validator.py):
# The corroboration loop iterates over surviving stations (current_ids &
# last_ids) and treats ANY survivor that moved in the delta direction by
# >= corroboration_epsilon_f as corroboration. The loop is now guarded by
# `if len(current_map) > 1:` so a SINGLE-station median can no longer
# self-corroborate its own jump. For the lone-station case the spike gate's
# `not corroborated` condition is True and the gate fires, holding the spike
# (`suppressed_spike`). The gate also works whenever there is no surviving
# corroborator (e.g. a full contributor rotation), proven below.


def _prev_state_raw(
    *,
    last_temp,
    contributors: dict | None = None,
    history: list | None = None,
    raw_history: list | None = None,
) -> dict:
    """Build a ``__global__`` state dict including OutdoorRawHistory.

    Mirrors exactly how the module persists the fields: every stored list/map
    is a JSON string. ``OutdoorRawHistory`` is the RAW observed fused temps,
    oldest->newest.
    """
    return {
        "LastOutdoorTemp": last_temp,
        "LastOutdoorContributors": (
            json.dumps(contributors) if contributors is not None else ""
        ),
        "OutdoorTempHistory": (
            json.dumps(history) if history is not None else ""
        ),
        "OutdoorRawHistory": (
            json.dumps(raw_history) if raw_history is not None else ""
        ),
    }


def test_single_station_spike_on_stable_set_is_suppressed():
    """A +2.3°F one-cycle spike on a flat single station is held.

    The production failure case. A lone station KAAA reads a flat 72.7°F for
    three cycles, then snaps to 75.0°F. Nothing independent supports it (no
    OTHER sensor, flat trend, prev_raw == last_temp so not sustained), so the
    spike gate holds it at 72.7°F.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=72.7,
        contributors={"KAAA": 72.7},
        history=[72.7, 72.7, 72.7],
        raw_history=[72.7, 72.7, 72.7],
    )
    contributors = _contributors({"KAAA": 75.0})

    # Act
    result = validate_outdoor_temperature(75.0, contributors, prev)

    # Assert
    assert result.reason == "suppressed_spike"
    assert result.suppressed is True
    assert result.temperature_f == pytest.approx(72.7)
    # Raw history records the observed spike even though the value is held.
    raw_hist = json.loads(result.state_fields["OutdoorRawHistory"])
    assert raw_hist[-1] == pytest.approx(75.0)
    # Validated history records the HELD value, not the raw spike.
    val_hist = json.loads(result.state_fields["OutdoorTempHistory"])
    assert val_hist[-1] == pytest.approx(72.7)


def test_sustained_high_reading_passes_next_cycle():
    """A spike that persists for a second cycle clears within one cycle.

    Continuation of the suppression scenario: last_temp is still 72.7°F (held),
    but the previous RAW observation was already 75.0°F. A fresh 75.1°F reading
    therefore has prev_raw(75.0) - last_temp(72.7) = +2.3 on the same side as
    the +2.4 delta and > jitter, so `sustained` is True and the gate is
    skipped. The reading passes through unheld.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=72.7,
        contributors={"KAAA": 72.7},
        history=[72.7, 72.7, 72.7],
        raw_history=[72.7, 72.7, 75.0],
    )
    contributors = _contributors({"KAAA": 75.1})

    # Act
    result = validate_outdoor_temperature(75.1, contributors, prev)

    # Assert
    assert result.suppressed is False
    assert result.temperature_f == pytest.approx(75.1)
    assert result.reason == "confident_stable_set"


def test_spike_reverts_no_false_value():
    """A spike that reverts within the jitter band never corrupts the value.

    After a held spike (last_temp 72.7°F, prev raw 75.0°F), the lone station
    reverts to 72.4°F. The delta vs the validated 72.7°F is only -0.3°F, inside
    the jitter band, so it passes as within_threshold — the transient 75.0°F
    never moved the validated temperature at all.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=72.7,
        contributors={"KAAA": 72.7},
        history=[72.7, 72.7, 72.7],
        raw_history=[72.7, 72.7, 75.0],
    )
    contributors = _contributors({"KAAA": 72.4})

    # Act
    result = validate_outdoor_temperature(72.4, contributors, prev)

    # Assert
    assert result.reason == "within_threshold"
    assert result.suppressed is False
    assert result.temperature_f == pytest.approx(72.4)


def test_large_jump_corroborated_passes():
    """A large jump confirmed by a surviving sensor passes immediately.

    The set changes (KBBB out, KCCC in) but KAAA survives and rose +2.5°F,
    independently corroborating the +2.5°F median jump. The spike gate is
    skipped (survivor corroboration) and, with the corroboration bar at 1, the
    confidence gate passes it as ``confident_corroborated``.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=[70.0, 70.0, 70.0],
        raw_history=[70.0, 70.0, 70.0],
    )
    contributors = _contributors({"KAAA": 72.5, "KCCC": 72.5})

    # Act
    result = validate_outdoor_temperature(
        72.5, contributors, prev, min_corroborating_sources=1
    )

    # Assert
    assert result.suppressed is False
    assert result.reason == "confident_corroborated"
    assert result.temperature_f == pytest.approx(72.5)


def test_large_jump_trend_aligned_churn_is_held():
    """A trend-aligned jump coincident with churn is HELD, not passed.

    A rising validated history ([66, 67.5, 69, 70]) plus a +2.5°F jump used to
    pass on trend alignment. Here the contributor set ALSO rotates (KBBB out,
    KCCC in) and no surviving sensor moves, so there is no independent
    corroboration. Trend alignment lets the move past the spike gate, but the
    confidence gate holds it at the last confident value
    (``held_uncorroborated_churn``) — trend alone no longer authorizes a
    churn-coincident move.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=[66.0, 67.5, 69.0, 70.0],
        raw_history=[66.0, 67.5, 69.0, 70.0],
    )
    contributors = _contributors({"KAAA": 70.0, "KCCC": 72.5})

    # Act
    result = validate_outdoor_temperature(72.5, contributors, prev)

    # Assert
    assert result.suppressed is True
    assert result.reason == "held_uncorroborated_churn"
    assert result.temperature_f == pytest.approx(70.0)


def test_jump_just_under_max_rate_passes_stable_set():
    """A sub-threshold jump (delta 1.9 < 2.0) on a flat stable set is unaffected.

    Proves the spike-rate boundary: only jumps STRICTLY greater than
    spike_max_rate_f are eligible to be gated, so a 1.9°F move passes normally
    as genuine_stable_set.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=70.0,
        contributors={"KAAA": 70.0},
        history=[70.0, 70.0, 70.0],
        raw_history=[70.0, 70.0, 70.0],
    )
    contributors = _contributors({"KAAA": 71.9})

    # Act
    result = validate_outdoor_temperature(71.9, contributors, prev)

    # Assert
    assert result.suppressed is False
    assert result.reason == "confident_stable_set"
    assert result.temperature_f == pytest.approx(71.9)


def test_custom_lower_max_rate_catches_smaller_spike():
    """Lowering spike_max_rate_f to 1.0 gates a 1.5°F jump.

    With the default 2.0°F/cycle ceiling a 1.5°F jump is not a spike, but with
    spike_max_rate_f=1.0 it exceeds the ceiling. On a flat, unsustained,
    non-corroborated single station it is held as suppressed_spike, proving the
    knob works.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=70.0,
        contributors={"KAAA": 70.0},
        history=[70.0, 70.0, 70.0],
        raw_history=[70.0, 70.0, 70.0],
    )
    contributors = _contributors({"KAAA": 71.5})

    # Act
    result = validate_outdoor_temperature(
        71.5, contributors, prev, spike_max_rate_f=1.0
    )

    # Assert
    assert result.reason == "suppressed_spike"
    assert result.suppressed is True
    assert result.temperature_f == pytest.approx(70.0)


def test_raw_history_recorded_on_every_path():
    """OutdoorRawHistory always appends the RAW fused temp, even on a pass.

    Uses a within_threshold pass and asserts the raw history's last element is
    the raw fused temp observed this cycle (not the validated value).
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=[70.0, 70.0, 70.0],
        raw_history=[70.0, 70.0, 70.0],
    )
    contributors = _contributors({"KAAA": 70.3, "KCCC": 70.3})

    # Act
    result = validate_outdoor_temperature(70.3, contributors, prev)

    # Assert
    assert result.reason == "within_threshold"
    raw_hist = json.loads(result.state_fields["OutdoorRawHistory"])
    assert raw_hist[-1] == pytest.approx(70.3)


# ------------------------------------------------------------------
# E2. Spike gate DOES fire when there is genuinely no corroborator.
#     (Demonstrates the gate works outside the single-station loophole.)
# ------------------------------------------------------------------


def test_full_rotation_spike_no_survivor_is_suppressed():
    """A +2.3°F jump on a full contributor rotation (no survivor) is held.

    With every station rotated out there is no surviving sensor to
    corroborate, the trend is flat, and prev_raw == last_temp (not sustained),
    so the spike gate fires and returns suppressed_spike, holding 72.7°F. This
    is the scenario where Fix 3's gate genuinely engages.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=72.7,
        contributors={"KAAA": 72.7, "KBBB": 72.7},
        history=[72.7, 72.7, 72.7],
        raw_history=[72.7, 72.7, 72.7],
    )
    contributors = _contributors({"KCCC": 75.0, "KDDD": 75.0})

    # Act
    result = validate_outdoor_temperature(75.0, contributors, prev)

    # Assert
    assert result.reason == "suppressed_spike"
    assert result.suppressed is True
    assert result.temperature_f == pytest.approx(72.7)
    # Raw history still records the observed spike despite the hold.
    raw_hist = json.loads(result.state_fields["OutdoorRawHistory"])
    assert raw_hist[-1] == pytest.approx(75.0)


def test_full_rotation_custom_max_rate_catches_smaller_spike():
    """Lowering spike_max_rate_f to 1.0 gates a 1.5°F jump on a full rotation.

    Proves the spike_max_rate_f knob: a 1.5°F move is below the default 2.0°F
    ceiling but above a custom 1.0°F ceiling. On a full rotation with no
    survivor/trend/sustain support, the lowered ceiling causes a
    suppressed_spike hold at 70.0°F.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=[70.0, 70.0, 70.0],
        raw_history=[70.0, 70.0, 70.0],
    )
    contributors = _contributors({"KCCC": 71.5, "KDDD": 71.5})

    # Act
    result = validate_outdoor_temperature(
        71.5, contributors, prev, spike_max_rate_f=1.0
    )

    # Assert
    assert result.reason == "suppressed_spike"
    assert result.suppressed is True
    assert result.temperature_f == pytest.approx(70.0)


# ==================================================================
# F. CONFIDENCE GATE (signal-trust layer — symmetric, NOT a close deadband)
# ==================================================================
#
# A supra-threshold move relative to the LAST CONFIDENT value is trusted only
# when independently backed (stable contributor set, OR >= N corroborating
# fresh sources incl. Open-Meteo); otherwise the last confident value is HELD.
# This suppresses station-rotation artifacts while letting a genuine,
# corroborated warm-up through on the FIRST poll (so the bare decision-engine
# close ` > coolest ` fires immediately). The decision-engine close is
# unchanged; all of this happens upstream in the validator.


def _contrib_entry(station_id, temp_f, *, source_type="nws_station", is_cached=False):
    """Build one rich contributor_log entry as the orchestrator would."""
    return {
        "station_id": station_id,
        "source_type": source_type,
        "temp_f": temp_f,
        "is_cached": is_cached,
    }


def _clog_payload(
    *,
    contributors,
    real_station_count,
    openmeteo_present=False,
    used_cache_fallback=False,
    stickiness_active=False,
):
    """Build the Phase-2 contributor_log payload the confidence gate reads."""
    return {
        "contributors": contributors,
        "real_station_count": real_station_count,
        "openmeteo_present": openmeteo_present,
        "used_cache_fallback": used_cache_fallback,
        "stickiness_active": stickiness_active,
    }


def _prev_state_conf(
    *,
    last_temp,
    last_confident=None,
    contributors=None,
    history=None,
    raw_history=None,
    hold_count=0,
    real_station_count=None,
):
    """Build a ``__global__`` state dict including the confidence-gate fields."""
    st = {
        "LastOutdoorTemp": last_temp,
        "LastOutdoorContributors": (
            json.dumps(contributors) if contributors is not None else ""
        ),
        "OutdoorTempHistory": json.dumps(history) if history is not None else "",
        "OutdoorRawHistory": (
            json.dumps(raw_history) if raw_history is not None else ""
        ),
        "ConfidentHoldCount": hold_count,
    }
    if last_confident is not None:
        st["LastConfidentOutdoorTemp"] = last_confident
    if real_station_count is not None:
        st["LastRealStationCount"] = real_station_count
    return st


def test_rotation_artifact_uncorroborated_is_held():
    """This morning's artifact: rotation churn + no corroboration → HELD.

    A station rotates (KBBB out, KCCC in) and the median snaps 69.7 → 71.6,
    but the surviving station (KAAA) did not move and no peer agrees. The
    confidence gate holds the last confident value (69.7) and emits it, so a
    spurious close is suppressed.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=69.7,
        last_confident=69.7,
        contributors={"KAAA": 69.7, "KBBB": 69.7},
        history=[69.7, 69.7, 69.7],
        raw_history=[69.7, 69.7, 69.7],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 69.7, "KCCC": 71.6})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 69.7),
            _contrib_entry("KCCC", 71.6),
        ],
        real_station_count=2,
    )

    # Act
    result = validate_outdoor_temperature(71.6, contributors, prev, contributor_log=clog)

    # Assert
    assert result.reason == "held_uncorroborated_churn"
    assert result.suppressed is True
    assert result.confident is False
    assert result.temperature_f == pytest.approx(69.7)  # emits last confident
    assert result.state_fields["LastConfidentOutdoorTemp"] == pytest.approx(69.7)
    assert result.state_fields["ConfidentHoldCount"] == 1


def test_genuine_corroborated_warmup_passes_first_poll():
    """Genuine warm-up: churn + C>=2 (surviving sensor + Open-Meteo) → passes.

    Even though the contributor set rotates, a surviving station (KAAA +2.0)
    AND the Open-Meteo peer (+2.2) both confirm the rise, so the +2.0 move is
    trusted on the FIRST poll as ``confident_corroborated`` — no lag, so the
    bare decision-engine close can fire this cycle.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=69.0,
        last_confident=69.0,
        contributors={"KAAA": 69.0, "KBBB": 69.0, "OPENMETEO": 69.0},
        history=[69.0, 69.0, 69.0],
        raw_history=[69.0, 69.0, 69.0],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 71.0, "KCCC": 71.0, "OPENMETEO": 71.2})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 71.0),
            _contrib_entry("KCCC", 71.0),
            _contrib_entry("OPENMETEO", 71.2, source_type="openmeteo"),
        ],
        real_station_count=2,
        openmeteo_present=True,
    )

    # Act
    result = validate_outdoor_temperature(71.0, contributors, prev, contributor_log=clog)

    # Assert
    assert result.reason == "confident_corroborated"
    assert result.suppressed is False
    assert result.confident is True
    assert result.temperature_f == pytest.approx(71.0)  # emits the fused value


def test_confident_stable_set_no_churn_passes():
    """No churn (identical set, no stickiness, no station drop) → trusted move.

    A supra-threshold move on a stable contributor set is genuine by
    construction, so it passes as ``confident_stable_set`` without needing
    corroboration.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=69.0,
        last_confident=69.0,
        contributors={"KAAA": 69.0, "KBBB": 69.0},
        history=[69.0, 69.0, 69.0],
        raw_history=[69.0, 69.0, 69.0],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 71.0, "KBBB": 71.0})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 71.0),
            _contrib_entry("KBBB", 71.0),
        ],
        real_station_count=2,
    )

    # Act
    result = validate_outdoor_temperature(71.0, contributors, prev, contributor_log=clog)

    # Assert
    assert result.reason == "confident_stable_set"
    assert result.suppressed is False
    assert result.confident is True
    assert result.temperature_f == pytest.approx(71.0)


def test_cache_fallback_move_is_held():
    """A move sourced from an LKG cache fallback is not trusted → HELD.

    ``used_cache_fallback`` (or any single ``is_cached`` contributor) marks a
    cache-driven cycle: the confidence gate refuses to move off the last
    confident value on cached data (``held_cache_only``).
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=69.0,
        last_confident=69.0,
        contributors={"KAAA": 69.0, "KBBB": 69.0},
        history=[69.0, 69.0, 69.0],
        raw_history=[69.0, 69.0, 69.0],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 70.5, "KCCC": 70.5})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 70.5),
            _contrib_entry("KCCC", 70.5, is_cached=True),
        ],
        real_station_count=2,
        used_cache_fallback=True,
    )

    # Act
    result = validate_outdoor_temperature(70.5, contributors, prev, contributor_log=clog)

    # Assert
    assert result.reason == "held_cache_only"
    assert result.suppressed is True
    assert result.confident is False
    assert result.temperature_f == pytest.approx(69.0)


def test_wide_spread_uncorroborated_move_is_held():
    """Churn + uncorroborated + wide contributor spread (>3°F) → HELD.

    When the surviving station disagrees so badly that the contributor spread
    exceeds ``confidence_max_spread_f`` (3.0°F) and nothing corroborates, the
    move is untrustworthy and held as ``held_wide_spread``.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=70.0,
        last_confident=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=[70.0, 70.0, 70.0],
        raw_history=[70.0, 70.0, 70.0],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 70.0, "KCCC": 74.5})  # spread 4.5°F
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 70.0),
            _contrib_entry("KCCC", 74.5),
        ],
        real_station_count=2,
    )

    # Act
    result = validate_outdoor_temperature(71.5, contributors, prev, contributor_log=clog)

    # Assert
    assert result.reason == "held_wide_spread"
    assert result.suppressed is True
    assert result.confident is False
    assert result.temperature_f == pytest.approx(70.0)


def test_openmeteo_only_corroboration_passes():
    """Survivors flat but Open-Meteo agrees + low bar → confident_openmeteo_agree.

    No surviving real sensor moves, but the Open-Meteo peer confirms the rise.
    With ``min_corroborating_sources=1`` the peer alone authorizes the move,
    and because ``survivor_c == 0`` the reason is the OM-specific
    ``confident_openmeteo_agree``.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=69.0,
        last_confident=69.0,
        contributors={"KAAA": 69.0, "OPENMETEO": 69.0},
        history=[69.0, 69.0, 69.0],
        raw_history=[69.0, 69.0, 69.0],
        real_station_count=1,
    )
    contributors = _contributors({"KAAA": 69.0, "KCCC": 70.5, "OPENMETEO": 70.8})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 69.0),
            _contrib_entry("KCCC", 70.5),
            _contrib_entry("OPENMETEO", 70.8, source_type="openmeteo"),
        ],
        real_station_count=2,
        openmeteo_present=True,
    )

    # Act
    result = validate_outdoor_temperature(
        70.5, contributors, prev, contributor_log=clog, min_corroborating_sources=1
    )

    # Assert
    assert result.reason == "confident_openmeteo_agree"
    assert result.suppressed is False
    assert result.confident is True
    assert result.temperature_f == pytest.approx(70.5)


def test_confidence_hold_safety_valve_releases_at_max_cycles():
    """A hold never lasts forever: at ``confidence_hold_max_cycles`` it releases.

    With one hold already recorded (``ConfidentHoldCount=1``) and the default
    max of 2 cycles, a second would-be hold instead RELEASES: the move is
    accepted as ``hold_expired_accept``, the value becomes confident, and the
    hold counter resets to 0.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=69.7,
        last_confident=69.7,
        contributors={"KAAA": 69.7, "KCCC": 71.6},
        history=[69.7, 69.7, 69.7],
        raw_history=[69.7, 69.7, 69.7],
        hold_count=1,
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 69.7, "KDDD": 71.6})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 69.7),
            _contrib_entry("KDDD", 71.6),
        ],
        real_station_count=2,
    )

    # Act
    result = validate_outdoor_temperature(71.6, contributors, prev, contributor_log=clog)

    # Assert
    assert result.reason == "hold_expired_accept"
    assert result.suppressed is False
    assert result.confident is True
    assert result.temperature_f == pytest.approx(71.6)
    assert result.state_fields["ConfidentHoldCount"] == 0


def test_uncorroborated_downward_churn_is_also_held():
    """Symmetry: a churn-driven uncorroborated DOWNWARD move is held too.

    The confidence gate is direction-symmetric (signal-trust, NOT a close-only
    margin). A rotation that drops the median 70.0 → 68.5 with no corroborating
    survivor is held exactly like the upward case.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=70.0,
        last_confident=70.0,
        contributors={"KAAA": 70.0, "KBBB": 70.0},
        history=[70.0, 70.0, 70.0],
        raw_history=[70.0, 70.0, 70.0],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 70.0, "KCCC": 68.0})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 70.0),
            _contrib_entry("KCCC", 68.0),
        ],
        real_station_count=2,
    )

    # Act
    result = validate_outdoor_temperature(68.5, contributors, prev, contributor_log=clog)

    # Assert
    assert result.reason == "held_uncorroborated_churn"
    assert result.suppressed is True
    assert result.confident is False
    assert result.temperature_f == pytest.approx(70.0)


def test_confidence_disabled_emits_fused_legacy():
    """``confidence_enabled=False`` → legacy pass-through (``disabled_legacy``).

    With the confidence gate bypassed, a churn-coincident uncorroborated move
    that would otherwise be held is emitted unchanged and marked confident.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=69.0,
        last_confident=69.0,
        contributors={"KAAA": 69.0, "KBBB": 69.0},
        history=[69.0, 69.0, 69.0],
        raw_history=[69.0, 69.0, 69.0],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 69.0, "KCCC": 71.0})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 69.0),
            _contrib_entry("KCCC", 71.0),
        ],
        real_station_count=2,
    )

    # Act
    result = validate_outdoor_temperature(
        71.0, contributors, prev, contributor_log=clog, confidence_enabled=False
    )

    # Assert
    assert result.reason == "disabled_legacy"
    assert result.suppressed is False
    assert result.confident is True
    assert result.temperature_f == pytest.approx(71.0)


def test_confidence_disabled_still_suppresses_spike():
    """Even with the confidence gate off, the spike gate still fires first.

    ``confidence_enabled=False`` bypasses only the confidence gate; the
    upstream spike gate (and within_threshold band) remain active. A full
    rotation +2.3°F spike with no survivor is still held as ``suppressed_spike``.
    """
    # Arrange
    prev = _prev_state_conf(
        last_temp=72.7,
        last_confident=72.7,
        contributors={"KAAA": 72.7, "KBBB": 72.7},
        history=[72.7, 72.7, 72.7],
        raw_history=[72.7, 72.7, 72.7],
        real_station_count=2,
    )
    contributors = _contributors({"KCCC": 75.0, "KDDD": 75.0})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KCCC", 75.0),
            _contrib_entry("KDDD", 75.0),
        ],
        real_station_count=2,
    )

    # Act
    result = validate_outdoor_temperature(
        75.0, contributors, prev, contributor_log=clog, confidence_enabled=False
    )

    # Assert
    assert result.reason == "suppressed_spike"
    assert result.suppressed is True
    assert result.confident is False
    assert result.temperature_f == pytest.approx(72.7)


# ==================================================================
# G. STATE ROUND-TRIP (confidence-gate persisted fields)
# ==================================================================


def test_confidence_state_round_trips_hold_then_release():
    """LastConfidentOutdoorTemp / ConfidentHoldCount / LastRealStationCount
    persist and re-read across cycles: a hold increments the counter, and the
    safety-valve release resets it.

    Cycle 1 feeds its persisted ``state_fields`` straight back in as the prior
    state for cycle 2 (exactly as the orchestrator persists to ``__global__``),
    proving the fields survive a round trip.
    """
    # --- Cycle 1: churn + uncorroborated → HELD, counter increments 0 → 1 ---
    prev1 = _prev_state_conf(
        last_temp=69.7,
        last_confident=69.7,
        contributors={"KAAA": 69.7, "KBBB": 69.7},
        history=[69.7, 69.7, 69.7],
        raw_history=[69.7, 69.7, 69.7],
        hold_count=0,
        real_station_count=2,
    )
    contributors1 = _contributors({"KAAA": 69.7, "KCCC": 71.6})
    clog1 = _clog_payload(
        contributors=[_contrib_entry("KAAA", 69.7), _contrib_entry("KCCC", 71.6)],
        real_station_count=2,
    )

    r1 = validate_outdoor_temperature(71.6, contributors1, prev1, contributor_log=clog1)

    assert r1.reason == "held_uncorroborated_churn"
    assert r1.suppressed is True
    assert r1.state_fields["ConfidentHoldCount"] == 1
    assert r1.state_fields["LastConfidentOutdoorTemp"] == pytest.approx(69.7)
    assert r1.state_fields["LastRealStationCount"] == 2

    # --- Cycle 2: re-read persisted state; another churn → safety-valve release
    prev2 = r1.state_fields  # round-trip the persisted __global__ fields verbatim
    contributors2 = _contributors({"KAAA": 69.7, "KDDD": 71.6})  # set changes again
    clog2 = _clog_payload(
        contributors=[_contrib_entry("KAAA", 69.7), _contrib_entry("KDDD", 71.6)],
        real_station_count=2,
    )

    r2 = validate_outdoor_temperature(71.6, contributors2, prev2, contributor_log=clog2)

    # The persisted hold_count (1) was read back; +1 hits the 2-cycle max → release.
    assert r2.reason == "hold_expired_accept"
    assert r2.suppressed is False
    assert r2.confident is True
    assert r2.temperature_f == pytest.approx(71.6)
    assert r2.state_fields["ConfidentHoldCount"] == 0  # emit resets the counter
    assert r2.state_fields["LastConfidentOutdoorTemp"] == pytest.approx(71.6)


# ==================================================================
# H. INTEGRATION — validator + bare decision-engine close (this morning's flap)
# ==================================================================


def _mini_engine():
    """Minimal DecisionEngine config exercising the bare close path."""
    from src.decision_engine import DecisionEngine

    return DecisionEngine(
        {
            "hysteresis_open_diff": 1.0,
            "hysteresis_close_diff": 1.0,  # ignored by the reverted bare close
            "max_outdoor_humidity": 80,
            "max_aqi_threshold": 100,
            "min_aqi_for_opening": 50,
            "allowed_hvac_modes": ["cool", "heatCool", "auto"],
        }
    )


def test_morning_artifact_replay_stays_open_no_flap():
    """Replay this morning's artifact: 71.6 rotation with Open-Meteo flat.

    The rotation artifact pushes the raw median to 71.6, but Open-Meteo stays
    flat and the surviving station doesn't move, so the validator HOLDS 69.7.
    Feeding the validated 69.7 into the bare decision-engine close ( > coolest
    of 71.2 ) keeps the window OPEN — no spurious close, no flap.
    """
    # Arrange — validator holds the artifact at the last confident 69.7.
    prev = _prev_state_conf(
        last_temp=69.7,
        last_confident=69.7,
        contributors={"KAAA": 69.7, "KBBB": 69.7, "OPENMETEO": 69.7},
        history=[69.7, 69.7, 69.7],
        raw_history=[69.7, 69.7, 69.7],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 69.7, "KCCC": 71.6, "OPENMETEO": 69.7})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 69.7),
            _contrib_entry("KCCC", 71.6),
            _contrib_entry("OPENMETEO", 69.7, source_type="openmeteo"),  # flat peer
        ],
        real_station_count=2,
        openmeteo_present=True,
    )

    # Act — validate, then run the bare decision-engine close.
    val = validate_outdoor_temperature(71.6, contributors, prev, contributor_log=clog)
    engine = _mini_engine()
    decision = engine.decide(
        floor="upstairs",
        floor_sensors=[{"name": "s", "temperature_f": 71.2, "is_online": True}],
        outdoor={"temperature_f": val.temperature_f, "humidity": 50.0},
        aqi={"aqi": 30},
        hvac_mode="cool",
        last_state="OPEN",
        floor_group=["s"],
    )

    # Assert — held at 69.7 and the window stays OPEN (69.7 < 71.2, no flap).
    assert val.suppressed is True
    assert val.temperature_f == pytest.approx(69.7)
    assert decision.new_state == "OPEN"
    assert decision.changed is False


def test_corroborated_warmup_replay_closes_first_poll():
    """A genuinely corroborated warm-up feeds the bare close and it fires now.

    When the rise is confirmed (surviving station + Open-Meteo), the validator
    emits the fused 71.6 on the FIRST poll, so the bare close ( 71.6 > coolest
    71.2 ) closes the window this cycle — the signal layer does not delay a
    real move.
    """
    # Arrange — corroborated rise so the validator passes it through.
    prev = _prev_state_conf(
        last_temp=69.7,
        last_confident=69.7,
        contributors={"KAAA": 69.7, "KBBB": 69.7, "OPENMETEO": 69.7},
        history=[69.7, 69.7, 69.7],
        raw_history=[69.7, 69.7, 69.7],
        real_station_count=2,
    )
    contributors = _contributors({"KAAA": 71.6, "KCCC": 71.6, "OPENMETEO": 71.8})
    clog = _clog_payload(
        contributors=[
            _contrib_entry("KAAA", 71.6),
            _contrib_entry("KCCC", 71.6),
            _contrib_entry("OPENMETEO", 71.8, source_type="openmeteo"),
        ],
        real_station_count=2,
        openmeteo_present=True,
    )

    # Act
    val = validate_outdoor_temperature(71.6, contributors, prev, contributor_log=clog)
    engine = _mini_engine()
    decision = engine.decide(
        floor="upstairs",
        floor_sensors=[{"name": "s", "temperature_f": 71.2, "is_online": True}],
        outdoor={"temperature_f": val.temperature_f, "humidity": 50.0},
        aqi={"aqi": 30},
        hvac_mode="cool",
        last_state="OPEN",
        floor_group=["s"],
    )

    # Assert — passed through and the bare close fires immediately.
    assert val.suppressed is False
    assert val.reason == "confident_corroborated"
    assert val.temperature_f == pytest.approx(71.6)
    assert decision.new_state == "CLOSED"
    assert decision.changed is True
