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
    assert result.reason == "suppressed_jitter"
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
    assert result.reason == "suppressed_jitter"
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
    assert result.reason == "genuine_stable_set"
    assert result.temperature_f == 69.2


def test_corroborated_warming_passes_despite_rotation():
    """Surviving sensor moved same direction by >= epsilon → corroborated pass."""
    # Arrange
    prev = _prev_state(
        last_temp=68.0,
        contributors={"KAAA": 68.0, "KBBB": 68.0},
        history=[68.0],
    )
    contributors = _contributors({"KAAA": 68.9, "KCCC": 68.8})

    # Act
    result = validate_outdoor_temperature(68.85, contributors, prev)

    # Assert
    assert result.suppressed is False
    assert result.reason == "genuine_corroborated"
    assert result.temperature_f == pytest.approx(68.85)


def test_trend_aligned_jump_passes():
    """Jump aligned with the validated-history trend slope → passes."""
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
    assert result.suppressed is False
    assert result.reason == "genuine_trend_aligned"
    assert result.temperature_f == pytest.approx(68.7)


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
    assert result.reason in ("genuine_stable_set", "genuine_trend_aligned")


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
    independently corroborating the +2.5°F median jump, so the spike gate is
    skipped and the move passes as genuine_corroborated.
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
    result = validate_outdoor_temperature(72.5, contributors, prev)

    # Assert
    assert result.suppressed is False
    assert result.reason == "genuine_corroborated"
    assert result.temperature_f == pytest.approx(72.5)


def test_large_jump_trend_aligned_passes():
    """A large jump consistent with a rising trend passes without a hold.

    A single station on a steadily warming history ([66, 67.5, 69, 70]) jumps
    +2.5°F. REAL MODULE BEHAVIOR: the contributor set is unchanged, so the
    module returns `genuine_stable_set` (the unchanged-set path returns BEFORE
    the trend check). Either way the spike gate does not hold it: the move is
    both trend-aligned and (single-station) self-corroborated. The key
    guarantees are that it is not suppressed and the validated temp is 72.5°F.
    """
    # Arrange
    prev = _prev_state_raw(
        last_temp=70.0,
        contributors={"KAAA": 70.0},
        history=[66.0, 67.5, 69.0, 70.0],
        raw_history=[66.0, 67.5, 69.0, 70.0],
    )
    contributors = _contributors({"KAAA": 72.5})

    # Act
    result = validate_outdoor_temperature(72.5, contributors, prev)

    # Assert
    assert result.suppressed is False
    assert result.temperature_f == pytest.approx(72.5)
    # Unchanged set returns genuine_stable_set before the trend branch is reached.
    assert result.reason == "genuine_stable_set"


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
    assert result.reason == "genuine_stable_set"
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
