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
