"""Outdoor-temperature jitter suppression for WindowBot.

Distinguishes genuine temperature movement from availability-driven jitter
(the median's contributor set rotating in/out) without lagging real moves.

Each 10-minute cycle the outdoor temperature is the median of a *changing*
set of sensors (the 3 nearest NWS stations plus an optional Open-Meteo peer).
When a contributor crosses the 20-minute freshness line, or an LKG-cache
station rotates in or out, the median can snap by a few tenths of a degree —
producing false open/close flapping.

A jump from the previous validated temperature is SUPPRESSED (held at the
previous validated value) only when ALL of these hold:

  1. ``abs(delta) > jitter_threshold_f``
  2. the contributor SET changed versus the last cycle
  3. NO surviving sensor (present in both cycles) moved in the same direction
     as ``delta`` by at least ``corroboration_epsilon_f``
  4. ``delta`` runs AGAINST the recent validated-history trend slope

Otherwise the jump PASSES THROUGH unchanged. This guarantees genuine
movement — corroborated by a surviving sensor or consistent with the trend —
is never delayed. The only cost is that a genuine spike coinciding with a
full sensor rotation AND zero surviving sensors AND against-trend is deferred
at most one cycle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("windowbot.outdoor_validator")


@dataclass
class OutdoorValidationResult:
    temperature_f: float          # validated temp to use this cycle
    reason: str                   # one of: cold_start, within_threshold,
                                  # genuine_stable_set, genuine_corroborated,
                                  # genuine_trend_aligned, suppressed_jitter,
                                  # no_contributors_passthrough
    suppressed: bool              # True if a jump was held back
    state_fields: dict            # fields to persist to the __global__ state key


def _sign(value: float) -> int:
    """Return -1, 0, or 1 for the sign of ``value``."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _parse_contributor_map(raw: object) -> dict:
    """Parse a stored ``{station_id: temperature_f}`` JSON string robustly."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _parse_history(raw: object) -> list:
    """Parse a stored list-of-floats JSON string robustly."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [float(v) for v in parsed if isinstance(v, (int, float))]


def validate_outdoor_temperature(
    fused_temp: float,
    contributors: list[dict],
    prev_state: dict,
    *,
    jitter_threshold_f: float = 0.5,
    trend_window: int = 6,
    max_history: int = 12,
    corroboration_epsilon_f: float = 0.05,
) -> OutdoorValidationResult:
    """Suppress availability-driven jitter in the fused outdoor temperature.

    Args:
        fused_temp: This cycle's median outdoor temperature.
        contributors: ``[{"station_id": str, "temperature_f": float}, ...]``
            — the sensors that fed the median this cycle.
        prev_state: The ``__global__`` floor-state dict (may be empty/defaults).
        jitter_threshold_f: Jumps within this band always pass through.
        trend_window: Number of recent validated temps used for the trend slope.
        max_history: Maximum validated temps retained in persisted history.
        corroboration_epsilon_f: Minimum own-reading move for a surviving
            sensor to count as corroborating.

    Returns:
        An :class:`OutdoorValidationResult` with the validated temperature, a
        machine-readable reason, the suppression flag, and the state fields to
        persist back to the ``__global__`` key.
    """
    last_temp = prev_state.get("LastOutdoorTemp")
    last_contrib_map = _parse_contributor_map(prev_state.get("LastOutdoorContributors"))
    parsed_history = _parse_history(prev_state.get("OutdoorTempHistory"))

    current_map = {
        c["station_id"]: c["temperature_f"]
        for c in contributors
        if c.get("station_id") is not None
    }
    current_ids = set(current_map.keys())

    def _result(validated: float, reason: str, suppressed: bool) -> OutdoorValidationResult:
        state_fields = {
            "LastOutdoorTemp": validated,
            "LastOutdoorContributors": json.dumps(current_map),
            "OutdoorTempHistory": json.dumps((parsed_history + [validated])[-max_history:]),
        }
        return OutdoorValidationResult(
            temperature_f=validated,
            reason=reason,
            suppressed=suppressed,
            state_fields=state_fields,
        )

    # Single-source / fallback cycles carry no contributor detail. Treat the
    # set as stable/unknown and pass through rather than risk suppressing a
    # legitimate single-source reading.
    if not contributors:
        return _result(fused_temp, "no_contributors_passthrough", False)

    # Cold start — no prior validated temp to compare against.
    if last_temp is None:
        return _result(fused_temp, "cold_start", False)

    delta = fused_temp - last_temp

    # Within the jitter band — always pass through.
    if abs(delta) <= jitter_threshold_f:
        return _result(fused_temp, "within_threshold", False)

    last_ids = set(last_contrib_map.keys())
    set_changed = current_ids != last_ids

    if not set_changed:
        return _result(fused_temp, "genuine_stable_set", False)

    # Corroboration: any surviving sensor moving the same direction as delta.
    corroborated = False
    for sid in current_ids & last_ids:
        own_delta = current_map[sid] - last_contrib_map[sid]
        if (
            own_delta
            and _sign(own_delta) == _sign(delta)
            and abs(own_delta) >= corroboration_epsilon_f
        ):
            corroborated = True
            break
    if corroborated:
        return _result(fused_temp, "genuine_corroborated", False)

    # Trend alignment: slope over the recent validated history.
    recent = parsed_history[-trend_window:]
    if len(recent) >= 2:
        trend = (recent[-1] - recent[0]) / (len(recent) - 1)
    else:
        trend = 0.0
    if _sign(trend) != 0 and _sign(delta) == _sign(trend):
        return _result(fused_temp, "genuine_trend_aligned", False)

    # All discriminators agree: this is availability-driven jitter. Hold.
    logger.warning(
        "Outdoor temp jitter suppressed: delta=%.2f°F held at %.1f°F "
        "(old_ids=%s new_ids=%s trend=%.3f)",
        delta, last_temp, sorted(last_ids), sorted(current_ids), trend,
    )
    return _result(last_temp, "suppressed_jitter", True)
