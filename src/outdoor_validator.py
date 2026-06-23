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

Independently of set stability, a SPIKE GATE holds any jump larger than
``spike_max_rate_f`` (a physically-implausible per-cycle rate) unless it is
corroborated by a surviving sensor, aligned with the validated-history trend,
or sustained — i.e. the previous RAW observation (tracked in a separate
``OutdoorRawHistory``) was already on the same side of the last validated
temp. This catches one-cycle spikes on a stable single-station median, which
the ``genuine_stable_set`` path would otherwise wave through, while delaying
genuine sustained warming by at most one cycle (``suppressed_spike``).
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
                                  # suppressed_spike,
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
    spike_max_rate_f: float = 2.0,
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
        spike_max_rate_f: Maximum physically-plausible per-cycle change. A
            jump larger than this is held unless corroborated, trend-aligned,
            or sustained across cycles (raw-history persistence).

    Returns:
        An :class:`OutdoorValidationResult` with the validated temperature, a
        machine-readable reason, the suppression flag, and the state fields to
        persist back to the ``__global__`` key.
    """
    last_temp = prev_state.get("LastOutdoorTemp")
    last_contrib_map = _parse_contributor_map(prev_state.get("LastOutdoorContributors"))
    parsed_history = _parse_history(prev_state.get("OutdoorTempHistory"))
    raw_history = _parse_history(prev_state.get("OutdoorRawHistory"))

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
            "OutdoorRawHistory": json.dumps((raw_history + [fused_temp])[-max_history:]),
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

    # Corroboration: any surviving sensor moving the same direction as delta.
    # Corroboration requires INDEPENDENT evidence: a surviving sensor that is
    # not the sole source of the median. A single-contributor median cannot
    # corroborate its own jump (its own_delta is trivially equal to delta), so
    # corroboration is only meaningful when more than one sensor contributes.
    corroborated = False
    if len(current_map) > 1:
        for sid in current_ids & last_ids:
            own_delta = current_map[sid] - last_contrib_map[sid]
            if (
                own_delta
                and _sign(own_delta) == _sign(delta)
                and abs(own_delta) >= corroboration_epsilon_f
            ):
                corroborated = True
                break

    # Validated-history trend slope.
    recent = parsed_history[-trend_window:]
    if len(recent) >= 2:
        trend = (recent[-1] - recent[0]) / (len(recent) - 1)
    else:
        trend = 0.0
    trend_aligned = _sign(trend) != 0 and _sign(delta) == _sign(trend)

    # Persistence: was the PREVIOUS raw observation already on the same side of
    # last_temp as this jump? If so the elevated reading has lasted >=2 cycles,
    # so it is a sustained move, not a one-cycle spike.
    prev_raw = raw_history[-1] if raw_history else last_temp
    sustained = (
        _sign(prev_raw - last_temp) == _sign(delta)
        and abs(prev_raw - last_temp) > jitter_threshold_f
    )

    # SPIKE GATE: an implausibly large single-cycle jump that nothing supports
    # is held. Applies even on a stable contributor set (the single-station
    # case), which the genuine_stable_set path below would otherwise wave
    # through. Genuine sustained warming is delayed at most one cycle: if the
    # high reading persists, `sustained` (or the trend) clears it next cycle.
    if (
        abs(delta) > spike_max_rate_f
        and not corroborated
        and not trend_aligned
        and not sustained
    ):
        logger.warning(
            "Outdoor temp spike suppressed: delta=%.2f°F (> %.1f/cycle) held at "
            "%.1f°F (raw=%.1f, prev_raw=%.1f, trend=%.3f)",
            delta, spike_max_rate_f, last_temp, fused_temp, prev_raw, trend,
        )
        return _result(last_temp, "suppressed_spike", True)

    if not set_changed:
        return _result(fused_temp, "genuine_stable_set", False)

    if corroborated:
        return _result(fused_temp, "genuine_corroborated", False)

    if trend_aligned:
        return _result(fused_temp, "genuine_trend_aligned", False)

    # All discriminators agree: this is availability-driven jitter. Hold.
    logger.warning(
        "Outdoor temp jitter suppressed: delta=%.2f°F held at %.1f°F "
        "(old_ids=%s new_ids=%s trend=%.3f)",
        delta, last_temp, sorted(last_ids), sorted(current_ids), trend,
    )
    return _result(last_temp, "suppressed_jitter", True)
