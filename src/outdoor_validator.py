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
    reason: str                   # machine-readable outcome (mirrors
                                  # ``confidence_reason`` — see vocabulary below)
    suppressed: bool              # True if a jump was held back
    state_fields: dict            # fields to persist to the __global__ state key
    confident: bool = True        # True when this cycle's value is trusted
                                  # (not held). False whenever a supra-threshold
                                  # move was HELD (spike OR confidence gate).
    confidence_reason: str = ""   # machine-readable outcome, one of:
                                  #   confident_stable_set,
                                  #   confident_corroborated,
                                  #   confident_openmeteo_agree,
                                  #   held_uncorroborated_churn,
                                  #   held_wide_spread, held_cache_only,
                                  #   hold_expired_accept, cold_start,
                                  #   within_threshold,
                                  #   no_contributors_passthrough,
                                  #   suppressed_spike, disabled_legacy


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
    contributor_log: dict | None = None,
    min_corroborating_sources: int = 2,
    confidence_max_spread_f: float = 3.0,
    confidence_hold_max_cycles: int = 2,
    confidence_enabled: bool = True,
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
        contributor_log: The Phase-2 rich contributor payload
            (``outdoor["contributor_log"]``) — carries ``stickiness_active``,
            ``real_station_count``, ``used_cache_fallback``,
            ``openmeteo_present`` and per-contributor ``is_cached``/``temp_f``.
            Optional; when absent the confidence gate degrades gracefully.
        min_corroborating_sources: Corroboration count required to trust a
            churn-coincident supra-threshold move (confidence gate).
        confidence_max_spread_f: Contributor spread (°F) above which an
            uncorroborated churn-coincident move is held.
        confidence_hold_max_cycles: Safety valve — a confidence hold is
            released (value accepted) once it has persisted this many cycles.
        confidence_enabled: When False the confidence gate is bypassed
            entirely (legacy pass-through behaviour, ``disabled_legacy``).

    Returns:
        An :class:`OutdoorValidationResult` with the validated temperature, a
        machine-readable reason, the suppression flag, the trust flag
        (``confident``), the ``confidence_reason``, and the state fields to
        persist back to the ``__global__`` key.
    """
    last_temp = prev_state.get("LastOutdoorTemp")
    last_contrib_map = _parse_contributor_map(prev_state.get("LastOutdoorContributors"))
    parsed_history = _parse_history(prev_state.get("OutdoorTempHistory"))
    raw_history = _parse_history(prev_state.get("OutdoorRawHistory"))

    # Confidence-gate state (all upstream of the bare decision-engine close).
    last_confident = prev_state.get("LastConfidentOutdoorTemp")
    if last_confident is None:
        last_confident = last_temp  # fall back to the last validated temp
    hold_count = prev_state.get("ConfidentHoldCount", 0) or 0
    prior_real_station_count = prev_state.get("LastRealStationCount")
    _clog = contributor_log or {}
    current_real_station_count = _clog.get("real_station_count", 0) or 0

    current_map = {
        c["station_id"]: c["temperature_f"]
        for c in contributors
        if c.get("station_id") is not None
    }
    current_ids = set(current_map.keys())

    def _result(
        validated: float,
        reason: str,
        suppressed: bool,
        *,
        confident: bool | None = None,
        confidence_reason: str | None = None,
        last_confident_out: float | None = None,
        hold_count_out: int = 0,
    ) -> OutdoorValidationResult:
        if confident is None:
            confident = not suppressed
        if confidence_reason is None:
            confidence_reason = reason
        if last_confident_out is None:
            last_confident_out = validated
        state_fields = {
            "LastOutdoorTemp": validated,
            "LastOutdoorContributors": json.dumps(current_map),
            "OutdoorTempHistory": json.dumps((parsed_history + [validated])[-max_history:]),
            "OutdoorRawHistory": json.dumps((raw_history + [fused_temp])[-max_history:]),
            "LastConfidentOutdoorTemp": last_confident_out,
            "ConfidentHoldCount": hold_count_out,
            "LastRealStationCount": current_real_station_count,
        }
        return OutdoorValidationResult(
            temperature_f=validated,
            reason=reason,
            suppressed=suppressed,
            state_fields=state_fields,
            confident=confident,
            confidence_reason=confidence_reason,
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
        return _result(
            last_temp, "suppressed_spike", True,
            confident=False, confidence_reason="suppressed_spike",
            last_confident_out=last_confident, hold_count_out=hold_count,
        )

    # ------------------------------------------------------------------
    # CONFIDENCE GATE (signal-layer, symmetric in direction — NOT a
    # close-side deadband). A supra-threshold move relative to the last
    # CONFIDENT value is trusted only when independently backed; otherwise the
    # last confident value is HELD. This suppresses station-rotation artifacts
    # while letting a genuine, corroborated move through on the first poll so
    # the bare decision-engine close ( > coolest ) fires immediately.
    # Only ONE suppression may fire per cycle: reaching here means the spike
    # gate did NOT fire.
    # ------------------------------------------------------------------
    if not confidence_enabled:
        return _result(
            fused_temp, "disabled_legacy", False,
            confident=True, confidence_reason="disabled_legacy",
            last_confident_out=fused_temp, hold_count_out=0,
        )

    conf_delta = fused_temp - last_confident

    # Churn: the contributor SET changed, stickiness held a source back, or the
    # real-station pool shrank versus the prior cycle.
    station_dropped = (
        prior_real_station_count is not None
        and current_real_station_count < prior_real_station_count
    )
    churn = (
        (current_ids != last_ids)
        or bool(_clog.get("stickiness_active"))
        or station_dropped
    )

    # Spread across the current contributor temperatures.
    _cur_temps = list(current_map.values())
    spread = (max(_cur_temps) - min(_cur_temps)) if len(_cur_temps) >= 2 else 0.0

    # Cache-driven cycle? (LKG fallback, or any single contributor served cached.)
    is_cache = bool(_clog.get("used_cache_fallback")) or any(
        c.get("is_cached") for c in _clog.get("contributors", [])
    )

    # --- Corroboration count C -----------------------------------------
    # 1) Surviving real sensors that moved the same direction as conf_delta.
    survivor_c = 0
    if len(current_map) > 1:
        for sid in current_ids & last_ids:
            if sid == "OPENMETEO":
                continue  # OM handled separately below (no double-count)
            own_move = current_map[sid] - last_contrib_map[sid]
            if (
                own_move
                and _sign(own_move) == _sign(conf_delta)
                and abs(own_move) >= corroboration_epsilon_f
            ):
                survivor_c += 1

    # 2) Open-Meteo peer agreement.
    om_now = None
    for c in _clog.get("contributors", []):
        if c.get("source_type") == "openmeteo":
            om_now = c.get("temp_f")
            break
    if om_now is None:
        om_now = current_map.get("OPENMETEO")
    om_prior = last_contrib_map.get("OPENMETEO")
    om_corroborated = False
    if _clog.get("openmeteo_present") and om_now is not None:
        if om_prior is not None:
            om_move = om_now - om_prior
            om_corroborated = bool(om_move) and _sign(om_move) == _sign(conf_delta)
        else:
            # No OM prior: OM level on the same side of last_confident as fused.
            om_corroborated = _sign(om_now - last_confident) == _sign(conf_delta)

    corroboration = survivor_c + (1 if om_corroborated else 0)

    # --- Decision -------------------------------------------------------
    if is_cache:
        held, reason = True, "held_cache_only"
    elif not churn:
        held, reason = False, "confident_stable_set"
    elif corroboration >= min_corroborating_sources:
        held = False
        reason = (
            "confident_openmeteo_agree"
            if (survivor_c == 0 and om_corroborated)
            else "confident_corroborated"
        )
    elif spread > confidence_max_spread_f:
        held, reason = True, "held_wide_spread"
    else:
        held, reason = True, "held_uncorroborated_churn"

    if held:
        # Safety valve — never hold a sustained change forever.
        if hold_count + 1 >= confidence_hold_max_cycles:
            logger.info(
                "Outdoor confidence hold expired after %d cycle(s) — accepting "
                "%.1f°F (was holding %.1f°F, reason=%s)",
                hold_count + 1, fused_temp, last_confident, reason,
            )
            return _result(
                fused_temp, "hold_expired_accept", False,
                confident=True, confidence_reason="hold_expired_accept",
                last_confident_out=fused_temp, hold_count_out=0,
            )
        logger.warning(
            "Outdoor confidence hold: delta=%.2f°F held at %.1f°F (reason=%s, "
            "C=%d, spread=%.2f°F, old_ids=%s new_ids=%s)",
            conf_delta, last_confident, reason, corroboration, spread,
            sorted(last_ids), sorted(current_ids),
        )
        return _result(
            last_confident, reason, True,
            confident=False, confidence_reason=reason,
            last_confident_out=last_confident, hold_count_out=hold_count + 1,
        )

    return _result(
        fused_temp, reason, False,
        confident=True, confidence_reason=reason,
        last_confident_out=fused_temp, hold_count_out=0,
    )
