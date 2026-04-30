"""Quiet hours helpers for WindowBot.

Pure functions — no I/O, no side effects. All time handling uses UTC inputs
and converts to local time via zoneinfo.ZoneInfo only where necessary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# If now - last_check exceeds this gap, boundary functions return False to
# prevent stale cold-start firings (e.g. function wakes at 09:00 after a
# 2-hour gap and would otherwise think quiet hours just ended at 07:00).
MAX_BOUNDARY_GAP = timedelta(minutes=30)


@dataclass(frozen=True, slots=True)
class QuietHoursConfig:
    """Parsed, validated quiet-hours configuration."""

    start: time
    end: time
    tz: ZoneInfo


def get_quiet_hours(config: dict) -> QuietHoursConfig | None:
    """Return a QuietHoursConfig if all three keys are present and valid.

    Returns None (feature disabled) if any key is absent, empty, or
    unparseable — or if start == end (would mean 24-hour quiet, treated
    as a misconfiguration).

    Args:
        config: Dict from ``src.config.get_config()``.
    """
    start_str = config.get("quiet_hours_start")
    end_str = config.get("quiet_hours_end")
    tz_str = config.get("quiet_hours_timezone")

    if not start_str or not end_str or not tz_str:
        return None

    try:
        start = time.fromisoformat(start_str)
        end = time.fromisoformat(end_str)
    except ValueError:
        return None

    try:
        tz = ZoneInfo(tz_str)
    except (ZoneInfoNotFoundError, KeyError):
        return None

    if start == end:
        # start == end would mean always-active (or never-active depending on
        # interpretation) — treat as disabled per spec.
        return None

    return QuietHoursConfig(start=start, end=end, tz=tz)


def is_active(now_utc: datetime, qh: QuietHoursConfig) -> bool:
    """Return True if *now_utc* falls within the quiet-hours window.

    Handles both same-day windows (start < end) and midnight-spanning
    windows (start > end, e.g. 22:00–07:00).

    Args:
        now_utc: Current time as a UTC-aware datetime.
        qh: Validated quiet-hours config.
    """
    local_dt = now_utc.astimezone(qh.tz)
    t = local_dt.time()

    if qh.start == qh.end:
        return False  # disabled

    if qh.start < qh.end:
        # Same-day window, e.g. 01:00–05:00
        return qh.start <= t < qh.end

    # Midnight-spanning window, e.g. 22:00–07:00
    return t >= qh.start or t < qh.end


def just_started(
    now_utc: datetime,
    last_check_utc: datetime,
    qh: QuietHoursConfig,
) -> bool:
    """Return True if quiet hours began between *last_check_utc* and *now_utc*.

    Applies the stale-cycle guard: if the gap exceeds MAX_BOUNDARY_GAP the
    function returns False to avoid spurious boundary firings on cold starts.

    Args:
        now_utc: Current time (UTC-aware).
        last_check_utc: Time of the previous decision cycle (UTC-aware).
        qh: Validated quiet-hours config.
    """
    if now_utc - last_check_utc > MAX_BOUNDARY_GAP:
        return False
    return is_active(now_utc, qh) and not is_active(last_check_utc, qh)


def just_ended(
    now_utc: datetime,
    last_check_utc: datetime,
    qh: QuietHoursConfig,
) -> bool:
    """Return True if quiet hours ended between *last_check_utc* and *now_utc*.

    Applies the stale-cycle guard: if the gap exceeds MAX_BOUNDARY_GAP the
    function returns False to avoid spurious boundary firings on cold starts.

    Args:
        now_utc: Current time (UTC-aware).
        last_check_utc: Time of the previous decision cycle (UTC-aware).
        qh: Validated quiet-hours config.
    """
    if now_utc - last_check_utc > MAX_BOUNDARY_GAP:
        return False
    return not is_active(now_utc, qh) and is_active(last_check_utc, qh)
