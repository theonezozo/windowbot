"""Tests for src/quiet_hours.py — quiet hours feature helpers.

Design spec: .squad/decisions/inbox/ava-quiet-hours-design.md
Barbara (Test Engineer) — Session: quiet-hours tests
"""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.quiet_hours import (
    MAX_BOUNDARY_GAP,
    QuietHoursConfig,
    get_quiet_hours,
    is_active,
    just_ended,
    just_started,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LA = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc


def _local_to_utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Build a UTC datetime from a local LA wall-clock time."""
    local_dt = datetime(year, month, day, hour, minute, tzinfo=LA)
    return local_dt.astimezone(UTC)


# Use a fixed non-DST reference date to keep tests deterministic.
# 2024-01-15 is a Monday in standard time (UTC-8).
_REF_DATE = (2024, 1, 15)


def _utc(*args) -> datetime:
    """Shorthand: _utc(hour, minute) → UTC datetime on reference date using LA tz."""
    h, m = args if len(args) == 2 else (args[0], 0)
    return _local_to_utc(*_REF_DATE, h, m)


# ---------------------------------------------------------------------------
# 1. get_quiet_hours — config parsing
# ---------------------------------------------------------------------------


class TestGetQuietHours:
    """get_quiet_hours() receives the output of get_config() — keys are lowercase."""

    def test_all_keys_missing_returns_none(self):
        assert get_quiet_hours({}) is None

    @pytest.mark.parametrize(
        "config",
        [
            {"quiet_hours_start": "22:00", "quiet_hours_end": "07:00"},                        # missing tz
            {"quiet_hours_start": "22:00", "quiet_hours_timezone": "America/Los_Angeles"},     # missing end
            {"quiet_hours_end": "07:00", "quiet_hours_timezone": "America/Los_Angeles"},       # missing start
        ],
    )
    def test_any_one_key_missing_returns_none(self, config):
        assert get_quiet_hours(config) is None

    def test_invalid_timezone_returns_none(self):
        config = {
            "quiet_hours_start": "22:00",
            "quiet_hours_end": "07:00",
            "quiet_hours_timezone": "Not/ATimezone",
        }
        assert get_quiet_hours(config) is None

    @pytest.mark.parametrize(
        "start, end",
        [
            ("25:00", "07:00"),   # hour out of range
            ("22:00", "99:59"),   # end hour out of range
            ("ab:cd", "07:00"),   # non-numeric
        ],
    )
    def test_invalid_time_format_returns_none(self, start, end):
        config = {
            "quiet_hours_start": start,
            "quiet_hours_end": end,
            "quiet_hours_timezone": "America/Los_Angeles",
        }
        assert get_quiet_hours(config) is None

    def test_all_valid_returns_quiet_hours_config(self):
        config = {
            "quiet_hours_start": "22:00",
            "quiet_hours_end": "07:00",
            "quiet_hours_timezone": "America/Los_Angeles",
        }
        qh = get_quiet_hours(config)
        assert qh is not None
        assert isinstance(qh, QuietHoursConfig)
        assert qh.start == time(22, 0)
        assert qh.end == time(7, 0)
        assert str(qh.tz) == "America/Los_Angeles"

    def test_valid_same_day_window_returns_config(self):
        config = {
            "quiet_hours_start": "08:00",
            "quiet_hours_end": "18:00",
            "quiet_hours_timezone": "America/Los_Angeles",
        }
        qh = get_quiet_hours(config)
        assert qh is not None
        assert qh.start == time(8, 0)
        assert qh.end == time(18, 0)


# ---------------------------------------------------------------------------
# 2. is_active — same-day window (08:00–18:00 local)
# ---------------------------------------------------------------------------


class TestIsActiveSameDay:
    """08:00–18:00 LA local — start < end."""

    @pytest.fixture
    def qh(self):
        return QuietHoursConfig(start=time(8, 0), end=time(18, 0), tz=LA)

    def test_before_start_returns_false(self, qh):
        now = _utc(7, 59)
        assert is_active(now, qh) is False

    def test_at_start_exactly_returns_true(self, qh):
        now = _utc(8, 0)
        assert is_active(now, qh) is True

    def test_during_window_returns_true(self, qh):
        now = _utc(13, 0)
        assert is_active(now, qh) is True

    def test_at_end_exactly_returns_false(self, qh):
        # end is exclusive
        now = _utc(18, 0)
        assert is_active(now, qh) is False

    def test_after_end_returns_false(self, qh):
        now = _utc(20, 0)
        assert is_active(now, qh) is False


# ---------------------------------------------------------------------------
# 3. is_active — midnight-spanning window (22:00–07:00 local)
# ---------------------------------------------------------------------------


class TestIsActiveMidnightSpanning:
    """22:00–07:00 LA local — start > end."""

    @pytest.fixture
    def qh(self):
        return QuietHoursConfig(start=time(22, 0), end=time(7, 0), tz=LA)

    def test_23_00_local_returns_true(self, qh):
        now = _utc(23, 0)
        assert is_active(now, qh) is True

    def test_00_30_local_returns_true(self, qh):
        # Next calendar day at 00:30 LA
        now = _local_to_utc(2024, 1, 16, 0, 30)
        assert is_active(now, qh) is True

    def test_06_59_local_returns_true(self, qh):
        now = _local_to_utc(2024, 1, 16, 6, 59)
        assert is_active(now, qh) is True

    def test_07_00_local_returns_false(self, qh):
        now = _local_to_utc(2024, 1, 16, 7, 0)
        assert is_active(now, qh) is False

    def test_12_00_local_returns_false(self, qh):
        now = _local_to_utc(2024, 1, 16, 12, 0)
        assert is_active(now, qh) is False

    def test_at_start_exactly_returns_true(self, qh):
        now = _utc(22, 0)
        assert is_active(now, qh) is True


# ---------------------------------------------------------------------------
# 4. is_active — edge cases
# ---------------------------------------------------------------------------


class TestIsActiveEdgeCases:
    def test_start_equals_end_always_false(self):
        qh = QuietHoursConfig(start=time(12, 0), end=time(12, 0), tz=LA)
        for hour in (0, 6, 12, 18, 23):
            now = _utc(hour, 0)
            assert is_active(now, qh) is False, f"Expected False at {hour}:00"

    @pytest.mark.parametrize("hour", [0, 1, 6, 7, 11, 12, 17, 18, 22, 23])
    def test_start_equals_end_parametrize_always_false(self, hour):
        qh = QuietHoursConfig(start=time(10, 0), end=time(10, 0), tz=LA)
        now = _utc(hour, 0)
        assert is_active(now, qh) is False


# ---------------------------------------------------------------------------
# 5. just_started / just_ended — boundary detection
# ---------------------------------------------------------------------------


class TestBoundaryDetection:
    """22:00–07:00 midnight-spanning window."""

    @pytest.fixture
    def qh(self):
        return QuietHoursConfig(start=time(22, 0), end=time(7, 0), tz=LA)

    def _within_gap(self, anchor_utc: datetime, minutes: int = 10) -> datetime:
        """Return a time `minutes` before anchor, within MAX_BOUNDARY_GAP."""
        return anchor_utc - timedelta(minutes=minutes)

    def test_just_started_crossing_inactive_to_active(self, qh):
        """Last check was just before 22:00, now it's just after."""
        now = _utc(22, 5)
        last = _utc(21, 55)
        assert just_started(now, last, qh) is True

    def test_just_ended_crossing_active_to_inactive(self, qh):
        """Last check was just before 07:00 (active), now it's 07:05 (inactive)."""
        now = _local_to_utc(2024, 1, 16, 7, 5)
        last = _local_to_utc(2024, 1, 16, 6, 55)
        assert just_ended(now, last, qh) is True

    def test_both_false_when_both_inactive(self, qh):
        now = _utc(14, 0)
        last = _utc(13, 50)
        assert just_started(now, last, qh) is False
        assert just_ended(now, last, qh) is False

    def test_both_false_when_both_active(self, qh):
        now = _utc(23, 0)
        last = _utc(22, 50)
        assert just_started(now, last, qh) is False
        assert just_ended(now, last, qh) is False

    def test_just_started_and_just_ended_cannot_both_be_true(self, qh):
        """For any (now, last, qh), just_started and just_ended are mutually exclusive."""
        # Test multiple scenarios
        test_cases = [
            (_utc(22, 5), _utc(21, 55)),                              # crossing → active
            (_local_to_utc(2024, 1, 16, 7, 5), _local_to_utc(2024, 1, 16, 6, 55)),  # crossing → inactive
            (_utc(14, 0), _utc(13, 50)),                              # both inactive
            (_utc(23, 0), _utc(22, 50)),                              # both active
        ]
        for now, last in test_cases:
            started = just_started(now, last, qh)
            ended = just_ended(now, last, qh)
            assert not (started and ended), f"Both True at now={now}, last={last}"

    def test_just_started_false_when_gap_exceeds_max(self, qh):
        """Stale-cycle guard: gap > MAX_BOUNDARY_GAP → just_started returns False."""
        now = _utc(22, 5)
        stale_last = now - MAX_BOUNDARY_GAP - timedelta(seconds=1)
        assert just_started(now, stale_last, qh) is False

    def test_just_ended_false_when_gap_exceeds_max(self, qh):
        """Stale-cycle guard: gap > MAX_BOUNDARY_GAP → just_ended returns False."""
        now = _local_to_utc(2024, 1, 16, 7, 5)
        stale_last = now - MAX_BOUNDARY_GAP - timedelta(seconds=1)
        assert just_ended(now, stale_last, qh) is False


# ---------------------------------------------------------------------------
# 6. Stale-cycle guard — parametrized gap boundary
# ---------------------------------------------------------------------------


class TestStaleCycleGuard:
    """Parametrize gaps just under and just over MAX_BOUNDARY_GAP."""

    @pytest.fixture
    def qh(self):
        return QuietHoursConfig(start=time(22, 0), end=time(7, 0), tz=LA)

    @pytest.mark.parametrize(
        "gap_delta, expect_detection",
        [
            (timedelta(seconds=1),          True),   # 1 s in → boundary detected
            (timedelta(minutes=15),         True),   # well within gap
            (MAX_BOUNDARY_GAP,              True),   # exactly at boundary (inclusive)
            (MAX_BOUNDARY_GAP + timedelta(seconds=1), False),  # 1 s over → stale
            (MAX_BOUNDARY_GAP + timedelta(minutes=5), False),  # clearly stale
            (timedelta(hours=2),            False),  # very stale
        ],
    )
    def test_just_started_stale_guard(self, qh, gap_delta, expect_detection):
        """just_started crossing suppressed when gap > MAX_BOUNDARY_GAP."""
        now = _utc(22, 5)          # active
        last = now - gap_delta     # inactive (pre-22:00 minus the gap)
        # Only run the detection assertion when last is actually inactive
        last_local_time = last.astimezone(LA).time()
        last_is_inactive = not (last_local_time >= time(22, 0) or last_local_time < time(7, 0))
        if not last_is_inactive:
            pytest.skip("last is active for this gap — not a crossing scenario")
        result = just_started(now, last, qh)
        assert result is expect_detection

    @pytest.mark.parametrize(
        "gap_delta, expect_detection",
        [
            (timedelta(seconds=1),          True),
            (timedelta(minutes=15),         True),
            (MAX_BOUNDARY_GAP,              True),   # exactly at boundary (inclusive)
            (MAX_BOUNDARY_GAP + timedelta(seconds=1), False),
            (MAX_BOUNDARY_GAP + timedelta(minutes=5), False),
            (timedelta(hours=2),            False),
        ],
    )
    def test_just_ended_stale_guard(self, qh, gap_delta, expect_detection):
        """just_ended crossing suppressed when gap > MAX_BOUNDARY_GAP."""
        now = _local_to_utc(2024, 1, 16, 7, 5)   # inactive (just after 07:00)
        last = now - gap_delta                     # should be active (pre-07:00)
        last_local_time = last.astimezone(LA).time()
        last_is_active = last_local_time >= time(22, 0) or last_local_time < time(7, 0)
        if not last_is_active:
            pytest.skip("last is inactive for this gap — not a crossing scenario")
        result = just_ended(now, last, qh)
        assert result is expect_detection

    def test_max_boundary_gap_is_30_minutes(self):
        """Design spec: MAX_BOUNDARY_GAP == 30 minutes."""
        assert MAX_BOUNDARY_GAP == timedelta(minutes=30)
