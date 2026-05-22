"""Tests for the status page renderer.

Focuses on the new temperature-history surface area
(see ``.squad/decisions.md`` 2026-05-19):

- ``_render_history_card`` omits the card entirely when there is no
  history, and otherwise renders a *collapsed* ``<details>`` with
  the canonical summary text and a table sized to the union of
  floor keys, with missing values shown as ``—``.
- ``render_status_page`` includes a top-level ``"history"`` key in
  JSON output.
- ``_pin_denied_response`` returns 401 (sanity check — the PIN gate
  is already exercised elsewhere; we only confirm history is not
  leaked behind the gate).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import azure.functions as func
import pytest

from src.diagnostic import (
    FloorSnapshot,
    GlobalSnapshot,
    SensorReading,
    TemperatureHistoryEntry,
)
from src.status_page import (
    _pin_denied_response,
    _render_history_card,
    render_status_page,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _entry(ts: str, outdoor: float | None, indoor: dict[str, float | None]) -> TemperatureHistoryEntry:
    return TemperatureHistoryEntry(
        timestamp=ts, outdoor_temp_f=outdoor, indoor_temps=indoor
    )


def _make_request(*, params: dict | None = None, headers: dict | None = None) -> func.HttpRequest:
    return func.HttpRequest(
        method="GET",
        url="http://localhost/api/status",
        body=b"",
        params=params or {},
        headers=headers or {},
    )


def _floor_snapshot(floor: str) -> FloorSnapshot:
    return FloorSnapshot(
        floor=floor,
        decision="CLOSED",
        reason="test",
        indoor_sensors=[SensorReading(name="s1", temperature_f=70.0, is_online=True)],
        outdoor_temp_f=65.0,
        outdoor_source="nws",
        outdoor_stations=[],
        outdoor_humidity=50.0,
        aqi_value=25,
        aqi_source="purpleair",
        aqi_stations=[],
        gates=[],
        last_notification_type=None,
        last_notification_time=None,
        timestamp="2026-05-19T14:30:00+00:00",
    )


def _global_snapshot() -> GlobalSnapshot:
    return GlobalSnapshot(
        poll_start="2026-05-19T14:30:00+00:00",
        poll_duration_seconds=1.2,
        hvac_mode="cool",
        quiet_hours_active=False,
        quiet_hours_next_transition=None,
        next_poll_eta="2026-05-19T14:40:00+00:00",
        errors=[],
    )


@pytest.fixture(autouse=True)
def _clear_pin_env(monkeypatch):
    """Status-page PIN gate is opt-in via env var; default these tests off."""
    monkeypatch.delenv("STATUS_PAGE_PIN", raising=False)


# ------------------------------------------------------------------
# _render_history_card
# ------------------------------------------------------------------


class TestRenderHistoryCard:
    """Direct unit tests of the card renderer."""

    def test_empty_history_omits_card_entirely(self):
        # No entries → no card markup at all (avoids an empty expando).
        assert _render_history_card([]) == ""

    def test_card_present_and_collapsed_by_default(self):
        entries = [_entry("2026-05-19T14:00:00+00:00", 70.0, {"upstairs": 71.0})]

        html = _render_history_card(entries)

        # Card was rendered.
        assert "history-card" in html
        # Contains the canonical summary.
        assert "<summary>📈 Temperature History (last 12h)</summary>" in html
        # The <details> element MUST NOT carry the `open` attribute —
        # collapsed by default is part of the design contract.
        assert "<details>" in html
        assert "<details open>" not in html

    def test_table_renders_one_row_per_entry_newest_first(self):
        # Entries are already newest-first when passed in (production
        # contract from get_temperature_history); the renderer preserves
        # that order.
        entries = [
            _entry("2026-05-19T14:00:00+00:00", 70.0, {"upstairs": 71.0}),
            _entry("2026-05-19T13:00:00+00:00", 68.0, {"upstairs": 70.5}),
            _entry("2026-05-19T12:00:00+00:00", 65.0, {"upstairs": 70.0}),
        ]

        html = _render_history_card(entries)

        # Three <tr> rows in <tbody> (the header <tr> is in <thead>).
        # Count by data-row content via the formatted timestamps.
        assert html.count("<tr>") == 4  # 1 thead + 3 tbody rows
        idx_14 = html.find("05-19 14:00")
        idx_13 = html.find("05-19 13:00")
        idx_12 = html.find("05-19 12:00")
        assert idx_14 != -1 and idx_13 != -1 and idx_12 != -1
        # Newest-first ordering preserved in the rendered HTML.
        assert idx_14 < idx_13 < idx_12

    def test_columns_are_union_of_floor_keys_with_missing_rendered_as_dash(self):
        # Entry 1 has both floors; entry 2 only has upstairs (downstairs
        # column should render as — for that row).  Verifies the union.
        entries = [
            _entry(
                "2026-05-19T14:00:00+00:00",
                70.0,
                {"upstairs": 71.0, "downstairs": 70.5},
            ),
            _entry(
                "2026-05-19T13:00:00+00:00",
                None,  # outdoor also missing → outdoor cell is —
                {"upstairs": 70.5},
            ),
        ]

        html = _render_history_card(entries)

        # Both floor columns appear in the header.
        assert "<th>Upstairs</th>" in html
        assert "<th>Downstairs</th>" in html
        # Missing readings render as the em-dash sentinel.
        assert "—" in html
        # The older row's downstairs cell is —, but its upstairs cell is 70.5.
        # And the outdoor cell on that row is —.
        # Locate the older row by its timestamp and verify cell contents.
        row2_start = html.find("05-19 13:00")
        # The next ~200 chars of the row should contain 70.5 (upstairs)
        # and at least two — sentinels (outdoor + downstairs missing).
        row2_chunk = html[row2_start:row2_start + 400]
        assert "70.5°F" in row2_chunk
        assert row2_chunk.count("—") >= 2


# ------------------------------------------------------------------
# render_status_page — JSON output includes history
# ------------------------------------------------------------------


class TestRenderStatusPageJsonHistory:
    """JSON output includes a top-level ``history`` key with entries serialized."""

    @patch("src.status_page.SnapshotManager")
    @patch("src.status_page.get_state_manager")
    def test_json_output_includes_history_key(
        self, mock_get_state, mock_snap_mgr_cls,
    ):
        # State manager exposes get_snapshot_table → not LocalStateManager.
        state_mgr = MagicMock()
        state_mgr.get_snapshot_table.return_value = MagicMock()
        mock_get_state.return_value = state_mgr

        snap_mgr = MagicMock()
        snap_mgr.get_all_floor_snapshots.return_value = [_floor_snapshot("upstairs")]
        snap_mgr.get_global_snapshot.return_value = _global_snapshot()
        snap_mgr.get_temperature_history.return_value = [
            _entry("2026-05-19T14:00:00+00:00", 70.0, {"upstairs": 71.0}),
            _entry("2026-05-19T13:00:00+00:00", 68.0, {"upstairs": 70.5}),
        ]
        mock_snap_mgr_cls.return_value = snap_mgr

        resp = render_status_page(_make_request(params={"format": "json"}))

        assert resp.status_code == 200
        assert resp.mimetype == "application/json"
        payload = json.loads(resp.get_body().decode())
        assert "history" in payload
        assert isinstance(payload["history"], list)
        assert len(payload["history"]) == 2
        # Entries serialized to dicts with the expected schema.
        first = payload["history"][0]
        assert first["timestamp"] == "2026-05-19T14:00:00+00:00"
        assert first["outdoor_temp_f"] == 70.0
        assert first["indoor_temps"] == {"upstairs": 71.0}
        # Fetched at 12-hour window (production default).
        snap_mgr.get_temperature_history.assert_called_once_with(hours=12)


# ------------------------------------------------------------------
# PIN gate — sanity check that 401 short-circuits before any data
# ------------------------------------------------------------------


class TestPinDeniedShortCircuits:
    """When the PIN is required but absent/wrong, history is never fetched."""

    def test_pin_denied_helper_returns_401_for_html_and_json(self):
        # Sanity check on the helper used by the gate.
        html_resp = _pin_denied_response(is_json=False)
        json_resp = _pin_denied_response(is_json=True)
        assert html_resp.status_code == 401
        assert json_resp.status_code == 401
        assert json_resp.mimetype == "application/json"

    @patch("src.status_page.SnapshotManager")
    @patch("src.status_page.get_state_manager")
    def test_missing_pin_returns_401_without_fetching_history(
        self, mock_get_state, mock_snap_mgr_cls, monkeypatch,
    ):
        monkeypatch.setenv("STATUS_PAGE_PIN", "secret")

        resp = render_status_page(_make_request(params={"format": "json"}))

        assert resp.status_code == 401
        # PIN gate ran before any snapshot work — nothing fetched.
        mock_get_state.assert_not_called()
        mock_snap_mgr_cls.assert_not_called()


# ------------------------------------------------------------------
# Outdoor freshness bucket + range-format rendering (Jacob's audit gates)
# ------------------------------------------------------------------


class TestOutdoorFreshnessBucketRendering:
    """The status page paints the outdoor ``data-freshness`` block based on
    the OLDEST contributor (worst-case bucket): ``data-fresh`` < 20 min,
    ``data-warn`` 20–45 min, ``data-stale`` ≥ 45 min. The AQI bucket is
    intentionally asymmetric (30/60 min) — this class pins that delta.
    """

    def _snapshot_with_outdoor_age(
        self,
        outdoor_minutes: int,
        *,
        newest_minutes: int | None = None,
        contributor_count: int | None = None,
        aqi_minutes: int = 5,
    ) -> FloorSnapshot:
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        snap = _floor_snapshot("upstairs")
        snap.outdoor_observation_time = (
            now - timedelta(minutes=outdoor_minutes)
        ).isoformat()
        if newest_minutes is not None:
            snap.outdoor_newest_observation_time = (
                now - timedelta(minutes=newest_minutes)
            ).isoformat()
        else:
            snap.outdoor_newest_observation_time = None
        snap.outdoor_contributor_count = contributor_count
        snap.aqi_observation_time = (
            now - timedelta(minutes=aqi_minutes)
        ).isoformat()
        return snap

    def test_outdoor_10_min_is_fresh(self):
        """10 min old → ``data-fresh`` class on the outdoor block."""
        from src.status_page import _render_environment_section

        snap = self._snapshot_with_outdoor_age(10)
        html = _render_environment_section(snap)
        assert "data-freshness data-fresh" in html
        # Sanity: not warn/stale.
        assert "data-freshness data-warn" not in html.split("Air Quality")[0]
        assert "data-freshness data-stale" not in html.split("Air Quality")[0]

    def test_outdoor_30_min_is_warn(self):
        """30 min old → ``data-warn`` on the outdoor block (past 20-min fresh,
        before 45-min stale).
        """
        from src.status_page import _render_environment_section

        snap = self._snapshot_with_outdoor_age(30)
        html = _render_environment_section(snap)
        outdoor_section = html.split("Air Quality")[0]
        assert "data-freshness data-warn" in outdoor_section

    def test_outdoor_50_min_is_stale(self):
        """50 min old → ``data-stale`` on the outdoor block (past 45-min cap)."""
        from src.status_page import _render_environment_section

        snap = self._snapshot_with_outdoor_age(50)
        html = _render_environment_section(snap)
        outdoor_section = html.split("Air Quality")[0]
        assert "data-freshness data-stale" in outdoor_section

    def test_aqi_25_min_is_fresh_despite_outdoor_20_min_threshold(self):
        """AQI bucket is 30/60 — asymmetric with outdoor 20/45. A 25-min-old
        AQI reading is still ``data-fresh`` even though it would be ``data-warn``
        under the outdoor thresholds. Pins the asymmetry from Jacob's audit.
        """
        from src.status_page import _render_environment_section

        # Force outdoor into a known bucket so it doesn't interfere with the AQI assertion.
        snap = self._snapshot_with_outdoor_age(5, aqi_minutes=25)
        html = _render_environment_section(snap)
        aqi_section = html.split("Air Quality")[1]
        assert "data-freshness data-fresh" in aqi_section

    def test_range_format_renders_when_oldest_differs_from_newest(self):
        """Multiple contributors spanning ages → renders the en-dash range
        string ``"observed Xmin–Ymin ago (N readings)"`` exactly as Gregory
        wrote it (note the U+2013 en-dash, not a hyphen).
        """
        from src.status_page import _render_environment_section

        snap = self._snapshot_with_outdoor_age(
            outdoor_minutes=25,
            newest_minutes=3,
            contributor_count=3,
        )
        html = _render_environment_section(snap)
        # Exact substring: 25min en-dash 3min ago (3 readings).
        # \u2013 is the en-dash from src/status_page.py line 320.
        assert "observed 25min\u20133min ago (3 readings)" in html
        # The outdoor block is in the warn bucket (oldest = 25 min).
        outdoor_section = html.split("Air Quality")[0]
        assert "data-freshness data-warn" in outdoor_section

    def test_single_contributor_uses_legacy_single_age_format(self):
        """When there is only one contributor (or the newest matches the
        oldest), the legacy ``"observed Xmin ago"`` string is used and the
        range-format substring does NOT appear.
        """
        from src.status_page import _render_environment_section

        snap = self._snapshot_with_outdoor_age(
            outdoor_minutes=10,
            newest_minutes=10,  # same as oldest → suppresses range format
            contributor_count=1,
        )
        html = _render_environment_section(snap)
        assert "observed 10min ago" in html
        # No range delimiter must appear in the outdoor block.
        outdoor_section = html.split("Air Quality")[0]
        assert "\u2013" not in outdoor_section
        assert "readings)" not in outdoor_section


# ------------------------------------------------------------------
# Build-info footer (deploy-stamped version surface)
# ------------------------------------------------------------------
#
# Contract: ``.squad/decisions/inbox/gregory-version-info-runtime.md``
#
# - ``_render_build_info`` returns a ``<div class="build-info">`` line.
# - Dev build (no ``_version.py``): shows ``Build: dev`` and ``worker up``,
#   without the ``build-stale`` modifier.
# - Stamped build: shows the short SHA (linked to ``commit_url``,
#   ``target="_blank" rel="noopener"`` for the new-tab safety pattern),
#   the branch, ``committed Xh ago`` from commit_time, ``deployed Yh ago``
#   from build_time, and ``worker up`` for the process uptime.
# - The ``build-stale`` modifier is added IFF ``build_time`` is older than
#   exactly 7 days (literal ``timedelta(days=N)`` — no production constant
#   rides).
# - ``_render_json`` includes a top-level ``"version"`` block carrying
#   ``commit_sha``, ``is_dev_build``, ``worker_started_at`` (parseable ISO),
#   and ``worker_uptime_seconds`` (non-negative int).
# - Bad timestamps (``commit_time = "not-a-date"``) must not crash the
#   render; the SHA still appears, the broken-age substring does not.


class TestRenderBuildInfo:
    """Direct unit tests for ``_render_build_info``."""

    def _patch_version(self, monkeypatch, version: dict, is_dev: bool) -> None:
        """Patch the build-info inputs as the renderer sees them.

        ``status_page`` did ``from src.version_info import VERSION, is_dev_build``
        so the bound names live in the ``status_page`` namespace — patch
        there, not in ``version_info``.
        """
        monkeypatch.setattr("src.status_page.VERSION", version)
        monkeypatch.setattr("src.status_page.is_dev_build", lambda: is_dev)

    def test_dev_mode_renders_build_info_without_stale_class(self, monkeypatch):
        # Force the dev-build code path. No version stamp present.
        self._patch_version(
            monkeypatch,
            version=dict(
                commit_sha="dev",
                commit_sha_full="dev",
                commit_time=None,
                build_time=None,
                branch="local",
                commit_url=None,
            ),
            is_dev=True,
        )
        from src.status_page import _render_build_info

        html = _render_build_info()

        assert 'class="build-info"' in html
        assert "Build:" in html
        assert "dev" in html
        # Dev build never gets the stale modifier.
        assert 'class="build-info build-stale"' not in html
        # Worker uptime is part of the dev-mode line too.
        assert "worker up" in html

    def test_deployed_version_shows_sha_branch_commit_and_deploy_ages(
        self, monkeypatch
    ):
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        commit_url = "https://github.com/foo/bar/commit/abc..."
        self._patch_version(
            monkeypatch,
            version=dict(
                commit_sha="abc1234",
                commit_sha_full="abc1234def567890abc1234def567890abc12345",
                # Literal timedelta — no production-constant ride.
                commit_time=(now - timedelta(hours=3)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                build_time=(now - timedelta(hours=2)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                branch="main",
                commit_url=commit_url,
            ),
            is_dev=False,
        )
        from src.status_page import _render_build_info

        html = _render_build_info()

        # Short SHA appears (rendered inside the anchor text).
        assert "abc1234" in html
        # Branch name appears.
        assert "main" in html
        # ``_format_age`` on a 3-hour delta produces ``"3h 0min"`` —
        # substring ``"committed 3h"`` is the stable contract surface.
        assert "committed 3h" in html
        assert "deployed 2h" in html
        # Worker uptime line is present.
        assert "worker up" in html
        # Commit URL is linkified, new-tab-safe, and the href appears intact.
        assert commit_url in html
        assert 'target="_blank"' in html
        # ``rel="noopener"`` is the security pattern that prevents the
        # opened tab from accessing window.opener — must always pair with
        # ``target="_blank"``.
        assert 'rel="noopener"' in html

    @pytest.mark.parametrize(
        "build_days_ago, expects_stale",
        [
            (6, False),  # Just under the 7-day cap → no stale class.
            (8, True),   # Past the 7-day cap → stale class applied.
        ],
    )
    def test_build_stale_class_applied_at_7_day_threshold(
        self, monkeypatch, build_days_ago, expects_stale
    ):
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        self._patch_version(
            monkeypatch,
            version=dict(
                commit_sha="abc1234",
                commit_sha_full="abc1234def567890abc1234def567890abc12345",
                commit_time=(now - timedelta(days=build_days_ago)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                build_time=(now - timedelta(days=build_days_ago)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                branch="main",
                commit_url="https://github.com/foo/bar/commit/abc",
            ),
            is_dev=False,
        )
        from src.status_page import _render_build_info

        html = _render_build_info()

        if expects_stale:
            assert 'class="build-info build-stale"' in html
        else:
            # Plain class — explicitly NOT the stale variant.
            assert 'class="build-info"' in html
            assert 'class="build-info build-stale"' not in html

    def test_render_build_info_does_not_crash_on_unparseable_timestamps(
        self, monkeypatch
    ):
        """Garbage timestamps must NOT crash the render. The SHA still
        appears; the broken-age substring does NOT appear.
        """
        self._patch_version(
            monkeypatch,
            version=dict(
                commit_sha="abc1234",
                commit_sha_full="abc1234def567890abc1234def567890abc12345",
                commit_time="not-a-date",
                build_time="also-not-a-date",
                branch="main",
                commit_url="https://github.com/foo/bar/commit/abc",
            ),
            is_dev=False,
        )
        from src.status_page import _render_build_info

        # No exception escapes.
        html = _render_build_info()

        # SHA still renders.
        assert "abc1234" in html
        # Broken age fragments are silently dropped — ``committed`` and
        # ``deployed`` substrings must not appear when their timestamps
        # failed to parse.
        assert "committed " not in html
        assert "deployed " not in html


class TestRenderJsonIncludesVersion:
    """``?format=json`` exposes a top-level ``"version"`` block."""

    @patch("src.status_page.SnapshotManager")
    @patch("src.status_page.get_state_manager")
    def test_json_output_includes_version_block(
        self, mock_get_state, mock_snap_mgr_cls, monkeypatch
    ):
        # State + snapshot wiring (mirrors TestRenderStatusPageJsonHistory).
        state_mgr = MagicMock()
        state_mgr.get_snapshot_table.return_value = MagicMock()
        mock_get_state.return_value = state_mgr

        snap_mgr = MagicMock()
        snap_mgr.get_all_floor_snapshots.return_value = [_floor_snapshot("upstairs")]
        snap_mgr.get_global_snapshot.return_value = _global_snapshot()
        snap_mgr.get_temperature_history.return_value = []
        mock_snap_mgr_cls.return_value = snap_mgr

        # Patch VERSION + is_dev_build at the status_page level (where
        # ``_render_json`` resolves them).
        monkeypatch.setattr(
            "src.status_page.VERSION",
            dict(
                commit_sha="abc1234",
                commit_sha_full="abc1234def567890abc1234def567890abc12345",
                commit_time="2026-05-21T18:30:00Z",
                build_time="2026-05-21T18:35:12Z",
                branch="main",
                commit_url="https://github.com/foo/bar/commit/abc1234",
            ),
        )
        monkeypatch.setattr("src.status_page.is_dev_build", lambda: False)

        resp = render_status_page(_make_request(params={"format": "json"}))

        assert resp.status_code == 200
        assert resp.mimetype == "application/json"
        payload = json.loads(resp.get_body().decode())

        assert "version" in payload
        v = payload["version"]
        assert v["commit_sha"] == "abc1234"
        assert v["is_dev_build"] is False
        # ``worker_uptime_seconds`` is a non-negative int (zero is allowed
        # if the test runs in the same wall-clock second as module import).
        assert isinstance(v["worker_uptime_seconds"], int)
        assert v["worker_uptime_seconds"] >= 0
        # ``worker_started_at`` is a parseable ISO-8601 datetime string.
        parsed = datetime.fromisoformat(v["worker_started_at"])
        assert parsed.tzinfo is not None
