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
