"""Tests for diagnostic snapshot persistence.

Validates design decisions for the temperature history feature
(see ``.squad/decisions.md`` 2026-05-19):

- ``TemperatureHistoryEntry`` is JSON round-trippable, including
  None values and empty dicts.
- ``SnapshotManager.record_temperature_history`` writes to the
  ``"history"`` partition with the entry timestamp as RowKey, and
  is best-effort (swallows exceptions).
- ``SnapshotManager.get_temperature_history`` issues a
  ``PartitionKey eq 'history' and RowKey ge '<cutoff>'`` range query,
  returns entries newest-first, skips unparseable rows, and returns
  ``[]`` when the underlying query raises.
- ``SnapshotManager.get_all_floor_snapshots`` excludes history-partition
  rows so they can never be misparsed as ``FloorSnapshot``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.diagnostic import (
    HISTORY_PARTITION,
    SnapshotManager,
    TemperatureHistoryEntry,
)


# ------------------------------------------------------------------
# TemperatureHistoryEntry serialization
# ------------------------------------------------------------------


class TestTemperatureHistoryEntrySerialization:
    """Round-trip ``to_json`` / ``from_json`` preserves all fields."""

    def test_round_trip_full_entry_returns_equivalent(self):
        entry = TemperatureHistoryEntry(
            timestamp="2026-05-19T14:30:00+00:00",
            outdoor_temp_f=72.5,
            indoor_temps={"upstairs": 71.0, "downstairs": 70.5},
        )

        restored = TemperatureHistoryEntry.from_json(entry.to_json())

        assert restored == entry

    def test_round_trip_none_outdoor_temp_preserves_none(self):
        entry = TemperatureHistoryEntry(
            timestamp="2026-05-19T14:30:00+00:00",
            outdoor_temp_f=None,
            indoor_temps={"upstairs": 71.0},
        )

        restored = TemperatureHistoryEntry.from_json(entry.to_json())

        assert restored.outdoor_temp_f is None
        assert restored.indoor_temps == {"upstairs": 71.0}

    def test_round_trip_none_indoor_temps_preserves_none(self):
        entry = TemperatureHistoryEntry(
            timestamp="2026-05-19T14:30:00+00:00",
            outdoor_temp_f=68.0,
            indoor_temps={"upstairs": None, "downstairs": 70.0},
        )

        restored = TemperatureHistoryEntry.from_json(entry.to_json())

        assert restored.indoor_temps == {"upstairs": None, "downstairs": 70.0}

    def test_round_trip_empty_indoor_temps_returns_empty_dict(self):
        entry = TemperatureHistoryEntry(
            timestamp="2026-05-19T14:30:00+00:00",
            outdoor_temp_f=68.0,
            indoor_temps={},
        )

        restored = TemperatureHistoryEntry.from_json(entry.to_json())

        assert restored.indoor_temps == {}
        assert restored == entry


# ------------------------------------------------------------------
# SnapshotManager.record_temperature_history
# ------------------------------------------------------------------


class TestRecordTemperatureHistory:
    """Writes to history partition; never raises."""

    def test_writes_entity_with_history_partition_and_timestamp_rowkey(self):
        table = MagicMock()
        mgr = SnapshotManager(table)
        entry = TemperatureHistoryEntry(
            timestamp="2026-05-19T14:30:00+00:00",
            outdoor_temp_f=72.5,
            indoor_temps={"upstairs": 71.0},
        )

        mgr.record_temperature_history(entry)

        table.upsert_entity.assert_called_once()
        written = table.upsert_entity.call_args[0][0]
        assert written["PartitionKey"] == HISTORY_PARTITION
        assert written["PartitionKey"] == "history"
        assert written["RowKey"] == entry.timestamp
        # Payload is the entry serialized back to a parseable dict.
        payload = json.loads(written["Data"])
        assert payload["outdoor_temp_f"] == 72.5
        assert payload["indoor_temps"] == {"upstairs": 71.0}

    def test_upsert_failure_does_not_propagate(self):
        table = MagicMock()
        table.upsert_entity.side_effect = RuntimeError("table unavailable")
        mgr = SnapshotManager(table)
        entry = TemperatureHistoryEntry(
            timestamp="2026-05-19T14:30:00+00:00",
            outdoor_temp_f=72.5,
            indoor_temps={"upstairs": 71.0},
        )

        # Must not raise — history writes are best-effort.
        mgr.record_temperature_history(entry)

        table.upsert_entity.assert_called_once()


# ------------------------------------------------------------------
# SnapshotManager.get_temperature_history
# ------------------------------------------------------------------


def _history_entity(ts: str, outdoor: float | None, indoor: dict) -> dict:
    """Build the dict shape that ``query_entities`` yields for a history row."""
    entry = TemperatureHistoryEntry(
        timestamp=ts, outdoor_temp_f=outdoor, indoor_temps=indoor
    )
    return {
        "PartitionKey": HISTORY_PARTITION,
        "RowKey": ts,
        "Data": entry.to_json(),
    }


class TestGetTemperatureHistory:
    """Read returns newest-first, uses a range filter, tolerates bad rows."""

    def test_returns_entries_sorted_newest_first(self):
        # Intentionally supplied out of order to confirm sort happens.
        rows = [
            _history_entity("2026-05-19T12:00:00+00:00", 65.0, {"upstairs": 70.0}),
            _history_entity("2026-05-19T14:00:00+00:00", 70.0, {"upstairs": 71.0}),
            _history_entity("2026-05-19T13:00:00+00:00", 68.0, {"upstairs": 70.5}),
        ]
        table = MagicMock()
        table.query_entities.return_value = iter(rows)
        mgr = SnapshotManager(table)

        result = mgr.get_temperature_history(hours=12)

        timestamps = [e.timestamp for e in result]
        assert timestamps == [
            "2026-05-19T14:00:00+00:00",
            "2026-05-19T13:00:00+00:00",
            "2026-05-19T12:00:00+00:00",
        ]

    @pytest.mark.parametrize("hours", [1, 12, 24])
    def test_query_filter_uses_history_partition_and_rowkey_cutoff(self, hours):
        table = MagicMock()
        table.query_entities.return_value = iter([])
        mgr = SnapshotManager(table)
        fake_now = datetime(2026, 5, 19, 14, 30, 0, tzinfo=timezone.utc)

        with patch("src.diagnostic.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mgr.get_temperature_history(hours=hours)

        table.query_entities.assert_called_once()
        query = table.query_entities.call_args[0][0]
        # PartitionKey clause is exact and uses the constant.
        assert f"PartitionKey eq '{HISTORY_PARTITION}'" in query
        # RowKey cutoff is now - hours, expressed in ISO 8601.
        cutoff_iso = (fake_now - __import__("datetime").timedelta(hours=hours)).isoformat()
        assert f"RowKey ge '{cutoff_iso}'" in query

    def test_skips_unparseable_rows_without_failing(self):
        rows = [
            _history_entity("2026-05-19T14:00:00+00:00", 70.0, {"upstairs": 71.0}),
            {
                "PartitionKey": HISTORY_PARTITION,
                "RowKey": "2026-05-19T13:30:00+00:00",
                "Data": "{not valid json",
            },
            _history_entity("2026-05-19T13:00:00+00:00", 68.0, {"upstairs": 70.5}),
        ]
        table = MagicMock()
        table.query_entities.return_value = iter(rows)
        mgr = SnapshotManager(table)

        result = mgr.get_temperature_history(hours=12)

        # The two valid rows come through; the broken one is silently skipped.
        assert len(result) == 2
        assert {e.timestamp for e in result} == {
            "2026-05-19T14:00:00+00:00",
            "2026-05-19T13:00:00+00:00",
        }

    def test_returns_empty_list_when_query_raises(self):
        table = MagicMock()
        table.query_entities.side_effect = RuntimeError("table down")
        mgr = SnapshotManager(table)

        result = mgr.get_temperature_history(hours=12)

        assert result == []


# ------------------------------------------------------------------
# get_all_floor_snapshots: history partition is excluded
# ------------------------------------------------------------------


class TestGetAllFloorSnapshotsExcludesHistory:
    """History rows must not leak into floor-snapshot results."""

    def test_query_filter_excludes_history_partition(self):
        table = MagicMock()
        table.query_entities.return_value = iter([])
        mgr = SnapshotManager(table)

        mgr.get_all_floor_snapshots()

        table.query_entities.assert_called_once()
        query = table.query_entities.call_args[0][0]
        assert "PartitionKey ne 'global'" in query
        assert "PartitionKey ne 'history'" in query
        assert "RowKey eq 'snapshot'" in query
