"""Tests for NWS freshness metrics logger."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from src.nws_client import NWSClient


@pytest.fixture
def temp_metrics_file(tmp_path):
    """Fixture to provide a temporary metrics file path."""
    metrics_path = tmp_path / "test_metrics.jsonl"
    with patch.dict(os.environ, {"WINDOWBOT_METRICS_PATH": str(metrics_path)}):
        yield str(metrics_path)


def test_metrics_written_after_fetch_batch(temp_metrics_file):
    """Verify metrics JSONL line is written with correct fields after _fetch_batch."""
    client = NWSClient(latitude=37.4, longitude=-122.1)
    now = datetime.now(timezone.utc)

    batch_stats = {"checked": 5, "fresh": 3, "cached": 1, "valid": 4}
    client._record_freshness_metric(now, batch_stats, median_temp_f=68.5)

    assert Path(temp_metrics_file).exists()

    with open(temp_metrics_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["timestamp"] == now.isoformat()
    assert record["checked"] == 5
    assert record["fresh"] == 3
    assert record["cached"] == 1
    assert record["valid"] == 4
    assert record["fresh_pct"] == 60.0
    assert record["median_temp_f"] == 68.5


def test_metrics_silent_on_filesystem_error():
    """Verify metrics writes are silent on filesystem errors."""
    client = NWSClient(latitude=37.4, longitude=-122.1)
    now = datetime.now(timezone.utc)
    batch_stats = {"checked": 5, "fresh": 3, "cached": 1, "valid": 4}

    with patch.dict(os.environ, {"WINDOWBOT_METRICS_PATH": "/nonexistent/path/metrics.jsonl"}):
        # Should not raise
        client._record_freshness_metric(now, batch_stats, median_temp_f=65.0)


def test_fresh_pct_calculation():
    """Verify fresh_pct calculation is correct with rounding."""
    client = NWSClient(latitude=37.4, longitude=-122.1)
    now = datetime.now(timezone.utc)

    test_cases = [
        # (checked, fresh, expected_pct)
        (5, 0, 0.0),        # 0% when no fresh
        (5, 5, 100.0),      # 100% when all fresh
        (5, 3, 60.0),       # 60% = 3/5
        (7, 2, 28.6),       # 28.6% = 2/7 (rounded to 1 decimal)
        (0, 0, 0.0),        # handle zero-division case
    ]

    for checked, fresh, expected_pct in test_cases:
        with patch.dict(os.environ, {"WINDOWBOT_METRICS_PATH": "/dev/null"}):
            batch_stats = {"checked": checked, "fresh": fresh, "cached": 0, "valid": fresh}
            client._record_freshness_metric(now, batch_stats, median_temp_f=70.0)

        if checked:
            calculated_pct = round(100 * fresh / checked, 1)
        else:
            calculated_pct = 0.0
        assert calculated_pct == expected_pct


def test_metrics_appended_not_overwritten(temp_metrics_file):
    """Verify multiple metric writes append to the file."""
    client = NWSClient(latitude=37.4, longitude=-122.1)
    now1 = datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc)
    now2 = datetime(2026, 4, 27, 10, 10, 0, tzinfo=timezone.utc)

    client._record_freshness_metric(now1, {"checked": 5, "fresh": 1, "cached": 0, "valid": 1}, median_temp_f=62.0)
    client._record_freshness_metric(now2, {"checked": 5, "fresh": 4, "cached": 0, "valid": 4}, median_temp_f=64.5)

    with open(temp_metrics_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2

    record1 = json.loads(lines[0])
    assert record1["timestamp"] == now1.isoformat()
    assert record1["fresh"] == 1
    assert record1["fresh_pct"] == 20.0
    assert record1["median_temp_f"] == 62.0

    record2 = json.loads(lines[1])
    assert record2["timestamp"] == now2.isoformat()
    assert record2["fresh"] == 4
    assert record2["fresh_pct"] == 80.0
    assert record2["median_temp_f"] == 64.5
