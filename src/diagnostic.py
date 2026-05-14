"""Diagnostic snapshot for status page.

Captures rich state from each poll cycle so the status endpoint can
show exactly what WindowBot was thinking without re-polling APIs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

from azure.data.tables import TableClient

logger = logging.getLogger("windowbot.diagnostic")

SNAPSHOT_TABLE = "windowbotsnapshot"


@dataclass
class SensorReading:
    """Indoor sensor reading snapshot."""
    name: str
    temperature_f: float | None
    is_online: bool
    is_coolest: bool = False


@dataclass
class OutdoorStation:
    """Outdoor weather station reading."""
    station_id: str
    distance_mi: float | None
    temperature_f: float
    age_minutes: int | None


@dataclass
class AQIStation:
    """AQI sensor reading."""
    sensor_id: str
    distance_mi: float | None
    aqi: int
    pm25: float | None


@dataclass
class GateEvaluation:
    """Gate evaluation status."""
    name: str
    passed: bool
    threshold: str | None = None
    actual: str | None = None


@dataclass
class FloorSnapshot:
    """Diagnostic snapshot for one floor."""
    floor: str
    decision: str  # OPEN/CLOSED
    reason: str
    indoor_sensors: list[SensorReading]
    outdoor_temp_f: float
    outdoor_source: str
    outdoor_stations: list[OutdoorStation]
    outdoor_humidity: float | None
    aqi_value: int
    aqi_source: str
    aqi_stations: list[AQIStation]
    gates: list[GateEvaluation]
    last_notification_type: str | None
    last_notification_time: str | None
    timestamp: str

    def to_json(self) -> str:
        """Serialize to JSON for Table Storage."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, data: str) -> FloorSnapshot:
        """Deserialize from JSON."""
        obj = json.loads(data)
        # Reconstruct dataclasses from dicts
        obj["indoor_sensors"] = [SensorReading(**s) for s in obj["indoor_sensors"]]
        obj["outdoor_stations"] = [OutdoorStation(**s) for s in obj["outdoor_stations"]]
        obj["aqi_stations"] = [AQIStation(**s) for s in obj["aqi_stations"]]
        obj["gates"] = [GateEvaluation(**g) for g in obj["gates"]]
        return cls(**obj)


@dataclass
class GlobalSnapshot:
    """Global cycle-level snapshot."""
    poll_start: str
    poll_duration_seconds: float
    hvac_mode: str
    quiet_hours_active: bool
    quiet_hours_next_transition: str | None
    next_poll_eta: str
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON for Table Storage."""
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, data: str) -> GlobalSnapshot:
        """Deserialize from JSON."""
        return cls(**json.loads(data))


class SnapshotManager:
    """Manages diagnostic snapshots in Azure Table Storage."""

    def __init__(self, table_client: TableClient) -> None:
        self._table = table_client

    def save_floor_snapshot(self, snapshot: FloorSnapshot) -> None:
        """Persist a floor snapshot."""
        entity = {
            "PartitionKey": snapshot.floor,
            "RowKey": "snapshot",
            "Data": snapshot.to_json(),
            "Timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._table.upsert_entity(entity)
            logger.info("Saved diagnostic snapshot for floor '%s'.", snapshot.floor)
        except Exception:
            logger.exception("Failed to save snapshot for floor '%s'.", snapshot.floor)

    def save_global_snapshot(self, snapshot: GlobalSnapshot) -> None:
        """Persist global cycle snapshot."""
        entity = {
            "PartitionKey": "global",
            "RowKey": "snapshot",
            "Data": snapshot.to_json(),
            "Timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._table.upsert_entity(entity)
            logger.info("Saved global diagnostic snapshot.")
        except Exception:
            logger.exception("Failed to save global snapshot.")

    def get_floor_snapshot(self, floor: str) -> FloorSnapshot | None:
        """Retrieve the latest floor snapshot."""
        try:
            entity = self._table.get_entity(partition_key=floor, row_key="snapshot")
            return FloorSnapshot.from_json(entity["Data"])
        except Exception:
            logger.warning("No snapshot found for floor '%s'.", floor)
            return None

    def get_global_snapshot(self) -> GlobalSnapshot | None:
        """Retrieve the latest global snapshot."""
        try:
            entity = self._table.get_entity(partition_key="global", row_key="snapshot")
            return GlobalSnapshot.from_json(entity["Data"])
        except Exception:
            logger.warning("No global snapshot found.")
            return None

    def get_all_floor_snapshots(self) -> list[FloorSnapshot]:
        """Retrieve all floor snapshots."""
        snapshots = []
        try:
            query = "PartitionKey ne 'global' and RowKey eq 'snapshot'"
            for entity in self._table.query_entities(query):
                try:
                    snapshots.append(FloorSnapshot.from_json(entity["Data"]))
                except Exception:
                    logger.exception("Failed to parse snapshot for partition '%s'.", entity.get("PartitionKey"))
        except Exception:
            logger.exception("Failed to query floor snapshots.")
        return snapshots
