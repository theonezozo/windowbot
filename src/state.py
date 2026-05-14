"""State management for WindowBot.

Primary backend: Azure Table Storage (production and local Azurite).
Fallback backend: Local JSON file (when Azurite is unavailable).
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from azure.data.tables import TableServiceClient, TableClient
from azure.core.exceptions import ServiceRequestError

logger = logging.getLogger("windowbot.state")

STATE_TABLE = "windowbotstate"
OAUTH_TABLE = "oauthtokens"
SNAPSHOT_TABLE = "windowbotsnapshot"

LOCAL_STATE_FILE = Path(".local_state.json")


# ======================================================================
# Local file-based fallback
# ======================================================================


class LocalStateManager:
    """File-backed state manager for local development without Azurite.

    Stores all state in a single JSON file. Thread-safe via a lock.
    Implements the same interface as StateManager.
    """

    def __init__(self, path: Path = LOCAL_STATE_FILE) -> None:
        self._path = path
        self._lock = threading.Lock()
        logger.info("Using LOCAL file-based state at '%s' (no Azure connection).", self._path)

    def _read(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {"floors": {}, "oauth": {}}

    def _write(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2, default=str))

    # Floor state -----------------------------------------------------------

    def get_floor_state(self, floor: str) -> dict:
        with self._lock:
            data = self._read()
        entity = data.get("floors", {}).get(floor)
        if entity:
            return entity
        logger.info("No local state for floor '%s' — returning defaults.", floor)
        return {
            "PartitionKey": floor,
            "RowKey": "current",
            "CurrentState": "UNKNOWN",
            "LastNotificationTime": None,
            "LastDecisionTime": None,
            "DecisionReason": "initial",
            "OpenedBeforeQuietHours": False,
            "QuietHoursActive": False,
            "LastQuietHoursStart": None,
            "LastQuietHoursEnd": None,
        }

    def update_floor_state(self, floor: str, state: dict) -> None:
        entity = {
            "PartitionKey": floor,
            "RowKey": "current",
            "LastDecisionTime": datetime.now(timezone.utc).isoformat(),
            **state,
        }
        with self._lock:
            data = self._read()
            data.setdefault("floors", {})[floor] = entity
            self._write(data)
        logger.info("Local state updated for floor '%s': %s", floor, state.get("CurrentState"))

    # OAuth tokens ----------------------------------------------------------

    def get_oauth_tokens(self) -> dict:
        with self._lock:
            data = self._read()
        tokens = data.get("oauth", {})
        if tokens:
            return tokens
        logger.info("No local OAuth tokens — returning empty defaults.")
        return {
            "PartitionKey": "ecobee",
            "RowKey": "token",
            "AccessToken": "",
            "RefreshToken": "",
            "ExpiresAt": None,
        }

    def update_oauth_tokens(self, tokens: dict) -> None:
        entity = {
            "PartitionKey": "ecobee",
            "RowKey": "token",
            **tokens,
        }
        with self._lock:
            data = self._read()
            data["oauth"] = entity
            self._write(data)
        logger.info("Local OAuth tokens updated.")

    def get_snapshot_table(self) -> None:
        """Snapshot table not available in local state mode."""
        raise NotImplementedError("Snapshot table requires Azure Table Storage.")


# ======================================================================
# Factory
# ======================================================================


def get_state_manager(connection_string: str | None = None) -> "StateManager | LocalStateManager":
    """Return the appropriate state manager for the current environment.

    Rules:
        1. No connection string and no env var → LocalStateManager.
        2. ``UseDevelopmentStorage=true`` but Azurite unreachable → LocalStateManager.
        3. Otherwise → StateManager (Azure Table Storage).
    """
    conn_str = connection_string or os.environ.get("AzureWebJobsStorage", "")

    if not conn_str:
        logger.warning("AzureWebJobsStorage is not set — using local file state.")
        return LocalStateManager()

    try:
        return StateManager(connection_string=conn_str)
    except Exception as exc:
        if conn_str.strip() == "UseDevelopmentStorage=true":
            logger.warning(
                "Azurite appears unavailable (%s) — falling back to local file state.", exc
            )
            return LocalStateManager()
        raise


# ======================================================================
# Azure Table Storage backend
# ======================================================================


class StateManager:
    """Manages persistent state in Azure Table Storage.

    Tables:
        windowbotstate — per-floor window open/close state
        oauthtokens    — Ecobee OAuth tokens
    """

    def __init__(self, connection_string: str | None = None) -> None:
        conn_str = connection_string or os.environ.get("AzureWebJobsStorage", "")
        if not conn_str:
            raise ValueError("AzureWebJobsStorage connection string is not configured.")

        self._service = TableServiceClient.from_connection_string(conn_str)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create tables if they don't already exist.

        Raises connection-level errors so the factory can detect Azurite
        being unavailable and fall back to local state.
        """
        for table_name in (STATE_TABLE, OAUTH_TABLE, SNAPSHOT_TABLE):
            try:
                self._service.create_table_if_not_exists(table_name)
            except (ServiceRequestError, ConnectionError, OSError):
                raise  # let connection failures propagate to factory
            except Exception:
                logger.exception("Failed to ensure table '%s' exists.", table_name)

    def _state_table(self) -> TableClient:
        return self._service.get_table_client(STATE_TABLE)

    def _oauth_table(self) -> TableClient:
        return self._service.get_table_client(OAUTH_TABLE)

    def _snapshot_table(self) -> TableClient:
        return self._service.get_table_client(SNAPSHOT_TABLE)

    # ------------------------------------------------------------------
    # Floor state
    # ------------------------------------------------------------------

    def get_floor_state(self, floor: str) -> dict:
        """Retrieve the current state for a floor.

        Args:
            floor: "upstairs" or "downstairs"

        Returns:
            Entity dict from Table Storage, or an empty default if none exists.
        """
        try:
            entity = self._state_table().get_entity(partition_key=floor, row_key="current")
            return dict(entity)
        except Exception:
            logger.info("No existing state for floor '%s' — returning defaults.", floor)
            return {
                "PartitionKey": floor,
                "RowKey": "current",
                "CurrentState": "UNKNOWN",
                "LastNotificationTime": None,
                "LastDecisionTime": None,
                "DecisionReason": "initial",
                "OpenedBeforeQuietHours": False,
                "QuietHoursActive": False,
                "LastQuietHoursStart": None,
                "LastQuietHoursEnd": None,
            }

    def update_floor_state(self, floor: str, state: dict) -> None:
        """Upsert the state for a floor.

        Args:
            floor: "upstairs" or "downstairs"
            state: Dict of fields to persist (will be merged with keys).
        """
        entity = {
            "PartitionKey": floor,
            "RowKey": "current",
            "LastDecisionTime": datetime.now(timezone.utc).isoformat(),
            **state,
        }
        try:
            self._state_table().upsert_entity(entity)
            logger.info("State updated for floor '%s': %s", floor, state.get("CurrentState"))
        except Exception:
            logger.exception("Failed to update state for floor '%s'.", floor)

    # ------------------------------------------------------------------
    # OAuth tokens
    # ------------------------------------------------------------------

    def get_oauth_tokens(self) -> dict:
        """Retrieve stored Ecobee OAuth tokens.

        Returns:
            Entity dict with AccessToken, RefreshToken, ExpiresAt — or empty
            defaults if no tokens are stored yet.
        """
        try:
            entity = self._oauth_table().get_entity(partition_key="ecobee", row_key="token")
            return dict(entity)
        except Exception:
            logger.info("No stored OAuth tokens — returning empty defaults.")
            return {
                "PartitionKey": "ecobee",
                "RowKey": "token",
                "AccessToken": "",
                "RefreshToken": "",
                "ExpiresAt": None,
            }

    def update_oauth_tokens(self, tokens: dict) -> None:
        """Upsert Ecobee OAuth tokens.

        Args:
            tokens: Dict with AccessToken, RefreshToken, and ExpiresAt.
        """
        entity = {
            "PartitionKey": "ecobee",
            "RowKey": "token",
            **tokens,
        }
        try:
            self._oauth_table().upsert_entity(entity)
            logger.info("OAuth tokens updated.")
        except Exception:
            logger.exception("Failed to update OAuth tokens.")

    def get_snapshot_table(self) -> TableClient:
        """Return the snapshot table client for diagnostic data.
        
        Returns:
            TableClient for the windowbotsnapshot table.
        """
        return self._snapshot_table()
