"""Azure Table Storage state manager for WindowBot."""

import logging
import os
from datetime import datetime, timezone

from azure.data.tables import TableServiceClient, TableClient

logger = logging.getLogger("windowbot.state")

STATE_TABLE = "windowbotstate"
OAUTH_TABLE = "oauthtokens"


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
        """Create tables if they don't already exist."""
        for table_name in (STATE_TABLE, OAUTH_TABLE):
            try:
                self._service.create_table_if_not_exists(table_name)
            except Exception:
                logger.exception("Failed to ensure table '%s' exists.", table_name)

    def _state_table(self) -> TableClient:
        return self._service.get_table_client(STATE_TABLE)

    def _oauth_table(self) -> TableClient:
        return self._service.get_table_client(OAUTH_TABLE)

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
