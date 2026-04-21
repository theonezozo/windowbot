"""Tests for the Azure Table Storage state manager.

Validates design decisions:
- Floor state reads/writes with partition key = floor name.
- Default state returned when no entity exists.
- OAuth token persistence and retrieval.
- Missing connection string raises ValueError.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.state import StateManager, STATE_TABLE, OAUTH_TABLE


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_table_service():
    """Mock the Azure TableServiceClient."""
    with patch("src.state.TableServiceClient") as mock_cls:
        mock_service = MagicMock()
        mock_cls.from_connection_string.return_value = mock_service
        yield mock_service


@pytest.fixture
def state_mgr(mock_table_service):
    """A StateManager backed by mocked Azure Table Storage."""
    return StateManager(connection_string="DefaultEndpointsProtocol=https;AccountName=test")


# ------------------------------------------------------------------
# Floor State
# ------------------------------------------------------------------


class TestFloorState:
    """Per-floor state reads and writes."""

    def test_get_floor_state_returns_entity(self, state_mgr, mock_table_service):
        """Existing entity → returned as dict."""
        table_client = MagicMock()
        table_client.get_entity.return_value = {
            "PartitionKey": "upstairs",
            "RowKey": "current",
            "CurrentState": "OPEN",
            "LastNotificationTime": "2024-01-01T00:00:00Z",
        }
        mock_table_service.get_table_client.return_value = table_client

        result = state_mgr.get_floor_state("upstairs")
        assert result["CurrentState"] == "OPEN"
        table_client.get_entity.assert_called_once_with(
            partition_key="upstairs", row_key="current"
        )

    def test_get_floor_state_default_on_missing(self, state_mgr, mock_table_service):
        """No entity → returns default with UNKNOWN state."""
        table_client = MagicMock()
        table_client.get_entity.side_effect = Exception("Not found")
        mock_table_service.get_table_client.return_value = table_client

        result = state_mgr.get_floor_state("downstairs")
        assert result["CurrentState"] == "UNKNOWN"
        assert result["LastNotificationTime"] is None

    def test_update_floor_state_upserts(self, state_mgr, mock_table_service):
        """update_floor_state upserts entity with partition key = floor."""
        table_client = MagicMock()
        mock_table_service.get_table_client.return_value = table_client

        state_mgr.update_floor_state("upstairs", {"CurrentState": "CLOSED"})

        table_client.upsert_entity.assert_called_once()
        entity = table_client.upsert_entity.call_args[0][0]
        assert entity["PartitionKey"] == "upstairs"
        assert entity["RowKey"] == "current"
        assert entity["CurrentState"] == "CLOSED"


# ------------------------------------------------------------------
# OAuth Tokens
# ------------------------------------------------------------------


class TestOAuthTokens:
    """OAuth token persistence."""

    def test_get_oauth_tokens_returns_entity(self, state_mgr, mock_table_service):
        """Stored tokens → returned as dict."""
        table_client = MagicMock()
        table_client.get_entity.return_value = {
            "PartitionKey": "ecobee",
            "RowKey": "token",
            "AccessToken": "access_123",
            "RefreshToken": "refresh_456",
        }
        mock_table_service.get_table_client.return_value = table_client

        result = state_mgr.get_oauth_tokens()
        assert result["AccessToken"] == "access_123"
        assert result["RefreshToken"] == "refresh_456"

    def test_get_oauth_tokens_default_on_missing(self, state_mgr, mock_table_service):
        """No stored tokens → empty defaults."""
        table_client = MagicMock()
        table_client.get_entity.side_effect = Exception("Not found")
        mock_table_service.get_table_client.return_value = table_client

        result = state_mgr.get_oauth_tokens()
        assert result["AccessToken"] == ""
        assert result["RefreshToken"] == ""

    def test_update_oauth_tokens_upserts(self, state_mgr, mock_table_service):
        """update_oauth_tokens upserts with partition key = ecobee."""
        table_client = MagicMock()
        mock_table_service.get_table_client.return_value = table_client

        state_mgr.update_oauth_tokens({
            "AccessToken": "new_access",
            "RefreshToken": "new_refresh",
            "ExpiresAt": "2024-01-01T00:00:00Z",
        })

        table_client.upsert_entity.assert_called_once()
        entity = table_client.upsert_entity.call_args[0][0]
        assert entity["PartitionKey"] == "ecobee"
        assert entity["RowKey"] == "token"
        assert entity["AccessToken"] == "new_access"


# ------------------------------------------------------------------
# Connection String Validation
# ------------------------------------------------------------------


class TestConnectionString:
    """Missing connection string raises ValueError."""

    @patch("src.state.os.environ", {})
    def test_no_connection_string_raises(self):
        """No connection string and no env var → ValueError."""
        with pytest.raises(ValueError, match="not configured"):
            StateManager()

    def test_explicit_connection_string_used(self, mock_table_service):
        """Explicitly passed connection string is used."""
        StateManager(connection_string="UseDevelopmentStorage=true")
        from src.state import TableServiceClient
        TableServiceClient.from_connection_string.assert_called_with(
            "UseDevelopmentStorage=true"
        )


# ------------------------------------------------------------------
# Table Initialization
# ------------------------------------------------------------------


class TestTableInit:
    """Tables are created if they don't exist."""

    def test_ensures_both_tables(self, mock_table_service):
        """Constructor creates both windowbotstate and oauthtokens tables."""
        StateManager(connection_string="test_conn")

        create_calls = mock_table_service.create_table_if_not_exists.call_args_list
        table_names = [c[0][0] for c in create_calls]
        assert STATE_TABLE in table_names
        assert OAUTH_TABLE in table_names
