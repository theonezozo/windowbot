"""Tests for the ntfy.sh notification client.

Validates design decisions:
- Priority mapping: urgent=True → priority "urgent"; normal → "default".
- Topic construction from NTFY_TOPIC environment variable.
- HTTP POST parameters (URL, headers, body encoding).
- Tags: urgent messages get "warning,window"; normal get "window".
- Returns False when NTFY_TOPIC is not set.
- Returns False on network errors (does not raise).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import os

import pytest

from src.notifier import send_notification, NTFY_BASE_URL


# ------------------------------------------------------------------
# Priority Mapping
# ------------------------------------------------------------------


class TestPriorityMapping:
    """urgent=True forces priority to 'urgent'."""

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_urgent_true_sets_priority_urgent(self, mock_post):
        """urgent=True → priority header = 'urgent'."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", priority="default", urgent=True)

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers", {})
        assert headers["Priority"] == "urgent"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_urgent_false_keeps_given_priority(self, mock_post):
        """urgent=False → uses the provided priority value."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", priority="high", urgent=False)

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers", {})
        assert headers["Priority"] == "high"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_default_priority(self, mock_post):
        """No priority or urgent → default priority."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body")

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers", {})
        assert headers["Priority"] == "default"


# ------------------------------------------------------------------
# Tags
# ------------------------------------------------------------------


class TestTags:
    """Urgent notifications get 'warning,window'; normal get 'window'."""

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_urgent_tags(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", urgent=True)

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers", {})
        assert headers["Tags"] == "warning,window"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_normal_tags(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", urgent=False)

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers", {})
        assert headers["Tags"] == "window"


# ------------------------------------------------------------------
# Topic Construction & HTTP Call
# ------------------------------------------------------------------


class TestHTTPCall:
    """URL, body, and header construction."""

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "my_window_topic"})
    def test_url_uses_topic(self, mock_post):
        """POST URL = NTFY_BASE_URL / topic."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body")

        url = mock_post.call_args[0][0]
        assert url == f"{NTFY_BASE_URL}/my_window_topic"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_title_in_header(self, mock_post):
        """Title is sent as a header."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("My Title", "My Body")

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[1].get("headers", {})
        assert headers["Title"] == "My Title"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_body_utf8_encoded(self, mock_post):
        """Message body is UTF-8 encoded bytes."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("T", "Hello 🪟")

        data = mock_post.call_args.kwargs.get("data") or mock_post.call_args[1].get("data")
        assert data == "Hello 🪟".encode("utf-8")

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_returns_true_on_success(self, mock_post):
        """Successful send → True."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        assert send_notification("T", "B") is True


# ------------------------------------------------------------------
# Error Handling
# ------------------------------------------------------------------


class TestNotifierErrors:
    """Missing topic and network errors handled gracefully."""

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_topic_returns_false(self):
        """NTFY_TOPIC not set → returns False, no exception."""
        # Ensure NTFY_TOPIC is not present
        os.environ.pop("NTFY_TOPIC", None)
        result = send_notification("Title", "Body")
        assert result is False

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_network_error_returns_false(self, mock_post):
        """Network exception → returns False."""
        import requests as real_requests

        mock_post.side_effect = real_requests.ConnectionError("DNS fail")

        result = send_notification("Title", "Body")
        assert result is False

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_http_error_returns_false(self, mock_post):
        """HTTP 500 → returns False."""
        import requests as real_requests

        mock_resp = MagicMock(status_code=500)
        mock_resp.raise_for_status.side_effect = real_requests.HTTPError("500")
        mock_post.return_value = mock_resp

        result = send_notification("Title", "Body")
        assert result is False
