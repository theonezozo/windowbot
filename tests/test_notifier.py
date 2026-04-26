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

from src.notifier import send_notification, NTFY_BASE_URL, _priority_to_int


# ------------------------------------------------------------------
# Priority Mapping
# ------------------------------------------------------------------


class TestPriorityMapping:
    """urgent=True forces priority to 'urgent'."""

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_urgent_true_sets_priority_urgent(self, mock_post):
        """urgent=True → priority = 5 (urgent)."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", priority="default", urgent=True)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["priority"] == 5

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_urgent_false_keeps_given_priority(self, mock_post):
        """urgent=False → uses the provided priority value."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", priority="high", urgent=False)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["priority"] == 4

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_default_priority(self, mock_post):
        """No priority or urgent → default priority (3)."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["priority"] == 3


# ------------------------------------------------------------------
# Tags
# ------------------------------------------------------------------


class TestTags:
    """Urgent notifications get ['warning', 'window']; normal get ['window']."""

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_urgent_tags(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", urgent=True)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["tags"] == ["warning", "window"]

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_normal_tags(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body", urgent=False)

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["tags"] == ["window"]


# ------------------------------------------------------------------
# Topic Construction & HTTP Call
# ------------------------------------------------------------------


class TestHTTPCall:
    """URL, body, and payload construction."""

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "my_window_topic"})
    def test_url_is_base(self, mock_post):
        """POST URL = NTFY_BASE_URL (topic in JSON body)."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body")

        url = mock_post.call_args[0][0]
        assert url == NTFY_BASE_URL

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "my_window_topic"})
    def test_topic_in_payload(self, mock_post):
        """Topic is sent in the JSON payload."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("Title", "Body")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["topic"] == "my_window_topic"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_title_in_payload(self, mock_post):
        """Title is sent in the JSON payload."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("My Title", "My Body")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["title"] == "My Title"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_message_in_payload(self, mock_post):
        """Message body with unicode is sent in JSON payload."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        send_notification("T", "Hello 🪟")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["message"] == "Hello 🪟"

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_returns_true_on_success(self, mock_post):
        """Successful send → True."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        assert send_notification("T", "B") is True

    @patch("src.notifier.requests.post")
    @patch.dict(os.environ, {"NTFY_TOPIC": "test_topic"})
    def test_unicode_title_works(self, mock_post):
        """Emoji in title works via JSON API (no latin-1 header encoding)."""
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        result = send_notification("🪟 WindowBot", "Open windows!")

        assert result is True
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json", {})
        assert payload["title"] == "🪟 WindowBot"


# ------------------------------------------------------------------
# Priority Integer Conversion
# ------------------------------------------------------------------


class TestPriorityConversion:
    """_priority_to_int maps string priorities to ntfy integers."""

    @pytest.mark.parametrize("name,expected", [
        ("min", 1), ("low", 2), ("default", 3), ("high", 4), ("urgent", 5),
    ])
    def test_known_priorities(self, name, expected):
        assert _priority_to_int(name) == expected

    def test_unknown_defaults_to_3(self):
        assert _priority_to_int("bogus") == 3


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
