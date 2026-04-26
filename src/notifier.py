"""ntfy.sh notification client for WindowBot."""

import logging
import os

import requests

logger = logging.getLogger("windowbot.notifier")

NTFY_BASE_URL = "https://ntfy.sh"


def send_notification(
    title: str,
    message: str,
    priority: str = "default",
    urgent: bool = False,
) -> bool:
    """Send a push notification via ntfy.sh.

    Args:
        title: Notification title (e.g. "WindowBot — Upstairs").
        message: Notification body text.
        priority: ntfy priority level — "min", "low", "default", "high", "urgent".
        urgent: When True the caller should ALWAYS send this notification,
                bypassing any cooldown timer.  This flag is informational to
                the caller; the notifier itself always attempts delivery.

    Returns:
        True if the notification was sent successfully, False otherwise.
    """
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        logger.error("NTFY_TOPIC not set — cannot send notification.")
        return False

    if urgent:
        priority = "urgent"

    url = f"{NTFY_BASE_URL}"
    payload = {
        "topic": topic,
        "title": title,
        "message": message,
        "priority": _priority_to_int(priority),
        "tags": ["warning", "window"] if urgent else ["window"],
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Notification sent: %s", title)
        return True
    except requests.RequestException:
        logger.exception("Failed to send notification to ntfy.sh.")
        return False


def _priority_to_int(priority: str) -> int:
    """Convert string priority to ntfy integer value."""
    return {
        "min": 1,
        "low": 2,
        "default": 3,
        "high": 4,
        "urgent": 5,
    }.get(priority, 3)
