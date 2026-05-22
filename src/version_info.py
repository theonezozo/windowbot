"""Runtime version + startup info for WindowBot.

The deployed bundle includes a build-time-generated ``_version.py`` containing
the git SHA and commit timestamp. This module loads that file when present and
falls back to "dev" placeholders for local development.

Worker startup time is captured at MODULE IMPORT TIME and reflects the current
Python worker process — on Azure Functions Consumption plan, workers cycle
frequently, so this is the cold-start timestamp of the currently-serving worker.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("windowbot.version")

# Captured exactly once at module import (= worker cold start).
WORKER_STARTED_AT: datetime = datetime.now(timezone.utc)

# Defaults for when _version.py is absent (local dev, fresh clone).
_DEFAULTS: dict[str, str | None] = {
    "commit_sha": "dev",
    "commit_sha_full": "dev",
    "commit_time": None,
    "build_time": None,
    "branch": "local",
    "commit_url": None,
}


def _load_version() -> dict[str, str | None]:
    try:
        from src import _version  # type: ignore[import-not-found]
    except ImportError:
        return dict(_DEFAULTS)
    except Exception:
        # A malformed _version.py must never crash the status page.
        logger.warning("Failed to import src/_version.py; using dev defaults.", exc_info=True)
        return dict(_DEFAULTS)
    return {
        "commit_sha": getattr(_version, "__commit_sha__", _DEFAULTS["commit_sha"]),
        "commit_sha_full": getattr(_version, "__commit_sha_full__", _DEFAULTS["commit_sha_full"]),
        "commit_time": getattr(_version, "__commit_time__", _DEFAULTS["commit_time"]),
        "build_time": getattr(_version, "__build_time__", _DEFAULTS["build_time"]),
        "branch": getattr(_version, "__branch__", _DEFAULTS["branch"]),
        "commit_url": getattr(_version, "__commit_url__", _DEFAULTS["commit_url"]),
    }


# Loaded once at module import.
VERSION: dict[str, str | None] = _load_version()


def is_dev_build() -> bool:
    """True when no deploy-stamped version is present (local dev)."""
    return VERSION["commit_sha"] == "dev"


def parse_iso_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC string (tolerates trailing 'Z'). Returns None on failure."""
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None
