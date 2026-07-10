"""Tests for ``src/version_info.py`` — defensive loader for the deploy-stamped
build manifest (``src/_version.py``).

Contract under test (see ``.squad/decisions/inbox/gregory-version-info-runtime.md``):

- ``_load_version()`` returns dev DEFAULTS when ``src/_version.py`` is absent
  (the normal local-dev state — file is gitignored, generated only at deploy).
- ``_load_version()`` reads each of the six contract fields when present.
- ``_load_version()`` tolerates a partial ``_version.py`` (missing fields
  fall back to per-field defaults via ``getattr(..., default)``).
- ``_load_version()`` tolerates a broken ``_version.py`` (import-time
  exception is caught, defaults returned — status page must never 500).
- ``WORKER_STARTED_AT`` is captured once at module import (tz-aware UTC).
- ``parse_iso_utc()`` accepts ``Z`` and offset suffixes; returns ``None`` for
  garbage WITHOUT raising.

All age / timestamp literals are pinned — no production-constant rides.
"""

from __future__ import annotations

import builtins
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src
from src.version_info import (
    WORKER_STARTED_AT,
    _DEFAULTS,
    _load_version,
    is_dev_build,
    parse_iso_utc,
)


# ------------------------------------------------------------------
# Isolation fixture — hide any on-disk deploy-stamped version file
# ------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _hide_stamped_version_file():
    """Guarantee these tests exercise the documented *dev-defaults* contract.

    ``src/_version.py`` is a gitignored, deploy-only build artifact produced by
    ``scripts/stamp_version.sh`` (and re-created by ``squad upgrade``). The whole
    module contracts that this file is ABSENT during local dev: the "file
    missing" tests expect ``from src import _version`` to raise ``ImportError``,
    and the fake-injection tests ``monkeypatch.setitem(sys.modules, ...)`` a
    stand-in module.

    A stray stamped file on disk breaks both assumptions:
      1. ``import src`` binds a real ``_version`` submodule *attribute* on the
         ``src`` package, so ``from src import _version`` returns the real
         stamped module and IGNORES the test's ``sys.modules`` fake.
      2. The on-disk file makes ``from src import _version`` succeed, so the
         "file missing" tests never see the expected ``ImportError``.

    So for the duration of this module we move any real ``src/_version.py``
    aside (never delete — the user's build artifact must survive verbatim) and
    scrub the cached module + bound package attribute. try/finally restores the
    file no matter what, even on test failure.
    """
    version_path = Path(__file__).resolve().parents[1] / "src" / "_version.py"
    backup_path = version_path.with_name("_version.py.pytest-bak")

    def _scrub_cached_module():
        # Drop a stale cached import and the attribute bound on the package
        # object so ``from src import _version`` can't resolve a real module.
        sys.modules.pop("src._version", None)
        if hasattr(src, "_version"):
            delattr(src, "_version")

    moved = False
    if version_path.exists():
        version_path.rename(backup_path)
        moved = True

    _scrub_cached_module()

    try:
        yield
    finally:
        # Restore the deploy artifact verbatim, then leave a clean cache state.
        if moved:
            backup_path.rename(version_path)
        _scrub_cached_module()


# ------------------------------------------------------------------
# _load_version — file absent (default local-dev state)
# ------------------------------------------------------------------


class TestLoadVersionFileMissing:
    """``src/_version.py`` does not exist in the source tree (gitignored,
    deploy-only). Loader must return the dev defaults without raising.
    """

    def test_returns_defaults_when_version_file_missing(self, monkeypatch):
        # Force a clean slate — drop any cached _version that might linger
        # from an earlier test that injected a fake module.
        monkeypatch.delitem(sys.modules, "src._version", raising=False)

        result = _load_version()

        # Each field matches the documented dev default exactly.
        assert result["commit_sha"] == "dev"
        assert result["commit_sha_full"] == "dev"
        assert result["commit_time"] is None
        assert result["build_time"] is None
        assert result["branch"] == "local"
        assert result["commit_url"] is None

    def test_is_dev_build_true_under_defaults(self, monkeypatch):
        """``is_dev_build()`` reads the cached module-level ``VERSION``.
        Under the default (no ``_version.py``) state it must report True.
        """
        # Force module-level VERSION back to a freshly-loaded default.
        monkeypatch.delitem(sys.modules, "src._version", raising=False)
        monkeypatch.setattr("src.version_info.VERSION", _load_version())

        assert is_dev_build() is True


# ------------------------------------------------------------------
# _load_version — fully-stamped _version.py present
# ------------------------------------------------------------------


class TestLoadVersionFilePresent:
    """When all six contract fields are present, each is read verbatim."""

    def test_reads_each_field_when_version_file_complete(self, monkeypatch):
        fake = types.ModuleType("src._version")
        fake.__commit_sha__ = "abc1234"
        fake.__commit_sha_full__ = "abc1234def567890abc1234def567890abc12345"
        fake.__commit_time__ = "2026-05-21T18:30:00Z"
        fake.__build_time__ = "2026-05-21T18:35:12Z"
        fake.__branch__ = "main"
        fake.__commit_url__ = "https://github.com/theonezozo/windowbot/commit/abc1234"
        monkeypatch.setitem(sys.modules, "src._version", fake)

        result = _load_version()

        assert result["commit_sha"] == "abc1234"
        assert result["commit_sha_full"] == "abc1234def567890abc1234def567890abc12345"
        assert result["commit_time"] == "2026-05-21T18:30:00Z"
        assert result["build_time"] == "2026-05-21T18:35:12Z"
        assert result["branch"] == "main"
        assert (
            result["commit_url"]
            == "https://github.com/theonezozo/windowbot/commit/abc1234"
        )
        # Sanity: a result with a real SHA would NOT register as dev build.
        assert result["commit_sha"] != _DEFAULTS["commit_sha"]


# ------------------------------------------------------------------
# _load_version — partial _version.py (only some fields set)
# ------------------------------------------------------------------


class TestLoadVersionPartialFile:
    """Missing fields fall back to per-field defaults via ``getattr``."""

    def test_missing_fields_fall_back_to_defaults(self, monkeypatch):
        fake = types.ModuleType("src._version")
        # Only the short SHA is set — everything else falls back.
        fake.__commit_sha__ = "deadbee"
        monkeypatch.setitem(sys.modules, "src._version", fake)

        result = _load_version()

        # The one set field flows through.
        assert result["commit_sha"] == "deadbee"
        # Every other field uses the dev default.
        assert result["commit_sha_full"] == "dev"
        assert result["commit_time"] is None
        assert result["build_time"] is None
        assert result["branch"] == "local"
        assert result["commit_url"] is None


# ------------------------------------------------------------------
# _load_version — broken _version.py (import raises)
# ------------------------------------------------------------------


class TestLoadVersionBrokenFile:
    """A malformed ``_version.py`` (syntax error, raises on import, etc.)
    must NEVER propagate — defaults are returned and the status page is
    saved from a 500.
    """

    def test_broken_version_module_returns_defaults_without_raising(
        self, monkeypatch, caplog
    ):
        # Make sure no real _version is cached, then intercept the import
        # itself so ``from src import _version`` raises a non-ImportError
        # exception. The loader's ``except Exception`` branch must catch it.
        monkeypatch.delitem(sys.modules, "src._version", raising=False)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "src" and fromlist and "_version" in fromlist:
                raise ValueError("simulated malformed _version.py")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        caplog.set_level(logging.WARNING, logger="windowbot.version")
        # No exception escapes.
        result = _load_version()

        # Defaults returned — every field matches the dev contract.
        assert result == dict(_DEFAULTS)
        # Gregory's defensive warning is emitted on the version logger.
        assert any(
            "Failed to import src/_version.py" in rec.getMessage()
            for rec in caplog.records
        ), "expected a warning log from the defensive _version loader"


# ------------------------------------------------------------------
# WORKER_STARTED_AT — captured at module import
# ------------------------------------------------------------------


class TestWorkerStartedAt:
    """Module-level constant captured once at import time."""

    def test_worker_started_at_is_tz_aware_utc(self):
        assert isinstance(WORKER_STARTED_AT, datetime)
        assert WORKER_STARTED_AT.tzinfo is not None
        # Specifically UTC offset, not just any tz-aware datetime.
        assert WORKER_STARTED_AT.utcoffset() == timezone.utc.utcoffset(None)

    def test_worker_started_at_is_recent(self):
        """The whole test session is well under 30 minutes; pin the constant
        to be within 30 minutes of "now" so we catch a regression that
        accidentally hard-codes the value or recomputes it on every read.
        """
        delta = (datetime.now(timezone.utc) - WORKER_STARTED_AT).total_seconds()
        assert 0 <= delta < 1800, f"WORKER_STARTED_AT delta out of range: {delta}s"


# ------------------------------------------------------------------
# parse_iso_utc — accepts trailing-Z and offset; rejects garbage
# ------------------------------------------------------------------


class TestParseIsoUtcValid:
    """Parametrized over real and synthetic timestamps."""

    @pytest.mark.parametrize(
        "value",
        [
            "2026-05-21T18:30:00Z",
            "2026-05-21T18:30:00+00:00",
            "2026-05-21T18:30:00-07:00",
        ],
    )
    def test_accepts_iso_with_z_or_offset(self, value):
        result = parse_iso_utc(value)
        assert isinstance(result, datetime)
        # All three inputs must land as tz-aware datetimes — the loader's
        # downstream arithmetic (``now - commit_time``) requires it.
        assert result.tzinfo is not None


class TestParseIsoUtcInvalid:
    """Garbage in → ``None`` out. Must NEVER raise."""

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "not-a-date",
            "2026-13-99T99:99:99Z",
        ],
    )
    def test_invalid_input_returns_none_without_raising(self, value):
        # Wrapped in a try/except to make the intent explicit — even if
        # the assertion fails, the test should NOT raise from the parser.
        try:
            result = parse_iso_utc(value)
        except Exception as exc:  # pragma: no cover — fail visibly
            pytest.fail(f"parse_iso_utc raised on {value!r}: {exc!r}")
        assert result is None
