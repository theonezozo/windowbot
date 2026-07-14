"""Tests for the WindowBot orchestrator.

Validates design decisions:
- Full pipeline: fetch → decide → notify with mocked dependencies.
- Notification cooldown: 1 hour between non-urgent notifications.
- Cooldown bypass: urgent AQI close ALWAYS sends immediately.
- Auth failure: EcobeeAuthError triggers urgent notification to user.
- Per-floor iteration: upstairs and downstairs evaluated independently.
- AQI source preference: PurpleAir first, AirNow fallback on failure.
- Graceful handling of NWS/AQI failures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from src.orchestrator import run_check, _fetch_aqi, _evaluate_floor, _NOTIFICATION_COOLDOWN
from src.ecobee_client import EcobeeAuthError, EcobeeApiError
from src.nws_client import NWSClient, NWSError
from src.openmeteo_client import OpenMeteoError
from src.outdoor_validator import OutdoorValidationResult
from src.decision_engine import DecisionEngine, FloorDecision, InsufficientDataError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _base_config(**overrides):
    """Build a minimal config dict for orchestrator tests."""
    cfg = {
        "ecobee_client_id": "cid",
        "ecobee_refresh_token": "rtoken",
        "airnow_api_key": "akey",
        "purpleair_api_key": "pkey",
        "user_latitude": 40.0,
        "user_longitude": -74.0,
        "ntfy_topic": "test",
        "upstairs_sensors": ["sensor_up"],
        "downstairs_sensors": ["sensor_down"],
        "hysteresis_open_diff": 1.0,
        "hysteresis_close_diff": 1.0,
        "max_outdoor_humidity": 80,
        "max_aqi_threshold": 100,
        "min_aqi_for_opening": 50,
        "allowed_hvac_modes": ["cool", "heatCool", "auto"],
        "enable_humidity_gate": True,
        "enable_aqi_gate": True,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------
# Auth Failure Notification
# ------------------------------------------------------------------


class TestAuthFailure:
    """EcobeeAuthError triggers urgent notification to user."""

    @patch("src.orchestrator.send_notification")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_config")
    def test_auth_error_sends_urgent_notification(
        self, mock_config, mock_ecobee_cls, mock_state_cls, mock_notify
    ):
        """Ecobee auth failure → urgent notification sent."""
        mock_config.return_value = _base_config()
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.side_effect = EcobeeAuthError("Token revoked")
        mock_ecobee_cls.return_value = mock_ecobee

        run_check()

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["urgent"] is True
        assert kwargs["priority"] == "urgent"
        assert "Auth" in kwargs["title"] or "auth" in kwargs["title"].lower()

    @patch("src.orchestrator.send_notification")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_config")
    def test_api_error_does_not_notify(
        self, mock_config, mock_ecobee_cls, mock_state_cls, mock_notify
    ):
        """EcobeeApiError (non-auth) → no notification sent, just returns."""
        mock_config.return_value = _base_config()
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.side_effect = EcobeeApiError("Server error")
        mock_ecobee_cls.return_value = mock_ecobee

        run_check()

        mock_notify.assert_not_called()


# ------------------------------------------------------------------
# AQI Source Preference — AirNow-first with open-windows carve-out (Option B)
# ------------------------------------------------------------------


class TestAQISourcePreference:
    """AirNow-first, cost-aware ordering.

    PurpleAir's read API is metered; AirNow is free. Query AirNow first and
    only spend PurpleAir points when its higher local sensitivity can change
    the decision — always querying it when windows are OPEN or AirNow is clean,
    skipping it when AirNow already blocks/force-closes and the value can't
    change the outcome. See _fetch_aqi Option B docstring.
    """

    # --- (a) AirNow already urgent → PurpleAir NOT called -----------------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_urgent_skips_purpleair(self, mock_pa_cls, mock_an_cls):
        """AirNow ≥ URGENT (100) → AirNow force-closes; PurpleAir not queried."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 155, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        config = _base_config()
        # Even with windows OPEN, an already-urgent AirNow needs no PurpleAir.
        result = _fetch_aqi(config, last_state="OPEN")

        assert result["aqi"] == 155
        assert result["source"] == "airnow"
        mock_pa_cls.assert_not_called()

    # --- (b) AirNow block-band + windows CLOSED → PurpleAir NOT called -----
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_blockband_closed_skips_purpleair(self, mock_pa_cls, mock_an_cls):
        """AirNow in [50, 100) + windows CLOSED → opening already blocked; no PurpleAir."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 72, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        config = _base_config()
        result = _fetch_aqi(config, last_state="CLOSED")

        assert result["aqi"] == 72
        assert result["source"] == "airnow"
        mock_pa_cls.assert_not_called()

    # --- (c) AirNow block-band + windows OPEN → PurpleAir IS called --------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_blockband_open_queries_purpleair_urgent(self, mock_pa_cls, mock_an_cls):
        """AirNow in [50, 100) + windows OPEN → PurpleAir queried; a local ≥100 drives urgent close."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 72, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 168, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        config = _base_config()
        result = _fetch_aqi(config, last_state="OPEN")

        # PurpleAir's higher-sensitivity reading wins and exceeds URGENT.
        assert result["aqi"] == 168
        assert result["source"] == "purpleair"
        mock_pa.get_aqi.assert_called_once()

    # --- (d) AirNow clean → PurpleAir IS called (catches lagging smoke) ----
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_clean_queries_purpleair(self, mock_pa_cls, mock_an_cls):
        """AirNow < 50 → PurpleAir queried since AirNow may lag a local smoke event."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 20, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 130, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        config = _base_config()
        # Clean-AirNow branch queries PurpleAir regardless of window state.
        result = _fetch_aqi(config, last_state="CLOSED")

        assert result["aqi"] == 130
        assert result["source"] == "purpleair"
        mock_pa.get_aqi.assert_called_once()

    # --- (e) AirNow fails → PurpleAir called as resilience fallback --------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_fails_purpleair_resilience(self, mock_pa_cls, mock_an_cls):
        """AirNow fails → PurpleAir becomes the reading (never lose the AQI gate)."""
        mock_an = MagicMock()
        mock_an.get_aqi.side_effect = Exception("AirNow down")
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 45, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        config = _base_config()
        result = _fetch_aqi(config, last_state="CLOSED")

        assert result["aqi"] == 45
        assert result["source"] == "purpleair"
        mock_pa.get_aqi.assert_called_once()

    # --- (f) PurpleAir queried but fails (402) → fall back to AirNow -------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_purpleair_402_falls_back_to_airnow(self, mock_pa_cls, mock_an_cls, caplog):
        """PurpleAir 402 (out of points) → fall back to AirNow value, reason logged, no crash.

        Regression guard: the payment/account failure must appear in logs so a
        depleted PurpleAir balance is diagnosable, not silently masked.
        """
        from src.purpleair_client import PurpleAirError

        # AirNow clean → PurpleAir gets queried; then it 402s.
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 30, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = PurpleAirError(
            "PurpleAir API error (402: Payment Required — out of points) body={}"
        )
        mock_pa_cls.return_value = mock_pa

        config = _base_config()
        with caplog.at_level("WARNING", logger="windowbot"):
            result = _fetch_aqi(config, last_state="OPEN")

        assert result["aqi"] == 30
        assert result["source"] == "airnow"
        assert "402" in caplog.text
        assert "Payment Required" in caplog.text

    def test_both_providers_fail_returns_zero(self):
        """Both providers fail → AQI defaults to {"aqi": 0, "source": "none"}."""
        with patch("src.orchestrator.AirNowClient") as mock_an_cls, \
                patch("src.orchestrator.PurpleAirClient") as mock_pa_cls:
            mock_an = MagicMock()
            mock_an.get_aqi.side_effect = Exception("AN down")
            mock_an_cls.return_value = mock_an

            mock_pa = MagicMock()
            mock_pa.get_aqi.side_effect = Exception("PA down")
            mock_pa_cls.return_value = mock_pa

            config = _base_config()
            result = _fetch_aqi(config, last_state="OPEN")

        assert result["aqi"] == 0
        assert result["source"] == "none"

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_no_airnow_key_goes_straight_to_purpleair(self, mock_pa_cls, mock_an_cls):
        """No AirNow API key → AirNow treated as failed; PurpleAir is the reading."""
        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 42, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        config = _base_config(airnow_api_key="")
        result = _fetch_aqi(config, last_state="CLOSED")

        assert result["source"] == "purpleair"
        mock_an_cls.assert_not_called()

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_no_purpleair_key_uses_airnow(self, mock_pa_cls, mock_an_cls):
        """No PurpleAir API key → AirNow value used even when PurpleAir would be queried."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 30, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        # AirNow clean would normally query PurpleAir, but no key → skip it.
        config = _base_config(purpleair_api_key="")
        result = _fetch_aqi(config, last_state="OPEN")

        assert result["aqi"] == 30
        assert result["source"] == "airnow"
        mock_pa_cls.assert_not_called()

    # --- (g) once-per-cycle: shared cache keeps each provider to one call --
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_shared_cache_one_call_per_provider(self, mock_pa_cls, mock_an_cls):
        """Two floors sharing an aqi_cache → AirNow and PurpleAir each queried once."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 30, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 44, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        config = _base_config()
        cache: dict = {}
        # Simulate two floors in one cycle (both clean-AirNow → both want PurpleAir).
        r1 = _fetch_aqi(config, last_state="OPEN", aqi_cache=cache)
        r2 = _fetch_aqi(config, last_state="CLOSED", aqi_cache=cache)

        assert r1["source"] == "purpleair"
        assert r2["source"] == "purpleair"
        # Providers constructed and queried at most once despite two floors.
        mock_an.get_aqi.assert_called_once()
        mock_pa.get_aqi.assert_called_once()


# ------------------------------------------------------------------
# AQI dual readings — carry BOTH providers' values for status-page display
# ------------------------------------------------------------------


class TestAQIDualReadings:
    """``_fetch_aqi`` attaches a display-only ``readings`` dict carrying each
    provider's AQI for the cycle. It never changes the authoritative
    ``aqi``/``source`` that drive the decision — it only records what BOTH
    providers reported (or None when a provider wasn't queried / failed).
    """

    # --- both checked: AirNow clean → PurpleAir queried, PurpleAir wins ----
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_both_checked_carries_both_readings(self, mock_pa_cls, mock_an_cls):
        """AirNow < 50 queries PurpleAir → readings holds BOTH values; the
        authoritative value stays PurpleAir (unchanged decision)."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 42, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 155, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        result = _fetch_aqi(_base_config(), last_state="CLOSED")

        # Authoritative value is unchanged: PurpleAir won.
        assert result["aqi"] == 155
        assert result["source"] == "purpleair"
        # Both readings preserved for display.
        assert result["readings"] == {"airnow": 42, "purpleair": 155}

    # --- both checked (block-band + OPEN): PurpleAir wins, both carried -----
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_both_checked_blockband_open(self, mock_pa_cls, mock_an_cls):
        """AirNow in [50,100) + windows OPEN → PurpleAir queried; both carried."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 72, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 168, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        result = _fetch_aqi(_base_config(), last_state="OPEN")

        assert result["source"] == "purpleair"
        assert result["readings"] == {"airnow": 72, "purpleair": 168}

    # --- AirNow-only: urgent short-circuit → PurpleAir NOT queried ---------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_only_urgent_shortcircuit(self, mock_pa_cls, mock_an_cls):
        """AirNow ≥ 100 skips PurpleAir → readings shows airnow value, purpleair None."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 155, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        result = _fetch_aqi(_base_config(), last_state="OPEN")

        assert result["aqi"] == 155
        assert result["source"] == "airnow"
        assert result["readings"] == {"airnow": 155, "purpleair": None}
        mock_pa_cls.assert_not_called()

    # --- AirNow-only: block-band + CLOSED → PurpleAir NOT queried ----------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_only_blockband_closed(self, mock_pa_cls, mock_an_cls):
        """AirNow in [50,100) + windows CLOSED skips PurpleAir → purpleair None."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 72, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        result = _fetch_aqi(_base_config(), last_state="CLOSED")

        assert result["readings"] == {"airnow": 72, "purpleair": None}
        mock_pa_cls.assert_not_called()

    # --- PurpleAir queried but fails → fall back, purpleair reading None ---
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_purpleair_failed_fallback_readings(self, mock_pa_cls, mock_an_cls):
        """AirNow clean queries PurpleAir; PurpleAir fails → fall back to AirNow.
        readings carries the AirNow value with purpleair None (queried, failed)."""
        from src.purpleair_client import PurpleAirError

        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 30, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = PurpleAirError("402: Payment Required — out of points")
        mock_pa_cls.return_value = mock_pa

        result = _fetch_aqi(_base_config(), last_state="OPEN")

        assert result["aqi"] == 30
        assert result["source"] == "airnow"
        assert result["readings"] == {"airnow": 30, "purpleair": None}

    # --- both providers fail → readings both None, no crash ---------------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_both_fail_readings_both_none(self, mock_pa_cls, mock_an_cls):
        """Both providers fail → readings is {airnow: None, purpleair: None}."""
        mock_an = MagicMock()
        mock_an.get_aqi.side_effect = Exception("AN down")
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = Exception("PA down")
        mock_pa_cls.return_value = mock_pa

        result = _fetch_aqi(_base_config(), last_state="OPEN")

        assert result["source"] == "none"
        assert result["readings"] == {"airnow": None, "purpleair": None}

    # --- AirNow reading is not lost when PurpleAir wins (regression) -------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_reading_survives_purpleair_win(self, mock_pa_cls, mock_an_cls):
        """Regression: before this change the PurpleAir dict replaced AirNow's
        entirely, losing AirNow's value. readings must still expose AirNow."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 18, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 140, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        result = _fetch_aqi(_base_config(), last_state="CLOSED")

        assert result["readings"]["airnow"] == 18
        assert result["readings"]["purpleair"] == 140


# ------------------------------------------------------------------
# AQI failure reasons — explain a "source: none" instead of a bare sentinel
# ------------------------------------------------------------------


class TestAQIFailureReasons:
    """When a provider produces no reading, ``_fetch_aqi`` records WHY in a
    display-only ``aqi_reasons`` dict so the status page / logs can distinguish
    a config/account gap (missing key, 402 out-of-points, no stations) from a
    silent breakage. Never affects the authoritative aqi/source.
    """

    # --- Regression guard: a valid AirNow reading must NOT collapse to none --
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_airnow_value_not_collapsed_to_none(self, mock_pa_cls, mock_an_cls):
        """AirNow returns a valid AQI and PurpleAir is NOT queried (urgent
        short-circuit) → source is "airnow" with that value, NEVER "none".

        This is the exact regression the "AQI 0, source: none" symptom would
        represent if the AirNow-first refactor dropped a good AirNow reading.
        """
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 155, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        result = _fetch_aqi(_base_config(), last_state="OPEN")

        assert result["source"] == "airnow"
        assert result["aqi"] == 155
        assert result["source"] != "none"
        # No failure reasons attached when AirNow succeeded and PA wasn't needed.
        assert "aqi_reasons" not in result
        mock_pa_cls.assert_not_called()

    # --- Both fail → reasons captured for BOTH providers ------------------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_both_fail_captures_reasons(self, mock_pa_cls, mock_an_cls):
        """Both providers raise → source "none" AND aqi_reasons explains each."""
        mock_an = MagicMock()
        mock_an.get_aqi.side_effect = Exception("AirNow API error (403): Invalid API_KEY")
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = Exception("402: Payment Required — out of points")
        mock_pa_cls.return_value = mock_pa

        result = _fetch_aqi(_base_config(), last_state="OPEN")

        assert result["source"] == "none"
        assert result["aqi"] == 0
        reasons = result["aqi_reasons"]
        assert "403" in reasons["airnow"]
        assert "402" in reasons["purpleair"]

    # --- Missing keys → reasons say so, not a bare sentinel ---------------
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_missing_keys_report_no_key(self, mock_pa_cls, mock_an_cls):
        """No AirNow key + no PurpleAir key → source none, reasons name the gap."""
        config = _base_config(airnow_api_key="", purpleair_api_key="")
        result = _fetch_aqi(config, last_state="OPEN")

        assert result["source"] == "none"
        assert result["aqi_reasons"]["airnow"] == "no API key configured"
        assert result["aqi_reasons"]["purpleair"] == "no API key configured"
        # Clients never even constructed when the key is absent.
        mock_an_cls.assert_not_called()
        mock_pa_cls.assert_not_called()

    # --- Partial failure surfaces in reasons even when a reading survives --
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_purpleair_402_reason_present_with_airnow_reading(self, mock_pa_cls, mock_an_cls):
        """AirNow succeeds but PurpleAir 402s → reading stands, PA reason recorded."""
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 30, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = Exception("402: Payment Required — out of points")
        mock_pa_cls.return_value = mock_pa

        result = _fetch_aqi(_base_config(), last_state="OPEN")

        assert result["source"] == "airnow"
        assert result["aqi"] == 30
        assert "402" in result["aqi_reasons"]["purpleair"]
        assert "airnow" not in result["aqi_reasons"]


# ------------------------------------------------------------------
# PurpleAir Client Persistence (cross-cycle caching)
# ------------------------------------------------------------------


class TestPurpleAirClientPersistence:
    """The PurpleAirClient is reused across cycles so its sensor-ID cache survives."""

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_client_reused_across_cycles(self, mock_pa_cls, mock_an_cls):
        """Multiple _fetch_aqi calls construct the client once, reuse it after."""
        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 42, "source": "purpleair"}
        mock_pa_cls.return_value = mock_pa

        # Clean AirNow each cycle → PurpleAir always queried (confirm no smoke).
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 10, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        config = _base_config()
        _fetch_aqi(config)
        _fetch_aqi(config)
        _fetch_aqi(config)

        # Constructed once; the same instance (and its cache) serves later cycles.
        assert mock_pa_cls.call_count == 1
        assert mock_pa.get_aqi.call_count == 3

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_cache_ttl_passed_from_config(self, mock_pa_cls, mock_an_cls):
        """The configurable cache TTL is threaded into the client constructor."""
        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 10, "source": "purpleair"}
        mock_pa_cls.return_value = mock_pa

        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 10, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        config = _base_config(purpleair_sensor_cache_hours=6.0)
        _fetch_aqi(config)

        _, kwargs = mock_pa_cls.call_args
        assert kwargs["sensor_cache_ttl_hours"] == 6.0

    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    def test_client_rebuilt_when_location_changes(self, mock_pa_cls, mock_an_cls):
        """Changing lat/lon invalidates the singleton and rebuilds the client."""
        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 10, "source": "purpleair"}
        mock_pa_cls.return_value = mock_pa

        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 10, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        _fetch_aqi(_base_config())
        _fetch_aqi(_base_config(user_latitude=41.0))

        assert mock_pa_cls.call_count == 2



# ------------------------------------------------------------------
# Notification Cooldown
# ------------------------------------------------------------------


class TestNotificationCooldown:
    """1-hour cooldown between non-urgent notifications."""

    def test_cooldown_constant_is_3600(self):
        """Cooldown is 1 hour = 3600 seconds."""
        assert _NOTIFICATION_COOLDOWN == 3600

    @patch("src.orchestrator.send_notification")
    def test_first_notification_always_sent(self, mock_notify):
        """First notification ever (no LastNotificationTime) → always sent."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()

    @patch("src.orchestrator.send_notification")
    def test_transition_notifies_within_legacy_cooldown_window(self, mock_notify):
        """Option A: a genuine CLOSED→OPEN transition notifies even 30 min after
        the last notification.

        The legacy 1-hour time cooldown no longer gates a real open↔close
        transition — only same-type dedup prevents spam. Here the last type is
        None (≠ 'open'), so the transition is delivered despite the notification
        sent 30 min ago, and the persisted record records
        LastNotificationType='open'.
        """
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()

        # Last notification was 30 minutes ago (within the legacy 1-hour window).
        recent = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": recent,
            "LastNotificationType": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        state_mgr.update_floor_state.assert_called_once()
        record = state_mgr.update_floor_state.call_args.args[1]
        assert record["LastNotificationType"] == "open"

    @patch("src.orchestrator.send_notification")
    def test_cooldown_expired_allows_notification(self, mock_notify):
        """Non-urgent notification after cooldown expires → sent."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()

        # Last notification was 2 hours ago (beyond 1-hour cooldown)
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": old,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()


# ------------------------------------------------------------------
# Urgent AQI Cooldown Bypass
# ------------------------------------------------------------------


class TestUrgentCooldownBypass:
    """Urgent AQI close ALWAYS sends immediately regardless of cooldown."""

    @patch("src.orchestrator.send_notification")
    def test_urgent_bypasses_cooldown(self, mock_notify):
        """AQI ≥100 on OPEN windows → urgent notification even during cooldown."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()

        # Last notification was just 5 minutes ago (within cooldown)
        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "OPEN",
            "LastNotificationTime": recent,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 150}  # triggers urgent close
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["urgent"] is True


# ------------------------------------------------------------------
# Per-Floor Iteration
# ------------------------------------------------------------------


class TestPerFloorIteration:
    """Each floor evaluated independently in run_check."""

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_both_floors_evaluated(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """run_check calls _evaluate_floor for both upstairs and downstairs."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": 70.0, "humidity": 50.0
        }
        mock_nws_cls.return_value = mock_nws
        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("test")

        mock_fetch_aqi.return_value = {"aqi": 30}

        run_check()

        # Should be called for both floors
        assert mock_eval_floor.call_count == 2
        floor_names = [c.args[0] for c in mock_eval_floor.call_args_list]
        assert "upstairs" in floor_names
        assert "downstairs" in floor_names

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_empty_floor_skipped(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Floor with empty sensor list → skipped."""
        mock_config.return_value = _base_config(downstairs_sensors=[])

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": 70.0, "humidity": 50.0
        }
        mock_nws_cls.return_value = mock_nws
        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("test")

        mock_fetch_aqi.return_value = {"aqi": 30}

        run_check()

        # Only upstairs evaluated (downstairs has empty sensor list)
        assert mock_eval_floor.call_count == 1
        assert mock_eval_floor.call_args.args[0] == "upstairs"


# ------------------------------------------------------------------
# State Persistence
# ------------------------------------------------------------------


class TestStatePersistence:
    """State is always updated after evaluation."""

    @patch("src.orchestrator.send_notification")
    def test_state_updated_on_change(self, mock_notify):
        """State change → update_floor_state called with new state."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        state_mgr.update_floor_state.assert_called_once()
        update_args = state_mgr.update_floor_state.call_args
        assert update_args.args[0] == "upstairs"
        assert update_args.args[1]["CurrentState"] == "OPEN"

    @patch("src.orchestrator.send_notification")
    def test_state_updated_even_without_change(self, mock_notify):
        """No state change → still updates state (timestamp, reason, etc.)."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 80.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]

        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        state_mgr.update_floor_state.assert_called_once()


# ------------------------------------------------------------------
# NWS Failure Handling
# ------------------------------------------------------------------


class TestNWSFailure:
    """NWS failure when it's the only remaining source gracefully aborts the check."""

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_failure_with_openmeteo_failure_aborts(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """NWSError + OpenMeteoError (peer and fallback both fail) → floors not evaluated."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("API down")
        mock_get_nws.return_value = mock_nws

        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("OM down")
        mock_om.get_outdoor_conditions.side_effect = OpenMeteoError("OM down")
        mock_om_cls.return_value = mock_om

        run_check()

        mock_eval_floor.assert_not_called()


# ------------------------------------------------------------------
# InsufficientDataError Handling
# ------------------------------------------------------------------


class TestInsufficientData:
    """InsufficientDataError for a floor is caught and logged."""

    @patch("src.orchestrator.send_notification")
    def test_offline_sensor_with_valid_temp_works(self, mock_notify):
        """Floor with offline sensor but valid temp → decision proceeds normally."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": None,
        }

        outdoor = {"temperature_f": 68.0, "humidity": 50.0}
        aqi_data = {"aqi": 30}
        # Sensor is offline but has valid temperature
        sensors = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": False}]

        # Should proceed normally — offline status doesn't block decisions anymore
        _evaluate_floor(
            "upstairs", ["sensor_up"], sensors, outdoor, aqi_data,
            "cool", engine, state_mgr,
        )

        # State gets updated with normal decision (warmest=74, threshold=73, outdoor 68 < 73 → OPEN)
        state_mgr.update_floor_state.assert_called_once()
        call_args = state_mgr.update_floor_state.call_args[0]
        assert call_args[1]["CurrentState"] == "OPEN"


# ------------------------------------------------------------------
# Conditional / Lazy AQI Polling
# ------------------------------------------------------------------


class TestConditionalAqiPolling:
    """Lazy AQI fetching — orchestrator skips PurpleAir/AirNow when unnecessary.

    Validates:
    - AQI skipped when windows CLOSED + indoor comfortable
    - AQI skipped when HVAC mode blocks action
    - AQI fetched when windows OPEN (safety net)
    - AQI fetched when temperature favors opening
    - AQI cached across floors in same run_check cycle
    - Fallback PurpleAir→AirNow still works in lazy path
    - Skip reason logged correctly
    """

    # -- Helpers --

    @staticmethod
    def _setup_run_check_mocks(
        mock_config,
        mock_state_cls,
        mock_ecobee_cls,
        mock_nws_cls,
        mock_om_cls=None,
        *,
        sensor_temps=None,
        hvac_mode="cool",
        outdoor_temp=65.0,
        outdoor_humidity=50.0,
        current_state="CLOSED",
    ):
        """Wire up the standard mocks for a run_check integration test."""
        if sensor_temps is None:
            sensor_temps = {"sensor_up": 70.0, "sensor_down": 70.0}

        mock_config.return_value = _base_config()

        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {
            "CurrentState": current_state,
        }
        mock_state_cls.return_value = mock_state

        sensors = [
            {"name": name, "temperature_f": temp, "is_online": True}
            for name, temp in sensor_temps.items()
        ]
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = sensors
        mock_ecobee.get_hvac_mode.return_value = hvac_mode
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": outdoor_temp,
            "humidity": outdoor_humidity,
        }
        mock_nws_cls.return_value = mock_nws

        if mock_om_cls is not None:
            mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")

    # ------------------------------------------------------------------
    # 1. AQI skipped when windows CLOSED + indoor comfortable (≤72°F)
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_skipped_when_closed_and_comfortable(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Windows CLOSED + indoor 70°F (≤ 72°F comfort max) → AQI not fetched."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        mock_fetch_aqi.assert_not_called()

    # ------------------------------------------------------------------
    # 1b. Skipped aqi_data carries the human-readable skip_reason
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_skipped_aqi_data_carries_skip_reason(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """When a floor skips AQI, the aqi_data handed downstream carries the
        human-readable skip_reason so the status page can show WHY (not a bare
        'AQI 0, source: skipped')."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        # AQI was skipped (fetch never called), and the aqi_data passed to
        # _evaluate_floor reflects the skip with a populated reason.
        mock_fetch_aqi.assert_not_called()
        assert mock_eval_floor.call_args is not None
        aqi_data = mock_eval_floor.call_args[0][4]
        assert aqi_data["source"] == "skipped"
        assert aqi_data["aqi"] == 0
        assert isinstance(aqi_data.get("skip_reason"), str)
        assert aqi_data["skip_reason"]  # non-empty

    # ------------------------------------------------------------------
    # 2. AQI skipped when HVAC mode not in allowed list
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_skipped_when_hvac_mode_not_allowed(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """HVAC mode 'heat' not in allowed modes → AQI not fetched."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 78.0, "sensor_down": 78.0},
            hvac_mode="heat",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        mock_fetch_aqi.assert_not_called()

    # ------------------------------------------------------------------
    # 3. AQI fetched when windows are OPEN
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_fetched_when_windows_open(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Windows OPEN → AQI always fetched (urgent close safety net)."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 74.0, "sensor_down": 74.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )
        mock_fetch_aqi.return_value = {"aqi": 30, "source": "purpleair"}

        run_check()

        # AQI is fetched (not skipped) — once per floor under the per-floor carve-out.
        assert mock_fetch_aqi.call_count == 2

    # ------------------------------------------------------------------
    # 3b. OPEN floor must NEVER receive the "skipped" sentinel
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_open_floor_never_gets_skipped_sentinel(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """OPEN + comfortable indoor (≤72°F) → real fetch, never source:skipped.

        Comfortable indoor would make the CLOSED branch SKIP; proving the OPEN
        safety rule wins over the comfort gate at the orchestrator level.
        """
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )
        mock_fetch_aqi.return_value = {"aqi": 42, "source": "airnow"}

        run_check()

        # Every floor was fetched, and _evaluate_floor never saw the sentinel.
        assert mock_fetch_aqi.call_count == 2
        for call in mock_eval_floor.call_args_list:
            aqi_arg = call.args[4]  # aqi_data is the 5th positional arg
            assert aqi_arg.get("source") != "skipped"

    # ------------------------------------------------------------------
    # 3c. Non-canonical OPEN state ("open") still fetches (normalisation)
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_lowercase_open_state_still_fetches_aqi(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """A stored 'open' (lowercase) state must NOT skip the AQI safety fetch.

        Regression for the casing/format hardening: with comfortable indoor
        temps the CLOSED branch would skip, so a fetch here proves 'open' is
        normalised to OPEN before the safety check.
        """
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="open",
        )
        mock_fetch_aqi.return_value = {"aqi": 42, "source": "airnow"}

        run_check()

        assert mock_fetch_aqi.call_count == 2
        for call in mock_eval_floor.call_args_list:
            aqi_arg = call.args[4]
            assert aqi_arg.get("source") != "skipped"

    # ------------------------------------------------------------------
    # 4. AQI fetched when windows closed but temperature favors opening
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_fetched_when_temp_favors_opening(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Indoor 78°F, outdoor 65°F, CLOSED → AQI fetched to confirm safe."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 78.0, "sensor_down": 78.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )
        mock_fetch_aqi.return_value = {"aqi": 25, "source": "purpleair"}

        run_check()

        # AQI is fetched (not skipped) — once per floor under the per-floor carve-out.
        assert mock_fetch_aqi.call_count == 2

    # ------------------------------------------------------------------
    # 5. AQI provider readings shared across floors in same run_check cycle
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_cache_shared_across_floors(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """Both floors need AQI (OPEN) → _fetch_aqi called per floor with ONE shared cache.

        Under AirNow-first the carve-out decision is per floor (it depends on
        each floor's window state), so _fetch_aqi is invoked once per floor —
        but they share a single aqi_cache dict so each provider is still fetched
        at most once per cycle.
        """
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 74.0, "sensor_down": 74.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )
        aqi_result = {"aqi": 42, "source": "purpleair"}
        mock_fetch_aqi.return_value = aqi_result

        run_check()

        # One _fetch_aqi call per floor (the carve-out is per-floor)...
        assert mock_fetch_aqi.call_count == 2
        # ...but all calls thread the SAME aqi_cache dict — so providers stay
        # capped at one fetch each per cycle.
        caches = [c.kwargs["aqi_cache"] for c in mock_fetch_aqi.call_args_list]
        assert caches[0] is caches[1]
        # Both floors evaluated with the returned result
        assert mock_eval_floor.call_count == 2
        for eval_call in mock_eval_floor.call_args_list:
            assert eval_call.args[4] == aqi_result  # aqi_data positional arg

    # ------------------------------------------------------------------
    # 6. Fallback PurpleAir→AirNow still works in the lazy path
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_fallback_works_when_aqi_needed(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_pa_cls, mock_an_cls, mock_eval_floor,
    ):
        """Windows OPEN + AirNow block-band, PurpleAir fails → AirNow fallback used.

        Both floors OPEN with AirNow at 55 (block band) → the carve-out queries
        PurpleAir; it fails → each floor falls back to AirNow's value. The shared
        cache keeps both providers to a single call across the two floors.
        """
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 74.0, "sensor_down": 74.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )

        # PurpleAir fails
        mock_pa = MagicMock()
        mock_pa.get_aqi.side_effect = Exception("PurpleAir down")
        mock_pa_cls.return_value = mock_pa

        # AirNow succeeds (block band)
        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 55, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        run_check()

        # Each provider queried at most once despite two floors (shared cache).
        mock_an.get_aqi.assert_called_once()
        mock_pa.get_aqi.assert_called_once()
        # _evaluate_floor received the AirNow fallback result for both floors
        assert mock_eval_floor.call_count == 2
        for eval_call in mock_eval_floor.call_args_list:
            assert eval_call.args[4]["source"] == "airnow"

    # ------------------------------------------------------------------
    # 6b. Per-floor carve-out: OPEN floor spends PurpleAir, CLOSED floor doesn't
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_per_floor_carveout_open_vs_closed(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_pa_cls, mock_an_cls, mock_eval_floor,
    ):
        """AirNow block-band (72): OPEN floor queries PurpleAir, CLOSED floor uses AirNow.

        upstairs=OPEN, downstairs=CLOSED, both need AQI. AirNow returns 72 (in
        [50, 100)). The OPEN floor must catch a real local ≥100 → PurpleAir is
        queried once (and its value wins). The CLOSED floor is already blocked
        from opening → it reuses AirNow's value, spending no extra points.
        """
        mock_config.return_value = _base_config()

        # Per-floor window state via a keyed side_effect.
        def _floor_state(key):
            if key == "upstairs":
                return {"CurrentState": "OPEN"}
            if key == "downstairs":
                return {"CurrentState": "CLOSED"}
            return {}  # __global__ cold-starts outdoor validation

        mock_state = MagicMock()
        mock_state.get_floor_state.side_effect = _floor_state
        mock_state_cls.return_value = mock_state

        # Indoor 78°F so the CLOSED floor's non-AQI gates still favor opening
        # (→ it genuinely needs AQI), outdoor 65°F cool enough to open.
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = [
            {"name": "sensor_up", "temperature_f": 78.0, "is_online": True},
            {"name": "sensor_down", "temperature_f": 78.0, "is_online": True},
        ]
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": 65.0, "humidity": 50.0,
        }
        mock_nws_cls.return_value = mock_nws
        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")

        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 72, "source": "airnow"}
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 168, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        run_check()

        # AirNow fetched once (shared cache); PurpleAir fetched once — only the
        # OPEN floor triggered it, and the CLOSED floor reused the cached reading.
        mock_an.get_aqi.assert_called_once()
        mock_pa.get_aqi.assert_called_once()

        # Map each _evaluate_floor call to its floor name → aqi source used.
        source_by_floor = {
            c.args[0]: c.args[4]["source"] for c in mock_eval_floor.call_args_list
        }
        assert source_by_floor["upstairs"] == "purpleair"   # OPEN → PurpleAir wins
        assert source_by_floor["downstairs"] == "airnow"    # CLOSED → AirNow reused

    # ------------------------------------------------------------------
    # 6c. Once-per-cycle: two OPEN floors → each provider queried once
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator.AirNowClient")
    @patch("src.orchestrator.PurpleAirClient")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_once_per_cycle_two_floors_single_provider_calls(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_pa_cls, mock_an_cls, mock_eval_floor,
    ):
        """Two OPEN floors, AirNow clean → AirNow and PurpleAir each called exactly once."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 74.0, "sensor_down": 74.0},
            hvac_mode="cool",
            outdoor_temp=68.0,
            current_state="OPEN",
        )

        mock_an = MagicMock()
        mock_an.get_aqi.return_value = {"aqi": 20, "source": "airnow"}  # clean
        mock_an_cls.return_value = mock_an

        mock_pa = MagicMock()
        mock_pa.get_aqi.return_value = {"aqi": 44, "source": "purpleair", "sensor_count": 3}
        mock_pa_cls.return_value = mock_pa

        run_check()

        mock_an.get_aqi.assert_called_once()
        mock_pa.get_aqi.assert_called_once()
        assert mock_eval_floor.call_count == 2
        for eval_call in mock_eval_floor.call_args_list:
            assert eval_call.args[4]["source"] == "purpleair"

    # ------------------------------------------------------------------
    # 7. AQI skip reason logged
    # ------------------------------------------------------------------

    @patch("src.orchestrator.logger")
    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_aqi_skipped_log_message(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor, mock_logger,
    ):
        """When AQI is skipped, logger records the floor name and skip reason."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        # At least one "Skipping AQI fetch" log per skipped floor
        skip_calls = [
            c for c in mock_logger.info.call_args_list
            if len(c.args) >= 2 and "Skipping AQI fetch" in str(c.args[0])
        ]
        assert len(skip_calls) >= 1
        # The log message includes the floor name
        assert any("upstairs" in str(c) for c in skip_calls)

    # ------------------------------------------------------------------
    # 8. Skipped AQI passes sentinel to _evaluate_floor
    # ------------------------------------------------------------------

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_skipped_aqi_passes_sentinel_to_evaluate(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_eval_floor,
    ):
        """When AQI is skipped, _evaluate_floor receives the skip sentinel
        ({"aqi": 0, "source": "skipped"}) now enriched with a human-readable
        skip_reason for the status page."""
        self._setup_run_check_mocks(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            hvac_mode="cool",
            outdoor_temp=65.0,
            current_state="CLOSED",
        )

        run_check()

        mock_fetch_aqi.assert_not_called()
        for eval_call in mock_eval_floor.call_args_list:
            aqi_arg = eval_call.args[4]
            assert aqi_arg["aqi"] == 0
            assert aqi_arg["source"] == "skipped"
            assert isinstance(aqi_arg.get("skip_reason"), str) and aqi_arg["skip_reason"]


# ------------------------------------------------------------------
# Weather Fallback Chain: NWS (+ OM peer) → Open-Meteo sole fallback
# ------------------------------------------------------------------


class TestWeatherFallbackChain:
    """Outdoor weather: Open-Meteo always attempted as a peer alongside NWS.

    Flow:
      1. om.get_observation() called first (peer blending attempt).
      2. nws.get_outdoor_conditions(peer_observations=...) called.
      3. If NWS fails + OM peer fresh → OM used as sole source.
      4. If NWS fails + OM peer stale → om.get_outdoor_conditions() (no age check).
      5. If everything fails → return early.
    """

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_succeeds_floors_evaluated(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS succeeds (OM peer optional) → floors evaluated."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": 68.0, "humidity": 55.0, "source": "nws",
        }
        mock_get_nws.return_value = mock_nws

        # OM peer fails — NWS should still work alone
        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("stale")
        mock_om_cls.return_value = mock_om

        run_check()

        mock_nws.get_outdoor_conditions.assert_called_once()
        assert mock_eval_floor.call_count >= 1

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_fails_om_peer_used_as_sole_source(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS fails → fresh OM peer used as sole outdoor source; get_outdoor_conditions not called."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("NWS down")
        mock_get_nws.return_value = mock_nws

        from datetime import datetime, timezone
        mock_om = MagicMock()
        mock_om.get_observation.return_value = {
            "station_id": "OPENMETEO",
            "temperature_f": 66.0,
            "humidity": 60.0,
            "wind_speed_mph": 5.0,
            "timestamp": datetime.now(timezone.utc),
        }
        mock_om_cls.return_value = mock_om

        run_check()

        mock_nws.get_outdoor_conditions.assert_called_once()
        mock_om.get_observation.assert_called_once()
        mock_om.get_outdoor_conditions.assert_not_called()
        assert mock_eval_floor.call_count >= 1

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_nws_fails_peer_stale_falls_back_to_om_conditions(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS fails, OM peer stale → get_outdoor_conditions() last-resort fallback."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("NWS down")
        mock_get_nws.return_value = mock_nws

        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("stale")
        mock_om.get_outdoor_conditions.return_value = {
            "temperature_f": 66.0, "humidity": 60.0, "wind_speed_mph": 5.0,
            "source": "openmeteo", "is_fallback": True,
            "station_count": 1, "used_cache": False,
        }
        mock_om_cls.return_value = mock_om

        run_check()

        mock_om.get_outdoor_conditions.assert_called_once()
        assert mock_eval_floor.call_count >= 1

    @patch("src.orchestrator._evaluate_floor")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_both_fail_returns_early(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls,
        mock_fetch_aqi, mock_eval_floor,
    ):
        """NWS fails, OM peer stale, OM fallback fails → returns early, no floor evaluation."""
        mock_config.return_value = _base_config()

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("NWS down")
        mock_get_nws.return_value = mock_nws

        mock_om = MagicMock()
        mock_om.get_observation.side_effect = OpenMeteoError("stale")
        mock_om.get_outdoor_conditions.side_effect = OpenMeteoError("OM down")
        mock_om_cls.return_value = mock_om

        run_check()

        mock_eval_floor.assert_not_called()


# ------------------------------------------------------------------
# Temperature history recording (Gregory's 2026-05-19 feature)
# ------------------------------------------------------------------


class TestTemperatureHistoryRecording:
    """``run_check`` writes one history entry per cycle, and a history-write
    failure never breaks the cycle."""

    @staticmethod
    def _wire(
        mock_config,
        mock_state_cls,
        mock_ecobee_cls,
        mock_nws_cls,
        mock_om_cls,
        *,
        sensor_temps,
        outdoor_temp,
        outdoor_humidity=50.0,
        hvac_mode="cool",
        current_state="CLOSED",
    ):
        mock_config.return_value = _base_config()

        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {"CurrentState": current_state}
        mock_state_cls.return_value = mock_state

        sensors = [
            {"name": name, "temperature_f": temp, "is_online": True}
            for name, temp in sensor_temps.items()
        ]
        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = sensors
        mock_ecobee.get_hvac_mode.return_value = hvac_mode
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = {
            "temperature_f": outdoor_temp,
            "humidity": outdoor_humidity,
            "source": "nws",
        }
        mock_nws_cls.return_value = mock_nws

        # OM peer fails — keeps the path simple; NWS is the sole source.
        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")
        return mock_state

    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_record_called_once_with_floors_and_outdoor_temp(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_snap_mgr_cls,
    ):
        """Successful cycle → ``record_temperature_history`` called once with
        an entry whose ``outdoor_temp_f`` matches the outdoor reading and
        whose ``indoor_temps`` covers each floor that produced a snapshot."""
        self._wire(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 72.0},
            outdoor_temp=65.0,
        )
        mock_fetch_aqi.return_value = {"aqi": 25, "source": "purpleair"}

        snap_mgr = MagicMock()
        mock_snap_mgr_cls.return_value = snap_mgr

        run_check()

        snap_mgr.record_temperature_history.assert_called_once()
        entry = snap_mgr.record_temperature_history.call_args[0][0]
        assert entry.outdoor_temp_f == 65.0
        # Both floors produced snapshots, so both keys present.
        assert set(entry.indoor_temps.keys()) == {"upstairs", "downstairs"}
        # Coolest valid reading per floor (single sensor each → that reading).
        assert entry.indoor_temps["upstairs"] == 70.0
        assert entry.indoor_temps["downstairs"] == 72.0

    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator._fetch_aqi")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator.NWSClient")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_history_write_failure_does_not_break_cycle(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_nws_cls, mock_om_cls, mock_fetch_aqi, mock_snap_mgr_cls,
    ):
        """If ``record_temperature_history`` raises, floor + global snapshots
        still persist and the cycle completes without propagating."""
        self._wire(
            mock_config, mock_state_cls, mock_ecobee_cls, mock_nws_cls, mock_om_cls,
            sensor_temps={"sensor_up": 70.0, "sensor_down": 70.0},
            outdoor_temp=65.0,
        )
        mock_fetch_aqi.return_value = {"aqi": 25, "source": "purpleair"}

        snap_mgr = MagicMock()
        snap_mgr.record_temperature_history.side_effect = RuntimeError("history exploded")
        mock_snap_mgr_cls.return_value = snap_mgr

        # Must not raise.
        run_check()

        # Floor + global snapshots persisted before the history write.
        assert snap_mgr.save_floor_snapshot.call_count >= 1
        snap_mgr.save_global_snapshot.assert_called_once()
        snap_mgr.record_temperature_history.assert_called_once()


# ------------------------------------------------------------------
# Fix 1: Notification-Type Dedup
# ------------------------------------------------------------------


class TestNotificationTypeDedup:
    """Suppress consecutive SAME-type non-urgent notifications.

    The gate in ``_evaluate_floor`` persists ``LastNotificationType``
    ("open"/"close"). A non-urgent notification whose type equals the
    previously-sent type is suppressed (same-type dedup). Urgent alerts
    bypass dedup. Under Option A, the opposite type is never deduped and is
    NOT gated by the legacy time cooldown: a genuine open↔close transition
    always notifies.

    Technique (mirrors ``TestNotificationCooldown``): drive ``_evaluate_floor``
    end-to-end with sensors/outdoor/aqi that yield the intended OPEN/CLOSED
    *changed* decision, patch ``src.orchestrator.send_notification``, and assert
    "sent" via ``mock_notify.assert_called_once()`` / "suppressed" via
    ``mock_notify.assert_not_called()``. The persisted record is inspected via
    ``state_mgr.update_floor_state.call_args`` (the single end-of-cycle write).
    """

    # Decision inputs that produce a non-urgent CLOSED *changed* decision:
    # last_state OPEN + outdoor warmer than coolest indoor + good AQI.
    _CLOSE_OUTDOOR = {"temperature_f": 78.0, "humidity": 50.0}
    _CLOSE_SENSORS = [{"name": "sensor_up", "temperature_f": 72.0, "is_online": True}]
    # Decision inputs that produce a non-urgent OPEN *changed* decision:
    # last_state CLOSED + outdoor cooler than warmest indoor (>hysteresis) + good AQI.
    _OPEN_OUTDOOR = {"temperature_f": 68.0, "humidity": 50.0}
    _OPEN_SENSORS = [{"name": "sensor_up", "temperature_f": 74.0, "is_online": True}]
    _GOOD_AQI = {"aqi": 30}

    @staticmethod
    def _persisted_record(state_mgr):
        """Return the state record from the end-of-cycle update_floor_state call."""
        state_mgr.update_floor_state.assert_called_once()
        return state_mgr.update_floor_state.call_args.args[1]

    @patch("src.orchestrator.send_notification")
    def test_first_close_notifies_immediately(self, mock_notify):
        """last type 'open', last time >1h ago, change → CLOSED: sends + persists 'close'.

        Proves the first notification of a (differing-type) transition fires
        once the cooldown has elapsed and the persisted type flips to 'close'.
        """
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "OPEN",
            "LastNotificationTime": old,
            "LastNotificationType": "open",
        }

        _evaluate_floor(
            "upstairs", ["sensor_up"], self._CLOSE_SENSORS,
            self._CLOSE_OUTDOOR, self._GOOD_AQI, "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["urgent"] is False
        record = self._persisted_record(state_mgr)
        assert record["LastNotificationType"] == "close"

    @patch("src.orchestrator.send_notification")
    def test_repeated_close_suppressed_after_cooldown(self, mock_notify):
        """last type 'close', cooldown elapsed, change → CLOSED again: suppressed.

        Dedup wins even though the time cooldown has elapsed.
        """
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "OPEN",
            "LastNotificationTime": old,
            "LastNotificationType": "close",
        }

        _evaluate_floor(
            "upstairs", ["sensor_up"], self._CLOSE_SENSORS,
            self._CLOSE_OUTDOOR, self._GOOD_AQI, "cool", engine, state_mgr,
        )

        mock_notify.assert_not_called()
        # No send → the persisted record must not carry a fresh notification type.
        record = self._persisted_record(state_mgr)
        assert "LastNotificationType" not in record

    @patch("src.orchestrator.send_notification")
    def test_open_after_close_not_deduped(self, mock_notify):
        """last type 'close', cooldown elapsed, change → OPEN: sends + persists 'open'.

        The opposite type is never deduped against the current type.
        """
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": old,
            "LastNotificationType": "close",
        }

        _evaluate_floor(
            "upstairs", ["sensor_up"], self._OPEN_SENSORS,
            self._OPEN_OUTDOOR, self._GOOD_AQI, "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        record = self._persisted_record(state_mgr)
        assert record["LastNotificationType"] == "open"

    @patch("src.orchestrator.send_notification")
    def test_urgent_close_bypasses_dedup(self, mock_notify):
        """last type 'close', urgent change → CLOSED (AQI unhealthy): sends despite same type.

        Urgent notifications bypass dedup AND the time cooldown.
        """
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "OPEN",
            "LastNotificationTime": recent,
            "LastNotificationType": "close",
        }

        # AQI ≥ close threshold (100) on OPEN windows → urgent close.
        urgent_aqi = {"aqi": 150}

        _evaluate_floor(
            "upstairs", ["sensor_up"], self._CLOSE_SENSORS,
            self._CLOSE_OUTDOOR, urgent_aqi, "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["urgent"] is True
        record = self._persisted_record(state_mgr)
        assert record["LastNotificationType"] == "close"

    @patch("src.orchestrator.send_notification")
    def test_open_flip_shortly_after_close_notifies(self, mock_notify):
        """last type 'close', last time 10 min ago, flip → OPEN: notifies.

        Option A: a differing-type transition is no longer gated by the legacy
        time cooldown. A 'close' was sent 10 min ago, but the flip to OPEN is a
        genuine transition (different type, so not deduped) and is delivered
        promptly; the persisted type becomes 'open'. (Previously this scenario
        was wrongly suppressed by the time cooldown — bug #10.)
        """
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        recent = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": recent,
            "LastNotificationType": "close",
        }

        _evaluate_floor(
            "upstairs", ["sensor_up"], self._OPEN_SENSORS,
            self._OPEN_OUTDOOR, self._GOOD_AQI, "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        record = self._persisted_record(state_mgr)
        assert record["LastNotificationType"] == "open"

    @patch("src.orchestrator.send_notification")
    def test_bug10_open_after_humidity_close_notifies(self, mock_notify):
        """Regression for bug #10: an OPEN alert is delivered promptly even
        though a 'close' was sent 15 min earlier.

        Reported flow: windows were CLOSED after a (humidity-driven) close 15
        min ago, so LastNotificationType='close'. Conditions now favor opening
        (outdoor cool vs warmest indoor, good AQI, humidity below the gate), so
        the decision flips CLOSED→OPEN. Under the old 1-hour time cooldown this
        open transition was silently dropped; Option A sends it. The persisted
        record records LastNotificationType='open'.
        """
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        recent = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastNotificationTime": recent,
            "LastNotificationType": "close",
        }

        _evaluate_floor(
            "upstairs", ["sensor_up"], self._OPEN_SENSORS,
            self._OPEN_OUTDOOR, self._GOOD_AQI, "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        _, kwargs = mock_notify.call_args
        assert kwargs["urgent"] is False
        assert "🪟" in kwargs["title"] or "open" in kwargs["title"].lower()
        record = self._persisted_record(state_mgr)
        assert record["LastNotificationType"] == "open"

    @patch("src.orchestrator.send_notification")
    def test_sent_notification_persists_type_and_time(self, mock_notify):
        """A successful send persists BOTH LastNotificationType and LastNotificationTime."""
        engine = DecisionEngine(_base_config())
        state_mgr = MagicMock()
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state_mgr.get_floor_state.return_value = {
            "CurrentState": "OPEN",
            "LastNotificationTime": old,
            "LastNotificationType": "open",
        }

        _evaluate_floor(
            "upstairs", ["sensor_up"], self._CLOSE_SENSORS,
            self._CLOSE_OUTDOOR, self._GOOD_AQI, "cool", engine, state_mgr,
        )

        mock_notify.assert_called_once()
        record = self._persisted_record(state_mgr)
        assert record["LastNotificationType"] == "close"
        assert "LastNotificationTime" in record
        # Persisted time is a fresh ISO-8601 timestamp parseable back to a datetime.
        parsed = datetime.fromisoformat(record["LastNotificationTime"])
        assert parsed.tzinfo is not None


# ------------------------------------------------------------------
# Phase 2 — per-contributor observability log (Gregory's feature)
# ------------------------------------------------------------------


class TestOutdoorContributorLog:
    """``run_check`` finalizes and writes exactly one per-contributor record
    per poll (checklist #13, #14, #15).

    All tests drive ``run_check`` with **empty floor sensor lists** so the
    per-floor loop is skipped entirely — the contributor-log write happens in
    the outdoor-validation block *before* any floor is evaluated, so no AQI /
    decision-engine / snapshot machinery is needed. External clients are
    mocked; no network. ``_record_contributor_log`` (the only I/O) is patched
    so the assembled record can be inspected in-memory.
    """

    @staticmethod
    def _nws_outdoor(raw_temp):
        """A realistic NWS ``get_outdoor_conditions`` return value carrying a
        fetch-time ``contributor_log`` whose ``median_temp_f`` is the RAW
        (pre-validation) median."""
        return {
            "temperature_f": raw_temp,
            "humidity": 55.0,
            "wind_speed_mph": 5.0,
            "station_count": 3,
            "is_fallback": False,
            "used_cache": False,
            "source": "nws",
            "observation_time": "2026-07-11T13:35:00+00:00",
            "contributors": [
                {"station_id": "KAAA", "temperature_f": raw_temp},
                {"station_id": "CW1", "temperature_f": raw_temp + 1.0},
            ],
            "contributor_log": {
                "contributors": [
                    {
                        "station_id": "KAAA", "source_type": "nws_station",
                        "station_class": "official", "temp_f": raw_temp,
                        "obs_time": "2026-07-11T13:35:00+00:00", "age_minutes": 5.0,
                        "distance_mi": 1.0, "included_in_median": True,
                        "is_cached": False, "excluded_reason": None,
                    },
                ],
                "median_temp_f": raw_temp,  # RAW median — must survive to the record
                "real_station_count": 1,
                "openmeteo_present": False,
                "openmeteo_included": False,
                "used_cache_fallback": False,
                "is_fallback": False,
                "source": "nws",
                "selected_source_ids": ["KAAA"],
                "stickiness_active": False,
                "sticky_source_id": None,
            },
        }

    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_prior_selected_ids_read_from_last_outdoor_contributors(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_snap_cls,
    ):
        """The prior cycle's ``LastOutdoorContributors`` (``__global__``) keys are
        read and forwarded as ``sticky_source_ids`` with ``stickiness_enabled``
        from config (checklist #13). OPENMETEO is filtered downstream inside
        ``get_outdoor_conditions`` (see the NWS-client stickiness tests)."""
        mock_config.return_value = _base_config(
            upstairs_sensors=[], downstairs_sensors=[],
            outdoor_source_stickiness=True,
        )
        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {
            "CurrentState": "CLOSED",
            "LastOutdoorContributors": json.dumps(
                {"KAAA": 70.0, "CW1": 71.0, "OPENMETEO": 72.0}
            ),
        }
        mock_state_cls.return_value = mock_state

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = self._nws_outdoor(70.0)
        mock_get_nws.return_value = mock_nws

        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")
        mock_snap_cls.return_value = MagicMock()

        run_check()

        mock_nws.get_outdoor_conditions.assert_called_once()
        kwargs = mock_nws.get_outdoor_conditions.call_args.kwargs
        # All prior contributor ids are forwarded (OPENMETEO is dropped from the
        # NWS priority set inside get_outdoor_conditions, not here).
        assert set(kwargs["sticky_source_ids"]) >= {"KAAA", "CW1"}
        assert kwargs["stickiness_enabled"] is True

    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_stickiness_disabled_forwarded_from_config(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_snap_cls,
    ):
        """``outdoor_source_stickiness=False`` in config → ``stickiness_enabled``
        forwarded as False (checklist #13)."""
        mock_config.return_value = _base_config(
            upstairs_sensors=[], downstairs_sensors=[],
            outdoor_source_stickiness=False,
        )
        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {"CurrentState": "CLOSED"}
        mock_state_cls.return_value = mock_state

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = self._nws_outdoor(70.0)
        mock_get_nws.return_value = mock_nws

        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")
        mock_snap_cls.return_value = MagicMock()

        run_check()

        kwargs = mock_nws.get_outdoor_conditions.call_args.kwargs
        assert kwargs["stickiness_enabled"] is False
        # Cold start (no prior contributors) → empty sticky set forwarded.
        assert kwargs["sticky_source_ids"] == []

    @patch.object(NWSClient, "_record_contributor_log")
    @patch("src.orchestrator.validate_outdoor_temperature")
    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_exactly_one_record_after_validation_median_is_raw(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_snap_cls, mock_validate, mock_record,
    ):
        """One record per poll, written AFTER validation, carrying
        ``validation_reason`` / ``suppressed`` / ``raw_temp_f`` /
        ``validated_temp_f``; ``median_temp_f`` is the RAW pre-validation median
        (checklist #14)."""
        mock_config.return_value = _base_config(
            upstairs_sensors=[], downstairs_sensors=[],
        )
        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {"CurrentState": "CLOSED"}
        mock_state_cls.return_value = mock_state

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        # RAW median 75.0; validation SUPPRESSES it down to 68.0 so raw and
        # validated are distinct and median_temp_f can be proven to be the raw.
        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = self._nws_outdoor(75.0)
        mock_get_nws.return_value = mock_nws

        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")
        mock_snap_cls.return_value = MagicMock()

        mock_validate.return_value = OutdoorValidationResult(
            temperature_f=68.0,
            reason="suppressed_spike",
            suppressed=True,
            state_fields={"LastOutdoorTemp": 68.0},
        )

        run_check()

        mock_record.assert_called_once()
        record = mock_record.call_args.args[0]
        assert record["type"] == "outdoor_contributors"
        assert "poll_id" in record and "timestamp" in record
        assert record["validation_reason"] == "suppressed_spike"
        assert record["suppressed"] is True
        assert record["raw_temp_f"] == 75.0
        assert record["validated_temp_f"] == 68.0
        # The record's median is the RAW (pre-validation) median, NOT 68.0.
        assert record["median_temp_f"] == 75.0

    @patch.object(NWSClient, "_record_contributor_log")
    @patch("src.orchestrator.validate_outdoor_temperature")
    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_om_only_fallback_emits_synthesized_single_contributor(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_snap_cls, mock_validate, mock_record,
    ):
        """NWS fails + fresh OM peer → sole-source path emits one synthesized
        single-contributor record for OPENMETEO (checklist #15)."""
        mock_config.return_value = _base_config(
            upstairs_sensors=[], downstairs_sensors=[],
        )
        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {"CurrentState": "CLOSED"}
        mock_state_cls.return_value = mock_state

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        # NWS unavailable — the fresh OM peer becomes the sole outdoor source
        # (inline dict with no fetch-time contributor_log → synthesized).
        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.side_effect = NWSError("NWS down")
        mock_get_nws.return_value = mock_nws

        mock_om = MagicMock()
        mock_om.get_observation.return_value = {
            "station_id": "OPENMETEO",
            "temperature_f": 66.0,
            "humidity": 60.0,
            "wind_speed_mph": 5.0,
            "timestamp": datetime.now(timezone.utc),
        }
        mock_om_cls.return_value = mock_om
        mock_snap_cls.return_value = MagicMock()

        mock_validate.return_value = OutdoorValidationResult(
            temperature_f=66.0,
            reason="cold_start",
            suppressed=False,
            state_fields={},
        )

        run_check()

        mock_record.assert_called_once()
        record = mock_record.call_args.args[0]
        assert record["openmeteo_present"] is True
        assert record["real_station_count"] == 0
        assert record["selected_source_ids"] == ["OPENMETEO"]
        assert record["stickiness_active"] is False
        contributors = record["contributors"]
        assert len(contributors) == 1
        only = contributors[0]
        assert only["station_id"] == "OPENMETEO"
        assert only["source_type"] == "openmeteo"
        assert only["included_in_median"] is True
        # Synthesized median mirrors the raw OM temperature.
        assert record["median_temp_f"] == 66.0
        assert record["raw_temp_f"] == 66.0


# ------------------------------------------------------------------
# Outdoor signal-confidence gate — orchestrator wiring
# ------------------------------------------------------------------


class TestOutdoorConfidenceIntegration:
    """``run_check`` threads the confidence params + ``contributor_log`` into
    the validator, stamps ``outdoor["confidence_reason"]``, and carries
    ``confidence_reason`` / ``confident`` on the per-contributor record.

    Same technique as ``TestOutdoorContributorLog``: empty floor sensor lists
    so only the outdoor-validation block runs; external clients mocked; the
    single I/O (``_record_contributor_log``) patched for in-memory inspection.
    """

    @staticmethod
    def _nws_outdoor(raw_temp):
        """NWS ``get_outdoor_conditions`` return value with a contributor_log."""
        return {
            "temperature_f": raw_temp,
            "humidity": 55.0,
            "wind_speed_mph": 5.0,
            "station_count": 2,
            "is_fallback": False,
            "used_cache": False,
            "source": "nws",
            "observation_time": "2026-07-11T14:45:00+00:00",
            "contributors": [
                {"station_id": "KAAA", "temperature_f": raw_temp},
                {"station_id": "KCCC", "temperature_f": raw_temp + 1.0},
            ],
            "contributor_log": {
                "contributors": [
                    {
                        "station_id": "KAAA", "source_type": "nws_station",
                        "station_class": "official", "temp_f": raw_temp,
                        "obs_time": "2026-07-11T14:45:00+00:00", "age_minutes": 5.0,
                        "distance_mi": 1.0, "included_in_median": True,
                        "is_cached": False, "excluded_reason": None,
                    },
                ],
                "median_temp_f": raw_temp,
                "real_station_count": 2,
                "openmeteo_present": False,
                "openmeteo_included": False,
                "used_cache_fallback": False,
                "is_fallback": False,
                "source": "nws",
                "selected_source_ids": ["KAAA", "KCCC"],
                "stickiness_active": False,
                "sticky_source_id": None,
            },
        }

    @patch.object(NWSClient, "_record_contributor_log")
    @patch("src.orchestrator.validate_outdoor_temperature")
    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_validator_receives_contributor_log_and_confidence_params(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_snap_cls, mock_validate, mock_record,
    ):
        """The validator call gets the fetch-time ``contributor_log`` plus the
        four confidence params sourced from config (checklist: signal gate)."""
        mock_config.return_value = _base_config(
            upstairs_sensors=[], downstairs_sensors=[],
            outdoor_min_corroborating_sources=3,
            outdoor_confidence_max_spread_f=4.0,
            outdoor_confidence_hold_max_cycles=5,
            outdoor_confidence_enabled=False,
        )
        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {"CurrentState": "CLOSED"}
        mock_state_cls.return_value = mock_state

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        outdoor = self._nws_outdoor(70.0)
        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = outdoor
        mock_get_nws.return_value = mock_nws

        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")
        mock_snap_cls.return_value = MagicMock()

        mock_validate.return_value = OutdoorValidationResult(
            temperature_f=70.0,
            reason="within_threshold",
            suppressed=False,
            state_fields={},
        )

        run_check()

        mock_validate.assert_called_once()
        kwargs = mock_validate.call_args.kwargs
        # The fetch-time contributor_log is forwarded verbatim.
        assert kwargs["contributor_log"] is outdoor["contributor_log"]
        # Confidence tuning params come from config.
        assert kwargs["min_corroborating_sources"] == 3
        assert kwargs["confidence_max_spread_f"] == pytest.approx(4.0)
        assert kwargs["confidence_hold_max_cycles"] == 5
        assert kwargs["confidence_enabled"] is False

    @patch.object(NWSClient, "_record_contributor_log")
    @patch("src.orchestrator.validate_outdoor_temperature")
    @patch("src.orchestrator.SnapshotManager")
    @patch("src.orchestrator.OpenMeteoClient")
    @patch("src.orchestrator._get_nws_client")
    @patch("src.orchestrator.EcobeeClient")
    @patch("src.orchestrator.get_state_manager")
    @patch("src.orchestrator.get_config")
    def test_record_carries_confidence_reason_and_confident(
        self, mock_config, mock_state_cls, mock_ecobee_cls,
        mock_get_nws, mock_om_cls, mock_snap_cls, mock_validate, mock_record,
    ):
        """The per-contributor record carries ``confidence_reason`` / ``confident``
        straight from the validation result (a HELD artifact here)."""
        mock_config.return_value = _base_config(
            upstairs_sensors=[], downstairs_sensors=[],
        )
        mock_state = MagicMock()
        mock_state.get_floor_state.return_value = {"CurrentState": "CLOSED"}
        mock_state_cls.return_value = mock_state

        mock_ecobee = MagicMock()
        mock_ecobee.get_sensors.return_value = []
        mock_ecobee.get_hvac_mode.return_value = "cool"
        mock_ecobee_cls.return_value = mock_ecobee

        mock_nws = MagicMock()
        mock_nws.get_outdoor_conditions.return_value = self._nws_outdoor(71.6)
        mock_get_nws.return_value = mock_nws

        mock_om_cls.return_value.get_observation.side_effect = OpenMeteoError("no peer")
        mock_snap_cls.return_value = MagicMock()

        # A confidence HOLD: validated back to the last confident value, with
        # confident=False and the held reason.
        mock_validate.return_value = OutdoorValidationResult(
            temperature_f=69.7,
            reason="held_uncorroborated_churn",
            suppressed=True,
            state_fields={"LastOutdoorTemp": 69.7},
            confident=False,
            confidence_reason="held_uncorroborated_churn",
        )

        run_check()

        mock_record.assert_called_once()
        record = mock_record.call_args.args[0]
        assert record["confidence_reason"] == "held_uncorroborated_churn"
        assert record["confident"] is False
        assert record["suppressed"] is True
        assert record["validated_temp_f"] == 69.7
        assert record["raw_temp_f"] == 71.6


# ------------------------------------------------------------------
# Observation-time normalization (Gregory's two-layer fix — writer layer)
#
# AirNow's Current Observations endpoint returns a human-readable
# observation_time ("2026-07-13 18:00 PDT" — local wall-clock + a bare TZ
# abbreviation) that datetime.fromisoformat cannot parse. The snapshot
# *_observation_time fields are re-parsed as ISO by the status page, so
# _normalize_observation_time converts the AirNow shape into a tz-aware ISO
# string with a numeric offset before it is ever persisted. This is the
# writer-side half of the fix that keeps a display string out of an
# ISO-typed field.
# ------------------------------------------------------------------


class TestNormalizeObservationTime:
    """Unit tests for ``_normalize_observation_time`` and its offset table."""

    def test_airnow_pdt_maps_to_minus_07_00_offset(self):
        from src.orchestrator import _normalize_observation_time

        assert (
            _normalize_observation_time("2026-07-13 18:00 PDT")
            == "2026-07-13T18:00:00-07:00"
        )

    def test_airnow_edt_maps_to_minus_04_00_offset(self):
        from src.orchestrator import _normalize_observation_time, _TZ_ABBREV_OFFSETS

        # Guard against the offset table drifting: EDT is -4 by definition.
        assert _TZ_ABBREV_OFFSETS["EDT"] == -4
        assert (
            _normalize_observation_time("2026-07-13 18:00 EDT")
            == "2026-07-13T18:00:00-04:00"
        )

    def test_hour_only_time_part_is_padded(self):
        from src.orchestrator import _normalize_observation_time

        # "HH" with no minutes/seconds → padded to HH:00:00.
        assert (
            _normalize_observation_time("2026-07-13 18 PDT")
            == "2026-07-13T18:00:00-07:00"
        )

    def test_hour_minute_time_part_is_padded(self):
        from src.orchestrator import _normalize_observation_time

        # "HH:MM" → padded to HH:MM:00.
        assert (
            _normalize_observation_time("2026-07-13 18:05 PDT")
            == "2026-07-13T18:05:00-07:00"
        )

    def test_hour_minute_second_time_part_preserved(self):
        from src.orchestrator import _normalize_observation_time

        # "HH:MM:SS" → seconds preserved.
        assert (
            _normalize_observation_time("2026-07-13 18:05:42 PDT")
            == "2026-07-13T18:05:42-07:00"
        )

    def test_already_iso_with_trailing_z_passes_through_unchanged(self):
        from src.orchestrator import _normalize_observation_time

        assert (
            _normalize_observation_time("2026-07-13T18:00:00Z")
            == "2026-07-13T18:00:00Z"
        )

    def test_already_iso_with_numeric_offset_passes_through_unchanged(self):
        from src.orchestrator import _normalize_observation_time

        assert (
            _normalize_observation_time("2026-07-13T18:00:00-07:00")
            == "2026-07-13T18:00:00-07:00"
        )

    def test_unknown_abbreviation_returns_none(self):
        from src.orchestrator import _normalize_observation_time

        assert _normalize_observation_time("2026-07-13 18:00 XYZ") is None

    @pytest.mark.parametrize("value", ["garbage", "", None])
    def test_junk_empty_and_none_return_none(self, value):
        from src.orchestrator import _normalize_observation_time

        assert _normalize_observation_time(value) is None


class TestBuildFloorSnapshotNormalizesObservationTime:
    """``_build_floor_snapshot`` must persist a re-parseable ISO string, never
    the raw AirNow display string, into ``aqi_observation_time``."""

    def test_airnow_display_string_round_trips_through_fromisoformat(self):
        from src.orchestrator import _build_floor_snapshot

        decision = FloorDecision(
            floor="upstairs",
            new_state="CLOSED",
            reason="test",
            urgent=False,
            changed=False,
        )
        outdoor = {
            "temperature_f": 65.0,
            "source": "nws",
            "humidity": 50.0,
            "station_count": 0,
            "observation_time": "2026-07-13 18:00 PDT",
        }
        aqi_data = {
            "aqi": 25,
            "source": "airnow",
            "sensor_count": 0,
            "observation_time": "2026-07-13 18:00 PDT",
        }
        engine = DecisionEngine(_base_config())
        now = datetime(2026, 7, 13, 19, 20, 0, tzinfo=timezone.utc)

        # Gate reconstruction is orthogonal to timestamp handling; stub it so
        # the test stays focused on the observation-time writer.
        with patch("src.orchestrator._evaluate_gates", return_value=[]):
            snap = _build_floor_snapshot(
                "upstairs",
                ["sensor_up"],
                [{"name": "sensor_up", "temperature_f": 70.0, "is_online": True}],
                decision,
                outdoor,
                aqi_data,
                engine,
                {},
                now,
            )

        # The persisted value is a string that fromisoformat CAN parse —
        # proving the writer no longer stores AirNow's display string.
        assert isinstance(snap.aqi_observation_time, str)
        parsed = datetime.fromisoformat(snap.aqi_observation_time)
        assert parsed.tzinfo is not None
        assert snap.aqi_observation_time == "2026-07-13T18:00:00-07:00"


