"""WindowBot decision engine.

Implements the per-floor open/close algorithm described in the architecture
spec (Section 4) with all user-specified refinements:

- Asymmetric hysteresis:
    • OPEN side keeps a 1 °F open hysteresis (outdoor must be >1 °F cooler
      than the warmest indoor temp before opening)
    • CLOSE side has **no** hysteresis — the close fires as soon as the
      outdoor temp exceeds the coolest indoor temp
- Bidirectional AQI gating:
    • AQI ≥ 100 → CLOSE immediately (urgent, bypasses cooldown)
    • AQI 50–99 → neutral (no AQI-driven state change)
    • AQI < 50 → allow opening (if temperature conditions favour it)
- Humidity gate: block opening if outdoor humidity > 80 %
- Comfort gate: skip opening when indoor temps are already comfortable
- HVAC mode gate: only act when in cooling/auto modes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("windowbot.decision")


class InsufficientDataError(Exception):
    """Raised when there aren't enough valid sensor readings to decide."""


# ------------------------------------------------------------------
# Result data-class
# ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FloorDecision:
    """The outcome of the decision engine for one floor.

    Attributes:
        floor: Floor identifier (e.g. ``"upstairs"``).
        new_state: Target window state — ``"OPEN"`` or ``"CLOSED"``.
        reason: Human-readable explanation for the decision.
        urgent: ``True`` when AQI triggers an immediate close
            (bypasses notification cooldown).
        changed: ``True`` when *new_state* differs from *last_state*.
    """

    floor: str
    new_state: str
    reason: str
    urgent: bool
    changed: bool


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class DecisionEngine:
    """Core algorithm that decides whether windows should be open or closed.

    All thresholds are read from the config dict produced by
    :func:`src.config.get_config`.
    """

    def __init__(self, config: dict) -> None:
        self.hysteresis_open: float = float(config.get("hysteresis_open_diff", 1.0))
        self.hysteresis_close: float = float(config.get("hysteresis_close_diff", 1.0))
        self.max_humidity: int = int(config.get("max_outdoor_humidity", 80))
        self.aqi_close_threshold: int = int(config.get("max_aqi_threshold", 100))
        self.aqi_open_threshold: int = int(config.get("min_aqi_for_opening", 50))
        self.allowed_hvac_modes: list[str] = list(
            config.get("allowed_hvac_modes", ["cool", "heatCool", "auto"])
        )
        self.comfort_temp_max: float = float(config.get("comfort_temp_max", 72.0))
        self.enable_humidity_gate: bool = config.get("enable_humidity_gate", True)
        self.enable_aqi_gate: bool = config.get("enable_aqi_gate", True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def decide(
        self,
        floor: str,
        floor_sensors: list[dict],
        outdoor: dict,
        aqi: dict,
        hvac_mode: str,
        last_state: str,
        floor_group: list[str],
    ) -> FloorDecision:
        """Run the full gate-chain for a single floor.

        Args:
            floor: Floor identifier (``"upstairs"`` / ``"downstairs"``).
            floor_sensors: List of sensor dicts from
                :meth:`EcobeeClient.get_sensors`.
            outdoor: Outdoor conditions from
                :meth:`NWSClient.get_outdoor_conditions`.
            aqi: AQI result from PurpleAir or AirNow client.
            hvac_mode: Current HVAC mode string from Ecobee.
            last_state: Previous window state (``"OPEN"`` / ``"CLOSED"`` /
                ``"UNKNOWN"``).
            floor_group: Sensor names assigned to this floor.

        Returns:
            A :class:`FloorDecision` describing the recommended action.
        """
        # Normalise unknown/initial state to CLOSED (safe default).
        if last_state not in ("OPEN", "CLOSED"):
            last_state = "CLOSED"

        outdoor_temp: float = outdoor["temperature_f"]
        outdoor_humidity: float | None = outdoor.get("humidity")
        aqi_value: int = aqi.get("aqi", 0)

        # ---- Gate 1: HVAC mode ----
        if hvac_mode not in self.allowed_hvac_modes:
            reason = f"HVAC mode '{hvac_mode}' not in allowed modes {self.allowed_hvac_modes}"
            return self._keep(floor, last_state, reason)

        # ---- Gate 2: AQI (bidirectional) ----
        if self.enable_aqi_gate:
            aqi_decision = self._check_aqi(floor, aqi_value, last_state)
            if aqi_decision is not None:
                return aqi_decision

        # ---- Gate 3: Humidity ----
        if self.enable_humidity_gate and outdoor_humidity is not None:
            humidity_decision = self._check_humidity(
                floor, outdoor_humidity, last_state
            )
            if humidity_decision is not None:
                return humidity_decision

        # ---- Gate 4: Comfort threshold ----
        if last_state == "CLOSED":
            try:
                warmest_pre, _ = self.get_floor_temps(floor_sensors, floor_group)
                if warmest_pre <= self.comfort_temp_max:
                    return self._keep(
                        floor,
                        "CLOSED",
                        f"Indoor already comfortable ({warmest_pre:.1f}°F ≤ {self.comfort_temp_max:.1f}°F)",
                    )
            except InsufficientDataError:
                pass  # let temperature logic handle the error

        # ---- Temperature logic with hysteresis ----
        try:
            warmest, coolest = self.get_floor_temps(floor_sensors, floor_group)
        except InsufficientDataError as exc:
            reason = f"Insufficient sensor data: {exc}"
            return self._keep(floor, last_state, reason)

        return self._temperature_decision(
            floor, outdoor_temp, warmest, coolest, aqi_value, last_state
        )

    # ------------------------------------------------------------------
    # Pre-check: is AQI data needed?
    # ------------------------------------------------------------------

    def needs_aqi(
        self,
        floor_sensors: list[dict],
        outdoor: dict,
        hvac_mode: str,
        last_state: str,
        floor_group: list[str],
    ) -> tuple[bool, str]:
        """Determine whether AQI data is needed before calling :meth:`decide`.

        Runs the non-AQI gates (HVAC mode, humidity, comfort, temperature)
        to see if AQI could influence the final outcome.  This lets the
        orchestrator skip PurpleAir/AirNow calls when they're irrelevant.

        Returns:
            ``(True, reason)`` if AQI should be fetched.
            ``(False, reason)`` if AQI can safely be skipped.
        """
        if not self.enable_aqi_gate:
            return False, "AQI gate disabled"

        if last_state not in ("OPEN", "CLOSED"):
            last_state = "CLOSED"

        # Windows OPEN → always need AQI (urgent close safety net)
        if last_state == "OPEN":
            return True, "windows open — need AQI for safety check"

        # ---- Windows CLOSED: would non-AQI gates even allow opening? ----

        if hvac_mode not in self.allowed_hvac_modes:
            return False, f"HVAC mode '{hvac_mode}' not in allowed modes"

        outdoor_humidity: float | None = outdoor.get("humidity")
        if self.enable_humidity_gate and outdoor_humidity is not None:
            if outdoor_humidity > self.max_humidity:
                return False, f"outdoor humidity too high ({outdoor_humidity:.0f}%)"

        try:
            warmest, _ = self.get_floor_temps(floor_sensors, floor_group)
        except InsufficientDataError:
            return False, "insufficient sensor data — decision won't depend on AQI"

        if warmest <= self.comfort_temp_max:
            return False, (
                f"indoor already comfortable ({warmest:.1f}°F "
                f"≤ {self.comfort_temp_max:.1f}°F)"
            )

        outdoor_temp: float = outdoor["temperature_f"]
        open_threshold = warmest - self.hysteresis_open
        if outdoor_temp >= open_threshold:
            return False, (
                f"outdoor not cool enough to open ({outdoor_temp:.1f}°F "
                f"≥ {open_threshold:.1f}°F)"
            )

        return True, "conditions favor opening — need AQI to confirm safe"

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _check_aqi(
        self, floor: str, aqi_value: int, last_state: str
    ) -> FloorDecision | None:
        """Gate 2 — bidirectional AQI check.

        Returns a FloorDecision if AQI alone determines the outcome,
        or ``None`` to continue to the next gate.
        """
        if aqi_value >= self.aqi_close_threshold:
            # Urgent close.
            if last_state == "OPEN":
                return FloorDecision(
                    floor=floor,
                    new_state="CLOSED",
                    reason=f"AQI unhealthy ({aqi_value}) — URGENT CLOSE",
                    urgent=True,
                    changed=True,
                )
            # Already closed — reinforce.
            return FloorDecision(
                floor=floor,
                new_state="CLOSED",
                reason=f"AQI prevents opening ({aqi_value})",
                urgent=False,
                changed=False,
            )

        if self.aqi_open_threshold <= aqi_value < self.aqi_close_threshold:
            # 50–99: neutral zone — no AQI-driven change.  Continue to
            # temperature logic but *block opening*.  If already open,
            # temperature may close; if closed, AQI prevents opening.
            if last_state == "CLOSED":
                return FloorDecision(
                    floor=floor,
                    new_state="CLOSED",
                    reason=f"AQI borderline ({aqi_value}) — maintaining closed",
                    urgent=False,
                    changed=False,
                )
            # If OPEN, let the temperature gate decide whether to close.
            return None

        # AQI < 50: good — continue to next gate.
        return None

    def _check_humidity(
        self, floor: str, outdoor_humidity: float, last_state: str
    ) -> FloorDecision | None:
        """Gate 3 — outdoor humidity check.

        Returns a FloorDecision if humidity blocks window changes,
        or ``None`` to continue.
        """
        if outdoor_humidity > self.max_humidity:
            if last_state == "OPEN":
                return FloorDecision(
                    floor=floor,
                    new_state="CLOSED",
                    reason=f"Outdoor humidity too high ({outdoor_humidity:.0f}%)",
                    urgent=False,
                    changed=True,
                )
            return FloorDecision(
                floor=floor,
                new_state="CLOSED",
                reason=f"Humidity prevents opening ({outdoor_humidity:.0f}%)",
                urgent=False,
                changed=False,
            )
        return None

    # ------------------------------------------------------------------
    # Temperature logic
    # ------------------------------------------------------------------

    def _temperature_decision(
        self,
        floor: str,
        outdoor_temp: float,
        warmest: float,
        coolest: float,
        aqi_value: int,
        last_state: str,
    ) -> FloorDecision:
        """Apply temperature comparison with open-side-only hysteresis.

        - CLOSED → OPEN: outdoor < warmest − hysteresis_open **and** AQI < 50
        - OPEN → CLOSED: outdoor > coolest (no close hysteresis)
        """
        if last_state == "CLOSED":
            open_threshold = warmest - self.hysteresis_open
            if outdoor_temp < open_threshold and aqi_value < self.aqi_open_threshold:
                return FloorDecision(
                    floor=floor,
                    new_state="OPEN",
                    reason=(
                        f"Outdoor ({outdoor_temp:.1f}°F) cooler than warmest indoor "
                        f"({warmest:.1f}°F) by >{self.hysteresis_open:.1f}°F, "
                        f"AQI good ({aqi_value})"
                    ),
                    urgent=False,
                    changed=True,
                )
            return FloorDecision(
                floor=floor,
                new_state="CLOSED",
                reason=(
                    f"Not cool enough to open (need <{open_threshold:.1f}°F, "
                    f"have {outdoor_temp:.1f}°F) or AQI not good enough ({aqi_value})"
                ),
                urgent=False,
                changed=False,
            )

        # last_state == "OPEN"
        if outdoor_temp > coolest:
            return FloorDecision(
                floor=floor,
                new_state="CLOSED",
                reason=(
                    f"Outdoor ({outdoor_temp:.1f}°F) warmer than coolest indoor "
                    f"({coolest:.1f}°F)"
                ),
                urgent=False,
                changed=True,
            )

        return FloorDecision(
            floor=floor,
            new_state="OPEN",
            reason=(
                f"Still beneficial (outdoor {outdoor_temp:.1f}°F "
                f"<= coolest indoor {coolest:.1f}°F)"
            ),
            urgent=False,
            changed=False,
        )

    # ------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_floor_temps(
        sensors: list[dict], floor_group: list[str]
    ) -> tuple[float, float]:
        """Extract warmest and coolest temperatures for a floor.

        Args:
            sensors: All sensor dicts from
                :meth:`EcobeeClient.get_sensors`.
            floor_group: Names of sensors that belong to this floor.

        Returns:
            ``(warmest, coolest)`` temperatures in °F.

        Raises:
            InsufficientDataError: If no valid sensors are found on the floor.
        """
        valid_temps: list[float] = []
        for sensor in sensors:
            if sensor["name"] not in floor_group:
                continue
            temp = sensor.get("temperature_f")
            if temp is not None:
                valid_temps.append(temp)

        if not valid_temps:
            raise InsufficientDataError(
                f"No valid sensors in floor group {floor_group}"
            )

        return max(valid_temps), min(valid_temps)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def _keep(floor: str, state: str, reason: str) -> FloorDecision:
        """Return a decision that maintains the current state."""
        return FloorDecision(
            floor=floor,
            new_state=state,
            reason=reason,
            urgent=False,
            changed=False,
        )
