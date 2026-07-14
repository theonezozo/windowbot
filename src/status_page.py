"""Status page endpoint for WindowBot.

Renders last persisted state as HTML or JSON, showing exactly what
WindowBot decided during its last poll cycle.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import azure.functions as func

from src.state import get_state_manager
from src.diagnostic import (
    SnapshotManager,
    FloorSnapshot,
    GlobalSnapshot,
    TemperatureHistoryEntry,
)
from src.version_info import (
    VERSION,
    WORKER_STARTED_AT,
    is_dev_build,
    parse_iso_utc,
)

logger = logging.getLogger("windowbot.status")


def _pin_denied_response(is_json: bool) -> func.HttpResponse:
    if is_json:
        return func.HttpResponse(
            json.dumps({"error": "Unauthorized"}),
            status_code=401,
            mimetype="application/json",
        )
    return func.HttpResponse(
        "<html><body><h1>401 Unauthorized</h1><p>Missing or incorrect PIN.</p></body></html>",
        status_code=401,
        mimetype="text/html",
    )


def render_status_page(req: func.HttpRequest) -> func.HttpResponse:
    """Render the status page as HTML or JSON.

    Args:
        req: HTTP request with optional ?format=json and ?pin=<pin> query params

    Returns:
        HTTP response with HTML (default) or JSON
    """
    # Determine output format
    accept_header = req.headers.get("Accept", "")
    format_param = req.params.get("format", "")
    wants_json = "application/json" in accept_header or format_param == "json"

    # PIN check — required when STATUS_PAGE_PIN is configured
    required_pin = os.environ.get("STATUS_PAGE_PIN", "").strip()
    if required_pin:
        provided_pin = req.params.get("pin", "").strip()
        if provided_pin != required_pin:
            return _pin_denied_response(wants_json)

    try:
        state_mgr = get_state_manager()

        # Check if snapshot table is available. The snapshot table only exists
        # in Azure Table Storage mode; when the app is running on (or has fallen
        # back to) local file state, ``get_snapshot_table`` is either absent or
        # raises ``NotImplementedError``. Both cases mean "no diagnostic data
        # source" and must render the friendly message — never a 500.
        get_snapshot_table = getattr(state_mgr, "get_snapshot_table", None)
        if get_snapshot_table is None:
            return _no_data_response(
                "Status page requires Azure Table Storage (not available in local state mode).",
                is_json=wants_json
            )

        try:
            snapshot_table = get_snapshot_table()
        except NotImplementedError:
            return _no_data_response(
                "Status page requires Azure Table Storage (not available in local state mode).",
                is_json=wants_json
            )
        mgr = SnapshotManager(snapshot_table)
        
        # Fetch snapshots
        floor_snapshots = mgr.get_all_floor_snapshots()
        global_snapshot = mgr.get_global_snapshot()

        # Fetch history (best-effort; absence must not break the page).
        try:
            history = mgr.get_temperature_history(hours=12)
        except Exception:
            logger.exception("Failed to fetch temperature history.")
            history = []

        if not floor_snapshots and not global_snapshot:
            return _no_data_response(
                "No diagnostic data available yet. WindowBot will populate this page after its first poll cycle.",
                is_json=wants_json
            )
        
        # Render as JSON or HTML
        if wants_json:
            return _render_json(floor_snapshots, global_snapshot, history)
        else:
            return _render_html(floor_snapshots, global_snapshot, history)
    
    except Exception as exc:
        logger.exception("Failed to render status page.")
        return func.HttpResponse(
            f"Error loading status: {exc}",
            status_code=500,
            mimetype="text/plain"
        )


def _no_data_response(message: str, is_json: bool) -> func.HttpResponse:
    """Return a friendly 'no data yet' response."""
    if is_json:
        return func.HttpResponse(
            json.dumps({"error": message}),
            status_code=200,
            mimetype="application/json"
        )
    else:
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WindowBot Status</title>
    <style>{_get_css()}</style>
</head>
<body>
    <div class="container">
        <h1>🪟 WindowBot Status</h1>
        <div class="card">
            <p><strong>{html.escape(message)}</strong></p>
        </div>
    </div>
</body>
</html>
"""
        return func.HttpResponse(html_content, status_code=200, mimetype="text/html")


def _render_json(
    floor_snapshots: list[FloorSnapshot],
    global_snapshot: GlobalSnapshot | None,
    history: list[TemperatureHistoryEntry] | None = None,
) -> func.HttpResponse:
    """Render status as JSON."""
    now = datetime.now(timezone.utc)
    tz = _get_display_tz()
    data = {
        "version": {
            **VERSION,
            "is_dev_build": is_dev_build(),
            "worker_started_at": WORKER_STARTED_AT.isoformat(),
            "worker_uptime_seconds": int((now - WORKER_STARTED_AT).total_seconds()),
        },
        "local_timezone": str(tz),
        "page_loaded_utc": now.isoformat(),
        "page_loaded_local": now.astimezone(tz).isoformat(),
        "global": json.loads(global_snapshot.to_json()) if global_snapshot else None,
        "floors": {s.floor: json.loads(s.to_json()) for s in floor_snapshots},
        "history": [json.loads(h.to_json()) for h in (history or [])],
    }
    return func.HttpResponse(
        json.dumps(data, indent=2),
        status_code=200,
        mimetype="application/json"
    )


def _render_html(
    floor_snapshots: list[FloorSnapshot],
    global_snapshot: GlobalSnapshot | None,
    history: list[TemperatureHistoryEntry] | None = None,
) -> func.HttpResponse:
    """Render status as HTML."""
    now = datetime.now(timezone.utc)
    tz = _get_display_tz()

    # --- Cycle timing -------------------------------------------------------
    # The timer fires every POLLING_INTERVAL_MINUTES minutes.  Add 90 s of
    # slop so the page is refreshed only after the new snapshot is likely
    # written (cold-start + poll latency can be ~60 s on the consumption plan).
    poll_interval_minutes = int(os.environ.get("POLLING_INTERVAL_MINUTES", "10"))
    slop_seconds = 90

    # Seconds until the top of the next N-minute window, plus slop
    seconds_since_epoch = now.timestamp()
    interval_seconds = poll_interval_minutes * 60
    seconds_into_interval = seconds_since_epoch % interval_seconds
    seconds_until_next = interval_seconds - seconds_into_interval
    refresh_seconds = math.ceil(seconds_until_next) + slop_seconds
    
    # Calculate data freshness
    freshness_html = ""
    freshness_class = "fresh"
    if global_snapshot:
        poll_time = _parse_ts(global_snapshot.poll_start)
        if poll_time is not None:
            age_minutes = (now - poll_time).total_seconds() / 60
            if age_minutes > 20:
                freshness_class = "stale"
            elif age_minutes > 12:
                freshness_class = "warning"

            freshness_html = f"""
        <div class="freshness {freshness_class}">
            <strong>Last poll:</strong> {_format_age(poll_time)} ago
            {f'<span class="freshness-warning">⚠️ Data may be stale</span>' if age_minutes > 20 else ''}
        </div>
        """
    
    # Global header
    header_html = ""
    if global_snapshot:
        next_poll_time = _parse_ts(global_snapshot.next_poll_eta)
        next_poll_in = (
            _format_age(next_poll_time, future=True)
            if next_poll_time is not None
            else "unknown"
        )
        
        header_html = f"""
        <div class="global-info">
            <div class="info-row">
                <span class="label">HVAC Mode:</span>
                <span class="value">{html.escape(global_snapshot.hvac_mode)}</span>
            </div>
            <div class="info-row">
                <span class="label">Quiet Hours:</span>
                <span class="value badge badge-{'active' if global_snapshot.quiet_hours_active else 'inactive'}">
                    {'Active' if global_snapshot.quiet_hours_active else 'Inactive'}
                </span>
            </div>
            <div class="info-row">
                <span class="label">Next Poll:</span>
                <span class="value">{next_poll_in}</span>
            </div>
            <div class="info-row">
                <span class="label">Poll Duration:</span>
                <span class="value">{global_snapshot.poll_duration_seconds:.1f}s</span>
            </div>
        </div>
        """
        
        if global_snapshot.errors:
            header_html += '<div class="errors"><strong>Errors:</strong><ul>'
            for err in global_snapshot.errors:
                header_html += f'<li>{html.escape(err)}</li>'
            header_html += '</ul></div>'
    
    # Floor cards
    sorted_floors = sorted(floor_snapshots, key=lambda s: s.floor)

    # Shared environmental conditions (outdoor + AQI are the same across floors,
    # since they come from the same coordinates).
    environment_html = ""
    if sorted_floors:
        environment_html = _render_environment_section(sorted_floors[0])

    floors_html = ""
    for snapshot in sorted_floors:
        floors_html += _render_floor_card(snapshot)

    history_html = _render_history_card(history or [], tz)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="{refresh_seconds}">
    <!-- DEPLOY-CHECK-v2 -->
    <title>WindowBot Status</title>
    <style>{_get_css()}</style>
</head>
<body>
    <div class="container">
        <h1>🪟 WindowBot Status</h1>
        {freshness_html}
        {header_html}
        {environment_html}
        {floors_html}
        {history_html}
        <div class="footer">
            {_render_build_info()}
            <p>Page loaded: {_format_local_and_utc(now, tz)} · auto-refreshes in {refresh_seconds}s</p>
            <p><a href="?format=json">View as JSON</a></p>
        </div>
    </div>
</body>
</html>
"""
    # Cache headers: expire at the next cycle boundary (without slop) so
    # shared caches don't serve a stale page after a new poll has fired.
    expires_seconds = math.ceil(seconds_until_next)
    from email.utils import formatdate
    expires_ts = now.timestamp() + expires_seconds
    headers = {
        "Cache-Control": f"public, max-age={expires_seconds}",
        "Expires": formatdate(timeval=expires_ts, usegmt=True),
    }

    return func.HttpResponse(
        html_content,
        status_code=200,
        mimetype="text/html",
        headers=headers,
    )


_AQI_PROVIDER_LABELS = {
    "airnow": "AirNow",
    "purpleair": "PurpleAir",
}


def _aqi_value_class(value: int) -> str:
    """AQI colour bucket used by the status page: good < 50, moderate < 100."""
    return "good" if value < 50 else ("moderate" if value < 100 else "unhealthy")


def _render_aqi_block(snapshot: FloorSnapshot, aqi_class: str, aqi_obs_age: str) -> str:
    """Render the Air Quality metrics.

    When both AirNow and PurpleAir were queried this cycle
    (``snapshot.aqi_readings`` holds two non-None values) both are shown, each
    labeled, and the authoritative provider (``snapshot.aqi_source`` — the value
    that drove the open/close decision) is marked "driving decision". When only
    one provider was checked, only that one is rendered — never a
    "PurpleAir: None" line. Snapshots predating the dual-reading field
    (``aqi_readings is None``) fall back to the original single AQI + Source view.

    When NEITHER provider produced a reading this cycle (``aqi_source ==
    "none"``), the per-provider failure reasons in ``snapshot.aqi_reasons`` are
    rendered ("AirNow: no stations in range", "PurpleAir: 402 out of points")
    so a config/account gap is visible instead of a bare "Source: none".

    When AQI was intentionally NOT fetched this cycle (``aqi_source ==
    "skipped"`` — the cost-skip working as designed because the floor couldn't
    open regardless), the human-readable ``snapshot.aqi_skip_reason`` is shown
    as "not checked — <reason>" so the skip reads as intentional, never as a
    failed "AQI 0" reading.
    """
    # AQI intentionally not fetched this cycle (cost-skip). The floor couldn't
    # open regardless (e.g. outdoor not cool enough, HVAC off, already
    # comfortable), so the precise AQI can't change the decision. Show the
    # reason instead of a misleading "AQI 0, source: skipped".
    if snapshot.aqi_source == "skipped":
        skip_reason = getattr(snapshot, "aqi_skip_reason", None) or "not needed this cycle"
        return f"""
    {aqi_obs_age}
    <div class="metric">
        <span class="label">AQI:</span>
        <span class="value">not checked \u2014 {html.escape(str(skip_reason))}</span>
    </div>
    """

    # No AQI this cycle — both providers empty. Explain WHY per provider so a
    # config/account gap (missing key, 402 out-of-points, no nearby stations)
    # is visible instead of a bare "Source: none".
    if snapshot.aqi_source == "none":
        reasons = (
            snapshot.aqi_reasons if isinstance(snapshot.aqi_reasons, dict) else {}
        )
        rows = ""
        for provider in ("airnow", "purpleair"):
            label = _AQI_PROVIDER_LABELS.get(provider, provider.title())
            reason = reasons.get(provider) or "unavailable"
            rows += f"""
    <div class="metric">
        <span class="label">{html.escape(label)}:</span>
        <span class="value aqi-unavailable">{html.escape(str(reason))}</span>
    </div>
    """
        return f"""
    {aqi_obs_age}
    <div class="metric">
        <span class="label">AQI:</span>
        <span class="value">unavailable — gate skipped</span>
    </div>
    {rows}
    """

    readings = snapshot.aqi_readings
    present = (
        [(p, v) for p, v in readings.items() if v is not None]
        if isinstance(readings, dict) else []
    )

    if len(present) >= 2:
        rows = ""
        for provider, value in present:
            label = _AQI_PROVIDER_LABELS.get(provider, provider.title())
            value_class = _aqi_value_class(value)
            auth_badge = (
                " <span class='aqi-authoritative'>· driving decision</span>"
                if provider == snapshot.aqi_source else ""
            )
            rows += f"""
    <div class="metric">
        <span class="label">{html.escape(label)}:</span>
        <span class="value aqi-{value_class}">{value}{auth_badge}</span>
    </div>
    """
        return f"""
    {aqi_obs_age}
    {rows}
    """

    # Single provider (or legacy snapshot): show one labeled value. Prefer the
    # authoritative value; label it by source when the source is known.
    label = _AQI_PROVIDER_LABELS.get(snapshot.aqi_source)
    aqi_label = f"AQI ({label})" if label else "AQI"
    return f"""
    {aqi_obs_age}
    <div class="metric">
        <span class="label">{html.escape(aqi_label)}:</span>
        <span class="value aqi-{aqi_class}">{snapshot.aqi_value}</span>
    </div>
    <div class="metric">
        <span class="label">Source:</span>
        <span class="value">{html.escape(snapshot.aqi_source)}</span>
    </div>
    """


def _render_environment_section(snapshot: FloorSnapshot) -> str:
    """Render the shared outdoor + AQI section (same data across all floors)."""
    now = datetime.now(timezone.utc)

    # Outdoor freshness.
    # Age class is driven by the OLDEST contributor (worst-case bucket) so
    # the colour matches the staleness of the pool, not its best member.
    # Thresholds match the 20-min peer cutoff (fresh < 20m, warn 20-45m, stale > 45m).
    outdoor_obs_age = ""
    obs_time = _parse_ts(snapshot.outdoor_observation_time)
    if obs_time is not None:
        obs_age_mins = (now - obs_time).total_seconds() / 60
        age_class = "data-fresh" if obs_age_mins < 20 else ("data-warn" if obs_age_mins < 45 else "data-stale")

        newest_iso = snapshot.outdoor_newest_observation_time
        contributor_count = snapshot.outdoor_contributor_count
        newest_time = _parse_ts(newest_iso)
        if (
            newest_time is not None
            and newest_iso != snapshot.outdoor_observation_time
            and contributor_count and contributor_count > 0
        ):
            # Multiple contributors spanning a range of ages — surface both ends
            # so the reader can tell a fresher peer was blended with an older one.
            label = (
                f"observed {_format_age(obs_time)}\u2013{_format_age(newest_time)} ago "
                f"({contributor_count} reading{'s' if contributor_count != 1 else ''})"
            )
        else:
            label = f"observed {_format_age(obs_time)} ago"
        outdoor_obs_age = f"<div class='data-freshness {age_class}'>{label}</div>"

    # Jitter-suppression indicator: only shown when a swing was actually held
    # back this cycle, so the page stays clean during normal operation.
    jitter_badge = ""
    if snapshot.outdoor_validation_reason == "suppressed_jitter":
        jitter_badge = (
            "<div class='data-freshness data-warn'>"
            "\U0001f6df Sensor swing suppressed \u2014 held steady "
            "(a rotating sensor disagreed with the recent trend)"
            "</div>"
        )
    elif snapshot.outdoor_validation_reason == "suppressed_spike":
        jitter_badge = (
            "<div class='data-freshness data-warn'>"
            "\U0001f6df Temperature spike suppressed \u2014 held steady "
            "(an implausible one-cycle jump, awaiting confirmation)"
            "</div>"
        )

    outdoor_html = f"""
    {outdoor_obs_age}
    {jitter_badge}
    <div class="metric">
        <span class="label">Temperature:</span>
        <span class="value">{snapshot.outdoor_temp_f:.1f}°F</span>
    </div>
    <div class="metric">
        <span class="label">Source:</span>
        <span class="value">{html.escape(snapshot.outdoor_source)}</span>
    </div>
    """
    if snapshot.outdoor_humidity is not None:
        outdoor_html += f"""
        <div class="metric">
            <span class="label">Humidity:</span>
            <span class="value">{snapshot.outdoor_humidity:.0f}%</span>
        </div>
        """

    # AQI. When BOTH AirNow and PurpleAir were queried this cycle, show both
    # readings clearly labeled and mark which one is authoritative (drove the
    # open/close decision). When only one provider was checked, show just that
    # one — never a "PurpleAir: None" line. Legacy snapshots (no aqi_readings)
    # fall back to the original single AQI + Source view.
    aqi_class = "good" if snapshot.aqi_value < 50 else ("moderate" if snapshot.aqi_value < 100 else "unhealthy")
    aqi_obs_age = ""
    obs_time = _parse_ts(snapshot.aqi_observation_time)
    if obs_time is not None:
        obs_age_mins = (now - obs_time).total_seconds() / 60
        age_class = "data-fresh" if obs_age_mins < 30 else ("data-warn" if obs_age_mins < 60 else "data-stale")
        aqi_obs_age = f"<div class='data-freshness {age_class}'>observed {_format_age(obs_time)} ago</div>"

    aqi_html = _render_aqi_block(snapshot, aqi_class, aqi_obs_age)

    return f"""
    <div class="floor-card environment-card">
        <div class="floor-header">
            <h2>🌤️ Environment</h2>
        </div>

        <details open>
            <summary>Outdoor Conditions</summary>
            {outdoor_html}
        </details>

        <details open>
            <summary>Air Quality</summary>
            {aqi_html}
        </details>
    </div>
    """


def _render_floor_card(snapshot: FloorSnapshot) -> str:
    """Render one floor as an HTML card."""
    decision_class = "open" if snapshot.decision == "OPEN" else "closed"

    # Indoor sensors — freshness derived from poll timestamp (sensors read at poll time)
    sensor_poll_time = _parse_ts(snapshot.timestamp)
    sensor_age = _format_age(sensor_poll_time) if sensor_poll_time is not None else "unknown"
    indoor_html = f"<div class='data-freshness'>read {sensor_age} ago</div><ul class='sensor-list'>"
    for sensor in snapshot.indoor_sensors:
        temp_str = f"{sensor.temperature_f:.1f}°F" if sensor.temperature_f else "N/A"
        status = "online" if sensor.is_online else "offline"
        coolest_mark = " 🌡️" if sensor.is_coolest else ""

        # Provenance + upstream sync age. Subtle, single line per sensor,
        # mirrors the outdoor section's .data-freshness pattern. Old
        # snapshots predating these fields render no extra line at all.
        provenance_html = _render_sensor_provenance(sensor.source, sensor.data_age_seconds)

        indoor_html += f"""
        <li>
            <span class="sensor-name">{html.escape(sensor.name)}{coolest_mark}</span>
            <span class="sensor-value">{temp_str}</span>
            <span class="sensor-status badge-{status}">{status}</span>
            {provenance_html}
        </li>
        """
    indoor_html += "</ul>"

    # Gates
    gates_html = "<ul class='gates-list'>"
    for gate in snapshot.gates:
        status_icon = "✓" if gate.passed else "✗"
        status_class = "pass" if gate.passed else "fail"
        gates_html += f"""
        <li class="gate-{status_class}">
            <span class="gate-icon">{status_icon}</span>
            <span class="gate-name">{html.escape(gate.name)}</span>
            <span class="gate-detail">
                {html.escape(gate.threshold or '')}
                {f' (actual: {html.escape(gate.actual)})' if gate.actual else ''}
            </span>
        </li>
        """
    gates_html += "</ul>"

    # Last notification
    notif_html = ""
    notif_time = _parse_ts(snapshot.last_notification_time)
    if notif_time is not None:
        notif_age = _format_age(notif_time)
        notif_type = snapshot.last_notification_type or "unknown"
        notif_html = f"""
        <div class="notification-info">
            <strong>Last notification:</strong> {html.escape(notif_type)} ({notif_age} ago)
        </div>
        """

    return f"""
    <div class="floor-card">
        <div class="floor-header">
            <h2>{html.escape(snapshot.floor.title())}</h2>
            <span class="decision-badge badge-{decision_class}">{snapshot.decision}</span>
        </div>
        <div class="reason">
            <strong>Reason:</strong> {html.escape(snapshot.reason)}
        </div>

        <details open>
            <summary>Indoor Sensors</summary>
            {indoor_html}
        </details>

        <details>
            <summary>Gate Evaluations</summary>
            {gates_html}
        </details>

        {notif_html}
    </div>
    """


def _render_history_card(history: list[TemperatureHistoryEntry], tz: ZoneInfo) -> str:
    """Render the collapsible 12-hour temperature history table.

    Returns "" when there are no entries so the page doesn't show an empty
    expando.
    """
    if not history:
        return ""

    # Union of floor keys across all entries, sorted for stable column order.
    floor_names: list[str] = sorted({
        floor for entry in history for floor in entry.indoor_temps.keys()
    })

    header_cells = "".join(
        f"<th>{html.escape(name.title())}</th>" for name in floor_names
    )

    rows_html = []
    for entry in history:  # already newest-first
        ts = _parse_ts(entry.timestamp)
        if ts is not None:
            time_label = _format_local_and_utc_compact(ts, tz)
        else:
            time_label = html.escape(str(entry.timestamp))

        outdoor_cell = (
            f"{entry.outdoor_temp_f:.1f}°F"
            if entry.outdoor_temp_f is not None
            else "—"
        )
        floor_cells = []
        for name in floor_names:
            val = entry.indoor_temps.get(name)
            floor_cells.append(
                f"<td>{val:.1f}°F</td>" if val is not None else "<td>—</td>"
            )
        rows_html.append(
            f"<tr><td>{time_label}</td><td>{outdoor_cell}</td>"
            + "".join(floor_cells)
            + "</tr>"
        )

    return f"""
    <div class="floor-card history-card">
        <details>
            <summary>📈 Temperature History (last 12h)</summary>
            <div class="history-scroll">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Time (Local / UTC)</th>
                            <th>Outdoor</th>
                            {header_cells}
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows_html)}
                    </tbody>
                </table>
            </div>
        </details>
    </div>
    """


_SENSOR_SOURCE_LABELS = {
    "beestat:live_temps": "via Beestat",
    "beestat:sensor-resource": "via Beestat",
    "ecobee:direct": "via Ecobee direct",
}


def _render_sensor_provenance(source: str | None, data_age_seconds: float | None) -> str:
    """Render the per-sensor provenance + sync-age line.

    Returns an empty string for back-compat snapshots (``source is None``)
    so the row falls back to the pre-change rendering exactly.

    Freshness buckets mirror the Beestat client's WARN threshold:
    ``data-fresh`` < 5 min, ``data-warn`` 5–10 min, ``data-stale`` > 10 min.
    When ``data_age_seconds`` is ``None`` (unknown — sync_status missing
    or this is Ecobee direct), no CSS bucket class is applied.
    """
    if source is None:
        return ""

    label = _SENSOR_SOURCE_LABELS.get(source, source)
    age_class = ""
    age_suffix = ""

    if source.startswith("beestat:"):
        if data_age_seconds is None:
            age_suffix = " · sync age unknown"
        else:
            age_minutes = data_age_seconds / 60
            if age_minutes < 5:
                age_class = "data-fresh"
            elif age_minutes < 10:
                age_class = "data-warn"
            else:
                age_class = "data-stale"
            # Reuse the same datetime-relative formatter outdoor uses so
            # "Xmin ago" / "Xs ago" formatting stays consistent.
            synced_at = datetime.now(timezone.utc) - timedelta(seconds=data_age_seconds)
            age_suffix = f" · synced {_format_age(synced_at)} ago"

    css_classes = "sensor-provenance data-freshness"
    if age_class:
        css_classes += f" {age_class}"
    return f"<div class='{css_classes}'>{html.escape(label)}{html.escape(age_suffix)}</div>"


def _get_display_tz() -> ZoneInfo:
    """Return the timezone used for user-visible local timestamps.

    Reuses the ``QUIET_HOURS_TIMEZONE`` env var (same IANA name the quiet-hours
    feature uses), defaulting to ``America/Los_Angeles``. Falls back to UTC
    silently if the configured name is not a valid IANA zone — a bad value
    must never crash the page.
    """
    name = os.environ.get("QUIET_HOURS_TIMEZONE", "").strip() or "America/Los_Angeles"
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        logger.warning("Invalid IANA timezone %r; falling back to UTC.", name)
        return ZoneInfo("UTC")


def _ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC (some historical snapshots predate tz-awareness)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_ts(value: str | None) -> datetime | None:
    """Tolerantly parse a snapshot timestamp field into an aware UTC datetime.

    Routes through :func:`parse_iso_utc` so valid ISO-8601 (including a trailing
    ``Z``) is parsed exactly as before, then normalizes naive values to UTC.
    Returns ``None`` for missing, empty, or non-ISO/legacy values — e.g. a
    human-formatted ``"2026-07-13 18:00 PDT"`` — so a single unparseable
    timestamp can never raise and error the whole status page.
    """
    dt = parse_iso_utc(value)
    return _ensure_utc(dt) if dt is not None else None


def _format_local_and_utc(dt_utc: datetime, tz: ZoneInfo) -> str:
    """Format an aware UTC datetime as ``YYYY-MM-DD HH:MM:SS PDT (HH:MM:SS UTC)``.

    The local-tz abbreviation (e.g. ``PST``/``PDT``) comes from ``strftime('%Z')``
    on the converted datetime, so DST is handled automatically.
    """
    dt_utc = _ensure_utc(dt_utc)
    local = dt_utc.astimezone(tz)
    local_label = local.strftime("%Y-%m-%d %H:%M:%S %Z")
    utc_label = dt_utc.strftime("%H:%M:%S UTC")
    return f"{local_label} ({utc_label})"


def _format_local_and_utc_compact(dt_utc: datetime, tz: ZoneInfo) -> str:
    """Compact variant for table cells: ``MM-DD HH:MM PDT / HH:MM UTC``."""
    dt_utc = _ensure_utc(dt_utc)
    local = dt_utc.astimezone(tz)
    local_label = local.strftime("%m-%d %H:%M %Z")
    utc_label = dt_utc.strftime("%H:%M UTC")
    return f"{local_label} / {utc_label}"


def _format_age(dt: datetime, future: bool = False) -> str:
    """Format datetime as human-readable age."""
    now = datetime.now(timezone.utc)
    if future:
        delta = (dt - now).total_seconds()
        prefix = "in "
    else:
        delta = (now - dt).total_seconds()
        prefix = ""

    abs_delta = abs(delta)
    
    if abs_delta < 60:
        return f"{prefix}{int(abs_delta)}s"
    elif abs_delta < 3600:
        return f"{prefix}{int(abs_delta / 60)}min"
    elif abs_delta < 86400:
        hours = int(abs_delta / 3600)
        mins = int((abs_delta % 3600) / 60)
        return f"{prefix}{hours}h {mins}min"
    else:
        days = int(abs_delta / 86400)
        return f"{prefix}{days}d"


def _render_build_info() -> str:
    """Render the small build/version footer line.

    Always renders something — never crashes the status page even if every
    version field is missing or malformed.
    """
    try:
        now = datetime.now(timezone.utc)
        worker_uptime = _format_age(WORKER_STARTED_AT)

        if is_dev_build():
            return (
                f'<div class="build-info">'
                f'Build: <strong>dev</strong> (local) · worker up {worker_uptime}'
                f'</div>'
            )

        sha = VERSION.get("commit_sha") or "unknown"
        branch = VERSION.get("branch") or "unknown"
        commit_url = VERSION.get("commit_url")

        if commit_url:
            sha_html = (
                f'<a href="{html.escape(commit_url)}" target="_blank" rel="noopener">'
                f'{html.escape(sha)}</a>'
            )
        else:
            sha_html = f'<strong>{html.escape(sha)}</strong>'

        parts = [f'Build: {sha_html} ({html.escape(branch)})']

        commit_time = parse_iso_utc(VERSION.get("commit_time"))
        if commit_time is not None:
            parts.append(f'committed {_format_age(commit_time)} ago')

        build_time = parse_iso_utc(VERSION.get("build_time"))
        stale_class = ""
        if build_time is not None:
            parts.append(f'deployed {_format_age(build_time)} ago')
            # Heuristic: flag if the build is older than 7 days.
            if (now - build_time).total_seconds() > 7 * 86400:
                stale_class = " build-stale"

        parts.append(f'worker up {worker_uptime}')

        return f'<div class="build-info{stale_class}">{" · ".join(parts)}</div>'
    except Exception:
        # Diagnostic info should NEVER break the page.
        logger.exception("Failed to render build info; omitting from footer.")
        return ""


def _get_css() -> str:
    """Return inline CSS for the status page."""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            font-size: 16px;
            line-height: 1.6;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        h1 {
            font-size: 2em;
            margin-bottom: 20px;
            color: #2c3e50;
            word-wrap: break-word;
        }
        
        h2 {
            font-size: 1.5em;
            color: #34495e;
            word-wrap: break-word;
        }
        
        .freshness {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
            font-size: 1em;
            word-wrap: break-word;
        }
        
        .freshness.warning {
            background: #fff3e0;
            border-left-color: #ff9800;
        }
        
        .freshness.stale {
            background: #ffebee;
            border-left-color: #f44336;
        }
        
        .freshness-warning {
            margin-left: 10px;
            color: #d32f2f;
            font-weight: bold;
            display: inline-block;
        }
        
        .global-info {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
            flex-wrap: wrap;
        }
        
        .info-row:last-child {
            border-bottom: none;
        }
        
        .label {
            font-weight: 600;
            color: #555;
            word-wrap: break-word;
        }
        
        .value {
            color: #333;
            word-wrap: break-word;
        }
        
        .badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            white-space: nowrap;
        }
        
        .badge-active {
            background: #ffe0b2;
            color: #e65100;
        }
        
        .badge-inactive {
            background: #e0e0e0;
            color: #666;
        }
        
        .badge-open {
            background: #c8e6c9;
            color: #2e7d32;
        }
        
        .badge-closed {
            background: #ffcdd2;
            color: #c62828;
        }
        
        .floor-card {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .floor-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
            flex-wrap: wrap;
        }
        
        .decision-badge {
            font-size: 1.1em;
            padding: 10px 18px;
            min-height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        
        .reason {
            background: #f9f9f9;
            padding: 12px;
            margin-bottom: 15px;
            border-radius: 4px;
            font-size: 0.95em;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        details {
            margin: 15px 0;
        }
        
        summary {
            cursor: pointer;
            font-weight: 600;
            padding: 14px;
            background: #f5f5f5;
            border-radius: 4px;
            user-select: none;
            min-height: 44px;
            display: flex;
            align-items: center;
        }
        
        summary:hover {
            background: #eeeeee;
        }
        
        summary:active {
            background: #e0e0e0;
        }
        
        details[open] summary {
            margin-bottom: 10px;
        }
        
        .sensor-list, .gates-list {
            list-style: none;
            padding: 10px 0;
        }
        
        .sensor-list li {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding: 10px;
            border-bottom: 1px solid #f0f0f0;
            align-items: center;
            flex-wrap: wrap;
            min-height: 44px;
        }
        
        .sensor-name {
            flex: 1;
            font-weight: 500;
            word-wrap: break-word;
            min-width: 120px;
        }
        
        .sensor-value {
            margin: 0 10px;
            font-weight: 600;
            white-space: nowrap;
        }
        
        .sensor-status {
            font-size: 0.75em;
            padding: 4px 10px;
        }
        
        .badge-online {
            background: #c8e6c9;
            color: #2e7d32;
        }
        
        .badge-offline {
            background: #ffcdd2;
            color: #c62828;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding: 8px 10px;
            flex-wrap: wrap;
        }
        
        .aqi-good {
            color: #2e7d32;
            font-weight: 600;
        }
        
        .aqi-moderate {
            color: #f57c00;
            font-weight: 600;
        }
        
        .aqi-unhealthy {
            color: #c62828;
            font-weight: 600;
        }

        .aqi-authoritative {
            font-size: 0.8em;
            font-weight: 600;
            color: #555;
        }

        .data-freshness {
            font-size: 0.8em;
            color: #777;
            padding: 4px 10px 8px;
            font-style: italic;
        }

        .data-freshness.data-warn {
            color: #f57c00;
        }

        .data-freshness.data-stale {
            color: #c62828;
            font-weight: 600;
        }

        .history-scroll {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
            margin-top: 4px;
        }

        .history-table th,
        .history-table td {
            border: 1px solid #e0e0e0;
            padding: 6px 10px;
            text-align: right;
            white-space: nowrap;
        }

        .history-table th {
            background: #f5f5f5;
            font-weight: 600;
            text-align: center;
        }

        .history-table th:first-child,
        .history-table td:first-child {
            text-align: left;
        }

        .history-table tbody tr:nth-child(even) {
            background: #fafafa;
        }
        
        .gates-list li {
            padding: 10px;
            border-bottom: 1px solid #f0f0f0;
            display: flex;
            gap: 10px;
            align-items: flex-start;
            flex-wrap: wrap;
            min-height: 44px;
        }
        
        .gate-icon {
            margin-right: 5px;
            font-weight: bold;
            font-size: 1.2em;
            flex-shrink: 0;
        }
        
        .gate-pass .gate-icon {
            color: #2e7d32;
        }
        
        .gate-fail .gate-icon {
            color: #c62828;
        }
        
        .gate-name {
            font-weight: 600;
            min-width: 100px;
            word-wrap: break-word;
        }
        
        .gate-detail {
            color: #666;
            font-size: 0.9em;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        .notification-info {
            margin-top: 15px;
            padding: 12px;
            background: #e3f2fd;
            border-radius: 4px;
            font-size: 0.9em;
            word-wrap: break-word;
        }
        
        .errors {
            background: #ffebee;
            padding: 15px;
            margin-top: 15px;
            border-radius: 4px;
            border-left: 4px solid #f44336;
        }
        
        .errors ul {
            margin-top: 10px;
            padding-left: 20px;
        }
        
        .errors li {
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        .footer a {
            color: #1976d2;
            text-decoration: none;
            padding: 8px;
            display: inline-block;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }

        .build-info {
            font-size: 0.8em;
            color: #888;
            margin-top: 8px;
            margin-bottom: 12px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.5;
        }

        .build-info a {
            color: #1976d2;
            text-decoration: none;
            padding: 0;
            display: inline;
        }

        .build-info a:hover {
            text-decoration: underline;
        }

        .build-info.build-stale {
            color: #b26a00;
        }
        
        @media (max-width: 600px) {
            body {
                padding: 12px;
            }
            
            h1 {
                font-size: 1.5em;
                margin-bottom: 15px;
            }
            
            h2 {
                font-size: 1.3em;
            }
            
            .freshness {
                padding: 12px;
                font-size: 0.95em;
            }
            
            .freshness strong {
                display: block;
                margin-bottom: 4px;
            }
            
            .freshness-warning {
                display: block;
                margin-left: 0;
                margin-top: 6px;
            }
            
            .global-info, .floor-card {
                padding: 15px;
                border-radius: 6px;
            }
            
            .floor-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            
            .decision-badge {
                margin-top: 0;
                font-size: 1em;
            }
            
            .sensor-list li, .gates-list li {
                padding: 12px 8px;
            }
            
            .sensor-name {
                min-width: 100%;
                margin-bottom: 4px;
            }
            
            .sensor-value {
                margin: 0;
            }
            
            .gate-name {
                min-width: 100%;
            }
            
            .info-row {
                flex-direction: column;
                gap: 4px;
                padding: 10px 0;
            }
            
            .metric {
                flex-direction: column;
                gap: 4px;
            }
        }
        
        @media (max-width: 480px) {
            body {
                padding: 8px;
                font-size: 15px;
            }
            
            h1 {
                font-size: 1.4em;
            }
            
            .global-info, .floor-card {
                padding: 12px;
            }
            
            summary {
                padding: 12px;
                font-size: 0.95em;
            }
        }
        
        @media (prefers-color-scheme: dark) {
            body {
                background: #1a1a1a;
                color: #e0e0e0;
            }
            
            h1, h2 {
                color: #e0e0e0;
            }
            
            .freshness {
                background: #1e3a1e;
                border-left-color: #66bb6a;
                color: #c8e6c9;
            }
            
            .freshness.warning {
                background: #3d2f1f;
                border-left-color: #ffb74d;
                color: #ffe0b2;
            }
            
            .freshness.stale {
                background: #3d1f1f;
                border-left-color: #e57373;
                color: #ffcdd2;
            }
            
            .freshness-warning {
                color: #ef5350;
            }
            
            .global-info, .floor-card {
                background: #2a2a2a;
                box-shadow: 0 2px 8px rgba(0,0,0,0.5);
            }
            
            .info-row {
                border-bottom-color: #3a3a3a;
            }
            
            .label {
                color: #aaa;
            }
            
            .value {
                color: #e0e0e0;
            }
            
            .badge-active {
                background: #5d4037;
                color: #ffcc80;
            }
            
            .badge-inactive {
                background: #3a3a3a;
                color: #aaa;
            }
            
            .badge-open {
                background: #2e5d2e;
                color: #a5d6a7;
            }
            
            .badge-closed {
                background: #5d2e2e;
                color: #ef9a9a;
            }
            
            .floor-header {
                border-bottom-color: #3a3a3a;
            }
            
            .reason {
                background: #252525;
                color: #d0d0d0;
            }
            
            summary {
                background: #252525;
                color: #e0e0e0;
            }
            
            summary:hover {
                background: #303030;
            }
            
            summary:active {
                background: #353535;
            }
            
            .sensor-list li, .gates-list li {
                border-bottom-color: #333;
            }
            
            .sensor-name {
                color: #d0d0d0;
            }
            
            .badge-online {
                background: #2e5d2e;
                color: #a5d6a7;
            }
            
            .badge-offline {
                background: #5d2e2e;
                color: #ef9a9a;
            }
            
            .aqi-good {
                color: #81c784;
            }
            
            .aqi-moderate {
                color: #ffb74d;
            }
            
            .aqi-unhealthy {
                color: #e57373;
            }
            
            .gate-pass .gate-icon {
                color: #81c784;
            }
            
            .gate-fail .gate-icon {
                color: #e57373;
            }
            
            .gate-detail {
                color: #999;
            }
            
            .notification-info {
                background: #1a2a3a;
                color: #b3d9ff;
            }
            
            .errors {
                background: #3d1f1f;
                border-left-color: #e57373;
                color: #ffcdd2;
            }
            
            .footer {
                color: #999;
            }
            
            .footer a {
                color: #64b5f6;
            }

            .build-info {
                color: #777;
            }

            .build-info a {
                color: #64b5f6;
            }

            .build-info.build-stale {
                color: #d4a056;
            }

            .history-table th,
            .history-table td {
                border-color: #3a3a3a;
            }

            .history-table th {
                background: #303030;
                color: #e0e0e0;
            }

            .history-table tbody tr:nth-child(even) {
                background: #252525;
            }

            .history-table td {
                color: #d0d0d0;
            }
        }
    """
