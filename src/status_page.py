"""Status page endpoint for WindowBot.

Renders last persisted state as HTML or JSON, showing exactly what
WindowBot decided during its last poll cycle.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone

import azure.functions as func

from src.state import get_state_manager
from src.diagnostic import SnapshotManager, FloorSnapshot, GlobalSnapshot

logger = logging.getLogger("windowbot.status")


def render_status_page(req: func.HttpRequest) -> func.HttpResponse:
    """Render the status page as HTML or JSON.
    
    Args:
        req: HTTP request with optional ?format=json query param
        
    Returns:
        HTTP response with HTML (default) or JSON
    """
    # Determine output format
    accept_header = req.headers.get("Accept", "")
    format_param = req.params.get("format", "")
    
    wants_json = "application/json" in accept_header or format_param == "json"
    
    try:
        state_mgr = get_state_manager()
        
        # Check if snapshot table is available
        if not hasattr(state_mgr, 'get_snapshot_table'):
            return _no_data_response(
                "Status page requires Azure Table Storage (not available in local state mode).",
                is_json=wants_json
            )
        
        snapshot_table = state_mgr.get_snapshot_table()
        mgr = SnapshotManager(snapshot_table)
        
        # Fetch snapshots
        floor_snapshots = mgr.get_all_floor_snapshots()
        global_snapshot = mgr.get_global_snapshot()
        
        if not floor_snapshots and not global_snapshot:
            return _no_data_response(
                "No diagnostic data available yet. WindowBot will populate this page after its first poll cycle.",
                is_json=wants_json
            )
        
        # Render as JSON or HTML
        if wants_json:
            return _render_json(floor_snapshots, global_snapshot)
        else:
            return _render_html(floor_snapshots, global_snapshot)
    
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
    global_snapshot: GlobalSnapshot | None
) -> func.HttpResponse:
    """Render status as JSON."""
    data = {
        "global": json.loads(global_snapshot.to_json()) if global_snapshot else None,
        "floors": {s.floor: json.loads(s.to_json()) for s in floor_snapshots}
    }
    return func.HttpResponse(
        json.dumps(data, indent=2),
        status_code=200,
        mimetype="application/json"
    )


def _render_html(
    floor_snapshots: list[FloorSnapshot],
    global_snapshot: GlobalSnapshot | None
) -> func.HttpResponse:
    """Render status as HTML."""
    now = datetime.now(timezone.utc)
    
    # Calculate data freshness
    freshness_html = ""
    freshness_class = "fresh"
    if global_snapshot:
        poll_time = datetime.fromisoformat(global_snapshot.poll_start)
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
        next_poll_time = datetime.fromisoformat(global_snapshot.next_poll_eta)
        next_poll_in = _format_age(next_poll_time, future=True)
        
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
    floors_html = ""
    for snapshot in sorted(floor_snapshots, key=lambda s: s.floor):
        floors_html += _render_floor_card(snapshot)
    
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
        {freshness_html}
        {header_html}
        {floors_html}
        <div class="footer">
            <p>Data source: Last persisted poll cycle</p>
            <p><a href="?format=json">View as JSON</a></p>
        </div>
    </div>
</body>
</html>
"""
    return func.HttpResponse(html_content, status_code=200, mimetype="text/html")


def _render_floor_card(snapshot: FloorSnapshot) -> str:
    """Render one floor as an HTML card."""
    decision_class = "open" if snapshot.decision == "OPEN" else "closed"
    
    # Indoor sensors
    indoor_html = "<ul class='sensor-list'>"
    for sensor in snapshot.indoor_sensors:
        temp_str = f"{sensor.temperature_f:.1f}°F" if sensor.temperature_f else "N/A"
        status = "online" if sensor.is_online else "offline"
        coolest_mark = " 🌡️" if sensor.is_coolest else ""
        indoor_html += f"""
        <li>
            <span class="sensor-name">{html.escape(sensor.name)}{coolest_mark}</span>
            <span class="sensor-value">{temp_str}</span>
            <span class="sensor-status badge-{status}">{status}</span>
        </li>
        """
    indoor_html += "</ul>"
    
    # Outdoor
    outdoor_html = f"""
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
    
    # AQI
    aqi_class = "good" if snapshot.aqi_value < 50 else ("moderate" if snapshot.aqi_value < 100 else "unhealthy")
    aqi_html = f"""
    <div class="metric">
        <span class="label">AQI:</span>
        <span class="value aqi-{aqi_class}">{snapshot.aqi_value}</span>
    </div>
    <div class="metric">
        <span class="label">Source:</span>
        <span class="value">{html.escape(snapshot.aqi_source)}</span>
    </div>
    """
    
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
    if snapshot.last_notification_time:
        notif_time = datetime.fromisoformat(snapshot.last_notification_time)
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
            <summary>Outdoor Conditions</summary>
            {outdoor_html}
        </details>
        
        <details>
            <summary>Air Quality</summary>
            {aqi_html}
        </details>
        
        <details>
            <summary>Gate Evaluations</summary>
            {gates_html}
        </details>
        
        {notif_html}
    </div>
    """


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


def _get_css() -> str:
    """Return inline CSS for the status page."""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
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
        }
        
        h2 {
            font-size: 1.5em;
            color: #34495e;
        }
        
        .freshness {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
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
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .info-row:last-child {
            border-bottom: none;
        }
        
        .label {
            font-weight: 600;
            color: #555;
        }
        
        .value {
            color: #333;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
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
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }
        
        .decision-badge {
            font-size: 1.1em;
            padding: 8px 16px;
        }
        
        .reason {
            background: #f9f9f9;
            padding: 12px;
            margin-bottom: 15px;
            border-radius: 4px;
            font-size: 0.95em;
        }
        
        details {
            margin: 15px 0;
        }
        
        summary {
            cursor: pointer;
            font-weight: 600;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
            user-select: none;
        }
        
        summary:hover {
            background: #eeeeee;
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
            padding: 8px 10px;
            border-bottom: 1px solid #f0f0f0;
            align-items: center;
        }
        
        .sensor-name {
            flex: 1;
            font-weight: 500;
        }
        
        .sensor-value {
            margin: 0 15px;
            font-weight: 600;
        }
        
        .sensor-status {
            font-size: 0.75em;
            padding: 3px 8px;
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
            padding: 6px 10px;
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
        
        .gates-list li {
            padding: 8px 10px;
            border-bottom: 1px solid #f0f0f0;
            display: flex;
            align-items: center;
        }
        
        .gate-icon {
            margin-right: 10px;
            font-weight: bold;
            font-size: 1.2em;
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
        }
        
        .gate-detail {
            color: #666;
            font-size: 0.9em;
        }
        
        .notification-info {
            margin-top: 15px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 4px;
            font-size: 0.9em;
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
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 600px) {
            body {
                padding: 10px;
            }
            
            h1 {
                font-size: 1.5em;
            }
            
            .floor-header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .decision-badge {
                margin-top: 10px;
            }
            
            .sensor-list li {
                flex-wrap: wrap;
            }
        }
    """
