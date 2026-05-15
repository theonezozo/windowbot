# WindowBot 🪟

Hyperlocal window advisory system — tells you when to open or close your windows based on indoor temperature sensors (Ecobee via [Beestat](https://beestat.io)), outdoor weather (Synoptic/MesoWest + Open-Meteo), and air quality (PurpleAir / AirNow).

Runs as an Azure Function on a 10-minute timer and sends push notifications via [ntfy.sh](https://ntfy.sh).

## How It Works

Every 10 minutes, WindowBot:

1. **Indoor temps** — reads your Ecobee remote sensors via the Beestat API, grouped by floor
2. **Outdoor weather** — fetches readings from the configured outdoor provider (default: Synoptic/MesoWest); Open-Meteo is blended in as a peer when its reading is ≤30 min old; NWS official stations are available as an alternative
3. **Air quality** — gets the median AQI from the 3 nearest PurpleAir sensors (falls back to AirNow)
4. **Per-floor decision** — runs the decision engine independently for each floor (upstairs, downstairs)
5. **Quiet hours** — suppresses notifications during a configurable sleep window; sends a precool opportunity alert when quiet hours end
6. **Notification** — sends a push notification via ntfy.sh if the recommendation changes

### Decision Logic

| Condition | Action | Details |
|-----------|--------|---------|
| Outdoor temp < warmest indoor − 1°F | **Open** | Must also satisfy all gates below |
| Outdoor temp ≥ coolest indoor + 1°F | **Close** | Temperature-based close |
| AQI ≥ 100 | **Urgent close** | Bypasses notification cooldown |
| AQI < 50 | Allow opening | AQI 50–99 is neutral (no change) |
| Outdoor humidity > 80% | Block opening | Prevents letting in humid air |
| Indoor temp ≤ 72°F | Block opening | Already comfortable — no need |
| HVAC not in cool/auto | Block all | Only active when cooling is relevant |
| Quiet hours active | Suppress notifications | Configurable sleep window (e.g. 23:00–07:00) |
| Quiet hours just ended | **Precool alert** | Notifies when morning air is cool enough to pre-chill the house |

The 1°F symmetric hysteresis prevents rapid open/close flip-flopping.

### Status Page

WindowBot exposes a `/api/status` endpoint that shows what it decided on its last poll cycle — indoor temps per sensor, outdoor conditions, AQI, and the open/close recommendation for each floor.

Access it at `https://<your-function>.azurewebsites.net/api/status` (or `http://localhost:7071/api/status` locally). Returns HTML by default; add `?format=json` or `Accept: application/json` for JSON. Set `STATUS_PAGE_PIN` to require a `?pin=` query parameter.

### Sensor Handling

- Sensors are grouped by floor (upstairs/downstairs) in config
- The warmest sensor on a floor drives the "open" decision; the coolest drives "close"
- All sensors are included unless explicitly marked `inactive` (decommissioned) in Ecobee
- The `in_use` flag is ignored — it reflects comfort-profile participation (e.g. Away mode) and is not a reliable hardware status indicator

## Project Structure

```
function_app.py              # Azure Function entry point (timer + HTTP triggers)
host.json                    # Azure Functions host config
requirements.txt             # Python dependencies
pyproject.toml               # Project metadata and pytest config
local.settings.json          # Local dev settings (git-ignored)

src/
  config.py                  # Environment variable loader with typed defaults
  orchestrator.py            # Main fetch → decide → notify pipeline
  decision_engine.py         # Per-floor open/close logic with hysteresis
  beestat_client.py          # Beestat API client (indoor temps via Ecobee)
  nws_client.py              # NWS client (personal + official weather stations)
  openmeteo_client.py        # Open-Meteo free weather peer (no API key, blended into outdoor median)
  purpleair_client.py        # PurpleAir AQI client (median of 3 nearest)
  airnow_client.py           # AirNow fallback AQI client
  notifier.py                # ntfy.sh push notification client (JSON API)
  state.py                   # State manager (Azure Table Storage + local fallback)
  synoptic_client.py         # Synoptic/MesoWest client (default outdoor weather provider)
  wu_client.py               # Weather Underground client (personal weather stations)
  quiet_hours.py             # Quiet hours helpers (suppress notifications during sleep, precool on wake)
  diagnostic.py              # Diagnostic snapshot captured each cycle for the status page
  status_page.py             # /api/status endpoint — PIN-protected HTML/JSON status view
  ecobee_client.py           # Direct Ecobee API client (alternative to Beestat)

tests/
  test_beestat_client.py     # Beestat client tests
  test_decision_engine.py    # Decision logic: all gates, hysteresis, edge cases
  test_orchestrator.py       # Pipeline integration tests
  test_nws_client.py         # NWS station selection and parsing
  test_nws_freshness_metrics.py # NWS freshness metrics JSONL tests
  test_openmeteo_client.py   # Open-Meteo client tests
  test_purpleair_client.py   # PurpleAir AQI conversion and median
  test_airnow_client.py      # AirNow client tests
  test_notifier.py           # ntfy JSON API, unicode, priorities
  test_state.py              # State manager (Azure + local fallback)
  test_synoptic_client.py    # Synoptic client tests
  test_config.py             # Config loader tests
  test_quiet_hours.py        # Quiet hours logic and boundary detection
  test_ecobee_client.py      # Direct Ecobee client tests
  test_e2e_live.py           # E2E tests with live APIs (run separately)
  test_wu_client.py          # Weather Underground client tests
  conftest.py                # Shared fixtures
```

## Prerequisites

- **Python 3.11+**
- API keys for:
  - [Beestat](https://beestat.io) — free, provides Ecobee sensor data
  - [Synoptic](https://synopticdata.com) — free tier available, default outdoor weather provider
  - [PurpleAir](https://develop.purpleair.com/) — free read key for AQI data
  - [AirNow](https://docs.airnowapi.org/) — free, fallback AQI source
- [ntfy](https://ntfy.sh) — free push notifications (install the app on your phone)
- An Ecobee thermostat with remote sensors

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/theonezozo/windowbot.git
cd windowbot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

### 2. Configure

Copy the example file and fill in your API keys and sensor names (the real `local.settings.json` is git-ignored):

```bash
cp local.settings.json.example local.settings.json
```

The template mirrors the shape below:

```json
{
  "IsEncrypted": false,
  "Values": {
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",

    "BEESTAT_API_KEY": "your-beestat-api-key",
    "INDOOR_PROVIDER": "beestat",

    "USER_LATITUDE": "37.7749",
    "USER_LONGITUDE": "-122.4194",

    "NTFY_TOPIC": "windowbot-yourname",

    "UPSTAIRS_SENSORS": "Bedroom,Office",
    "DOWNSTAIRS_SENSORS": "Living Room",

    "PURPLEAIR_API_KEY": "your-purpleair-read-key",
    "AIRNOW_API_KEY": "your-airnow-api-key",
    "AQ_PROVIDER": "purpleair",

    "OUTDOOR_PROVIDER": "synoptic",
    "SYNOPTIC_API_KEY": "your-synoptic-api-key",

    "HYSTERESIS_OPEN_DIFF": "1.0",
    "HYSTERESIS_CLOSE_DIFF": "1.0",
    "MAX_OUTDOOR_HUMIDITY": "80",
    "MAX_AQI_THRESHOLD": "100",
    "MIN_AQI_FOR_OPENING": "50",
    "POLLING_INTERVAL_MINUTES": "10",
    "NOTIFICATION_COOLDOWN_HOURS": "1",
    "ALLOWED_HVAC_MODES": "cool,heatCool,auto",

    "QUIET_HOURS_START": "23:00",
    "QUIET_HOURS_END": "07:00",
    "QUIET_HOURS_TIMEZONE": "America/Los_Angeles",

    "STATUS_PAGE_PIN": ""
  }
}
```

**Finding your sensor names:** Your Beestat dashboard shows your Ecobee remote sensor names. Use those exact names (case-sensitive) in `UPSTAIRS_SENSORS` and `DOWNSTAIRS_SENSORS`, comma-separated.

**Setting up ntfy:** Install the [ntfy app](https://ntfy.sh) on your phone and subscribe to your topic name (e.g., `windowbot-yourname`). Pick something unique — ntfy topics are public.

### 3. Run tests

```bash
# Unit tests (519 tests, runs in <1s)
pytest

# E2E tests with live APIs (requires configured API keys)
pytest -m e2e -v
```

### 4. Run locally

**Option A — Quick test (no Azure tools needed):**

```bash
python -c "
import os, json
with open('local.settings.json') as f:
    for k, v in json.load(f)['Values'].items():
        os.environ[k] = v
from src.orchestrator import run_check
run_check()
"
```

This uses a local JSON file (`.local_state.json`) for state storage instead of Azure Table Storage. You'll see a notification on your phone if conditions warrant opening or closing windows.

**Option B — Azure Functions Core Tools:**

```bash
# Install (macOS)
brew tap azure/functions
brew install azure-functions-core-tools@4

# Run the function locally
func start
```

This runs the full Azure Functions runtime with the 10-minute timer trigger.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BEESTAT_API_KEY` | — | Beestat API key (required for `beestat` indoor provider) |
| `INDOOR_PROVIDER` | `beestat` | Indoor data source (`beestat` or `ecobee`) |
| `USER_LATITUDE` | — | Your location latitude (required) |
| `USER_LONGITUDE` | — | Your location longitude (required) |
| `NTFY_TOPIC` | — | ntfy.sh topic name (required) |
| `UPSTAIRS_SENSORS` | — | Comma-separated sensor names for upstairs |
| `DOWNSTAIRS_SENSORS` | — | Comma-separated sensor names for downstairs |
| `OUTDOOR_PROVIDER` | `synoptic` | Primary outdoor weather source (`synoptic`, `nws`, or `wu`) |
| `SYNOPTIC_API_KEY` | — | Synoptic/MesoWest API key (required for `synoptic` provider) |
| `WU_API_KEY` | — | Weather Underground API key (required for `wu` provider) |
| `PURPLEAIR_API_KEY` | — | PurpleAir read API key |
| `AIRNOW_API_KEY` | — | AirNow API key (fallback AQI source) |
| `AQ_PROVIDER` | `purpleair` | Primary AQI source (`purpleair` or `airnow`) |
| `HYSTERESIS_OPEN_DIFF` | `1.0` | °F below indoor temp to trigger open |
| `HYSTERESIS_CLOSE_DIFF` | `1.0` | °F above indoor temp to trigger close |
| `COMFORT_TEMP_MAX` | `72.0` | Don't suggest opening below this indoor temp |
| `MAX_OUTDOOR_HUMIDITY` | `80` | Block opening above this humidity % |
| `MAX_AQI_THRESHOLD` | `100` | AQI ≥ this triggers urgent close |
| `MIN_AQI_FOR_OPENING` | `50` | AQI must be below this to allow opening |
| `MAX_OBSERVATION_AGE_MINUTES` | `30` | Discard outdoor readings older than this |
| `POLLING_INTERVAL_MINUTES` | `10` | Timer trigger interval |
| `NOTIFICATION_COOLDOWN_HOURS` | `1` | Min hours between non-urgent notifications |
| `ALLOWED_HVAC_MODES` | `cool,heatCool,auto` | HVAC modes where WindowBot is active |
| `ENABLE_HUMIDITY_GATE` | `true` | Whether to block opening on high outdoor humidity |
| `ENABLE_AQI_GATE` | `true` | Whether to block opening on poor air quality |
| `ENABLE_WIND_CHECK` | `false` | Whether to factor wind into the outdoor reading |
| `QUIET_HOURS_START` | — | Start of quiet window, 24-hour HH:MM local time (e.g. `23:00`) |
| `QUIET_HOURS_END` | — | End of quiet window, 24-hour HH:MM local time (e.g. `07:00`) |
| `QUIET_HOURS_TIMEZONE` | — | IANA timezone for quiet hours (e.g. `America/Los_Angeles`) — all three quiet hours keys required to enable |
| `STATUS_PAGE_PIN` | — | Optional PIN to protect `/api/status`; leave empty for unauthenticated access |

## Deployment

Deploy to Azure Functions (Consumption plan — free tier):

```bash
# Create resources (first time only)
az group create --name rg-windowbot --location westus2
az storage account create --name stwindowbot --resource-group rg-windowbot --sku Standard_LRS
az functionapp create --name func-windowbot-prod \
  --resource-group rg-windowbot \
  --storage-account stwindowbot \
  --consumption-plan-location westus2 \
  --runtime python --runtime-version 3.11 \
  --functions-version 4

# Deploy
func azure functionapp publish func-windowbot-prod

# Set app settings (your API keys)
az functionapp config appsettings set --name func-windowbot-prod \
  --resource-group rg-windowbot \
  --settings BEESTAT_API_KEY=xxx PURPLEAIR_API_KEY=xxx ...
```

### Estimated Cost

**$0.00–0.05/month** on the Azure Functions Consumption plan. The free grant covers 1M executions/month — WindowBot uses ~4,320 (every 10 min × 30 days).

## Data Sources

| Source | What | Endpoint | Auth |
|--------|------|----------|------|
| [Beestat](https://beestat.io) | Indoor temps, humidity, HVAC mode | `beestat.io/api/` | API key |
| [Synoptic](https://synopticdata.com) | Outdoor temp, humidity, wind (default provider) | `api.synopticdata.com` | API key |
| [NWS](https://www.weather.gov) | Outdoor temp, humidity, wind (alternative provider) | `api.weather.gov` | None (free) |
| [Weather Underground](https://www.wunderground.com/member/api-keys) | Outdoor temp, humidity, wind (alternative provider) | `api.weather.com` | API key |
| [Open-Meteo](https://open-meteo.com) | Outdoor temp, humidity, wind (peer blended into median) | `open-meteo.com/v1/forecast` | None (free) |
| [PurpleAir](https://www.purpleair.com) | AQI (PM2.5) | `api.purpleair.com` | Read key |
| [AirNow](https://www.airnow.gov) | AQI (fallback) | `airnowapi.org` | API key |
| [ntfy](https://ntfy.sh) | Push notifications | `ntfy.sh` | None (free) |

## License

MIT
