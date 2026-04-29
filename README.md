# WindowBot 🪟

Hyperlocal window advisory system — tells you when to open or close your windows based on indoor temperature sensors (Ecobee via [Beestat](https://beestat.io)), outdoor weather (NWS), and air quality (PurpleAir / AirNow).

Runs as an Azure Function on a 10-minute timer and sends push notifications via [ntfy.sh](https://ntfy.sh).

## How It Works

Every 10 minutes, WindowBot:

1. **Indoor temps** — reads your Ecobee remote sensors via the Beestat API, grouped by floor
2. **Outdoor weather** — fetches the median of the 3 nearest NWS personal weather stations (falls back to the nearest official station); Open-Meteo is also blended into the outdoor median when its reading is ≤30 min old
3. **Air quality** — gets the median AQI from the 3 nearest PurpleAir sensors (falls back to AirNow)
4. **Per-floor decision** — runs the decision engine independently for each floor (upstairs, downstairs)
5. **Notification** — sends a push notification via ntfy.sh if the recommendation changes

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

The 1°F symmetric hysteresis prevents rapid open/close flip-flopping.

### Sensor Handling

- Sensors are grouped by floor (upstairs/downstairs) in config
- The warmest sensor on a floor drives the "open" decision; the coolest drives "close"
- Ecobee rotates which remote sensors are `in_use` throughout the day
- If all sensors on a floor are marked `in_use=false`, WindowBot ignores the filter and uses all sensors for that floor

## Project Structure

```
function_app.py              # Azure Function entry point (timer trigger)
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
  openmeteo_client.py        # Open-Meteo free weather peer (no API key, blended into NWS median)
  purpleair_client.py        # PurpleAir AQI client (median of 3 nearest)
  airnow_client.py           # AirNow fallback AQI client
  notifier.py                # ntfy.sh push notification client (JSON API)
  state.py                   # State manager (Azure Table Storage + local fallback)
  synoptic_client.py         # Synoptic/MesoWest client (personal weather stations)
  ecobee_client.py           # Direct Ecobee API client (unused — Beestat preferred)
  wu_client.py               # Weather Underground client (personal weather stations)

tests/
  test_beestat_client.py     # Beestat client tests incl. in_use fallback
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
  test_e2e_live.py           # E2E tests with live APIs (run separately)
  test_wu_client.py          # Weather Underground client tests
  conftest.py                # Shared fixtures
```

## Prerequisites

- **Python 3.11+**
- API keys for:
  - [Beestat](https://beestat.io) — free, provides Ecobee sensor data
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

    "HYSTERESIS_OPEN_DIFF": "1.0",
    "HYSTERESIS_CLOSE_DIFF": "1.0",
    "MAX_OUTDOOR_HUMIDITY": "80",
    "MAX_AQI_THRESHOLD": "100",
    "MIN_AQI_FOR_OPENING": "50",
    "POLLING_INTERVAL_MINUTES": "10",
    "NOTIFICATION_COOLDOWN_HOURS": "1",
    "ALLOWED_HVAC_MODES": "cool,heatCool,auto"
  }
}
```

**Finding your sensor names:** Your Beestat dashboard shows your Ecobee remote sensor names. Use those exact names (case-sensitive) in `UPSTAIRS_SENSORS` and `DOWNSTAIRS_SENSORS`, comma-separated.

**Setting up ntfy:** Install the [ntfy app](https://ntfy.sh) on your phone and subscribe to your topic name (e.g., `windowbot-yourname`). Pick something unique — ntfy topics are public.

### 3. Run tests

```bash
# Unit tests (465 tests, runs in <1s)
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
| `BEESTAT_API_KEY` | — | Beestat API key (required) |
| `INDOOR_PROVIDER` | `beestat` | Indoor data source (`beestat` or `ecobee`) |
| `USER_LATITUDE` | — | Your location latitude (required) |
| `USER_LONGITUDE` | — | Your location longitude (required) |
| `NTFY_TOPIC` | — | ntfy.sh topic name (required) |
| `UPSTAIRS_SENSORS` | — | Comma-separated sensor names for upstairs |
| `DOWNSTAIRS_SENSORS` | — | Comma-separated sensor names for downstairs |
| `PURPLEAIR_API_KEY` | — | PurpleAir read API key |
| `AIRNOW_API_KEY` | — | AirNow API key (fallback) |
| `AQ_PROVIDER` | `purpleair` | Primary AQI source (`purpleair` or `airnow`) |
| `HYSTERESIS_OPEN_DIFF` | `1.0` | °F below indoor temp to trigger open |
| `HYSTERESIS_CLOSE_DIFF` | `1.0` | °F above indoor temp to trigger close |
| `COMFORT_TEMP_MAX` | `72.0` | Don't suggest opening below this indoor temp |
| `MAX_OUTDOOR_HUMIDITY` | `80` | Block opening above this humidity % |
| `MAX_AQI_THRESHOLD` | `100` | AQI ≥ this triggers urgent close |
| `MIN_AQI_FOR_OPENING` | `50` | AQI must be below this to allow opening |
| `POLLING_INTERVAL_MINUTES` | `10` | Timer trigger interval |
| `NOTIFICATION_COOLDOWN_HOURS` | `1` | Min hours between non-urgent notifications |
| `ALLOWED_HVAC_MODES` | `cool,heatCool,auto` | HVAC modes where WindowBot is active |

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
| [NWS](https://www.weather.gov) | Outdoor temp, humidity, wind | `api.weather.gov` | None (free) |
| [Open-Meteo](https://open-meteo.com) | Outdoor temp, humidity, wind (peer) | `open-meteo.com/v1/forecast` | None (free) |
| [PurpleAir](https://www.purpleair.com) | AQI (PM2.5) | `api.purpleair.com` | Read key |
| [AirNow](https://www.airnow.gov) | AQI (fallback) | `airnowapi.org` | API key |
| [ntfy](https://ntfy.sh) | Push notifications | `ntfy.sh` | None (free) |

## License

MIT
