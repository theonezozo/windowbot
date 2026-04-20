# WindowBot

Hyperlocal window advisory system — tells you when to open or close your windows based on indoor sensors (Ecobee), outdoor weather (NWS), and air quality (PurpleAir / AirNow).

Runs as an Azure Function on a 10-minute timer and sends push notifications via [ntfy.sh](https://ntfy.sh).

## How It Works

Every 10 minutes, WindowBot:
1. Reads indoor temperature from your Ecobee remote sensors
2. Fetches outdoor conditions from the 3 nearest NWS personal weather stations (median)
3. Gets AQI from the 3 nearest PurpleAir sensors (median), falling back to AirNow
4. Decides per-floor whether windows should be open or closed
5. Sends a push notification via ntfy.sh if the recommendation changes

### Decision Logic

- **Open** when outdoor temp < warmest indoor sensor − 1°F AND AQI < 50
- **Close** when outdoor temp > coolest indoor sensor + 1°F
- **Urgent close** when AQI ≥ 100 (bypasses notification cooldown)
- **Block opening** when outdoor humidity > 80%
- **Only active** when HVAC is in cooling/auto mode

## Project Structure

```
function_app.py          # Azure Function entry point (timer trigger)
host.json                # Azure Functions host configuration
requirements.txt         # Python dependencies
local.settings.json      # Local dev settings (not committed)

src/
  config.py              # Environment variable loader
  orchestrator.py        # Main fetch → decide → notify pipeline
  notifier.py            # ntfy.sh push notification client
  state.py               # Azure Table Storage state manager
  ecobee_client.py       # Ecobee thermostat API client (OAuth2)
  nws_client.py          # National Weather Service client (personal stations)
  purpleair_client.py    # PurpleAir AQI client (median of 3)
  airnow_client.py       # AirNow fallback AQI client
  decision_engine.py     # Per-floor open/close logic with hysteresis

tests/
  test_decision_engine.py  # 82 tests covering all decision logic
```

## Local Development

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install pytest
   ```

2. **Run tests:**
   ```bash
   pytest
   ```

3. **Configure settings:**
   Copy `local.settings.json.example` to `local.settings.json` and fill in your API keys and sensor names:
   ```bash
   cp local.settings.json.example local.settings.json
   ```

4. **Run locally with Azure Functions Core Tools:**
   ```bash
   func start
   ```

## Deployment

Deploy to Azure using the Azure Functions Core Tools:

```bash
func azure functionapp publish func-windowbot-prod
```

## Cost

$0.00–0.05/month on Azure consumption plan free tier.
