"""WindowBot Azure Function — Timer-triggered window advisory system."""

import azure.functions as func
import logging

from src.orchestrator import run_check
from src.status_page import render_status_page
from src.version_info import VERSION, WORKER_STARTED_AT

# Suppress Azure SDK HTTP transport logs (request/response traces) unless errors occur.
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

logging.getLogger("windowbot").info(
    "WindowBot worker started: commit=%s branch=%s build_time=%s started_at=%s",
    VERSION["commit_sha"],
    VERSION["branch"],
    VERSION["build_time"],
    WORKER_STARTED_AT.isoformat(),
)

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 */10 * * * *",
    arg_name="timer",
    run_on_startup=True,
)
def windowbot_check(timer: func.TimerRequest) -> None:
    """Runs every 10 minutes to evaluate window open/close conditions."""
    if timer.past_due:
        logging.warning("Timer trigger is past due — running anyway.")

    run_check()


@app.route(route="check", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def windowbot_check_http(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger for local development — same logic as the timer."""
    run_check()
    return func.HttpResponse("OK", status_code=200)


@app.route(route="status", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def windowbot_status(req: func.HttpRequest) -> func.HttpResponse:
    """Status page showing last persisted state.

    Returns HTML by default, JSON if Accept: application/json or ?format=json.
    Protected by a PIN stored in the STATUS_PAGE_PIN app setting.
    Access via: /api/status?pin=<your-pin>
    """
    return render_status_page(req)
