"""WindowBot Azure Function — Timer-triggered window advisory system."""

import azure.functions as func
import logging

from src.orchestrator import run_check

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
