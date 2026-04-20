"""WindowBot Azure Function — Timer-triggered window advisory system."""

import azure.functions as func
import logging

from src.orchestrator import run_check

app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 */10 * * * *",
    arg_name="timer",
    run_on_startup=False,
)
def windowbot_check(timer: func.TimerRequest) -> None:
    """Runs every 10 minutes to evaluate window open/close conditions."""
    if timer.past_due:
        logging.warning("Timer trigger is past due — running anyway.")

    run_check()
