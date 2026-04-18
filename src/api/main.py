from __future__ import annotations

import logging

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes.explain import router as explain_router
from src.api.routes.health import router as health_router
from src.api.routes.history import router as history_router
from src.api.routes.model_info import router as model_info_router
from src.api.routes.predict import router as predict_router
from src.api.routes.summarize import router as summarize_router
from src.config.settings import MODEL_INFO_PATH
from src.db.models import create_tables
from src.monitoring.helpers import set_active_model_metric
from src.monitoring.logging_config import setup_logging
from src.api.routes.drift import router as drift_router
from src.api.routes.retraining import router as retraining_router
from src.api.routes.live_drift import router as live_drift_router

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loan Approval Agent API",
    version="1.0.0",
    description="FastAPI backend for loan approval prediction, explanation, summary, persistence, and model metadata.",
)


@app.on_event("startup")
def startup_event() -> None:
    create_tables()
    set_active_model_metric(MODEL_INFO_PATH)
    logger.info("Loan Approval Agent API startup complete.")


@app.get("/")
def root():
    return {"message": "Loan Approval Agent API is running."}


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(summarize_router)
app.include_router(history_router)
app.include_router(model_info_router)
app.include_router(drift_router)
app.include_router(retraining_router)
app.include_router(live_drift_router)

Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)