from __future__ import annotations

import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.dependencies import get_preprocessor_and_model
from src.api.schemas import LoanApplicationRequest, PredictResponse
from src.db.repository import insert_prediction_record
from src.drift.live_monitor import run_live_drift_check_and_optional_retraining
from src.inference.pipeline import run_single_inference
from src.monitoring.metrics import (
    APPROVALS_TOTAL,
    BACKEND_EXCEPTIONS_TOTAL,
    MODEL_LOAD_FAILURES_TOTAL,
    PREDICT_LATENCY_SECONDS,
    PREDICT_REQUESTS_TOTAL,
    PREDICTION_PROBABILITY,
    REJECTIONS_TOTAL,
)

router = APIRouter(tags=["prediction"])
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictResponse)
def predict_loan(
    request: LoanApplicationRequest,
    background_tasks: BackgroundTasks,
    artifacts=Depends(get_preprocessor_and_model),
):
    start_time = time.perf_counter()
    PREDICT_REQUESTS_TOTAL.inc()

    try:
        preprocessor, model = artifacts

        if preprocessor is None or model is None:
            MODEL_LOAD_FAILURES_TOTAL.inc()
            raise HTTPException(
                status_code=500,
                detail="Inference artifacts could not be loaded.",
            )

        input_data = request.model_dump()

        inference_output = run_single_inference(
            input_data=input_data,
            preprocessor=preprocessor,
            model=model,
        )

        prediction_result = inference_output["prediction_result"]

        prediction = str(prediction_result.get("prediction", "Unknown"))
        probability = prediction_result.get("probability")

        if probability is not None:
            try:
                PREDICTION_PROBABILITY.observe(float(probability))
            except (TypeError, ValueError):
                logger.warning("Could not record prediction probability: %s", probability)

        if prediction.lower() == "approved":
            APPROVALS_TOTAL.inc()
        else:
            REJECTIONS_TOTAL.inc()

        # Persist prediction history for live drift monitoring
        insert_prediction_record(
            input_data=input_data,
            prediction_result=prediction_result,
            local_explanation=None,
            summary=None,
        )

        # Non-blocking live drift + optional retraining trigger
        background_tasks.add_task(run_live_drift_check_and_optional_retraining)

        return PredictResponse(
            prediction_result=prediction_result
        )

    except HTTPException:
        BACKEND_EXCEPTIONS_TOTAL.labels(endpoint="/predict").inc()
        raise

    except Exception as e:
        BACKEND_EXCEPTIONS_TOTAL.labels(endpoint="/predict").inc()
        logger.exception("Predict endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        PREDICT_LATENCY_SECONDS.observe(time.perf_counter() - start_time)