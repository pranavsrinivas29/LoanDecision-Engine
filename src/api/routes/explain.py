from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_preprocessor_and_model
from src.api.schemas import LoanApplicationRequest, ExplainResponse
from src.explainability.explainer import (
    compute_model_contributions,
    extract_local_reason_summary,
)
from src.inference.pipeline import run_single_inference
from src.monitoring.metrics import (
    APPROVALS_TOTAL,
    BACKEND_EXCEPTIONS_TOTAL,
    EXPLAIN_LATENCY_SECONDS,
    EXPLAIN_REQUESTS_TOTAL,
    MODEL_LOAD_FAILURES_TOTAL,
    PREDICTION_PROBABILITY,
    REJECTIONS_TOTAL,
)

router = APIRouter(tags=["explanation"])
logger = logging.getLogger(__name__)


@router.post("/explain", response_model=ExplainResponse)
def explain_loan(
    request: LoanApplicationRequest,
    artifacts=Depends(get_preprocessor_and_model),
):
    start_time = time.perf_counter()
    EXPLAIN_REQUESTS_TOTAL.inc()

    try:
        preprocessor, model = artifacts

        if preprocessor is None or model is None:
            MODEL_LOAD_FAILURES_TOTAL.inc()
            raise HTTPException(
                status_code=500,
                detail="Inference artifacts could not be loaded.",
            )

        inference_output = run_single_inference(
            input_data=request.model_dump(),
            preprocessor=preprocessor,
            model=model,
        )

        processed_features = inference_output["processed_features"]
        prediction_result = inference_output["prediction_result"]

        prediction = str(prediction_result.get("prediction", "Unknown"))
        probability = prediction_result.get("probability")

        if probability is not None:
            try:
                PREDICTION_PROBABILITY.observe(float(probability))
            except (TypeError, ValueError):
                logger.warning(
                    "Could not record prediction probability in /explain: %s",
                    probability,
                )

        if prediction.lower() == "approved":
            APPROVALS_TOTAL.inc()
        else:
            REJECTIONS_TOTAL.inc()

        contrib_df = compute_model_contributions(
        model=model,
        X_processed_df=processed_features,
        )

        local_explanation = extract_local_reason_summary(
            contrib_df=contrib_df,
            row_idx=processed_features.index[0],
            top_n=5,
        )

        return ExplainResponse(
            prediction_result=prediction_result,
            local_explanation=local_explanation,
        )

    except HTTPException:
        BACKEND_EXCEPTIONS_TOTAL.labels(endpoint="/explain").inc()
        raise

    except Exception as e:
        BACKEND_EXCEPTIONS_TOTAL.labels(endpoint="/explain").inc()
        logger.exception("Explain endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        EXPLAIN_LATENCY_SECONDS.observe(time.perf_counter() - start_time)