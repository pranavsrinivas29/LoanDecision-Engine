from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import (
    get_llm_dependency,
    get_preprocessor_and_model,
    get_retriever_dependency,
)
from src.api.schemas import LoanApplicationRequest, SummarizeResponse
from src.agent.summarizer import answer_loan_question
from src.db.repository import insert_prediction_record
from src.explainability.explainer import (
    compute_model_contributions,
    extract_local_reason_summary,
)
from src.inference.pipeline import run_single_inference
from src.monitoring.metrics import (
    APPROVALS_TOTAL,
    BACKEND_EXCEPTIONS_TOTAL,
    MODEL_LOAD_FAILURES_TOTAL,
    PREDICTION_PROBABILITY,
    REJECTIONS_TOTAL,
    SUMMARIZE_LATENCY_SECONDS,
    SUMMARIZE_REQUESTS_TOTAL,
)

router = APIRouter(tags=["summary"])
logger = logging.getLogger(__name__)


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_loan(
    request: LoanApplicationRequest,
    artifacts=Depends(get_preprocessor_and_model),
    retriever=Depends(get_retriever_dependency),
    llm=Depends(get_llm_dependency),
):
    start_time = time.perf_counter()
    SUMMARIZE_REQUESTS_TOTAL.inc()

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

        processed_features = inference_output["processed_features"]
        prediction_result = inference_output["prediction_result"]

        prediction = str(prediction_result.get("prediction", "Unknown"))
        probability = prediction_result.get("probability")

        if probability is not None:
            try:
                PREDICTION_PROBABILITY.observe(float(probability))
            except (TypeError, ValueError):
                logger.warning(
                    "Could not record prediction probability in /summarize: %s",
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

        summary_result = answer_loan_question(
            question="Summarize this loan application for an analyst.",
            input_data=input_data,
            prediction_result=prediction_result,
            local_explanation=local_explanation,
            retriever=retriever,
            llm=llm,
        )

        summary_text = summary_result["answer"]

        insert_prediction_record(
            input_data=input_data,
            prediction_result=prediction_result,
            local_explanation=local_explanation,
            summary=summary_text,
        )

        return SummarizeResponse(
            prediction_result=prediction_result,
            local_explanation=local_explanation,
            summary=summary_text,
        )

    except HTTPException:
        BACKEND_EXCEPTIONS_TOTAL.labels(endpoint="/summarize").inc()
        raise

    except Exception as e:
        BACKEND_EXCEPTIONS_TOTAL.labels(endpoint="/summarize").inc()
        logger.exception("Summarize endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        SUMMARIZE_LATENCY_SECONDS.observe(time.perf_counter() - start_time)