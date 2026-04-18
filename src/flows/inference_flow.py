from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, task, get_run_logger

from src.config.settings import ARTIFACTS_DIR, PROJECT_ROOT
from src.agent.summarizer import answer_loan_question, load_llm
from src.explainability.explainer import (
    compute_xgb_contributions,
    extract_local_reason_summary,
)
from src.inference.pipeline import (
    get_actual_label_from_dataset,
    get_input_row_from_dataset,
    load_dataset_for_testing,
    load_inference_artifacts,
    run_single_inference,
)
from src.inference.preprocess import transform_features
from src.retrieval.knowledge_loader import (
    load_knowledge_documents,
    split_knowledge_documents,
)
from src.retrieval.vectorstore import (
    get_retriever,
    load_or_build_vectorstore,
)

load_dotenv(PROJECT_ROOT / ".env")


@task
def load_artifacts_task():
    preprocessor, model = load_inference_artifacts()
    return preprocessor, model


@task
def get_sample_input_task(row_idx: int) -> dict:
    input_data = get_input_row_from_dataset(row_idx=row_idx)
    actual_label = get_actual_label_from_dataset(row_idx=row_idx)
    return {
        "row_idx": row_idx,
        "input_data": input_data,
        "actual_label": actual_label,
    }


@task
def run_inference_task(input_data: dict, preprocessor, model) -> dict:
    return run_single_inference(
        input_data=input_data,
        preprocessor=preprocessor,
        model=model,
    )


@task
def run_local_explanation_task(row_idx: int, preprocessor, model) -> dict:
    X_full, _ = load_dataset_for_testing()
    X_processed_df = transform_features(preprocessor, X_full)

    contrib_df = compute_xgb_contributions(
        model=model,
        X_processed_df=X_processed_df,
    )

    local_explanation = extract_local_reason_summary(
        contrib_df=contrib_df,
        row_idx=row_idx,
        top_n=5,
    )
    return local_explanation


@task
def load_retriever_task():
    docs = load_knowledge_documents()
    split_docs = split_knowledge_documents(docs)
    vectorstore = load_or_build_vectorstore(split_docs=split_docs)
    retriever = get_retriever(vectorstore)
    return retriever


@task
def run_summary_task(
    input_data: dict,
    prediction_result: dict,
    local_explanation: dict,
    retriever,
) -> dict:
    llm = load_llm()
    result = answer_loan_question(
        question="Summarize this loan application for an analyst.",
        input_data=input_data,
        prediction_result=prediction_result,
        local_explanation=local_explanation,
        retriever=retriever,
        llm=llm,
    )
    return result


@task
def save_inference_output_task(payload: dict) -> str:
    output_path = ARTIFACTS_DIR / "prefect_inference_flow_output.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(output_path)


@task
def log_inference_summary_task(output_path: str, result_payload: dict):
    logger = get_run_logger()
    logger.info(f"Inference flow output saved to: {output_path}")
    logger.info(
        f"Prediction: {result_payload['prediction_result']['prediction']} "
        f"with probability {result_payload['prediction_result']['probability']:.4f}"
    )


@flow(name="loan-approval-inference-flow")
def inference_flow(row_idx: int = 0) -> dict:
    """
    Prefect flow for:
    - loading active artifacts
    - running one real inference
    - generating explanation
    - generating agent summary
    """
    preprocessor, model = load_artifacts_task()
    sample = get_sample_input_task(row_idx=row_idx)

    inference_output = run_inference_task(
        input_data=sample["input_data"],
        preprocessor=preprocessor,
        model=model,
    )

    local_explanation = run_local_explanation_task(
        row_idx=row_idx,
        preprocessor=preprocessor,
        model=model,
    )

    retriever = load_retriever_task()

    summary_result = run_summary_task(
        input_data=sample["input_data"],
        prediction_result=inference_output["prediction_result"],
        local_explanation=local_explanation,
        retriever=retriever,
    )

    result_payload = {
        "row_idx": sample["row_idx"],
        "actual_label": sample["actual_label"],
        "input_data": sample["input_data"],
        "prediction_result": inference_output["prediction_result"],
        "local_explanation": local_explanation,
        "summary": summary_result["answer"],
        "retrieved_docs": summary_result["retrieved_docs"],
    }

    output_path = save_inference_output_task(result_payload)
    log_inference_summary_task(output_path, result_payload)

    return result_payload


if __name__ == "__main__":
    result = inference_flow(row_idx=0)
    print(json.dumps(result, indent=2, default=str))