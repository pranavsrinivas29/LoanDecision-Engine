from __future__ import annotations

import json

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


def main(row_idx: int = 0):
    print("=" * 100)
    print("LOADING ARTIFACTS")
    print("=" * 100)
    preprocessor, model = load_inference_artifacts()

    print("=" * 100)
    print("LOADING REAL INPUT ROW")
    print("=" * 100)
    input_data = get_input_row_from_dataset(row_idx=row_idx)
    actual_label = get_actual_label_from_dataset(row_idx=row_idx)

    print("Input data:")
    print(json.dumps(input_data, indent=2, default=str))
    print(f"Actual label: {actual_label}")

    print("=" * 100)
    print("RUNNING INFERENCE")
    print("=" * 100)
    inference_output = run_single_inference(
        input_data=input_data,
        preprocessor=preprocessor,
        model=model,
    )

    prediction_result = inference_output["prediction_result"]
    print("Prediction result:")
    print(json.dumps(prediction_result, indent=2))

    print("=" * 100)
    print("COMPUTING LOCAL EXPLANATION")
    print("=" * 100)
    X_full, _ = load_dataset_for_testing()
    X_processed_df = transform_features(preprocessor, X_full)

    contrib_df = compute_xgb_contributions(model=model, X_processed_df=X_processed_df)
    local_explanation = extract_local_reason_summary(contrib_df, row_idx=row_idx, top_n=5)

    print("Local explanation:")
    print(json.dumps(local_explanation, indent=2))

    print("=" * 100)
    print("LOADING RETRIEVAL PIPELINE")
    print("=" * 100)
    docs = load_knowledge_documents()
    split_docs = split_knowledge_documents(docs)
    vectorstore = load_or_build_vectorstore(split_docs=split_docs)
    retriever = get_retriever(vectorstore)

    print("=" * 100)
    print("RUNNING AGENT SUMMARY")
    print("=" * 100)
    llm = load_llm()

    question = "Summarize this loan application for an analyst."
    summary_result = answer_loan_question(
        question=question,
        input_data=input_data,
        prediction_result=prediction_result,
        local_explanation=local_explanation,
        retriever=retriever,
        llm=llm,
    )

    print("Agent answer:")
    print(summary_result["answer"])

    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main(row_idx=0)