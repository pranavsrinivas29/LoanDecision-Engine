from __future__ import annotations

from pathlib import Path

from prefect import flow, task, get_run_logger

from src.config.settings import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_INFO_PATH,
    DB_PATH,
)
from src.mlflow_utils.model_info import load_active_model_metadata


@task
def validate_file_exists(path: Path, label: str) -> dict:
    exists = path.exists()
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
    }


@task
def validate_model_metadata_task() -> dict:
    metadata = load_active_model_metadata()
    required_keys = [
        "model_type",
        "selected_metric",
        "selected_metric_value",
        "training_timestamp",
        "active_model_path",
        "active_preprocessor_path",
    ]

    missing_keys = [key for key in required_keys if key not in metadata]
    return {
        "exists": bool(metadata),
        "missing_keys": missing_keys,
        "metadata": metadata,
    }


@task
def log_validation_results_task(file_results: list[dict], metadata_result: dict):
    logger = get_run_logger()

    for result in file_results:
        if result["exists"]:
            logger.info(f"[OK] {result['label']} exists at {result['path']}")
        else:
            logger.warning(f"[MISSING] {result['label']} expected at {result['path']}")

    if metadata_result["exists"] and not metadata_result["missing_keys"]:
        logger.info("[OK] Active model metadata is valid.")
    else:
        logger.warning(
            f"[WARN] Metadata missing keys: {metadata_result['missing_keys']}"
        )


@flow(name="loan-approval-validation-flow")
def validation_flow() -> dict:
    """
    Prefect flow for validating core artifacts and metadata.
    """
    file_results = [
        validate_file_exists(MODEL_PATH, "Active model artifact"),
        validate_file_exists(PREPROCESSOR_PATH, "Active preprocessor artifact"),
        validate_file_exists(MODEL_COMPARISON_PATH, "Model comparison CSV"),
        validate_file_exists(MODEL_INFO_PATH, "Active model metadata JSON"),
        validate_file_exists(DB_PATH, "SQLite database"),
    ]

    metadata_result = validate_model_metadata_task()
    log_validation_results_task(file_results, metadata_result)

    return {
        "file_results": file_results,
        "metadata_result": metadata_result,
    }


if __name__ == "__main__":
    result = validation_flow()
    print(result)