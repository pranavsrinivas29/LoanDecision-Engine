from __future__ import annotations

from prefect import flow, task, get_run_logger

from src.training.train_all import train_all_models


@task(retries=1, retry_delay_seconds=5)
def run_training_task() -> dict:
    """
    Run the full multi-model training pipeline.
    """
    output = train_all_models()
    return {
        "best_model_type": output["metadata"]["model_type"],
        "selected_metric": output["metadata"]["selected_metric"],
        "selected_metric_value": output["metadata"]["selected_metric_value"],
        "training_timestamp": output["metadata"]["training_timestamp"],
        "comparison_rows": output["comparison_df"].to_dict(orient="records"),
        "metadata": output["metadata"],
    }


@task
def log_training_summary_task(training_output: dict):
    """
    Log a readable summary of the training results.
    """
    logger = get_run_logger()
    logger.info("Training pipeline completed successfully.")
    logger.info(f"Best model: {training_output['best_model_type']}")
    logger.info(
        f"Selected metric: {training_output['selected_metric']} = "
        f"{training_output['selected_metric_value']}"
    )
    logger.info(f"Training timestamp: {training_output['training_timestamp']}")


@flow(name="loan-approval-training-flow")
def training_flow() -> dict:
    """
    Prefect flow for:
    - training LR/RF/XGB
    - comparing models
    - selecting best model
    - saving artifacts and metadata
    """
    training_output = run_training_task()
    log_training_summary_task(training_output)
    return training_output


if __name__ == "__main__":
    result = training_flow()
    print(result)