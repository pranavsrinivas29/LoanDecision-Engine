from __future__ import annotations

import mlflow

from src.config.settings import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def setup_mlflow(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> str:
    """
    Configure MLflow tracking and experiment.
    Returns the experiment name.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def get_tracking_uri() -> str:
    return mlflow.get_tracking_uri()