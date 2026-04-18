from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn


def log_params(params: dict[str, Any]):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict[str, float], prefix: str = ""):
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if prefix else key
        mlflow.log_metric(metric_name, float(value))


def log_artifact_if_exists(path: str | Path):
    path = Path(path)
    if path.exists():
        mlflow.log_artifact(str(path))


def log_sklearn_model(model, artifact_path: str = "model"):
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)