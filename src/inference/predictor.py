from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config.settings import MODEL_PATH, PREDICTION_THRESHOLD


def load_model(path: str | None = None) -> Any:
    """
    Load the saved best model.
    """
    model_path = path if path is not None else MODEL_PATH
    return joblib.load(model_path)


def predict_with_model(
    model: Any,
    X_processed: pd.DataFrame,
    threshold: float = PREDICTION_THRESHOLD,
) -> dict:
    """
    Run prediction and return structured result.
    """
    probabilities = model.predict_proba(X_processed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    label_map = {1: "approved", 0: "rejected"}

    result = {
        "predicted_label_numeric": int(predictions[0]),
        "prediction": label_map[int(predictions[0])],
        "probability": float(probabilities[0]),
        "threshold": float(threshold),
    }
    return result