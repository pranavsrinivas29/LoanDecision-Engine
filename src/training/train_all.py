from __future__ import annotations

from datetime import datetime

import joblib
import pandas as pd

from src.config.settings import (
    MODEL_COMPARISON_PATH,
    MODEL_INFO_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)
from src.mlflow_utils.model_info import save_active_model_metadata
from src.training.common import save_model_comparison
from src.training.train_lr import train_logistic_regression
from src.training.train_rf import train_random_forest
from src.training.train_xgb import train_xgboost


def choose_best_model(results: list[dict], metric_name: str = "f1") -> dict:
    """
    Choose best model based on a metric from test_metrics.
    """
    best_result = max(results, key=lambda x: x["test_metrics"][metric_name])
    return best_result


def train_all_models():
    lr_result = train_logistic_regression()
    rf_result = train_random_forest()
    xgb_result = train_xgboost()

    all_results = [lr_result, rf_result, xgb_result]

    comparison_rows = []
    for result in all_results:
        row = {
            "model": result["model_name"],
            **result["test_metrics"],
            "mlflow_run_id": result["mlflow_run_id"],
            "model_path": result["model_path"],
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="f1", ascending=False)
    save_model_comparison(comparison_df)

    best_result = choose_best_model(all_results, metric_name="f1")

    # Save chosen model and preprocessor as active deployment artifacts
    joblib.dump(best_result["model"], MODEL_PATH)
    joblib.dump(best_result["preprocessor"], PREPROCESSOR_PATH)

    metadata = {
        "model_type": best_result["model_name"],
        "selected_metric": "f1",
        "selected_metric_value": best_result["test_metrics"]["f1"],
        "test_metrics": best_result["test_metrics"],
        "mlflow_run_id": best_result["mlflow_run_id"],
        "source_model_path": best_result["model_path"],
        "active_model_path": str(MODEL_PATH),
        "active_preprocessor_path": str(PREPROCESSOR_PATH),
        "comparison_csv_path": str(MODEL_COMPARISON_PATH),
        "training_timestamp": datetime.utcnow().isoformat(),
    }
    save_active_model_metadata(metadata, path=MODEL_INFO_PATH)

    return {
        "comparison_df": comparison_df,
        "best_result": best_result,
        "metadata": metadata,
    }


if __name__ == "__main__":
    output = train_all_models()
    print(output["comparison_df"])
    print("\nBest model:")
    print(output["metadata"])