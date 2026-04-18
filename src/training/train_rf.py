from __future__ import annotations

import mlflow
from sklearn.ensemble import RandomForestClassifier

from src.config.settings import ARTIFACTS_DIR
from src.mlflow_utils.logging import log_metrics, log_params, log_sklearn_model
from src.mlflow_utils.tracking import setup_mlflow
from src.training.common import (
    build_preprocessor,
    evaluate_model,
    fit_transform_datasets,
    load_training_data,
    save_model,
    save_preprocessor,
    split_data,
)


def train_random_forest():
    setup_mlflow()

    X, y = load_training_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)
    X_train_processed, X_val_processed, X_test_processed = fit_transform_datasets(
        preprocessor,
        X_train,
        X_val,
        X_test,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )

    with mlflow.start_run(run_name="random_forest"):
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": 42,
        }
        log_params(params)

        model.fit(X_train_processed, y_train)

        val_metrics, _, _ = evaluate_model(model, X_val_processed, y_val)
        test_metrics, _, _ = evaluate_model(model, X_test_processed, y_test)

        log_metrics(val_metrics, prefix="val_")
        log_metrics(test_metrics, prefix="test_")
        log_sklearn_model(model, artifact_path="model")

        save_preprocessor(preprocessor)
        model_path = ARTIFACTS_DIR / "random_forest_model.joblib"
        save_model(model, model_path)

        result = {
            "model_name": "RandomForest",
            "model": model,
            "model_path": str(model_path),
            "preprocessor": preprocessor,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }

    return result


if __name__ == "__main__":
    result = train_random_forest()
    print(result["test_metrics"])