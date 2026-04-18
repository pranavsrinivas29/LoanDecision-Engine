from __future__ import annotations

import mlflow
from xgboost import XGBClassifier

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


def train_xgboost():
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

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    with mlflow.start_run(run_name="xgboost"):
        params = {
            "model_type": "XGBClassifier",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        log_params(params)

        model.fit(X_train_processed, y_train)

        val_metrics, _, _ = evaluate_model(model, X_val_processed, y_val)
        test_metrics, _, _ = evaluate_model(model, X_test_processed, y_test)

        log_metrics(val_metrics, prefix="val_")
        log_metrics(test_metrics, prefix="test_")
        log_sklearn_model(model, artifact_path="model")

        save_preprocessor(preprocessor)
        model_path = ARTIFACTS_DIR / "xgboost_model.joblib"
        save_model(model, model_path)

        result = {
            "model_name": "XGBoost",
            "model": model,
            "model_path": str(model_path),
            "preprocessor": preprocessor,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }

    return result


if __name__ == "__main__":
    result = train_xgboost()
    print(result["test_metrics"])