from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import (
    CATEGORICAL_TEXT_COLUMNS,
    MODEL_COMPARISON_PATH,
    MODEL_INFO_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RAW_DATA_PATH,
    TARGET_COL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.retraining.promotion import promote_winning_model
from src.retraining.selector import select_best_model
from src.retraining.trainer import train_and_evaluate_candidate_models


RETRAINING_DIR = Path("artifacts/retraining")
LATEST_DRIFT_REPORT_PATH = Path("artifacts/drift/latest_drift_report.json")


def _prepare_training_dataframe() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH).copy()

    # Clean column names first
    df.columns = [col.strip() for col in df.columns]

    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Normalize target values safely
    raw_target = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_map = {
        "approved": 1,
        "rejected": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
    }

    df[TARGET_COL] = raw_target.map(target_map)

    if df[TARGET_COL].isna().any():
        bad_values = (
            raw_target[df[TARGET_COL].isna()]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Unmapped target values found in {TARGET_COL}: {bad_values}"
        )

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def run_retraining_pipeline(force: bool = False) -> dict:
    if not force:
        if not LATEST_DRIFT_REPORT_PATH.exists():
            return {
                "status": "skipped",
                "reason": "No drift report found.",
            }

        drift_report = json.loads(LATEST_DRIFT_REPORT_PATH.read_text())
        if not drift_report.get("retraining_required", False):
            return {
                "status": "skipped",
                "reason": "Drift report does not require retraining.",
            }

    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = _prepare_training_dataframe()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    run_timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    run_dir = RETRAINING_DIR / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"retraining_{run_timestamp}") as run:
        comparison_df = train_and_evaluate_candidate_models(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            categorical_columns=CATEGORICAL_TEXT_COLUMNS,
            run_dir=run_dir,
        )

        comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
        winner = select_best_model(comparison_df)

        promote_winning_model(
            source_model_path=winner["model_path"],
            source_preprocessor_path=winner["preprocessor_path"],
            active_model_path=MODEL_PATH,
            active_preprocessor_path=PREPROCESSOR_PATH,
            metadata_path=MODEL_INFO_PATH,
            winner_payload=winner.to_dict(),
        )

        for _, row in comparison_df.iterrows():
            mlflow.log_metrics(
                {
                    f"{row['model_name']}_accuracy": float(row["accuracy"]),
                    f"{row['model_name']}_precision": float(row["precision"]),
                    f"{row['model_name']}_recall": float(row["recall"]),
                    f"{row['model_name']}_f1": float(row["f1"]),
                    f"{row['model_name']}_roc_auc": float(row["roc_auc"]),
                }
            )

        mlflow.log_param("winner_model_name", winner["model_name"])
        mlflow.log_artifact(str(MODEL_COMPARISON_PATH))
        mlflow.log_artifact(str(MODEL_INFO_PATH))

        return {
            "status": "success",
            "winner_model_name": winner["model_name"],
            "winner_metrics": {
                "accuracy": float(winner["accuracy"]),
                "precision": float(winner["precision"]),
                "recall": float(winner["recall"]),
                "f1": float(winner["f1"]),
                "roc_auc": float(winner["roc_auc"]),
            },
            "run_id": run.info.run_id,
            "comparison_csv": str(MODEL_COMPARISON_PATH),
            "metadata_path": str(MODEL_INFO_PATH),
        }