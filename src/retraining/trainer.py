from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.retraining.evaluator import evaluate_classifier, get_positive_class_probability


def build_preprocessor(
    X_train: pd.DataFrame,
    categorical_columns: list[str],
) -> ColumnTransformer:
    numeric_columns = [col for col in X_train.columns if col not in categorical_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    return preprocessor


def get_candidate_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=4,
        ),
    }


def train_and_evaluate_candidate_models(
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    y_valid,
    categorical_columns: list[str],
    run_dir: str | Path,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor(X_train, categorical_columns)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    preprocessor_path = run_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    rows: list[dict[str, Any]] = []
    models = get_candidate_models()

    for model_name, model in models.items():
        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_valid_processed)
        y_proba = get_positive_class_probability(model, X_valid_processed)

        metrics = evaluate_classifier(
            y_true=y_valid,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        model_path = run_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        rows.append(
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "preprocessor_path": str(preprocessor_path),
                **metrics,
            }
        )

    return pd.DataFrame(rows)