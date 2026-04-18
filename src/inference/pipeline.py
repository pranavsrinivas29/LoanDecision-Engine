from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.settings import TARGET_COL
from src.data.loader import (
    load_raw_data,
    prepare_raw_dataframe,
    split_features_target,
)
from src.inference.preprocess import load_preprocessor, transform_features
from src.inference.predictor import load_model, predict_with_model


def load_inference_artifacts() -> tuple[Any, Any]:
    """
    Load the saved preprocessor and model artifacts.
    """
    preprocessor = load_preprocessor()
    model = load_model()
    return preprocessor, model


def build_feature_frame_from_input(input_data: dict) -> pd.DataFrame:
    """
    Convert a single input dictionary into a one-row DataFrame.
    """
    return pd.DataFrame([input_data])


def run_single_inference(
    input_data: dict,
    preprocessor: Any,
    model: Any,
) -> dict:
    """
    Run preprocessing + prediction for a single input record.
    """
    X_input = build_feature_frame_from_input(input_data)
    X_processed_df = transform_features(preprocessor, X_input)
    prediction_result = predict_with_model(model, X_processed_df)

    return {
        "input_data": input_data,
        "processed_features": X_processed_df,
        "prediction_result": prediction_result,
    }


def load_dataset_for_testing() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare full dataset for local testing.
    """
    df = load_raw_data()
    df = prepare_raw_dataframe(df, target_col=TARGET_COL)
    X, y = split_features_target(df, target_col=TARGET_COL)
    return X, y


def get_input_row_from_dataset(row_idx: int) -> dict:
    """
    Fetch one real input record from the prepared dataset.
    """
    X, _ = load_dataset_for_testing()
    return X.loc[row_idx].to_dict()


def get_actual_label_from_dataset(row_idx: int) -> int:
    """
    Fetch actual label for a given row index from the prepared dataset.
    """
    _, y = load_dataset_for_testing()
    return int(y.loc[row_idx])