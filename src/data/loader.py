from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config.settings import (
    RAW_DATA_PATH,
    TARGET_COL,
    ID_COLUMNS,
    CATEGORICAL_TEXT_COLUMNS,
)


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw dataset from CSV.
    """
    csv_path = path if path is not None else RAW_DATA_PATH
    return pd.read_csv(csv_path)


def clean_target_column(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Standardize the target column to binary:
    approved -> 1
    rejected -> 0
    """
    df = df.copy()

    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"approved": 1, "rejected": 0})
    )

    return df


def clean_categorical_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from known categorical text columns.
    """
    df = df.copy()

    for col in CATEGORICAL_TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop ID-like columns if present.
    """
    df = df.copy()
    return df.drop(columns=ID_COLUMNS, errors="ignore")


def prepare_raw_dataframe(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Apply standard cleaning steps to the raw dataframe.
    """
    df = drop_id_columns(df)
    df = clean_categorical_text_columns(df)

    if target_col in df.columns:
        df = clean_target_column(df, target_col=target_col)

    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y