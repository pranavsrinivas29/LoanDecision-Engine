from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config.settings import PREPROCESSOR_PATH


def load_preprocessor(path: str | None = None) -> Any:
    """
    Load the saved preprocessing artifact.
    """
    preprocessor_path = path if path is not None else PREPROCESSOR_PATH
    return joblib.load(preprocessor_path)


def transform_features(
    preprocessor: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply preprocessing and return a DataFrame with transformed feature names.
    """
    X_processed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    X_processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names,
        index=X.index,
    )

    return X_processed_df