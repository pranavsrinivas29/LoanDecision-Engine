from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.settings import LOCAL_EXPLANATION_TOP_N
from src.explainability.explainer_factory import get_explainer_function


def compute_model_contributions(
    model: Any,
    X_processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Unified entry point for local contribution computation across supported models.
    """
    explainer_fn = get_explainer_function(model)
    return explainer_fn(model=model, X_processed_df=X_processed_df)


def extract_local_reason_summary(
    contrib_df: pd.DataFrame,
    row_idx: int,
    top_n: int = LOCAL_EXPLANATION_TOP_N,
) -> dict:
    """
    Extract top positive and negative contributors for one row.
    Assumes contrib_df contains a 'bias' column.
    """
    row_contrib = contrib_df.loc[row_idx].drop("bias").sort_values()

    top_negative = row_contrib.head(top_n)
    top_positive = row_contrib.tail(top_n).sort_values(ascending=False)

    return {
        "top_positive": {k: float(v) for k, v in top_positive.to_dict().items()},
        "top_negative": {k: float(v) for k, v in top_negative.to_dict().items()},
    }


def compute_global_importance(contrib_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global mean absolute contribution importance.
    Assumes contrib_df contains a 'bias' column.
    """
    global_importance_df = (
        contrib_df.drop(columns=["bias"])
        .abs()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    global_importance_df.columns = ["feature", "mean_abs_contribution"]
    return global_importance_df