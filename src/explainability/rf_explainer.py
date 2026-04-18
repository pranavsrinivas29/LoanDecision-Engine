from __future__ import annotations

from typing import Any

import pandas as pd
import shap


def compute_rf_contributions(
    model: Any,
    X_processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Random Forest feature contributions using SHAP TreeExplainer.
    Returns a DataFrame with one extra 'bias' column.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed_df)

    # Binary classification can return a list of arrays
    if isinstance(shap_values, list):
        # take positive-class explanations if available
        if len(shap_values) > 1:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    contrib_df = pd.DataFrame(
        shap_values,
        columns=X_processed_df.columns.tolist(),
        index=X_processed_df.index,
    )

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    contrib_df["bias"] = float(expected_value)
    return contrib_df