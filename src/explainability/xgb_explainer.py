from __future__ import annotations

from typing import Any

import pandas as pd
import xgboost as xgb


def compute_xgb_contributions(
    model: Any,
    X_processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute native XGBoost feature contributions using pred_contribs=True.
    Returns a DataFrame with one extra 'bias' column.
    """
    booster = model.get_booster()

    dmat = xgb.DMatrix(
        X_processed_df,
        feature_names=X_processed_df.columns.tolist(),
    )

    contribs = booster.predict(dmat, pred_contribs=True)

    contrib_df = pd.DataFrame(
        contribs,
        columns=X_processed_df.columns.tolist() + ["bias"],
        index=X_processed_df.index,
    )
    return contrib_df