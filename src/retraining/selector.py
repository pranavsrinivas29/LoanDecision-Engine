from __future__ import annotations

import math
from typing import Any

import pandas as pd


RANKING_COLUMNS = ["roc_auc", "f1", "accuracy"]


def _safe_value(value: Any) -> float:
    try:
        value = float(value)
        if math.isnan(value):
            return float("-inf")
        return value
    except Exception:
        return float("-inf")


def select_best_model(comparison_df: pd.DataFrame) -> pd.Series:
    if comparison_df.empty:
        raise ValueError("comparison_df is empty")

    ranked = comparison_df.copy()

    ranked["_roc_auc_rank"] = ranked["roc_auc"].apply(_safe_value)
    ranked["_f1_rank"] = ranked["f1"].apply(_safe_value)
    ranked["_accuracy_rank"] = ranked["accuracy"].apply(_safe_value)

    ranked = ranked.sort_values(
        by=["_roc_auc_rank", "_f1_rank", "_accuracy_rank"],
        ascending=False,
    )

    return ranked.iloc[0]