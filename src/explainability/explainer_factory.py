from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.explainability.lr_explainer import compute_lr_contributions
from src.explainability.rf_explainer import compute_rf_contributions
from src.explainability.xgb_explainer import compute_xgb_contributions


def get_explainer_function(model: Any):
    """
    Return the correct contribution function based on model type.
    """
    if isinstance(model, XGBClassifier):
        return compute_xgb_contributions

    if isinstance(model, RandomForestClassifier):
        return compute_rf_contributions

    if isinstance(model, LogisticRegression):
        return compute_lr_contributions

    raise ValueError(f"Unsupported model type for explanation: {type(model).__name__}")