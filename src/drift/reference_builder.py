from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.drift.schemas import FeatureReferenceStats


NUMERIC_DRIFT_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0)) if len(series.dropna()) > 0 else 0.0
    return value if value > 0 else 1e-8


def build_reference_stats(df: pd.DataFrame) -> dict:
    feature_stats: dict[str, dict] = {}

    for col in NUMERIC_DRIFT_FEATURES:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        stats = FeatureReferenceStats(
            mean=float(series.mean()),
            std=_safe_std(series),
            min=float(series.min()),
            max=float(series.max()),
            q25=float(series.quantile(0.25)),
            median=float(series.quantile(0.50)),
            q75=float(series.quantile(0.75)),
            missing_fraction=float(series.isna().mean()),
        )

        feature_stats[col] = stats.model_dump()

    reference = {
        "reference_source": "training_data",
        "features": feature_stats,
        "prediction_reference": {
            "approval_rate": None,
            "probability_mean": None,
        },
    }
    return reference


def save_reference_stats(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference = build_reference_stats(df)
    output_path.write_text(json.dumps(reference, indent=2))
    return output_path