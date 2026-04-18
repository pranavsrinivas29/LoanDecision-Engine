from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

from src.drift.reference_builder import NUMERIC_DRIFT_FEATURES
from src.drift.schemas import (
    DriftReport,
    FeatureCurrentStats,
    FeatureDriftResult,
    FeatureReferenceStats,
    PredictionDriftResult,
)


DEFAULT_THRESHOLDS = {
    "mean_shift_ratio": 0.30,
    "std_shift_ratio": 0.50,
    "missing_shift_abs": 0.10,
    "drifted_feature_count_for_retraining": 2,
    "approval_rate_shift_abs": 0.15,
    "probability_mean_shift_abs": 0.10,
}


def _safe_std(series: pd.Series) -> float:
    value = float(series.std(ddof=0)) if len(series.dropna()) > 0 else 0.0
    return value if value > 0 else 1e-8


def _current_stats(series: pd.Series) -> FeatureCurrentStats:
    return FeatureCurrentStats(
        mean=float(series.mean()),
        std=_safe_std(series),
        min=float(series.min()),
        max=float(series.max()),
        q25=float(series.quantile(0.25)),
        median=float(series.quantile(0.50)),
        q75=float(series.quantile(0.75)),
        missing_fraction=float(series.isna().mean()),
    )


def _detect_feature_drift(
    feature_name: str,
    ref_stats: FeatureReferenceStats,
    current_stats: FeatureCurrentStats,
    thresholds: dict,
) -> FeatureDriftResult:
    mean_shift_ratio = abs(current_stats.mean - ref_stats.mean) / max(abs(ref_stats.mean), 1e-8)
    std_shift_ratio = abs(current_stats.std - ref_stats.std) / max(abs(ref_stats.std), 1e-8)
    missing_shift_abs = abs(current_stats.missing_fraction - ref_stats.missing_fraction)

    reasons: list[str] = []

    if mean_shift_ratio > thresholds["mean_shift_ratio"]:
        reasons.append("mean_shift_exceeded")

    if std_shift_ratio > thresholds["std_shift_ratio"]:
        reasons.append("std_shift_exceeded")

    if missing_shift_abs > thresholds["missing_shift_abs"]:
        reasons.append("missing_fraction_shift_exceeded")

    return FeatureDriftResult(
        feature_name=feature_name,
        reference=ref_stats,
        current=current_stats,
        mean_shift_ratio=float(mean_shift_ratio),
        std_shift_ratio=float(std_shift_ratio),
        missing_shift_abs=float(missing_shift_abs),
        drift_detected=len(reasons) > 0,
        reasons=reasons,
    )


def detect_prediction_drift(
    reference_probability_mean: float | None,
    current_probability_mean: float | None,
    reference_approval_rate: float | None,
    current_approval_rate: float | None,
    thresholds: dict,
) -> PredictionDriftResult:
    probability_mean_shift_abs = None
    approval_rate_shift_abs = None
    reasons: list[str] = []

    if (
        reference_probability_mean is not None
        and current_probability_mean is not None
    ):
        probability_mean_shift_abs = abs(current_probability_mean - reference_probability_mean)
        if probability_mean_shift_abs > thresholds["probability_mean_shift_abs"]:
            reasons.append("probability_mean_shift_exceeded")

    if (
        reference_approval_rate is not None
        and current_approval_rate is not None
    ):
        approval_rate_shift_abs = abs(current_approval_rate - reference_approval_rate)
        if approval_rate_shift_abs > thresholds["approval_rate_shift_abs"]:
            reasons.append("approval_rate_shift_exceeded")

    return PredictionDriftResult(
        reference_probability_mean=reference_probability_mean,
        current_probability_mean=current_probability_mean,
        probability_mean_shift_abs=probability_mean_shift_abs,
        reference_approval_rate=reference_approval_rate,
        current_approval_rate=current_approval_rate,
        approval_rate_shift_abs=approval_rate_shift_abs,
        drift_detected=len(reasons) > 0,
        reasons=reasons,
    )


def detect_feature_drift_from_dataframe(
    df: pd.DataFrame,
    reference_stats_path: str | Path,
    thresholds: dict | None = None,
    current_probability_mean: float | None = None,
    current_approval_rate: float | None = None,
) -> DriftReport:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    reference_stats_path = Path(reference_stats_path)

    reference_payload = json.loads(reference_stats_path.read_text())
    reference_features = reference_payload["features"]

    results: list[FeatureDriftResult] = []
    drifted_features: list[str] = []

    for col in NUMERIC_DRIFT_FEATURES:
        if col not in df.columns or col not in reference_features:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        current_stats = _current_stats(series)
        ref_stats = FeatureReferenceStats(**reference_features[col])

        feature_result = _detect_feature_drift(
            feature_name=col,
            ref_stats=ref_stats,
            current_stats=current_stats,
            thresholds=thresholds,
        )

        results.append(feature_result)
        if feature_result.drift_detected:
            drifted_features.append(col)

    pred_ref = reference_payload.get("prediction_reference", {})
    prediction_drift = detect_prediction_drift(
        reference_probability_mean=pred_ref.get("probability_mean"),
        current_probability_mean=current_probability_mean,
        reference_approval_rate=pred_ref.get("approval_rate"),
        current_approval_rate=current_approval_rate,
        thresholds=thresholds,
    )

    retraining_required = (
        len(drifted_features) >= thresholds["drifted_feature_count_for_retraining"]
        or prediction_drift.drift_detected
    )

    summary = (
        f"Detected drift in {len(drifted_features)} feature(s): {', '.join(drifted_features)}."
        if drifted_features
        else "No feature drift detected."
    )

    if prediction_drift.drift_detected:
        summary += f" Prediction drift detected: {', '.join(prediction_drift.reasons)}."

    return DriftReport(
        reference_source=reference_payload.get("reference_source", "unknown"),
        batch_size=len(df),
        drifted_features=drifted_features,
        feature_results=results,
        prediction_drift=prediction_drift,
        retraining_required=retraining_required,
        summary=summary,
        created_at=datetime.now(UTC).isoformat(),
        thresholds=thresholds,
    )