from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FeatureReferenceStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    q25: float
    median: float
    q75: float
    missing_fraction: float


class FeatureCurrentStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    q25: float
    median: float
    q75: float
    missing_fraction: float


class FeatureDriftResult(BaseModel):
    feature_name: str
    reference: FeatureReferenceStats
    current: FeatureCurrentStats
    mean_shift_ratio: float
    std_shift_ratio: float
    missing_shift_abs: float
    drift_detected: bool
    reasons: list[str]


class PredictionDriftResult(BaseModel):
    reference_probability_mean: float | None = None
    current_probability_mean: float | None = None
    probability_mean_shift_abs: float | None = None
    reference_approval_rate: float | None = None
    current_approval_rate: float | None = None
    approval_rate_shift_abs: float | None = None
    drift_detected: bool = False
    reasons: list[str] = []


class DriftReport(BaseModel):
    reference_source: str
    batch_size: int
    drifted_features: list[str]
    feature_results: list[FeatureDriftResult]
    prediction_drift: PredictionDriftResult | None = None
    retraining_required: bool
    summary: str
    created_at: str
    thresholds: dict[str, Any]