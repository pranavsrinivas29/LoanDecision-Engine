from __future__ import annotations

#import json
import logging
from pathlib import Path

import pandas as pd

from src.config.settings import (
    DRIFT_MONITOR_WINDOW_SIZE,
    ENABLE_AUTO_RETRAIN_ON_DRIFT,
    MIN_RECORDS_FOR_DRIFT_CHECK,
)
from src.db.repository import count_prediction_records, fetch_recent_prediction_records
from src.drift.detector import detect_feature_drift_from_dataframe
from src.drift.report_writer import save_drift_report
from src.retraining.pipeline import run_retraining_pipeline

logger = logging.getLogger(__name__)

DRIFT_DIR = Path("artifacts/drift")
REFERENCE_STATS_PATH = DRIFT_DIR / "reference_stats.json"
LATEST_LIVE_REPORT_PATH = DRIFT_DIR / "latest_live_drift_report.json"


def _compute_prediction_stats(df: pd.DataFrame) -> tuple[float | None, float | None]:
    probability_mean = None
    approval_rate = None

    if "probability" in df.columns and not df["probability"].dropna().empty:
        probability_mean = float(pd.to_numeric(df["probability"], errors="coerce").dropna().mean())

    if "prediction" in df.columns and not df["prediction"].dropna().empty:
        pred_series = df["prediction"].astype(str).str.strip().str.lower()
        approval_rate = float((pred_series == "approved").mean())

    return probability_mean, approval_rate


def run_live_drift_check_and_optional_retraining() -> dict:
    """
    Uses recent inference history as the live batch.
    Saves a drift report and optionally triggers retraining.
    """
    if not REFERENCE_STATS_PATH.exists():
        logger.info("Skipping live drift check: reference stats not found.")
        return {
            "status": "skipped",
            "reason": "Reference stats not found.",
        }

    total_records = count_prediction_records()
    if total_records < MIN_RECORDS_FOR_DRIFT_CHECK:
        logger.info(
            "Skipping live drift check: only %s prediction records available, need at least %s.",
            total_records,
            MIN_RECORDS_FOR_DRIFT_CHECK,
        )
        return {
            "status": "skipped",
            "reason": f"Not enough prediction records. Found {total_records}, need {MIN_RECORDS_FOR_DRIFT_CHECK}.",
        }

    df = fetch_recent_prediction_records(limit=DRIFT_MONITOR_WINDOW_SIZE)
    if df.empty:
        return {
            "status": "skipped",
            "reason": "No recent prediction records found.",
        }

    probability_mean, approval_rate = _compute_prediction_stats(df)

    report = detect_feature_drift_from_dataframe(
        df=df,
        reference_stats_path=REFERENCE_STATS_PATH,
        current_probability_mean=probability_mean,
        current_approval_rate=approval_rate,
    )
    save_drift_report(report, LATEST_LIVE_REPORT_PATH)

    result = {
        "status": "success",
        "drift_report_path": str(LATEST_LIVE_REPORT_PATH),
        "retraining_required": report.retraining_required,
        "drifted_features": report.drifted_features,
        "summary": report.summary,
    }

    if report.retraining_required and ENABLE_AUTO_RETRAIN_ON_DRIFT:
        logger.info("Live drift requires retraining. Triggering retraining pipeline.")
        retraining_result = run_retraining_pipeline(force=True)
        result["retraining_result"] = retraining_result
    else:
        result["retraining_result"] = {
            "status": "not_triggered",
            "reason": "Retraining not required or auto-retraining disabled.",
        }

    return result