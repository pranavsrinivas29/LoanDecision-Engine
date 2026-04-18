from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config.settings import RAW_DATA_PATH
from src.drift.detector import detect_feature_drift_from_dataframe
from src.drift.reference_builder import save_reference_stats
from src.drift.report_writer import save_drift_report

router = APIRouter(prefix="/drift", tags=["drift"])

DRIFT_DIR = Path("artifacts/drift")
REFERENCE_STATS_PATH = DRIFT_DIR / "reference_stats.json"
LATEST_REPORT_PATH = DRIFT_DIR / "latest_drift_report.json"


@router.post("/build-reference")
def build_reference():
    if not Path(RAW_DATA_PATH).exists():
        raise HTTPException(status_code=404, detail=f"Training data not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    save_reference_stats(df, REFERENCE_STATS_PATH)

    return {
        "message": "Reference stats created successfully.",
        "reference_stats_path": str(REFERENCE_STATS_PATH),
    }


@router.get("/status")
def drift_status():
    if not REFERENCE_STATS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Reference stats not found. Run /drift/build-reference first.",
        )

    if not Path(RAW_DATA_PATH).exists():
        raise HTTPException(status_code=404, detail=f"Training data not found: {RAW_DATA_PATH}")

    # For first-pass testing, this compares the same dataset to reference.
    # Later you can replace this with a live inference/history batch.
    df = pd.read_csv(RAW_DATA_PATH).tail(100)
    #df = pd.read_csv("artifacts/drift/artificial_drift_batch.csv")
    report = detect_feature_drift_from_dataframe(
        df=df,
        reference_stats_path=REFERENCE_STATS_PATH,
    )
    save_drift_report(report, LATEST_REPORT_PATH)

    return report.model_dump()


@router.get("/latest-report")
def latest_report():
    if not LATEST_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="No drift report found yet.")
    return pd.read_json(LATEST_REPORT_PATH, typ="series").to_dict()