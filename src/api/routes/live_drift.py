from __future__ import annotations

from fastapi import APIRouter

from src.drift.live_monitor import run_live_drift_check_and_optional_retraining

router = APIRouter(prefix="/live-drift", tags=["live-drift"])


@router.post("/check")
def check_live_drift():
    return run_live_drift_check_and_optional_retraining()