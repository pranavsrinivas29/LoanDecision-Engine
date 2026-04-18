from __future__ import annotations

from fastapi import APIRouter, Query

from src.retraining.pipeline import run_retraining_pipeline

router = APIRouter(prefix="/retraining", tags=["retraining"])


@router.post("/run")
def run_retraining(force: bool = Query(default=False)):
    result = run_retraining_pipeline(force=force)
    return result