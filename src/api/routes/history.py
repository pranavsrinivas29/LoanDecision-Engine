from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.schemas import HistoryListResponse
from src.db.repository import get_all_prediction_records

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryListResponse)
def get_history(limit: int = Query(default=20, ge=1, le=200)):
    records = get_all_prediction_records(limit=limit)
    return HistoryListResponse(records=records)