from __future__ import annotations

from fastapi import APIRouter

from src.mlflow_utils.model_info import load_active_model_metadata

router = APIRouter(tags=["model-info"])


@router.get("/model-info")
def get_model_info():
    metadata = load_active_model_metadata()
    if not metadata:
        return {
            "status": "not_available",
            "message": "No active model metadata found yet. Run training first.",
        }

    return {
        "status": "ok",
        "model_info": metadata,
    }