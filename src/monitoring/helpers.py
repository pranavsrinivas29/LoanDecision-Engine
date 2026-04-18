import json
from pathlib import Path

from src.monitoring.metrics import ACTIVE_MODEL_INFO


def set_active_model_metric(model_info_path: Path) -> None:
    model_name = "unknown"
    model_version = "unknown"

    if model_info_path.exists():
        try:
            payload = json.loads(model_info_path.read_text())
            model_name = str(payload.get("model_name", "unknown"))
            model_version = str(payload.get("model_version", "unknown"))
        except Exception:
            pass

    ACTIVE_MODEL_INFO.labels(
        model_name=model_name,
        model_version=model_version,
    ).set(1)