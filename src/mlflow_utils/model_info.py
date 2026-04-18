from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config.settings import MODEL_INFO_PATH


def save_active_model_metadata(metadata: dict, path: str | Path = MODEL_INFO_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = metadata.copy()
    metadata["saved_at"] = datetime.utcnow().isoformat()

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_active_model_metadata(path: str | Path = MODEL_INFO_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)