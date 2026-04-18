from __future__ import annotations

import json
import shutil
from datetime import datetime, UTC
from pathlib import Path


def promote_winning_model(
    source_model_path: str | Path,
    source_preprocessor_path: str | Path,
    active_model_path: str | Path,
    active_preprocessor_path: str | Path,
    metadata_path: str | Path,
    winner_payload: dict,
) -> None:
    source_model_path = Path(source_model_path)
    source_preprocessor_path = Path(source_preprocessor_path)
    active_model_path = Path(active_model_path)
    active_preprocessor_path = Path(active_preprocessor_path)
    metadata_path = Path(metadata_path)

    active_model_path.parent.mkdir(parents=True, exist_ok=True)
    active_preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_model_path, active_model_path)
    shutil.copy2(source_preprocessor_path, active_preprocessor_path)

    metadata = {
        "model_name": winner_payload["model_name"],
        "model_version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        "selected_at": datetime.now(UTC).isoformat(),
        "selection_metrics": {
            "accuracy": winner_payload.get("accuracy"),
            "precision": winner_payload.get("precision"),
            "recall": winner_payload.get("recall"),
            "f1": winner_payload.get("f1"),
            "roc_auc": winner_payload.get("roc_auc"),
        },
        "artifact_paths": {
            "model_path": str(active_model_path),
            "preprocessor_path": str(active_preprocessor_path),
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))