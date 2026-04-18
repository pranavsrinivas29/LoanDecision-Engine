from __future__ import annotations

import json
from pathlib import Path

from src.drift.schemas import DriftReport


def save_drift_report(report: DriftReport, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.model_dump(), indent=2))
    return output_path