from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.db.database import connection_context


def insert_prediction_record(
    input_data: dict[str, Any],
    prediction_result: dict[str, Any],
    local_explanation: dict[str, Any] | None = None,
    summary: str | None = None,
) -> None:
    sql = """
    INSERT INTO loan_predictions (
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        predicted_label_numeric,
        prediction,
        probability,
        threshold,
        local_explanation_json,
        summary
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    prediction_label = str(prediction_result.get("prediction", "")).strip()
    predicted_label_numeric = 1 if prediction_label.lower() == "approved" else 0

    values = (
        input_data["no_of_dependents"],
        input_data["education"],
        input_data["self_employed"],
        input_data["income_annum"],
        input_data["loan_amount"],
        input_data["loan_term"],
        input_data["cibil_score"],
        input_data["residential_assets_value"],
        input_data["commercial_assets_value"],
        input_data["luxury_assets_value"],
        input_data["bank_asset_value"],
        predicted_label_numeric,
        prediction_result.get("prediction"),
        prediction_result.get("probability"),
        prediction_result.get("threshold"),
        json.dumps(local_explanation) if local_explanation is not None else None,
        summary,
    )

    with connection_context() as conn:
        conn.execute(sql, values)


def get_all_prediction_records() -> list[dict[str, Any]]:
    sql = """
    SELECT
        id,
        created_at,
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        predicted_label_numeric,
        prediction,
        probability,
        threshold,
        local_explanation_json,
        summary
    FROM loan_predictions
    ORDER BY id DESC
    """

    with connection_context() as conn:
        rows = conn.execute(sql).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)

        if row_dict.get("local_explanation_json"):
            try:
                row_dict["local_explanation"] = json.loads(row_dict["local_explanation_json"])
            except Exception:
                row_dict["local_explanation"] = None
        else:
            row_dict["local_explanation"] = None

        records.append(row_dict)

    return records


def count_prediction_records() -> int:
    sql = "SELECT COUNT(*) AS total FROM loan_predictions"

    with connection_context() as conn:
        row = conn.execute(sql).fetchone()

    return int(row["total"])


def fetch_recent_prediction_records(limit: int = 100) -> pd.DataFrame:
    sql = """
    SELECT
        created_at,
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        prediction,
        probability
    FROM loan_predictions
    ORDER BY id DESC
    LIMIT ?
    """

    with connection_context() as conn:
        rows = conn.execute(sql, (int(limit),)).fetchall()

    records = [dict(row) for row in rows]
    return pd.DataFrame(records)