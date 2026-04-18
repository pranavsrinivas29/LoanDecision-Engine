from __future__ import annotations

from src.db.database import connection_context


CREATE_PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS loan_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    no_of_dependents INTEGER NOT NULL,
    education TEXT NOT NULL,
    self_employed TEXT NOT NULL,
    income_annum REAL NOT NULL,
    loan_amount REAL NOT NULL,
    loan_term REAL NOT NULL,
    cibil_score REAL NOT NULL,
    residential_assets_value REAL NOT NULL,
    commercial_assets_value REAL NOT NULL,
    luxury_assets_value REAL NOT NULL,
    bank_asset_value REAL NOT NULL,

    predicted_label_numeric INTEGER NOT NULL,
    prediction TEXT NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,

    local_explanation_json TEXT,
    summary TEXT
);
"""


def create_tables():
    """
    Create all required database tables.
    """
    with connection_context() as conn:
        conn.execute(CREATE_PREDICTIONS_TABLE_SQL)