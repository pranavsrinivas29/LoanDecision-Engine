from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from src.config.settings import DB_PATH


@contextmanager
def connection_context():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()