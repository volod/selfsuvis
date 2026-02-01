import json
import os
import sqlite3
import time
from typing import Dict, Any, Optional

from pipeline.config import settings
from pipeline.utils import ensure_dir


def _connect() -> sqlite3.Connection:
    ensure_dir(os.path.dirname(settings.JOB_DB_PATH))
    conn = sqlite3.connect(settings.JOB_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT,
                progress_json TEXT,
                payload_json TEXT,
                created_at REAL,
                started_at REAL,
                finished_at REAL,
                error TEXT
            )
            """
        )
    conn.close()


def create_job(job_id: str, payload: Dict[str, Any]) -> None:
    conn = _connect()
    now = time.time()
    with conn:
        conn.execute(
            "INSERT INTO jobs (id, status, progress_json, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "pending", json.dumps({}), json.dumps(payload), now),
        )
    conn.close()


def update_job(job_id: str, **kwargs: Any) -> None:
    conn = _connect()
    fields = []
    values = []
    for k, v in kwargs.items():
        if k in {"progress", "payload"}:
            v = json.dumps(v)
            col = f"{k}_json"
        else:
            col = k
        fields.append(f"{col} = ?")
        values.append(v)
    values.append(job_id)
    with conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)
    conn.close()


def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def fetch_next_pending() -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1").fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "status": row["status"],
        "progress": json.loads(row["progress_json"] or "{}"),
        "payload": json.loads(row["payload_json"] or "{}"),
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
    }
