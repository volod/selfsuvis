import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional

from pipeline.config import settings
from pipeline.utils import ensure_dir

# Allowed columns for update_job (whitelist to prevent injection)
_UPDATE_JOB_COLUMNS = frozenset(
    {"status", "progress", "payload", "started_at", "finished_at", "error"}
)

_conn_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get thread-local connection. Creates one if not present."""
    if not hasattr(_conn_local, "conn") or _conn_local.conn is None:
        ensure_dir(os.path.dirname(settings.JOB_DB_PATH))
        _conn_local.conn = sqlite3.connect(
            settings.JOB_DB_PATH,
            check_same_thread=False,
            timeout=settings.SQLITE_TIMEOUT,
        )
        _conn_local.conn.row_factory = sqlite3.Row
    return _conn_local.conn


def _connect() -> sqlite3.Connection:
    """Return connection for use in with-blocks. Prefer _get_conn for simple ops."""
    return _get_conn()


def init_db() -> None:
    conn = _get_conn()
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


def create_job(job_id: str, payload: Dict[str, Any]) -> None:
    conn = _get_conn()
    now = time.time()
    with conn:
        conn.execute(
            "INSERT INTO jobs (id, status, progress_json, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, "pending", json.dumps({}), json.dumps(payload), now),
        )


def update_job(job_id: str, **kwargs: Any) -> None:
    """Update job. Only whitelisted columns are accepted."""
    allowed = {k: v for k, v in kwargs.items() if k in _UPDATE_JOB_COLUMNS}
    if not allowed:
        return
    conn = _get_conn()
    fields = []
    values = []
    for k, v in allowed.items():
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


def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def fetch_next_pending() -> Optional[Dict[str, Any]]:
    """Fetch next pending job without claiming. Prefer fetch_and_claim_next_pending for workers."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return _row_to_dict(row)


def fetch_and_claim_next_pending() -> Optional[Dict[str, Any]]:
    """Atomically claim the next pending job (set status to running). Returns None if none pending or already claimed by another worker."""
    conn = _get_conn()
    now = time.time()
    with conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        job_id = row["id"]
        cursor = conn.execute(
            "UPDATE jobs SET status = ?, started_at = ? WHERE id = ? AND status = 'pending'",
            ("running", now, job_id),
        )
        if cursor.rowcount != 1:
            return None
    out = _row_to_dict(row)
    out["started_at"] = now
    out["status"] = "running"
    return out


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
