"""asyncpg-backed PostgreSQL job queue.

Replaces the SQLite job_db.py for production use with PostgreSQL 16.
The worker claims jobs with SELECT FOR UPDATE SKIP LOCKED for concurrency-safe
polling — no external lock manager required.

All functions accept an asyncpg Connection (or pool) as their first argument
so callers control connection lifecycle and transaction boundaries.
"""
import json
import time
from typing import Any, Dict, Optional

from pipeline.logging_utils import get_logger

logger = get_logger(__name__)

# Columns that update_job is allowed to touch (prevents injection of arbitrary column names)
_UPDATE_JOB_COLUMNS = frozenset(
    {"status", "progress", "payload", "started_at", "finished_at", "error"}
)

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS jobs (
        id            TEXT PRIMARY KEY,
        status        TEXT NOT NULL DEFAULT 'pending',
        progress_json TEXT NOT NULL DEFAULT '{}',
        payload_json  TEXT NOT NULL DEFAULT '{}',
        created_at    DOUBLE PRECISION,
        started_at    DOUBLE PRECISION,
        finished_at   DOUBLE PRECISION,
        error         TEXT
    )
"""


async def init_db(conn) -> None:
    """Create the jobs table if it does not already exist."""
    await conn.execute(_CREATE_TABLE_SQL)


async def create_job(conn, job_id: str, payload: Dict[str, Any]) -> None:
    """Insert a new job in 'pending' state."""
    now = time.time()
    await conn.execute(
        "INSERT INTO jobs (id, status, progress_json, payload_json, created_at)"
        " VALUES ($1, $2, $3, $4, $5)",
        job_id,
        "pending",
        "{}",
        json.dumps(payload),
        now,
    )


async def update_job(conn, job_id: str, **kwargs: Any) -> None:
    """Update whitelisted job fields.

    Unknown keyword arguments are silently ignored to prevent column-name injection.
    """
    allowed = {k: v for k, v in kwargs.items() if k in _UPDATE_JOB_COLUMNS}
    if not allowed:
        return

    parts = []
    values: list = []
    placeholder = 1
    for k, v in allowed.items():
        if k in {"progress", "payload"}:
            v = json.dumps(v)
            col = f"{k}_json"
        else:
            col = k
        parts.append(f"{col} = ${placeholder}")
        values.append(v)
        placeholder += 1

    values.append(job_id)
    await conn.execute(
        f"UPDATE jobs SET {', '.join(parts)} WHERE id = ${placeholder}",
        *values,
    )


async def fetch_job(conn, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single job by id. Returns None if not found."""
    row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
    return _row_to_dict(row) if row else None


async def fetch_and_claim_next_pending(conn) -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest pending job.

    Uses SELECT FOR UPDATE SKIP LOCKED so multiple workers can poll concurrently
    without stepping on each other. Must be called inside a transaction.
    Returns None if no pending jobs exist.
    """
    row = await conn.fetchrow(
        """
        SELECT * FROM jobs
        WHERE  status = 'pending'
        ORDER  BY created_at ASC
        LIMIT  1
        FOR UPDATE SKIP LOCKED
        """
    )
    if not row:
        return None

    now = time.time()
    await conn.execute(
        "UPDATE jobs SET status = 'running', started_at = $1 WHERE id = $2",
        now,
        row["id"],
    )
    d = _row_to_dict(row)
    d["status"] = "running"
    d["started_at"] = now
    return d


async def fetch_queue_depth(conn) -> int:
    """Return the number of pending jobs."""
    return await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE status = 'pending'")


def _row_to_dict(row) -> Dict[str, Any]:
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
