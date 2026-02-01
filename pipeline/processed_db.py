import json
import os
import sqlite3
import time
from typing import Dict, Any, Optional

from pipeline.config import settings
from pipeline.utils import ensure_dir

DB_PATH = os.path.join(settings.DATA_DIR, "processed.db")


def _connect() -> sqlite3.Connection:
    ensure_dir(os.path.dirname(DB_PATH))
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed (
                file_hash TEXT PRIMARY KEY,
                video_id TEXT,
                path TEXT,
                size_bytes INTEGER,
                mtime REAL,
                status TEXT,
                meta_json TEXT,
                created_at REAL,
                updated_at REAL
            )
            """
        )
    conn.close()


def get_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM processed WHERE file_hash = ?", (file_hash,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def get_by_url(url: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM processed WHERE json_extract(meta_json, '$.url') = ?",
        (url,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def get_by_size(size_bytes: int) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM processed WHERE size_bytes = ? ORDER BY updated_at DESC LIMIT 1",
        (size_bytes,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def upsert(file_hash: str, video_id: str, path: str, size_bytes: int, mtime: float, status: str, meta: Dict[str, Any]):
    conn = _connect()
    now = time.time()
    with conn:
        conn.execute(
            """
            INSERT INTO processed (file_hash, video_id, path, size_bytes, mtime, status, meta_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                video_id=excluded.video_id,
                path=excluded.path,
                size_bytes=excluded.size_bytes,
                mtime=excluded.mtime,
                status=excluded.status,
                meta_json=excluded.meta_json,
                updated_at=excluded.updated_at
            """,
            (
                file_hash,
                video_id,
                path,
                size_bytes,
                mtime,
                status,
                json.dumps(meta),
                now,
                now,
            ),
        )
    conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "file_hash": row["file_hash"],
        "video_id": row["video_id"],
        "path": row["path"],
        "size_bytes": row["size_bytes"],
        "mtime": row["mtime"],
        "status": row["status"],
        "meta": json.loads(row["meta_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
