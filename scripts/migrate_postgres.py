#!/usr/bin/env python3
"""PostgreSQL schema migration for selfsuvis.

Creates all tables on a fresh database or safely upgrades an existing one.
Run once after `make up postgres` starts the PostgreSQL container:

    python scripts/migrate_postgres.py

Reads DATABASE_URL from environment or env/prod.env.
"""
import asyncio
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
from dotenv import load_dotenv

# Load env file so DATABASE_URL is available when run locally
_env_name = os.getenv("APP_ENV", "prod")
_env_file = Path(__file__).parent.parent / "env" / f"{_env_name}.env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis")

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_MIGRATIONS = [
    # ── jobs ─────────────────────────────────────────────────────────────────
    # asyncpg-backed job queue (replaces SQLite job_db.py in production).
    # Worker claims rows with SELECT FOR UPDATE SKIP LOCKED.
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id            TEXT PRIMARY KEY,
        status        TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending','running','done','error')),
        progress_json TEXT NOT NULL DEFAULT '{}',
        payload_json  TEXT NOT NULL DEFAULT '{}',
        created_at    DOUBLE PRECISION NOT NULL,
        started_at    DOUBLE PRECISION,
        finished_at   DOUBLE PRECISION,
        error         TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at)",

    # ── processed_files ──────────────────────────────────────────────────────
    # Dedup registry: tracks SHA-256 of every processed video to prevent
    # re-indexing the same file.
    """
    CREATE TABLE IF NOT EXISTS processed_files (
        file_hash   TEXT PRIMARY KEY,
        video_id    TEXT NOT NULL,
        path        TEXT,
        size_bytes  BIGINT,
        mtime       DOUBLE PRECISION,
        status      TEXT NOT NULL DEFAULT 'done',
        meta_json   TEXT NOT NULL DEFAULT '{}',
        created_at  DOUBLE PRECISION NOT NULL,
        updated_at  DOUBLE PRECISION NOT NULL
    )
    """,

    # ── missions ─────────────────────────────────────────────────────────────
    # One row per indexed video / mission.
    """
    CREATE TABLE IF NOT EXISTS missions (
        id              TEXT PRIMARY KEY,
        video_id        TEXT NOT NULL,
        video_path      TEXT,
        job_id          TEXT REFERENCES jobs(id),
        status          TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending','indexing','done','error')),
        pose_status     TEXT NOT NULL DEFAULT 'pending'
                            CHECK (pose_status IN ('pending','running','success','failed','skipped')),
        map_status      TEXT NOT NULL DEFAULT 'pending'
                            CHECK (map_status IN ('pending','running','success','failed','skipped')),
        frame_count     INTEGER NOT NULL DEFAULT 0,
        duration_sec    DOUBLE PRECISION,
        gps_origin_json TEXT,   -- {lat, lon, alt} of first GPS-valid frame (ENU origin)
        created_at      DOUBLE PRECISION NOT NULL,
        updated_at      DOUBLE PRECISION NOT NULL,
        error           TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_missions_status ON missions (status)",
    "CREATE INDEX IF NOT EXISTS idx_missions_video_id ON missions (video_id)",

    # ── frames ───────────────────────────────────────────────────────────────
    # One row per keyframe extracted from a mission video.
    # pose_json:  pycolmap output  {R: [[...]], t: [...], camera_id: ...}
    # gps_json:   {lat, lon, alt, timestamp_ms}
    # global_pose_json: ENU pose after GPS-to-ENU registration (Phase 1)
    #             or ICP-registered pose (Phase 2, when populated)
    """
    CREATE TABLE IF NOT EXISTS frames (
        id                  TEXT PRIMARY KEY,
        mission_id          TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
        frame_path          TEXT NOT NULL,
        t_sec               DOUBLE PRECISION NOT NULL,
        segment_id          INTEGER,
        caption             TEXT,
        caption_confidence  DOUBLE PRECISION,
        al_score            DOUBLE PRECISION,
        al_tag              TEXT NOT NULL DEFAULT 'none'
                                CHECK (al_tag IN ('none','needs_annotation','novel','annotated')),
        pose_status         TEXT NOT NULL DEFAULT 'pending'
                                CHECK (pose_status IN ('pending','success','failed')),
        pose_json           TEXT,
        gps_json            TEXT,
        global_pose_json    TEXT,
        qdrant_id           BIGINT,
        created_at          DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_frames_mission_id   ON frames (mission_id)",
    "CREATE INDEX IF NOT EXISTS idx_frames_al_tag       ON frames (al_tag)",
    "CREATE INDEX IF NOT EXISTS idx_frames_pose_status  ON frames (pose_status)",
    # Payload indexes for GPS bounding-box queries (required for change detection
    # and robot API; Qdrant needs corresponding payload indexes on gps.lat/lon)
    "CREATE INDEX IF NOT EXISTS idx_frames_t_sec        ON frames (t_sec)",

    # ── embedding_clusters ───────────────────────────────────────────────────
    # k-means cluster assignments for DINOv3 embeddings (k=20 default).
    # Used by active learning to compute dino_dist (distance to nearest cluster centroid).
    """
    CREATE TABLE IF NOT EXISTS embedding_clusters (
        id              SERIAL PRIMARY KEY,
        mission_id      TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
        cluster_id      INTEGER NOT NULL,
        centroid_json   TEXT NOT NULL,  -- float[] JSON array
        frame_count     INTEGER NOT NULL DEFAULT 0,
        created_at      DOUBLE PRECISION NOT NULL,
        updated_at      DOUBLE PRECISION NOT NULL,
        UNIQUE (mission_id, cluster_id)
    )
    """,

    # ── change_detections ────────────────────────────────────────────────────
    # Records visual changes between GPS-overlapping frames across missions.
    """
    CREATE TABLE IF NOT EXISTS change_detections (
        id              SERIAL PRIMARY KEY,
        frame_id        TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
        mission_id      TEXT NOT NULL,
        ref_frame_id    TEXT NOT NULL,
        ref_mission_id  TEXT NOT NULL,
        change_score    DOUBLE PRECISION NOT NULL,
        threshold       DOUBLE PRECISION NOT NULL,
        detected_at     DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_change_detections_mission_id ON change_detections (mission_id)",
    "CREATE INDEX IF NOT EXISTS idx_change_detections_score      ON change_detections (change_score)",

    # ── global_map ───────────────────────────────────────────────────────────
    # One row per geographic site / ENU coordinate origin.
    # Phase 1: GPS-to-ENU registration (registration_error = NULL).
    # Phase 2: ICP fusion (registration_error populated from ICP residual).
    """
    CREATE TABLE IF NOT EXISTS global_map (
        id              SERIAL PRIMARY KEY,
        origin_lat      DOUBLE PRECISION NOT NULL,
        origin_lon      DOUBLE PRECISION NOT NULL,
        origin_alt      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        splat_path      TEXT,   -- path to fused splat.ply (Phase 2)
        created_at      DOUBLE PRECISION NOT NULL,
        updated_at      DOUBLE PRECISION NOT NULL
    )
    """,

    # ── global_map_missions ──────────────────────────────────────────────────
    # Junction: which missions are registered into which global map.
    # transform_json: 4×4 SE(3) matrix as nested float[][] JSON array.
    # registration_error: NULL for Phase 1 GPS; ICP residual for Phase 2.
    """
    CREATE TABLE IF NOT EXISTS global_map_missions (
        id                      SERIAL PRIMARY KEY,
        global_map_id           INTEGER NOT NULL REFERENCES global_map(id) ON DELETE CASCADE,
        mission_id              TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
        registration_transform_json TEXT NOT NULL,
        registration_error      DOUBLE PRECISION,
        registered_at           DOUBLE PRECISION NOT NULL,
        UNIQUE (global_map_id, mission_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_gmm_global_map_id ON global_map_missions (global_map_id)",
    "CREATE INDEX IF NOT EXISTS idx_gmm_mission_id    ON global_map_missions (mission_id)",

    # ── cvat_tasks ───────────────────────────────────────────────────────────
    # Maps CVAT task IDs to selfsuvis frame IDs.
    # Populated by POST /admin/cvat/task when a user creates a CVAT annotation task.
    # Read by POST /webhook/cvat to mark frames annotated when a job completes.
    """
    CREATE TABLE IF NOT EXISTS cvat_tasks (
        cvat_task_id  INTEGER      NOT NULL,
        frame_id      TEXT         NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
        created_at    DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
        PRIMARY KEY (cvat_task_id, frame_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cvat_tasks_task_id ON cvat_tasks (cvat_task_id)",

    # ── multi-robot: add robot_id to missions ────────────────────────────────
    # Added in P3: multi-robot shared world model.
    # Idempotent via IF NOT EXISTS — safe to re-run on existing databases.
    "ALTER TABLE missions ADD COLUMN IF NOT EXISTS robot_id TEXT NOT NULL DEFAULT 'robot_0'",
    "CREATE INDEX IF NOT EXISTS idx_missions_robot_id ON missions (robot_id)",
]


async def migrate(url: str) -> None:
    print(f"Connecting to: {url.split('@')[-1]}")
    conn = await asyncpg.connect(url)
    try:
        for i, sql in enumerate(_MIGRATIONS, 1):
            stmt = sql.strip()
            label = stmt.split("\n")[0][:80].strip()
            await conn.execute(stmt)
            print(f"  [{i:02d}/{len(_MIGRATIONS)}] {label}")
        print(f"\nMigration complete — {len(_MIGRATIONS)} statements applied.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate(DATABASE_URL))
