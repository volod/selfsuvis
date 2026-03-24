"""Admin statistics and mission listing endpoints.

GET /admin/stats              — queue depth, job status counts, al_tag distribution.
GET /admin/missions           — list missions with map_status and splat_path(s).
GET /admin/robots             — list distinct robot_ids contributing to the map.
GET /admin/global-maps        — list all global map sites (one entry per geographic site).
GET /admin/export/map-cache   — download NPZ cache of all indexed frames for onboard use.

al_tag distribution comes from the PostgreSQL frames table (available after migration).
Until the migration runs, all al_tag counts default to 0.
"""
import glob
import os
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.deps import rate_limit, require_api_key
from pipeline.config import settings
from pipeline.job_db import _get_conn as _job_db_conn

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


def _job_counts() -> Dict[str, int]:
    """Return counts of jobs by status from the SQLite job store."""
    try:
        conn = _job_db_conn()
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
        ).fetchall()
        counts = {row["status"]: row["cnt"] for row in rows}
    except Exception:
        counts = {}
    return {
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "done": counts.get("done", 0),
        "error": counts.get("error", 0),
    }


def _al_tag_counts() -> Dict[str, int]:
    """Return al_tag distribution from the PostgreSQL frames table.

    Returns zeros until the PostgreSQL migration has been applied and frames are indexed.
    """
    db_url = settings.DATABASE_URL
    if not db_url:
        return {"needs_annotation": 0, "novel": 0, "none": 0}
    try:
        import asyncpg  # noqa: F401  — only used if DATABASE_URL is set
        import asyncio

        async def _query():
            conn = await asyncpg.connect(db_url, timeout=5)
            try:
                rows = await conn.fetch(
                    "SELECT al_tag, COUNT(*) AS cnt FROM frames GROUP BY al_tag"
                )
                return {row["al_tag"]: row["cnt"] for row in rows}
            finally:
                await conn.close()

        counts = asyncio.run(_query())
    except Exception:
        counts = {}
    return {
        "needs_annotation": counts.get("needs_annotation", 0),
        "novel": counts.get("novel", 0),
        "none": counts.get("none", 0),
    }


def _discover_splat_paths(mission_id: str) -> List[str]:
    """Return all splat.ply paths for a mission, sorted by scene index.

    Checks for:
      - maps/{mission_id}/scene-*/splat.ply  (multi-scene, chunked)
      - maps/{mission_id}/splat.ply          (single-scene, legacy)
    """
    maps_dir = settings.MAPS_DIR
    chunked = sorted(
        glob.glob(os.path.join(maps_dir, mission_id, "scene-*", "splat.ply"))
    )
    if chunked:
        return chunked
    single = os.path.join(maps_dir, mission_id, "splat.ply")
    if os.path.isfile(single):
        return [single]
    return []


def _list_missions() -> List[Dict[str, Any]]:
    """Return missions from PostgreSQL with splat path discovery."""
    db_url = settings.DATABASE_URL
    missions = []
    if db_url:
        try:
            import asyncio
            import asyncpg

            async def _query():
                conn = await asyncpg.connect(db_url, timeout=5)
                try:
                    rows = await conn.fetch(
                        "SELECT id, video_id, status, map_status, frame_count, created_at "
                        "FROM missions ORDER BY created_at DESC LIMIT 100"
                    )
                    return [dict(r) for r in rows]
                finally:
                    await conn.close()

            missions = asyncio.run(_query())
        except Exception:
            missions = []

    # Attach splat paths discovered from the filesystem
    for m in missions:
        paths = _discover_splat_paths(m["id"])
        m["splat_paths"] = paths
        m["splat_path"] = paths[0] if paths else None
        m["scene_count"] = len(paths)
    return missions


@router.get("/missions", summary="List missions with map status and splat paths")
def admin_missions() -> List[Dict[str, Any]]:
    """Return up to 100 most recent missions with map status and discovered splat.ply paths."""
    return _list_missions()


@router.get("/robots", summary="List distinct robot_ids contributing to the map")
def admin_robots() -> List[str]:
    """Return all distinct robot_ids from the missions table.

    Returns an empty list if the robot_id column has not been added yet
    (migration not yet applied) or if DATABASE_URL is not configured.
    """
    db_url = settings.DATABASE_URL
    if not db_url:
        return []
    try:
        import asyncio
        import asyncpg

        async def _query():
            conn = await asyncpg.connect(db_url, timeout=5)
            try:
                rows = await conn.fetch(
                    "SELECT DISTINCT robot_id FROM missions "
                    "WHERE robot_id IS NOT NULL ORDER BY robot_id"
                )
                return [row["robot_id"] for row in rows]
            finally:
                await conn.close()

        return asyncio.run(_query())
    except Exception:
        return []


@router.get("/global-maps", summary="List all global map sites")
def admin_global_maps() -> List[Dict[str, Any]]:
    """Return all global_map rows — one entry per geographic site.

    Each entry has: id, origin_lat, origin_lon, origin_alt, splat_path, created_at.
    Returns an empty list if DATABASE_URL is not configured or the table does not exist.
    """
    db_url = settings.DATABASE_URL
    if not db_url:
        return []
    try:
        import asyncio
        import asyncpg
        from pipeline.global_map_db import list_global_maps

        async def _query():
            conn = await asyncpg.connect(db_url, timeout=5)
            try:
                return await list_global_maps(conn)
            finally:
                await conn.close()

        return asyncio.run(_query())
    except Exception:
        return []


@router.get(
    "/export/map-cache",
    summary="Download NPZ map cache for onboard robot use",
    response_class=Response,
)
def export_map_cache(
    mission_ids: Optional[str] = Query(
        default=None,
        description="Comma-separated mission IDs to include (all if omitted)",
    ),
    lat_min: Optional[float] = Query(default=None, description="GPS bounding box — min latitude"),
    lat_max: Optional[float] = Query(default=None, description="GPS bounding box — max latitude"),
    lon_min: Optional[float] = Query(default=None, description="GPS bounding box — min longitude"),
    lon_max: Optional[float] = Query(default=None, description="GPS bounding box — max longitude"),
) -> Response:
    """Export all indexed frames as a compressed NPZ cache file.

    The robot loads this file pre-flight for local nearest-neighbour search
    without a network round-trip.  NPZ layout:

        clip_vectors : float32 (N, D)   CLIP embeddings
        gps          : float32 (N, 3)   [lat, lon, alt], NaN if unavailable
        enu          : float32 (N, 3)   [tx, ty, tz] metres, NaN if unavailable
        t_sec        : float32 (N,)     frame timestamp in source video
        meta_json    : uint8  (M,)      JSON bytes → list of N {mission_id, frame_path, robot_id}

    Optional query params to narrow the export:
        mission_ids  — comma-separated list of mission IDs
        lat_min / lat_max / lon_min / lon_max — GPS bbox (all four required for bbox filter)
    """
    from app.state import qdrant_store
    from pipeline.map_cache import build_map_cache

    parsed_mission_ids = [m.strip() for m in mission_ids.split(",") if m.strip()] if mission_ids else None

    npz_bytes = build_map_cache(
        qdrant_store,
        mission_ids=parsed_mission_ids,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    return Response(
        content=npz_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=\"map_cache.npz\""},
    )


@router.get("/stats", summary="Queue depth, worker status, and al_tag distribution")
def admin_stats() -> Dict[str, Any]:
    """Return operational statistics for the admin dashboard.

    Fields:
        jobs:      {pending, running, done, error} counts from the job queue.
        al_tags:   {needs_annotation, novel, none} frame counts (PostgreSQL frames table).
        worker_active: true if any jobs are currently in 'running' state.
    """
    jobs = _job_counts()
    al_tags = _al_tag_counts()
    return {
        "jobs": jobs,
        "al_tags": al_tags,
        "worker_active": jobs["running"] > 0,
    }
