"""Admin statistics and mission listing endpoints.

GET /admin/stats              — queue depth, job status counts, al_tag distribution.
GET /admin/missions           — list missions with map_status and splat_path(s).
GET /admin/robots             — list distinct robot_ids contributing to the map.
GET /admin/global-maps        — list all global map sites (one entry per geographic site).
GET /admin/export/map-cache   — download NPZ cache of all indexed frames for onboard use.
GET /admin/automation-roi     — annotation frequency, finetune trigger rate, ops time saved.
POST /admin/reload-model      — hot-swap DINOv3 backbone weights from a checkpoint file.
POST /admin/reembed-all       — enqueue a full re-embedding sweep (all frames → new dino vectors).

al_tag distribution comes from the PostgreSQL frames table.
"""
import glob
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel

from app.deps import rate_limit, require_api_key
from pipeline.config import settings

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


def _job_counts() -> Dict[str, int]:
    """Return counts of jobs by status from PostgreSQL."""
    db_url = settings.DATABASE_URL
    if not db_url:
        return {"pending": 0, "running": 0, "finished": 0, "error": 0}
    try:
        import asyncio
        async def _query():
            conn = await asyncpg.connect(db_url, timeout=5)
            try:
                rows = await conn.fetch(
                    "SELECT status, COUNT(*) AS cnt FROM jobs GROUP BY status"
                )
                return {row["status"]: row["cnt"] for row in rows}
            finally:
                await conn.close()

        counts = asyncio.run(_query())
    except Exception:
        counts = {}
    return {
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "finished": counts.get("finished", 0),
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
        jobs:      {pending, running, finished, error} counts from the job queue.
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


# ── Automation ROI endpoint ──────────────────────────────────────────────────
#
# Evidence-based answer to: "is the auto-trigger pipeline worth maintaining?"
#
# Derived entirely from existing jobs + frames tables — no schema changes.
# Key question: annotation frequency per week.
#   < 0.5  → LOW_FREQUENCY   — manual restart may be simpler
#   0.5–2  → MODERATE        — automation pays off within months
#   > 2    → HIGH_FREQUENCY  — automation clearly valuable
#
# Ops time saved = accepted_finetune_count × MANUAL_RESTART_MIN (default 3 min).
# This is the lower bound: excludes time saved by not monitoring for threshold
# crossings and not manually kicking off reembed sweeps.

_MANUAL_RESTART_MINUTES = 3  # estimated time per manual `docker restart api`


@router.get("/automation-roi", summary="Annotation frequency, finetune trigger rate, ops time saved")
async def automation_roi() -> Dict[str, Any]:
    """Measure whether the auto-trigger pipeline is paying its maintenance cost.

    All metrics are derived from the jobs and frames tables — no extra
    instrumentation required.  Call this after 4+ weeks of production use.

    Returns:
        total_annotated_frames:       frames with al_tag='annotated' (cumulative)
        annotation_campaigns:         distinct months with at least one annotation
        finetune_jobs_triggered:      supervised_finetune jobs ever enqueued
        finetune_jobs_accepted:       checkpoints that passed the eval gate
        finetune_acceptance_rate:     accepted / triggered (null if no jobs)
        model_reloads:                accepted jobs (each triggers a hot-reload)
        reembed_sweeps:               reembed jobs ever completed
        estimated_ops_minutes_saved:  model_reloads × 3 min manual restart
        first_annotation_at:          ISO timestamp of first annotation (null if none)
        last_annotation_at:           ISO timestamp of most recent annotation (null)
        days_observed:                calendar days since first annotation (null if none)
        annotation_frequency_per_week: annotations per 7-day window (null if <7 days)
        verdict:                      LOW_FREQUENCY | MODERATE_FREQUENCY |
                                      HIGH_FREQUENCY | INSUFFICIENT_DATA
        verdict_detail:               plain-English interpretation
    """
    import json as _json

    db_url = settings.DATABASE_URL
    if not db_url:
        return {"error": "DATABASE_URL not configured"}

    try:
        conn = await asyncpg.connect(db_url, timeout=5)
        try:
            # Annotated frame count
            total_annotated = await conn.fetchval(
                "SELECT COUNT(*) FROM frames WHERE al_tag = 'annotated'"
            ) or 0

            # Annotation timeline: first and last annotation timestamps
            timeline = await conn.fetchrow(
                "SELECT MIN(created_at) AS first_at, MAX(created_at) AS last_at "
                "FROM frames WHERE al_tag = 'annotated'"
            )
            first_at = timeline["first_at"] if timeline else None
            last_at = timeline["last_at"] if timeline else None

            # Number of distinct calendar months with at least one annotation
            # (proxy for "annotation campaigns")
            campaigns = await conn.fetchval(
                "SELECT COUNT(DISTINCT TO_CHAR(created_at, 'YYYY-MM')) "
                "FROM frames WHERE al_tag = 'annotated'"
            ) or 0

            # Finetune job stats
            ft_rows = await conn.fetch(
                "SELECT status, progress_json FROM jobs "
                "WHERE type = 'supervised_finetune'"
            )
            ft_triggered = len(ft_rows)
            ft_accepted = sum(
                1 for r in ft_rows
                if (
                    (r["progress_json"] if isinstance(r["progress_json"], dict)
                     else _json.loads(r["progress_json"] or "{}"))
                    .get("accepted") is True
                )
            )

            # Reembed sweeps completed
            reembed_done = await conn.fetchval(
                "SELECT COUNT(*) FROM jobs "
                "WHERE type = 'reembed' AND status = 'finished'"
            ) or 0

            d = {
                "total_annotated": total_annotated,
                "first_at": first_at,
                "last_at": last_at,
                "campaigns": campaigns,
                "ft_triggered": ft_triggered,
                "ft_accepted": ft_accepted,
                "reembed_done": reembed_done,
            }
        finally:
            await conn.close()
    except Exception as exc:
        return {"error": f"DB query failed: {exc}"}

    # Derived metrics
    first_at = d["first_at"]
    last_at = d["last_at"]
    days_observed: Optional[float] = None
    freq_per_week: Optional[float] = None

    if first_at is not None and last_at is not None:
        days_observed = max((last_at - first_at) / 86400.0, 1.0)
        if days_observed >= 7:
            freq_per_week = d["total_annotated"] / (days_observed / 7.0)

    ft_triggered = d["ft_triggered"]
    ft_accepted = d["ft_accepted"]
    acceptance_rate: Optional[float] = (ft_accepted / ft_triggered) if ft_triggered else None
    ops_saved_minutes = ft_accepted * _MANUAL_RESTART_MINUTES

    # Verdict
    if freq_per_week is None:
        verdict = "INSUFFICIENT_DATA"
        verdict_detail = (
            "Less than 7 days of annotation data. Re-check after 4+ weeks of production use."
        )
    elif freq_per_week < 0.5:
        verdict = "LOW_FREQUENCY"
        verdict_detail = (
            f"Annotations happen ~{freq_per_week:.1f}×/week. "
            "Manual `docker restart api` after each fine-tune may be simpler to maintain "
            "than the auto-trigger pipeline."
        )
    elif freq_per_week < 2.0:
        verdict = "MODERATE_FREQUENCY"
        verdict_detail = (
            f"Annotations happen ~{freq_per_week:.1f}×/week. "
            "Automation is paying off but the benefit is modest. "
            "Keep the pipeline; revisit if annotation cadence drops."
        )
    else:
        verdict = "HIGH_FREQUENCY"
        verdict_detail = (
            f"Annotations happen ~{freq_per_week:.1f}×/week. "
            "Automation is clearly valuable — the compounding improvement loop is active."
        )

    return {
        "total_annotated_frames": d["total_annotated"],
        "annotation_campaigns": d["campaigns"],
        "finetune_jobs_triggered": ft_triggered,
        "finetune_jobs_accepted": ft_accepted,
        "finetune_acceptance_rate": round(acceptance_rate, 3) if acceptance_rate is not None else None,
        "model_reloads": ft_accepted,
        "reembed_sweeps_completed": d["reembed_done"],
        "estimated_ops_minutes_saved": ops_saved_minutes,
        "first_annotation_at": first_at,
        "last_annotation_at": last_at,
        "days_observed": round(days_observed, 1) if days_observed is not None else None,
        "annotation_frequency_per_week": round(freq_per_week, 2) if freq_per_week is not None else None,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }


# ── Hot-reload model endpoint ────────────────────────────────────────────────

class ReloadModelRequest(BaseModel):
    checkpoint: Optional[str] = None  # path to checkpoint; falls back to active_checkpoint.txt


@router.post("/reload-model", summary="Hot-swap DINOv3 backbone weights")
async def reload_model(body: ReloadModelRequest = ReloadModelRequest()) -> Dict[str, Any]:
    """Load a DINOv3 backbone checkpoint and swap the live model reference.

    The swap is GIL-atomic: in-flight inference calls hold their captured
    dino_model reference and complete normally with old weights. Only new
    requests after the swap see the new weights.

    - 400: checkpoint path not found on disk.
    - 400: no checkpoint specified and active_checkpoint.txt is missing.
    - 409: another reload is already in progress (lock is held).
    - 500: checkpoint loaded but weights failed to parse (original model unchanged).
    """
    import app.state as state
    from fastapi import HTTPException

    if state.dino_model is None:
        raise HTTPException(status_code=400, detail="DINO model is not loaded (MODEL_NAME not dinov2/dinov3)")

    # Resolve checkpoint path
    ckpt_path = body.checkpoint
    if not ckpt_path:
        active_txt = Path(settings.SUP_CHECKPOINT_DIR) / "active_checkpoint.txt"
        if not active_txt.exists():
            raise HTTPException(
                status_code=400,
                detail="No checkpoint specified and active_checkpoint.txt not found",
            )
        ckpt_path = active_txt.read_text().strip()

    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {ckpt_path}")

    if state.dino_model_lock.locked():
        raise HTTPException(status_code=409, detail="Reload already in progress")

    async with state.dino_model_lock:
        try:
            state.dino_model.load_backbone_checkpoint(ckpt_path)
            # Write active checkpoint so the next restart picks it up
            active_txt = Path(settings.SUP_CHECKPOINT_DIR) / "active_checkpoint.txt"
            active_txt.parent.mkdir(parents=True, exist_ok=True)
            active_txt.write_text(ckpt_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {exc}") from exc

    return {"status": "ok", "checkpoint": ckpt_path}


# ── Re-embed all endpoint ────────────────────────────────────────────────────

@router.post("/reembed-all", summary="Enqueue full re-embedding sweep (dino vectors)")
async def reembed_all() -> Dict[str, Any]:
    """Enqueue a background job to re-embed all indexed frames with the current DINOv3 model.

    Returns the job_id immediately. Monitor progress via GET /jobs/{job_id}.
    Each batch of 256 frames checkpoints last_offset so the sweep is resumable.
    """
    from pipeline.job_db_pg import create_job

    db_url = settings.DATABASE_URL
    if not db_url:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    job_id = uuid.uuid4().hex
    conn = await asyncpg.connect(db_url, timeout=5)
    try:
        await create_job(conn, job_id, {}, job_type="reembed")
    finally:
        await conn.close()

    return {"job_id": job_id}


# ── Reembed status endpoint ───────────────────────────────────────────────────

@router.get("/reembed-status", summary="Check whether a reembed sweep is active")
async def reembed_status() -> Dict[str, Any]:
    """Return whether a re-embedding sweep is currently running.

    Search uses this to suppress dino vector reranking during the sweep window,
    because Qdrant contains a mix of old-model and new-model dino vectors while
    the sweep is in progress — cosine similarity between them is meaningless.

    Returns:
        active: true if a reembed job has status='running'.
        job_id: the running job's id, or null if none.
        frames_reembedded: progress counter from the job payload, or null.
    """
    db_url = settings.DATABASE_URL
    if not db_url:
        return {"active": False, "job_id": None, "frames_reembedded": None}

    try:
        conn = await asyncpg.connect(db_url, timeout=5)
        try:
            row = await conn.fetchrow(
                "SELECT id, progress_json FROM jobs "
                "WHERE type = 'reembed' AND status = 'running' "
                "ORDER BY created_at DESC LIMIT 1"
            )
        finally:
            await conn.close()
    except Exception:
        return {"active": False, "job_id": None, "frames_reembedded": None}

    if row is None:
        return {"active": False, "job_id": None, "frames_reembedded": None}

    import json as _json
    progress = row["progress_json"] or {}
    if isinstance(progress, str):
        progress = _json.loads(progress)
    return {
        "active": True,
        "job_id": row["id"],
        "frames_reembedded": progress.get("frames_reembedded"),
    }
