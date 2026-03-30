import asyncio
import os
import time
import uuid
from datetime import timedelta
from typing import Optional

import asyncpg

from pipeline.config import get_dino_model_name, settings, validate_settings
from pipeline.indexer import VideoIndexer
from pipeline.job_db_pg import create_job, fetch_and_claim_next_pending, update_job
from pipeline.logging_utils import get_logger
import pipeline.processed_db as processed_db_mod
from pipeline.processed_db import init_db as init_processed_db, get_by_hash
from pipeline.downloader import download_url
from pipeline.mission_db import (
    apply_gps_registration,
    list_frames_after,
    mark_mission_finished,
    replace_frames,
    upsert_mission,
)
from pipeline.utils import datetime_to_ts, file_sha256, utcnow


def _resolve_site_origin(video_path: str, logger) -> tuple:
    """Extract the first GPS fix from a video and look up (or create) its global_map.

    Returns (global_map_id, (origin_lat, origin_lon, origin_alt)) on success,
    or (None, None) if GPS is unavailable or DB is unreachable.

    This runs synchronously before index_video so that ENU coordinates stored
    in Qdrant are relative to the site's canonical ENU origin rather than each
    mission's own local first-frame origin.
    """
    try:
        from pipeline.gps_extractor import extract_gps
        # Request GPS at t=1s; extract_gps will return the nearest available fix
        gps_list = extract_gps(video_path, [1_000.0])
        first_gps = next((g for g in gps_list if g is not None), None)
        if first_gps is None:
            logger.debug("Multi-site ENU: no GPS in video path=%s", video_path)
            return None, None
    except Exception as exc:
        logger.debug("Multi-site ENU: GPS extraction failed path=%s err=%s", video_path, exc)
        return None, None

    try:
        from pipeline.global_map_db import get_or_create_global_map, get_global_map_origin

        async def _lookup():
            conn = await asyncpg.connect(settings.DATABASE_URL)
            try:
                gmap_id = await get_or_create_global_map(
                    conn,
                    first_gps["lat"],
                    first_gps["lon"],
                    float(first_gps.get("alt", 0.0)),
                )
                origin = await get_global_map_origin(conn, gmap_id)
                return gmap_id, origin
            finally:
                await conn.close()

        gmap_id, origin = asyncio.run(_lookup())
        if origin is not None:
            logger.info(
                "Multi-site ENU: video assigned to global_map_id=%d origin=(%.6f, %.6f, %.1f)",
                gmap_id, origin[0], origin[1], origin[2],
            )
        return gmap_id, origin
    except Exception as exc:
        logger.debug("Multi-site ENU: site origin lookup failed: %s", exc)
        return None, None


def _run_pass_a(video_path: str, video_id: str, mission_id: str, index_result: dict, logger, global_map_id: int = None) -> None:
    """Run SfM → GPS registration → nerfstudio 3DGS for a completed indexing job.

    All steps degrade gracefully: missing optional deps (pycolmap, open3d) or
    unreachable containers (nerfstudio, mapper, postgres) are logged and skipped.
    """
    try:
        from pipeline.sfm import run_sfm
    except ImportError:
        logger.debug("Pass A: pycolmap not installed — skipping SfM")
        return

    try:
        sfm_out = run_sfm(video_path, video_id, mission_id)
    except Exception as exc:
        logger.warning("Pass A: SfM failed mission=%s: %s", mission_id, exc)
        return

    sfm_results = sfm_out.get("frames", [])
    scene_count = sfm_out.get("scene_count", 1)
    logger.info(
        "Pass A: SfM done mission=%s scene_count=%d registered=%d",
        mission_id, scene_count,
        sum(1 for r in sfm_results if r.get("pose_status") == "success"),
    )

    try:
        from pipeline.gps_registration import register_mission_gps
        keyed_frames = sfm_results
        if index_result.get("frame_records"):
            keyed_frames = []
            indexed_by_t = {
                round(frame["t_sec"], 3): frame
                for frame in index_result.get("frame_records", [])
            }
            for row in sfm_results:
                keyed = dict(row)
                matched = indexed_by_t.get(round(row.get("t_sec", 0.0), 3))
                keyed["gps_json"] = matched.get("gps_json") if matched else None
                keyed["id"] = matched.get("id") if matched else None
                keyed_frames.append(keyed)
        enu_origin, global_poses = register_mission_gps(keyed_frames)
        logger.info("Pass A: GPS registration done mission=%s enu_origin=%s", mission_id, enu_origin)
    except Exception as exc:
        logger.warning("Pass A: GPS registration failed mission=%s: %s", mission_id, exc)
        enu_origin = None
        global_poses = {}

    # 3DGS mapper (requires nerfstudio container — soft skip on ConnectionError)
    try:
        from pipeline.mapper import run_mapper
        from pipeline.global_map_db import (
            get_or_create_global_map,
            get_global_map_splats,
            register_mission,
            update_mission_splat_path,
            update_global_map_splat,
        )
    except ImportError as exc:
        logger.debug("Pass A: mapper deps unavailable (%s) — skipping 3DGS", exc)
        return

    async def _db_and_map():
        nonlocal global_map_id
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            await mark_mission_finished(
                conn,
                mission_id,
                status="indexing",
                pose_status="running",
                map_status="running",
            )
            if index_result.get("frame_records"):
                await apply_gps_registration(conn, mission_id, enu_origin, global_poses)

            # Determine target splats from global map
            target_splat_paths = []
            if global_map_id is None and enu_origin is not None:
                if isinstance(enu_origin, dict):
                    lat, lon, alt = enu_origin["lat"], enu_origin["lon"], enu_origin.get("alt", 0.0)
                else:
                    lat, lon, alt = enu_origin
                global_map_id = await get_or_create_global_map(conn, lat, lon, alt)
            if global_map_id is not None:
                target_splat_paths = await get_global_map_splats(conn, global_map_id)

            mapper_result = run_mapper(
                mission_id,
                sfm_results,
                scene_count=scene_count,
                target_splat_paths=target_splat_paths,
            )
            logger.info(
                "Pass A: mapper done mission=%s status=%s splats=%d icp=%d",
                mission_id,
                mapper_result["map_status"],
                len(mapper_result.get("splat_paths", [])),
                len(mapper_result.get("icp_results", [])),
            )

            primary_splat = mapper_result.get("splat_path")

            # Record splat path in missions table so get_global_map_splats can
            # return it to the next mission at this site as an ICP target.
            if primary_splat is not None and global_map_id is not None:
                await update_mission_splat_path(conn, mission_id, primary_splat)

            # Persist ICP registrations; update global map fused splat when produced.
            if global_map_id is not None:
                icp_registered = False
                for icp in mapper_result.get("icp_results", []):
                    if icp.get("converged"):
                        await register_mission(
                            conn,
                            global_map_id,
                            mission_id,
                            icp.get("transform_4x4"),
                            icp.get("rmse"),
                        )
                        icp_registered = True
                        if icp.get("fused_splat"):
                            await update_global_map_splat(
                                conn, global_map_id, icp["fused_splat"]
                            )

                # Bootstrap: no ICP targets existed (first mission at site) or
                # ICP did not converge.  Register with GPS-identity transform so
                # the next mission's get_global_map_splats call finds this splat.
                if not icp_registered and primary_splat is not None:
                    _identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
                    await register_mission(conn, global_map_id, mission_id, _identity, None)

            await mark_mission_finished(
                conn,
                mission_id,
                status="indexing",
                pose_status="success" if sfm_results else "skipped",
                map_status=mapper_result["map_status"],
            )
        finally:
            await conn.close()

    try:
        asyncio.run(_db_and_map())
    except Exception as exc:
        logger.warning("Pass A: mapper/DB step failed mission=%s: %s", mission_id, exc)


# ── GPU resource isolation ───────────────────────────────────────────────────
#
# gpu_jobs table acts as a semaphore: workers check in before allocating GPU
# memory and check out on completion.  Stale entries older than
# GPU_JOB_TIMEOUT_SEC are evicted on every check-in so a crashed worker does
# not permanently block GPU access.

def _gpu_checkin(job_id: str, job_type: str, conn_url: str, logger) -> bool:
    """Reserve the GPU for this job.

    Returns True if the GPU was successfully reserved (no other active job).
    Returns False if another job is holding the GPU.
    """
    async def _checkin():
        conn = await asyncpg.connect(conn_url)
        try:
            now = utcnow()
            stale_cutoff = now - timedelta(seconds=settings.GPU_JOB_TIMEOUT_SEC)
            # Evict stale entries first
            evicted = await conn.execute(
                "DELETE FROM gpu_jobs WHERE started_at < $1", stale_cutoff
            )
            evicted_count = int(evicted.split()[-1])
            if evicted_count:
                logger.info("GPU isolation: evicted %d stale gpu_jobs entry(s)", evicted_count)
            # Try to insert (fails if another row exists due to PK uniqueness on
            # any existing row — we rely on COUNT to detect contention)
            active = await conn.fetchval("SELECT COUNT(*) FROM gpu_jobs")
            if active > 0:
                holder = await conn.fetchrow("SELECT job_id, job_type FROM gpu_jobs LIMIT 1")
                logger.warning(
                    "GPU isolation: GPU busy (job_id=%s type=%s) — job %s will proceed anyway "
                    "(consider scaling down concurrent GPU jobs)",
                    holder["job_id"] if holder else "?",
                    holder["job_type"] if holder else "?",
                    job_id,
                )
                # We insert anyway (the table is informational, not a hard lock)
                # but log the contention so the operator is aware.
            await conn.execute(
                "INSERT INTO gpu_jobs (job_id, job_type, worker_id, started_at) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (job_id) DO NOTHING",
                job_id, job_type, settings.WORKER_ID, now,
            )
            return True
        finally:
            await conn.close()

    try:
        return asyncio.run(_checkin())
    except Exception as exc:
        logger.warning("GPU isolation: check-in failed (non-fatal): %s", exc)
        return True  # fail-open: don't block GPU work on DB errors


def _gpu_checkout(job_id: str, conn_url: str, logger) -> None:
    """Release the GPU reservation for this job."""
    async def _checkout():
        conn = await asyncpg.connect(conn_url)
        try:
            await conn.execute("DELETE FROM gpu_jobs WHERE job_id = $1", job_id)
        finally:
            await conn.close()

    try:
        asyncio.run(_checkout())
    except Exception as exc:
        logger.warning("GPU isolation: check-out failed (non-fatal): %s", exc)


# ── Supervised finetune job handler ─────────────────────────────────────────

def handle_finetune_job(job_id: str, payload: dict, conn_url: str, logger) -> None:
    """Run supervised contrastive fine-tuning on CVAT-annotated frames.

    Expects payload = {} (no fields required; frames fetched from DB via from_db()).
    On success: promotes checkpoint, updates system_state.last_retrain_watermark,
    calls POST /admin/reload-model via HTTP.
    """
    import httpx
    from pipeline.supervised_finetune import config_from_settings, run_supervised_finetune

    def _pg_run(coro):
        return asyncio.run(coro)

    async def _update(conn, **kwargs):
        await update_job(conn, job_id, **kwargs)

    try:
        cfg = config_from_settings(frames_dir=settings.FRAMES_DIR)
        logger.info("Finetune job started id=%s", job_id)

        async def _mark_running():
            conn = await asyncpg.connect(conn_url)
            try:
                await update_job(conn, job_id, status="running", started_at=time.time())
            finally:
                await conn.close()

        _pg_run(_mark_running())

        _gpu_checkin(job_id, "supervised_finetune", conn_url, logger)
        try:
            result = run_supervised_finetune(cfg)
        finally:
            _gpu_checkout(job_id, conn_url, logger)

        if not result["accepted"]:
            logger.info(
                "Finetune job id=%s — checkpoint rejected (accuracy=%.4f < gate=%.4f)",
                job_id, result["best_accuracy"], settings.SUP_EVAL_GATE_THRESHOLD,
            )
            async def _mark_finished_rejected():
                conn = await asyncpg.connect(conn_url)
                try:
                    await update_job(conn, job_id,
                                     status="finished",
                                     progress={"accepted": False,
                                               "best_accuracy": result["best_accuracy"]},
                                     finished_at=time.time())
                finally:
                    await conn.close()
            _pg_run(_mark_finished_rejected())
            return

        # Checkpoint accepted — hot-reload via admin API
        ckpt_path = result["path"]
        api_base = f"http://localhost:{os.environ.get('API_PORT', '8000')}"
        try:
            resp = httpx.post(
                f"{api_base}/admin/reload-model",
                json={"checkpoint": ckpt_path},
                headers={"X-API-Key": settings.API_KEY},
                timeout=30,
            )
            resp.raise_for_status()
            logger.info("Finetune job id=%s — model reloaded checkpoint=%s", job_id, ckpt_path)
        except Exception as exc:
            logger.warning("Finetune job id=%s — reload HTTP call failed: %s", job_id, exc)

        # Update retrain watermark + record provenance + set authoritative checkpoint source
        async def _update_watermark_and_finish():
            conn = await asyncpg.connect(conn_url)
            try:
                now = utcnow()
                total_annotated = await conn.fetchval(
                    "SELECT COUNT(*) FROM frames WHERE al_tag = 'annotated'"
                )
                # Retrain watermark
                await conn.execute(
                    "INSERT INTO system_state (key, value, updated_at) VALUES ($1, $2, $3) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
                    "last_retrain_watermark",
                    str(total_annotated),
                    now,
                )
                # Authoritative active checkpoint source (single source of truth)
                await conn.execute(
                    "INSERT INTO system_state (key, value, updated_at) VALUES ($1, $2, $3) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
                    "active_dino_checkpoint",
                    ckpt_path,
                    now,
                )
                # Model version provenance registry
                model_version_id = f"sup_{job_id[:8]}"
                await conn.execute(
                    "INSERT INTO model_checkpoints "
                    "(checkpoint_path, model_version_id, annotation_count, best_accuracy, "
                    " distribution_shift, created_at, notes) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7) "
                    "ON CONFLICT (checkpoint_path) DO NOTHING",
                    ckpt_path,
                    model_version_id,
                    total_annotated,
                    result["best_accuracy"],
                    result.get("distribution_shift", 0.0),
                    now,
                    f"finetune job_id={job_id}",
                )
                # Update MODEL_VERSION_ID in settings so subsequent indexing is tagged
                settings.MODEL_VERSION_ID = model_version_id
                logger.info(
                    "Finetune job id=%s — provenance registered model_version_id=%s "
                    "annotation_count=%d distribution_shift=%.4f",
                    job_id, model_version_id, total_annotated,
                    result.get("distribution_shift", 0.0),
                )
                await update_job(conn, job_id,
                                 status="finished",
                                 progress={"accepted": True,
                                           "best_accuracy": result["best_accuracy"],
                                           "epochs": result["epochs"],
                                           "checkpoint": ckpt_path,
                                           "model_version_id": model_version_id},
                                 finished_at=now)
            finally:
                await conn.close()

        _pg_run(_update_watermark_and_finish())
        logger.info("Finetune job finished id=%s checkpoint=%s accuracy=%.4f",
                    job_id, ckpt_path, result["best_accuracy"])

    except Exception as exc:
        logger.exception("Finetune job failed id=%s error=%s", job_id, exc)
        async def _mark_error():
            conn = await asyncpg.connect(conn_url)
            try:
                await update_job(conn, job_id, status="error", error=str(exc), finished_at=time.time())
            finally:
                await conn.close()
        _pg_run(_mark_error())


# ── Reembed job handler ──────────────────────────────────────────────────────

def handle_reembed_job(job_id: str, payload: dict, conn_url: str, logger) -> None:
    """Re-embed all indexed frames with the current DINOv3 model.

    Processes frames in batches of REEMBED_BATCH_SIZE (default 256).
    Checkpoints last_offset after each batch so the sweep is resumable.
    """
    from pipeline.qdrant_utils import QdrantStore

    batch_size = settings.REEMBED_BATCH_SIZE

    _gpu_checkin(job_id, "reembed", conn_url, logger)
    try:
        # Load DINO model for embedding
        from models.dino_model import DINOEmbedder
        from models.openclip_model import OpenCLIPEmbedder
        from PIL import Image as PILImage

        dino_name = get_dino_model_name(settings.MODEL_NAME)
        if dino_name is None:
            raise ValueError(f"Unsupported DINO model family: {settings.MODEL_NAME}")
        dino = DINOEmbedder(dino_name)
        clip = OpenCLIPEmbedder()
        qdrant = QdrantStore(
            clip_dim=clip.image_dim(),
            dino_dim=dino.image_dim(),
        )

        async def _run_reembed_with_single_conn() -> int:
            import json

            conn = await asyncpg.connect(conn_url)
            try:
                row = await conn.fetchrow("SELECT progress_json FROM jobs WHERE id = $1", job_id)
                progress = {}
                if row:
                    progress = row["progress_json"] or {}
                    if isinstance(progress, str):
                        progress = json.loads(progress)

                last_cursor_raw = progress.get("last_cursor")
                last_cursor = None
                if isinstance(last_cursor_raw, list) and len(last_cursor_raw) == 2:
                    last_cursor = (last_cursor_raw[0], last_cursor_raw[1])

                logger.info(
                    "Reembed job started id=%s resuming_from_cursor=%s",
                    job_id,
                    last_cursor,
                )

                cursor = tuple(last_cursor) if last_cursor else None
                frames_reembedded = progress.get("frames_reembedded", 0)

                while True:
                    batch = await list_frames_after(conn, cursor, batch_size)
                    if not batch:
                        break

                    images = []
                    valid_rows = []
                    for frame_row in batch:
                        try:
                            img = PILImage.open(frame_row["frame_path"]).convert("RGB")
                            images.append(img)
                            valid_rows.append(frame_row)
                        except Exception as exc:
                            logger.debug(
                                "Reembed: skipping unreadable frame id=%s err=%s",
                                frame_row["id"],
                                exc,
                            )

                    if images:
                        dino_vecs = dino.encode_images(images)
                        clip_vecs = clip.encode_images(images)

                        from qdrant_client.http import models as qmodels

                        points = [
                            qmodels.PointStruct(
                                id=frame_row["qdrant_id"],
                                vector={
                                    "clip": clip_vecs[i].tolist(),
                                    "dino": dino_vecs[i].tolist(),
                                },
                                payload={
                                    "frame_id": frame_row["id"],
                                    "mission_id": frame_row["mission_id"],
                                },
                            )
                            for i, frame_row in enumerate(valid_rows)
                        ]

                        try:
                            qdrant.upsert_points(points)
                        except Exception as exc:
                            logger.error(
                                "Reembed: Qdrant upsert failed cursor=%s err=%s",
                                cursor,
                                exc,
                            )
                            await update_job(
                                conn,
                                job_id,
                                status="error",
                                error=str(exc),
                                progress={
                                    "last_cursor": (
                                        [datetime_to_ts(cursor[0]), cursor[1]]
                                        if cursor
                                        else None
                                    ),
                                    "frames_reembedded": frames_reembedded,
                                },
                                finished_at=time.time(),
                            )
                            return frames_reembedded

                        frames_reembedded += len(valid_rows)

                    last_row = batch[-1]
                    cursor = (last_row["created_at"], last_row["id"])
                    await update_job(
                        conn,
                        job_id,
                        progress={
                            "last_cursor": [datetime_to_ts(cursor[0]), cursor[1]],
                            "frames_reembedded": frames_reembedded,
                        },
                    )
                    logger.debug(
                        "Reembed: cursor=%s frames_reembedded=%d",
                        cursor,
                        frames_reembedded,
                    )

                await update_job(
                    conn,
                    job_id,
                    status="finished",
                    progress={
                        "last_cursor": (
                            [datetime_to_ts(cursor[0]), cursor[1]] if cursor else None
                        ),
                        "frames_reembedded": frames_reembedded,
                    },
                    finished_at=time.time(),
                )
                return frames_reembedded
            finally:
                await conn.close()

        frames_reembedded = asyncio.run(_run_reembed_with_single_conn())
        logger.info(
            "Reembed job finished id=%s frames_reembedded=%d",
            job_id,
            frames_reembedded,
        )

    except Exception as exc:
        logger.exception("Reembed job failed id=%s error=%s", job_id, exc)
        async def _mark_error():
            conn = await asyncpg.connect(conn_url)
            try:
                await update_job(conn, job_id, status="error", error=str(exc),
                                 finished_at=time.time())
            finally:
                await conn.close()
        asyncio.run(_mark_error())
    finally:
        _gpu_checkout(job_id, conn_url, logger)


# ── Main loop ────────────────────────────────────────────────────────────────

def _claim_next_job(pool) -> Optional[dict]:
    """Atomically claim the next pending job using SELECT FOR UPDATE SKIP LOCKED."""
    async def _claim():
        async with pool.acquire() as conn:
            async with conn.transaction():
                return await fetch_and_claim_next_pending(conn)

    return asyncio.run(_claim())


def _update_job_sync(pool, job_id: str, **kwargs) -> None:
    async def _upd():
        async with pool.acquire() as conn:
            await update_job(conn, job_id, **kwargs)

    asyncio.run(_upd())


def main() -> None:
    init_processed_db()
    validate_settings()
    logger = get_logger(__name__)
    logger.info("Worker started")

    conn_url = settings.DATABASE_URL
    if not conn_url:
        logger.error("DATABASE_URL not configured — worker cannot start")
        return

    pool = asyncio.run(
        asyncpg.create_pool(
            dsn=conn_url,
            min_size=1,
            max_size=10,
            timeout=10,
        )
    )

    try:
        while True:
            job = _claim_next_job(pool)
            if not job:
                time.sleep(settings.WORKER_POLL_INTERVAL)
                continue

            job_id = job["id"]
            job_type = job.get("type")
            payload = job["payload"]
            logger.info("Job claimed id=%s type=%s", job_id, job_type)

            # Route by job type
            if job_type == "supervised_finetune":
                handle_finetune_job(job_id, payload, conn_url, logger)
                continue

            if job_type == "reembed":
                handle_reembed_job(job_id, payload, conn_url, logger)
                continue

            if job_type not in (None, "index"):
                logger.warning("Unknown job type=%s id=%s — marking error", job_type, job_id)
                _update_job_sync(pool, job_id, status="error",
                                 error=f"unknown job type: {job_type}",
                                 finished_at=time.time())
                continue

            # Default: index job (type=None for legacy rows, type='index' for new rows)
            logger.info("Index job started id=%s video_id=%s", job_id, payload.get("video_id"))
            indexer = VideoIndexer(enable_tiles=payload.get("enable_tiles", True))

            def progress_cb(progress):
                _update_job_sync(pool, job_id, progress=progress)

            video_path: Optional[str] = None
            try:
                video_id = payload["video_id"]
                video_path = payload.get("video_path")
                url = payload.get("video_url")

                if url and not video_path:
                    video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
                    if payload.get("ingest_mode") == "rtsp":
                        from pipeline.rtsp_ingest import record_rtsp

                        record_rtsp(
                            url,
                            video_path,
                            duration_sec=payload.get("duration_sec"),
                        )
                    else:
                        download_url(url, video_path)

                if not video_path or not os.path.exists(video_path):
                    raise RuntimeError("video_path not found")

                size_bytes = os.path.getsize(video_path)
                mtime = os.path.getmtime(video_path)
                file_hash = file_sha256(video_path)
                existing = get_by_hash(file_hash)
                if existing and existing.get("status") == "processed":
                    if url and video_path and os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except OSError as e:
                            logger.warning(
                                "Could not remove duplicate video file path=%s err=%s",
                                video_path,
                                e,
                            )
                    logger.info(
                        "Skipping duplicate video_id=%s hash=%s",
                        payload.get("video_id"),
                        file_hash,
                    )
                    _update_job_sync(
                        pool,
                        job_id,
                        status="finished",
                        progress={
                            "skipped": True,
                            "reason": "duplicate",
                            "video_id": existing.get("video_id"),
                        },
                        finished_at=time.time(),
                    )
                    continue

                mission_id = payload.get("mission_id") or video_id

                # Resolve site ENU origin before indexing so Qdrant ENU coords are site-consistent
                global_map_id, site_enu_origin = _resolve_site_origin(video_path, logger)

                result = indexer.index_video(
                    video_path,
                    video_id,
                    mission_id=mission_id,
                    robot_id=settings.ROBOT_ID,
                    site_enu_origin=site_enu_origin,
                    global_map_id=global_map_id,
                    progress_cb=progress_cb,
                )
                result_summary = {k: v for k, v in result.items() if k != "frame_records"}

                async def _persist_index_result():
                    conn = await asyncpg.connect(conn_url)
                    try:
                        async with conn.transaction():
                            await upsert_mission(
                                conn,
                                mission_id=mission_id,
                                video_id=video_id,
                                video_path=video_path,
                                job_id=job_id,
                                robot_id=settings.ROBOT_ID,
                                status="indexing",
                                frame_count=result_summary.get("frames", 0),
                                duration_sec=result_summary.get("duration_sec"),
                                gps_origin=result_summary.get("gps_origin"),
                            )
                            await replace_frames(conn, mission_id, result.get("frame_records", []))
                    finally:
                        await conn.close()

                asyncio.run(_persist_index_result())

                # Pass A: SfM → GPS registration → 3DGS mapper (GPU optional)
                _run_pass_a(
                    video_path,
                    video_id,
                    mission_id,
                    result,
                    logger,
                    global_map_id=global_map_id,
                )

                async def _finalize_success():
                    conn = await asyncpg.connect(conn_url)
                    try:
                        async with conn.transaction():
                            await processed_db_mod.aupsert(
                                file_hash,
                                video_id,
                                video_path,
                                size_bytes,
                                mtime,
                                "processed",
                                {"url": url},
                                conn=conn,
                            )
                            await mark_mission_finished(
                                conn,
                                mission_id,
                                status="done",
                                error=None,
                            )
                            await update_job(
                                conn,
                                job_id,
                                status="finished",
                                progress={**(job.get("progress") or {}), **result_summary},
                                finished_at=time.time(),
                            )
                    finally:
                        await conn.close()

                asyncio.run(_finalize_success())
                logger.info("Index job finished id=%s video_id=%s", job_id, video_id)
            except Exception as exc:
                logger.exception("Index job failed id=%s error=%s", job_id, exc)
                if video_path and os.path.exists(video_path):
                    try:
                        size_bytes = os.path.getsize(video_path)
                        mtime = os.path.getmtime(video_path)
                        file_hash = file_sha256(video_path)

                        async def _finalize_error():
                            conn = await asyncpg.connect(conn_url)
                            try:
                                async with conn.transaction():
                                    await processed_db_mod.aupsert(
                                        file_hash,
                                        payload.get("video_id", uuid.uuid4().hex),
                                        video_path,
                                        size_bytes,
                                        mtime,
                                        "error",
                                        {"error": str(exc)},
                                        conn=conn,
                                    )
                                    if payload.get("video_id"):
                                        await mark_mission_finished(
                                            conn,
                                            payload.get("mission_id") or payload["video_id"],
                                            status="error",
                                            error=str(exc),
                                        )
                            finally:
                                await conn.close()

                        asyncio.run(_finalize_error())
                    except OSError as e:
                        logger.warning(
                            "Could not read video for error record path=%s err=%s",
                            video_path,
                            e,
                        )
                    except Exception as e:
                        logger.warning(
                            "Could not upsert error record for path=%s err=%s",
                            video_path,
                            e,
                        )
                _update_job_sync(
                    pool, job_id, status="error", error=str(exc), finished_at=time.time()
                )
    finally:
        asyncio.run(pool.close())


if __name__ == "__main__":
    main()
