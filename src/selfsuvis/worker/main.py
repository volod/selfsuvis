import asyncio
import os
import time
import uuid
from datetime import timedelta
from enum import Enum

import asyncpg

import selfsuvis.pipeline.storage.processed as processed_db_mod
from selfsuvis.pipeline.core import (
    datetime_to_ts,
    file_sha256,
    get_dino_model_name,
    get_logger,
    log_preflight,
    run_production_preflight,
    settings,
    utcnow,
    validate_settings,
)
from selfsuvis.pipeline.media import download_url
from selfsuvis.pipeline.storage import create_job, fetch_and_claim_next_pending, update_job
from selfsuvis.pipeline.storage.missions import (
    apply_gps_registration,
    fetch_mission,
    list_frames_after,
    list_mission_frames,
    mark_mission_finished,
    replace_frames,
    upsert_mission,
)
from selfsuvis.pipeline.storage.processed import ainit_db as init_processed_db_async
from selfsuvis.pipeline.storage.processed import get_by_hash
from selfsuvis.pipeline.workflows import VideoIndexer

# Persistent event loop for the worker process.  _run() creates and
# closes a new loop on every call; asyncpg pools are tied to the loop they were
# created in, so mixing multiple _run() calls with a shared pool breaks.
_loop: asyncio.AbstractEventLoop | None = None


def _run(coro):
    """Run *coro* on the worker's persistent event loop."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


class JobType(str, Enum):
    INDEX = "index"
    FINETUNE = "supervised_finetune"
    REEMBED = "reembed"
    POSTFLIGHT_MAPPING = "postflight_mapping"
    POSTFLIGHT_SEMANTIC_GRAPH = "postflight_semantic_graph"


def _normalize_postflight_job_names(value) -> list[str]:
    names = []
    for item in value or []:
        name = str(item or "").strip()
        if not name:
            continue
        if name not in names:
            names.append(name)
    return names


async def _enqueue_postflight_jobs(conn, payload: dict, logger) -> list[str]:
    requested = _normalize_postflight_job_names(payload.get("postflight_jobs"))
    if not requested:
        return []

    created: list[str] = []
    base_payload = {
        "video_id": payload.get("video_id"),
        "video_path": payload.get("video_path"),
        "mission_id": payload.get("mission_id") or payload.get("video_id"),
        "realtime_session_id": payload.get("realtime_session_id"),
    }
    remaining = list(requested)
    while remaining:
        job_name = remaining.pop(0)
        next_jobs = list(remaining)
        if job_name not in {
            JobType.POSTFLIGHT_MAPPING.value,
            JobType.POSTFLIGHT_SEMANTIC_GRAPH.value,
        }:
            logger.warning("Skipping unknown postflight job type=%s", job_name)
            continue
        child_job_id = uuid.uuid4().hex
        child_payload = {**base_payload, "next_postflight_jobs": next_jobs}
        await create_job(conn, child_job_id, child_payload, job_type=job_name)
        created.append(child_job_id)
        logger.info(
            "Post-flight job enqueued parent_mission=%s job_id=%s type=%s",
            base_payload["mission_id"],
            child_job_id,
            job_name,
        )
        break
    return created


async def _finalize_postflight_job_success(
    conn,
    *,
    job_id: str,
    mission_id: str,
    payload: dict,
    progress: dict,
    logger,
) -> None:
    await update_job(
        conn,
        job_id,
        status="finished",
        progress=progress,
        finished_at=time.time(),
    )
    next_payload = {
        **payload,
        "postflight_jobs": payload.get("next_postflight_jobs", []),
    }
    created = await _enqueue_postflight_jobs(conn, next_payload, logger)
    if not created:
        await mark_mission_finished(conn, mission_id, status="done", error=None)


async def _finalize_postflight_job_error(
    conn,
    *,
    job_id: str,
    mission_id: str,
    error: str,
) -> None:
    await update_job(
        conn,
        job_id,
        status="error",
        error=error,
        finished_at=time.time(),
    )
    await mark_mission_finished(conn, mission_id, status="error", error=error)


def _resolve_site_origin(video_path: str, logger) -> tuple:
    """Extract the first GPS fix from a video and look up (or create) its global_map.

    Returns (global_map_id, (origin_lat, origin_lon, origin_alt)) on success,
    or (None, None) if GPS is unavailable or DB is unreachable.

    This runs synchronously before index_video so that ENU coordinates stored
    in Qdrant are relative to the site's canonical ENU origin rather than each
    mission's own local first-frame origin.
    """
    try:
        from selfsuvis.pipeline.gps_extractor import extract_gps

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
        from selfsuvis.pipeline.global_map_db import get_global_map_origin, get_or_create_global_map

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

        gmap_id, origin = _run(_lookup())
        if origin is not None:
            logger.info(
                "Multi-site ENU: video assigned to global_map_id=%d origin=(%.6f, %.6f, %.1f)",
                gmap_id,
                origin[0],
                origin[1],
                origin[2],
            )
        return gmap_id, origin
    except Exception as exc:
        logger.debug("Multi-site ENU: site origin lookup failed: %s", exc)
        return None, None


def _run_pass_a(
    video_path: str,
    video_id: str,
    mission_id: str,
    index_result: dict,
    logger,
    global_map_id: int = None,
) -> None:
    """Run SfM → GPS registration → nerfstudio 3DGS for a completed indexing job.

    All steps degrade gracefully: missing optional deps (pycolmap, open3d) or
    unreachable containers (nerfstudio, mapper, postgres) are logged and skipped.
    """
    try:
        from selfsuvis.pipeline.sfm import run_sfm
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
        mission_id,
        scene_count,
        sum(1 for r in sfm_results if r.get("pose_status") == "success"),
    )

    try:
        from selfsuvis.pipeline.gps_registration import register_mission_gps

        keyed_frames = sfm_results
        if index_result.get("frame_records"):
            keyed_frames = []
            indexed_by_t = {
                round(frame["t_sec"], 3): frame for frame in index_result.get("frame_records", [])
            }
            for row in sfm_results:
                keyed = dict(row)
                matched = indexed_by_t.get(round(row.get("t_sec", 0.0), 3))
                keyed["gps_json"] = matched.get("gps_json") if matched else None
                keyed["id"] = matched.get("id") if matched else None
                keyed_frames.append(keyed)
        enu_origin, global_poses = register_mission_gps(keyed_frames)
        logger.info(
            "Pass A: GPS registration done mission=%s enu_origin=%s", mission_id, enu_origin
        )
    except Exception as exc:
        logger.warning("Pass A: GPS registration failed mission=%s: %s", mission_id, exc)
        enu_origin = None
        global_poses = {}

    # 3DGS mapper (requires nerfstudio container — soft skip on ConnectionError)
    try:
        from selfsuvis.pipeline.global_map_db import (
            get_global_map_splats,
            get_or_create_global_map,
            register_mission,
            update_global_map_splat,
            update_mission_splat_path,
        )
        from selfsuvis.pipeline.mapper import run_mapper
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
                            await update_global_map_splat(conn, global_map_id, icp["fused_splat"])

                # Bootstrap: no ICP targets existed (first mission at site) or
                # ICP did not converge.  Register with GPS-identity transform so
                # the next mission's get_global_map_splats call finds this splat.
                if not icp_registered and primary_splat is not None:
                    _identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
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
        _run(_db_and_map())
    except Exception as exc:
        logger.warning("Pass A: mapper/DB step failed mission=%s: %s", mission_id, exc)


# -- GPU resource isolation ---------------------------------------------------
#
# gpu_jobs table acts as a semaphore: workers check in before allocating GPU
# memory and check out on completion.  Stale entries older than
# GPU_JOB_TIMEOUT_SEC are evicted on every check-in so a crashed worker does
# not permanently block GPU access.
#
# The semaphore is advisory (fail-open): DB errors never block GPU work.
# Contention is logged so operators are aware.


class GPULock:
    """Context manager that registers/deregisters a job in the gpu_jobs table."""

    def __init__(self, job_id: str, job_type: str, conn_url: str, logger):
        self.job_id = job_id
        self.job_type = job_type
        self.conn_url = conn_url
        self.logger = logger

    async def _checkin(self) -> None:
        conn = await asyncpg.connect(self.conn_url)
        try:
            now = utcnow()
            stale_cutoff = now - timedelta(seconds=settings.GPU_JOB_TIMEOUT_SEC)
            evicted = await conn.execute("DELETE FROM gpu_jobs WHERE started_at < $1", stale_cutoff)
            evicted_count = int(evicted.split()[-1])
            if evicted_count:
                self.logger.info("GPU isolation: evicted %d stale gpu_jobs entry(s)", evicted_count)
            active = await conn.fetchval("SELECT COUNT(*) FROM gpu_jobs")
            if active > 0:
                holder = await conn.fetchrow("SELECT job_id, job_type FROM gpu_jobs LIMIT 1")
                self.logger.warning(
                    "GPU isolation: GPU busy (job_id=%s type=%s) — job %s will proceed anyway "
                    "(consider scaling down concurrent GPU jobs)",
                    holder["job_id"] if holder else "?",
                    holder["job_type"] if holder else "?",
                    self.job_id,
                )
            await conn.execute(
                "INSERT INTO gpu_jobs (job_id, job_type, worker_id, started_at) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (job_id) DO NOTHING",
                self.job_id,
                self.job_type,
                settings.WORKER_ID,
                now,
            )
        finally:
            await conn.close()

    async def _checkout(self) -> None:
        conn = await asyncpg.connect(self.conn_url)
        try:
            await conn.execute("DELETE FROM gpu_jobs WHERE job_id = $1", self.job_id)
        finally:
            await conn.close()

    def __enter__(self):
        try:
            _run(self._checkin())
        except Exception as exc:
            self.logger.warning("GPU isolation: check-in failed (non-fatal): %s", exc)
        return self

    def __exit__(self, *_):
        try:
            _run(self._checkout())
        except Exception as exc:
            self.logger.warning("GPU isolation: check-out failed (non-fatal): %s", exc)


# -- Backward-compat wrappers (used by tests and legacy call sites) ------------


def _gpu_checkin(job_id: str, job_type: str, conn_url: str, logger) -> bool:
    """Synchronous wrapper around GPULock._checkin. Always returns True (fail-open)."""
    lock = GPULock(job_id, job_type, conn_url, logger)
    try:
        _run(lock._checkin())
    except Exception as exc:
        logger.warning("GPU isolation: check-in failed (non-fatal): %s", exc)
    return True


def _gpu_checkout(job_id: str, conn_url: str, logger) -> None:
    """Synchronous wrapper around GPULock._checkout. Fail-open on error."""
    lock = GPULock(job_id, "unknown", conn_url, logger)
    try:
        _run(lock._checkout())
    except Exception as exc:
        logger.warning("GPU isolation: check-out failed (non-fatal): %s", exc)


# -- Supervised finetune job handler -----------------------------------------

_UPSERT_SYSTEM_STATE_SQL = (
    "INSERT INTO system_state (key, value, updated_at) VALUES ($1, $2, $3) "
    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at"
)

_API_RELOAD_TIMEOUT_SEC = 30


async def _persist_finetune_acceptance(
    conn, job_id: str, ckpt_path: str, result: dict, logger
) -> str:
    """Persist watermark, checkpoint provenance, and mark job finished.

    Returns the model_version_id assigned to this checkpoint.
    """
    now = utcnow()
    total_annotated = await conn.fetchval("SELECT COUNT(*) FROM frames WHERE al_tag = 'annotated'")
    model_version_id = f"sup_{job_id[:8]}"

    await conn.execute(
        _UPSERT_SYSTEM_STATE_SQL, "last_retrain_watermark", str(total_annotated), now
    )
    await conn.execute(_UPSERT_SYSTEM_STATE_SQL, "active_dino_checkpoint", ckpt_path, now)
    await conn.execute(
        "INSERT INTO model_checkpoints "
        "(checkpoint_path, model_version_id, annotation_count, best_accuracy, "
        " distribution_shift, created_at, notes) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (checkpoint_path) DO NOTHING",
        ckpt_path,
        model_version_id,
        total_annotated,
        result["best_accuracy"],
        result.get("distribution_shift", 0.0),
        now,
        f"finetune job_id={job_id}",
    )
    settings.MODEL_VERSION_ID = model_version_id
    logger.info(
        "Finetune job id=%s — provenance registered model_version_id=%s "
        "annotation_count=%d distribution_shift=%.4f",
        job_id,
        model_version_id,
        total_annotated,
        result.get("distribution_shift", 0.0),
    )
    await update_job(
        conn,
        job_id,
        status="finished",
        progress={
            "accepted": True,
            "best_accuracy": result["best_accuracy"],
            "epochs": result["epochs"],
            "checkpoint": ckpt_path,
            "model_version_id": model_version_id,
        },
        finished_at=now,
    )
    return model_version_id


def _hot_reload_model(ckpt_path: str, job_id: str, logger) -> None:
    """Call POST /admin/reload-model to swap DINOv3 weights in the API process."""
    import httpx

    api_base = f"http://localhost:{os.environ.get('API_PORT', '8000')}"
    try:
        resp = httpx.post(
            f"{api_base}/admin/reload-model",
            json={"checkpoint": ckpt_path},
            headers={"X-API-Key": settings.API_KEY},
            timeout=_API_RELOAD_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        logger.info("Finetune job id=%s — model reloaded checkpoint=%s", job_id, ckpt_path)
    except Exception as exc:
        logger.warning("Finetune job id=%s — reload HTTP call failed: %s", job_id, exc)


def handle_finetune_job(job_id: str, payload: dict, db_pool, conn_url: str, logger) -> None:
    """Run supervised contrastive fine-tuning on CVAT-annotated frames.

    Expects payload = {} (no fields required; frames fetched from DB via from_db()).
    On success: promotes checkpoint, updates system_state.last_retrain_watermark,
    calls POST /admin/reload-model via HTTP.
    """
    from selfsuvis.pipeline.training.supervised import config_from_settings, run_supervised_finetune

    def _pg_run(coro):
        return _run(coro)

    try:
        cfg = config_from_settings(frames_dir=settings.FRAMES_DIR)
        logger.info("Finetune job started id=%s", job_id)

        async def _mark_running():
            async with db_pool.acquire() as conn:
                await update_job(conn, job_id, status="running", started_at=time.time())

        _pg_run(_mark_running())

        with GPULock(job_id, "supervised_finetune", conn_url, logger):
            result = run_supervised_finetune(cfg)

        if not result["accepted"]:
            logger.info(
                "Finetune job id=%s — checkpoint rejected (accuracy=%.4f < gate=%.4f)",
                job_id,
                result["best_accuracy"],
                settings.SUP_EVAL_GATE_THRESHOLD,
            )

            async def _mark_rejected():
                async with db_pool.acquire() as conn:
                    await update_job(
                        conn,
                        job_id,
                        status="finished",
                        progress={"accepted": False, "best_accuracy": result["best_accuracy"]},
                        finished_at=time.time(),
                    )

            _pg_run(_mark_rejected())
            return

        ckpt_path = result["path"]
        _hot_reload_model(ckpt_path, job_id, logger)

        async def _finish_accepted():
            async with db_pool.acquire() as conn:
                await _persist_finetune_acceptance(conn, job_id, ckpt_path, result, logger)

        _pg_run(_finish_accepted())
        logger.info(
            "Finetune job finished id=%s checkpoint=%s accuracy=%.4f",
            job_id,
            ckpt_path,
            result["best_accuracy"],
        )

    except Exception as exc:
        logger.exception("Finetune job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            async with db_pool.acquire() as conn:
                await update_job(
                    conn, job_id, status="error", error=error_message, finished_at=time.time()
                )

        _pg_run(_mark_error())


# -- Reembed job handler ------------------------------------------------------


async def _load_reembed_cursor(conn, job_id: str) -> tuple:
    """Return (cursor, frames_reembedded) restored from stored job progress."""
    import json as _json

    row = await conn.fetchrow("SELECT progress_json FROM jobs WHERE id = $1", job_id)
    progress: dict = {}
    if row:
        progress = row["progress_json"] or {}
        if isinstance(progress, str):
            progress = _json.loads(progress)
    raw = progress.get("last_cursor")
    cursor = (raw[0], raw[1]) if isinstance(raw, list) and len(raw) == 2 else None
    return cursor, progress.get("frames_reembedded", 0)


def _load_batch_images(batch, logger) -> tuple:
    """Open PIL images for a frame batch, skipping unreadable files.

    Returns (images, valid_rows).
    """
    from PIL import Image as PILImage

    images, valid_rows = [], []
    for frame_row in batch:
        try:
            img = PILImage.open(frame_row["frame_path"]).convert("RGB")
            images.append(img)
            valid_rows.append(frame_row)
        except Exception as exc:
            logger.warning("Reembed: skipping unreadable frame id=%s err=%s", frame_row["id"], exc)
    return images, valid_rows


def _build_reembed_points(valid_rows, dino_vecs, clip_vecs):
    """Assemble Qdrant PointStructs from embedding vectors and frame metadata."""
    from qdrant_client.http import models as qmodels

    return [
        qmodels.PointStruct(
            id=row["qdrant_id"],
            vector={"clip": clip_vecs[i].tolist(), "dino": dino_vecs[i].tolist()},
            payload={"frame_id": row["id"], "mission_id": row["mission_id"]},
        )
        for i, row in enumerate(valid_rows)
    ]


async def _run_reembed(conn, job_id: str, dino, clip, qdrant, batch_size: int, logger) -> int:
    """Iterate all frames in cursor order, re-embed, and checkpoint progress.

    Returns the total number of frames re-embedded.
    """
    cursor, frames_reembedded = await _load_reembed_cursor(conn, job_id)
    logger.info("Reembed job started id=%s resuming_from_cursor=%s", job_id, cursor)

    while True:
        batch = await list_frames_after(conn, cursor, batch_size)
        if not batch:
            break

        images, valid_rows = _load_batch_images(batch, logger)
        if images:
            dino_vecs = dino.encode_images(images)
            clip_vecs = clip.encode_images(images)
            points = _build_reembed_points(valid_rows, dino_vecs, clip_vecs)
            try:
                qdrant.upsert_points(points)
            except Exception as exc:
                logger.error("Reembed: Qdrant upsert failed cursor=%s err=%s", cursor, exc)
                cursor_serial = [datetime_to_ts(cursor[0]), cursor[1]] if cursor else None
                await update_job(
                    conn,
                    job_id,
                    status="error",
                    error=str(exc),
                    progress={"last_cursor": cursor_serial, "frames_reembedded": frames_reembedded},
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
        logger.debug("Reembed: cursor=%s frames_reembedded=%d", cursor, frames_reembedded)

    cursor_serial = [datetime_to_ts(cursor[0]), cursor[1]] if cursor else None
    await update_job(
        conn,
        job_id,
        status="finished",
        progress={"last_cursor": cursor_serial, "frames_reembedded": frames_reembedded},
        finished_at=time.time(),
    )
    return frames_reembedded


def handle_reembed_job(job_id: str, payload: dict, conn_url: str, logger) -> None:
    """Re-embed all indexed frames with the current DINOv3 model.

    Processes frames in batches of REEMBED_BATCH_SIZE (default 256).
    Checkpoints last_cursor after each batch so the sweep is resumable.
    """
    from selfsuvis.models.dino_model import DINOEmbedder
    from selfsuvis.models.openclip_model import OpenCLIPEmbedder
    from selfsuvis.pipeline.storage.qdrant import QdrantStore

    try:
        dino_name = get_dino_model_name(settings.MODEL_NAME)
        if dino_name is None:
            raise ValueError(f"Unsupported DINO model family: {settings.MODEL_NAME}")
        dino = DINOEmbedder(dino_name)
        clip = OpenCLIPEmbedder()
        qdrant = QdrantStore(clip_dim=clip.image_dim(), dino_dim=dino.image_dim())

        async def _connect_and_run() -> int:
            conn = await asyncpg.connect(conn_url)
            try:
                return await _run_reembed(
                    conn, job_id, dino, clip, qdrant, settings.REEMBED_BATCH_SIZE, logger
                )
            finally:
                await conn.close()

        with GPULock(job_id, "reembed", conn_url, logger):
            frames_reembedded = _run(_connect_and_run())
        logger.info("Reembed job finished id=%s frames_reembedded=%d", job_id, frames_reembedded)

    except Exception as exc:
        logger.exception("Reembed job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            conn = await asyncpg.connect(conn_url)
            try:
                await update_job(
                    conn, job_id, status="error", error=error_message, finished_at=time.time()
                )
            finally:
                await conn.close()

        _run(_mark_error())


def handle_postflight_mapping_job(job_id: str, payload: dict, pool, logger) -> None:
    video_path = payload.get("video_path")
    video_id = payload.get("video_id")
    mission_id = payload.get("mission_id") or video_id
    if not video_path or not video_id or not mission_id:
        _update_job_sync(
            pool,
            job_id,
            status="error",
            error="postflight_mapping requires video_id, mission_id, and video_path",
            finished_at=time.time(),
        )
        return

    try:
        if not os.path.exists(video_path):
            raise RuntimeError("video_path not found")

        async def _load_frames():
            async with pool.acquire() as conn:
                return await list_mission_frames(conn, mission_id)

        frame_records = _run(_load_frames())
        global_map_id, _site_enu_origin = _resolve_site_origin(video_path, logger)
        _run_pass_a(
            video_path,
            video_id,
            mission_id,
            {"frame_records": frame_records},
            logger,
            global_map_id=global_map_id,
        )

        async def _finish():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_success(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        payload=payload,
                        progress={"mission_id": mission_id, "video_id": video_id},
                        logger=logger,
                    )

        _run(_finish())
        logger.info("Post-flight mapping job finished id=%s mission=%s", job_id, mission_id)
    except Exception as exc:
        logger.exception("Post-flight mapping job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_error(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        error=error_message,
                    )

        _run(_mark_error())


def handle_postflight_semantic_graph_job(job_id: str, payload: dict, pool, logger) -> None:
    mission_id = payload.get("mission_id") or payload.get("video_id")
    if not mission_id:
        _update_job_sync(
            pool,
            job_id,
            status="error",
            error="postflight_semantic_graph requires mission_id",
            finished_at=time.time(),
        )
        return

    try:
        from selfsuvis.pipeline.mapping import (
            build_semantic_environment_graph,
            write_semantic_graph_markdown,
        )

        async def _load_mission_state():
            async with pool.acquire() as conn:
                mission = await fetch_mission(conn, mission_id)
                frames = await list_mission_frames(conn, mission_id)
                return mission, frames

        mission, frames = _run(_load_mission_state())
        if mission is None:
            raise LookupError(f"mission not found: {mission_id}")

        graph_dir = os.path.join(settings.MAPS_DIR, mission_id)
        graph_json = os.path.join(graph_dir, "semantic_environment_graph.json")
        graph_md = os.path.join(graph_dir, "semantic_environment_graph.md")
        graph = build_semantic_environment_graph(
            frames,
            graph_id=mission_id,
            output_path=graph_json,
        )
        write_semantic_graph_markdown(
            graph,
            graph_md,
            title=f"{mission_id} — Semantic Environment Graph",
        )

        async def _finish():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_success(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        payload=payload,
                        progress={
                            "mission_id": mission_id,
                            "graph_json": graph_json,
                            "graph_markdown": graph_md,
                            "node_count": graph.get("summary", {}).get("node_count", 0),
                            "edge_count": graph.get("summary", {}).get("edge_count", 0),
                        },
                        logger=logger,
                    )

        _run(_finish())
        logger.info("Post-flight semantic graph job finished id=%s mission=%s", job_id, mission_id)
    except Exception as exc:
        logger.exception("Post-flight semantic graph job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_error(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        error=error_message,
                    )

        _run(_mark_error())


# -- Main loop ----------------------------------------------------------------


def _claim_next_job(pool) -> dict | None:
    """Atomically claim the next pending job using SELECT FOR UPDATE SKIP LOCKED."""

    async def _claim():
        async with pool.acquire() as conn:
            async with conn.transaction():
                return await fetch_and_claim_next_pending(conn)

    return _run(_claim())


def _update_job_sync(pool, job_id: str, **kwargs) -> None:
    async def _upd():
        async with pool.acquire() as conn:
            await update_job(conn, job_id, **kwargs)

    _run(_upd())


def main() -> None:
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    validate_settings()
    logger = get_logger(__name__)
    report = run_production_preflight("worker")
    log_preflight(report)
    if os.getenv("STARTUP_PREFLIGHT_STRICT", "false").lower() == "true":
        report.raise_for_errors()
    logger.info("Worker started")

    conn_url = settings.DATABASE_URL
    if not conn_url:
        logger.error("DATABASE_URL not configured — worker cannot start")
        return

    async def _bootstrap():
        await init_processed_db_async()
        return await asyncpg.create_pool(
            dsn=conn_url,
            min_size=1,
            max_size=10,
            timeout=10,
        )

    pool = _run(_bootstrap())

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
            if job_type == JobType.FINETUNE:
                handle_finetune_job(job_id, payload, pool, conn_url, logger)
                continue

            if job_type == JobType.REEMBED:
                handle_reembed_job(job_id, payload, conn_url, logger)
                continue

            if job_type == JobType.POSTFLIGHT_MAPPING:
                handle_postflight_mapping_job(job_id, payload, pool, logger)
                continue

            if job_type == JobType.POSTFLIGHT_SEMANTIC_GRAPH:
                handle_postflight_semantic_graph_job(job_id, payload, pool, logger)
                continue

            if job_type not in (None, JobType.INDEX):
                logger.warning("Unknown job type=%s id=%s — marking error", job_type, job_id)
                _update_job_sync(
                    pool,
                    job_id,
                    status="error",
                    error=f"unknown job type: {job_type}",
                    finished_at=time.time(),
                )
                continue

            # Default: index job (type=None for legacy rows, type='index' for new rows)
            logger.info("Index job started id=%s video_id=%s", job_id, payload.get("video_id"))
            indexer = VideoIndexer(enable_tiles=payload.get("enable_tiles", True))

            def progress_cb(progress):
                _update_job_sync(pool, job_id, progress=progress)

            video_path: str | None = None
            try:
                video_id = payload["video_id"]
                video_path = payload.get("video_path")
                url = payload.get("video_url")

                if url and not video_path:
                    video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
                    if payload.get("ingest_mode") == "rtsp":
                        from selfsuvis.pipeline.media.rtsp_ingest import record_rtsp

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
                    async with pool.acquire() as conn:
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

                _run(_persist_index_result())

                postflight_jobs = _normalize_postflight_job_names(payload.get("postflight_jobs"))

                async def _finalize_success():
                    async with pool.acquire() as conn:
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
                            if postflight_jobs:
                                await mark_mission_finished(
                                    conn,
                                    mission_id,
                                    status="indexing",
                                    error=None,
                                )
                                await _enqueue_postflight_jobs(conn, payload, logger)
                            else:
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

                _run(_finalize_success())
                logger.info("Index job finished id=%s video_id=%s", job_id, video_id)
            except Exception as exc:
                logger.exception("Index job failed id=%s error=%s", job_id, exc)
                if video_path and os.path.exists(video_path):
                    try:
                        size_bytes = os.path.getsize(video_path)
                        mtime = os.path.getmtime(video_path)
                        file_hash = file_sha256(video_path)
                        error_message = str(exc)

                        async def _finalize_error():
                            async with pool.acquire() as conn:
                                async with conn.transaction():
                                    await processed_db_mod.aupsert(
                                        file_hash,
                                        payload.get("video_id", uuid.uuid4().hex),
                                        video_path,
                                        size_bytes,
                                        mtime,
                                        "error",
                                        {"error": error_message},
                                        conn=conn,
                                    )
                                    if payload.get("video_id"):
                                        await mark_mission_finished(
                                            conn,
                                            payload.get("mission_id") or payload["video_id"],
                                            status="error",
                                            error=error_message,
                                        )

                        _run(_finalize_error())
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
        _run(pool.close())


if __name__ == "__main__":
    main()
