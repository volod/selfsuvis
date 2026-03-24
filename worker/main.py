import asyncio
import os
import time
import uuid
from typing import Optional

from pipeline.job_db import init_db, fetch_and_claim_next_pending, update_job
from pipeline.indexer import VideoIndexer
from pipeline.utils import file_sha256
from pipeline.logging_utils import get_logger
from pipeline.processed_db import init_db as init_processed_db, get_by_hash, upsert
from pipeline.downloader import download_url
from pipeline.config import settings, validate_settings


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
        import asyncpg
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
        enu_origin, global_poses = register_mission_gps(sfm_results)
        logger.info("Pass A: GPS registration done mission=%s enu_origin=%s", mission_id, enu_origin)
    except Exception as exc:
        logger.warning("Pass A: GPS registration failed mission=%s: %s", mission_id, exc)
        enu_origin = None

    # 3DGS mapper (requires nerfstudio container — soft skip on ConnectionError)
    try:
        import asyncpg  # type: ignore
        from pipeline.mapper import run_mapper
        from pipeline.global_map_db import (
            get_or_create_global_map,
            get_global_map_splats,
            register_mission,
        )
    except ImportError as exc:
        logger.debug("Pass A: mapper deps unavailable (%s) — skipping 3DGS", exc)
        return

    async def _db_and_map():
        nonlocal global_map_id
        db_url = settings.DATABASE_URL
        conn = await asyncpg.connect(db_url)
        try:
            # Determine target splats from global map
            target_splat_paths = []
            if global_map_id is None and enu_origin is not None:
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

            # Persist ICP results
            if global_map_id is not None:
                for icp in mapper_result.get("icp_results", []):
                    if icp.get("converged"):
                        await register_mission(
                            conn,
                            global_map_id,
                            mission_id,
                            icp.get("transform_4x4"),
                            icp.get("rmse"),
                        )
        finally:
            await conn.close()

    try:
        asyncio.run(_db_and_map())
    except Exception as exc:
        logger.warning("Pass A: mapper/DB step failed mission=%s: %s", mission_id, exc)


def main() -> None:
    init_db()
    init_processed_db()
    validate_settings()
    logger = get_logger(__name__)
    logger.info("Worker started")
    while True:
        job = fetch_and_claim_next_pending()
        if not job:
            time.sleep(settings.WORKER_POLL_INTERVAL)
            continue

        job_id = job["id"]
        payload = job["payload"]
        logger.info("Job started id=%s video_id=%s", job_id, payload.get("video_id"))

        indexer = VideoIndexer(enable_tiles=payload.get("enable_tiles", True))

        def progress_cb(progress):
            update_job(job_id, progress=progress)

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
                        logger.warning("Could not remove duplicate video file path=%s err=%s", video_path, e)
                logger.info("Skipping duplicate video_id=%s hash=%s", payload.get("video_id"), file_hash)
                update_job(
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
                video_path, video_id,
                mission_id=mission_id,
                robot_id=settings.ROBOT_ID,
                site_enu_origin=site_enu_origin,
                global_map_id=global_map_id,
                progress_cb=progress_cb,
            )

            # Pass A: SfM → GPS registration → 3DGS mapper (GPU optional)
            _run_pass_a(video_path, video_id, mission_id, result, logger, global_map_id=global_map_id)

            upsert(file_hash, video_id, video_path, size_bytes, mtime, "processed", {"url": url})
            logger.info("Job finished id=%s video_id=%s", job_id, video_id)
            update_job(
                job_id,
                status="finished",
                progress={**(job.get("progress") or {}), **result},
                finished_at=time.time(),
            )
        except Exception as exc:
            logger.exception("Job failed id=%s error=%s", job_id, exc)
            # Only upsert error state if video_path was set and the file exists
            if video_path and os.path.exists(video_path):
                try:
                    size_bytes = os.path.getsize(video_path)
                    mtime = os.path.getmtime(video_path)
                    file_hash = file_sha256(video_path)
                    upsert(file_hash, payload.get("video_id", uuid.uuid4().hex), video_path, size_bytes, mtime, "error", {"error": str(exc)})
                except OSError as e:
                    logger.warning("Could not read video for error record path=%s err=%s", video_path, e)
                except Exception as e:
                    logger.warning("Could not upsert error record for path=%s err=%s", video_path, e)
            update_job(job_id, status="error", error=str(exc), finished_at=time.time())


if __name__ == "__main__":
    main()
