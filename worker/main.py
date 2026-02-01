import os
import time
import uuid

from pipeline.job_db import init_db, fetch_next_pending, update_job
from pipeline.indexer import VideoIndexer
from pipeline.utils import file_sha256
from pipeline.logging_utils import get_logger
from pipeline.processed_db import init_db as init_processed_db, get_by_hash, upsert
from pipeline.downloader import download_url
from pipeline.config import settings


POLL_INTERVAL = 2.0


def main():
    init_db()
    init_processed_db()
    logger = get_logger(__name__)
    logger.info("Worker started")
    while True:
        job = fetch_next_pending()
        if not job:
            time.sleep(POLL_INTERVAL)
            continue

        job_id = job["id"]
        payload = job["payload"]
        update_job(job_id, status="running", started_at=time.time())
        logger.info("Job started id=%s video_id=%s", job_id, payload.get("video_id"))

        indexer = VideoIndexer(enable_tiles=payload.get("enable_tiles", True))

        def progress_cb(progress):
            update_job(job_id, progress=progress)

        try:
            video_id = payload["video_id"]
            video_path = payload.get("video_path")
            url = payload.get("video_url")

            if url and not video_path:
                video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
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
                    except Exception:
                        pass
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

            result = indexer.index_video(video_path, video_id, progress_cb=progress_cb)
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
            if "video_path" in locals() and video_path and os.path.exists(video_path):
                size_bytes = os.path.getsize(video_path)
                mtime = os.path.getmtime(video_path)
                try:
                    file_hash = file_sha256(video_path)
                    upsert(file_hash, payload.get("video_id", uuid.uuid4().hex), video_path, size_bytes, mtime, "error", {"error": str(exc)})
                except Exception:
                    pass
            update_job(job_id, status="error", error=str(exc), finished_at=time.time())


if __name__ == "__main__":
    main()
