import os
import pathlib
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api_utils import ERROR_RESPONSES, error_response
from app.deps import rate_limit, require_api_key
from app.services.upload_utils import hash_upload_limited
from app.state import logger
from pipeline.config import settings
from pipeline.net_utils import safe_request, validate_url
from pipeline.processed_db import get_by_hash, get_by_size, get_by_url
from pipeline.utils import ensure_dir, file_sha256, resolve_allowed_path
from pipeline.job_db import create_job

router = APIRouter(tags=["index"], dependencies=[Depends(require_api_key), Depends(rate_limit)])


def _enqueue_job(payload: dict) -> str:
    job_id = uuid.uuid4().hex
    create_job(job_id, payload)
    return job_id


def _check_dir_limits(root: str, base: str, file_count: int, total_bytes: int) -> str:
    rel = os.path.relpath(root, base)
    depth = 0 if rel == "." else rel.count(os.sep) + 1
    if settings.MAX_DIR_DEPTH and depth > settings.MAX_DIR_DEPTH:
        return "directory depth limit exceeded"
    if settings.MAX_DIR_FILES and file_count > settings.MAX_DIR_FILES:
        return "directory file count limit exceeded"
    if settings.MAX_DIR_BYTES and total_bytes > settings.MAX_DIR_BYTES:
        return "directory byte limit exceeded"
    return ""


@router.post(
    "/index/video",
    summary="Index a video from upload or allowed path",
    responses={400: ERROR_RESPONSES[400], 403: ERROR_RESPONSES[403], 413: ERROR_RESPONSES[413]},
)
async def index_video(
    file: Optional[UploadFile] = File(default=None),
    path: Optional[str] = Form(default=None),
    enable_tiles: bool = Form(default=True),
):
    """Accept video file upload or a path (within ALLOWED_INDEX_PATHS). Returns video_id and job_id."""
    if file is None and path is None:
        return error_response("file or path required")
    ensure_dir(settings.VIDEOS_DIR)
    video_id = uuid.uuid4().hex
    if file is not None:
        upload_ext = pathlib.Path(file.filename or "").suffix.lower()
        if upload_ext not in settings.VIDEO_EXTS:
            upload_ext = ".mp4"
        video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}{upload_ext}")
        total = 0
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.MAX_UPLOAD_BYTES:
                    if os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except OSError as e:
                            logger.warning("Could not remove partial upload path=%s err=%s", video_path, e)
                    return error_response(
                        f"Upload exceeds max size {settings.MAX_UPLOAD_BYTES} bytes",
                        status_code=413,
                    )
                f.write(chunk)
    else:
        if path is None:
            return error_response("path required")
        resolved = resolve_allowed_path(path, must_be_file=True)
        if resolved is None:
            return error_response("path not allowed or not a file", status_code=403)
        video_path = resolved

    job_id = _enqueue_job({"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles})
    logger.info("Enqueued video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@router.post("/index/url", summary="Index a video from URL", responses={400: ERROR_RESPONSES[400]})
async def index_url(
    url: str = Form(...),
    enable_tiles: bool = Form(default=True),
):
    try:
        validate_url(url)
    except ValueError as exc:
        return error_response(str(exc))
    video_id = uuid.uuid4().hex
    job_id = _enqueue_job({"video_id": video_id, "video_url": url, "enable_tiles": enable_tiles})
    logger.info("Enqueued url video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@router.post(
    "/index/dir",
    summary="Index all videos in an allowed directory",
    responses={403: ERROR_RESPONSES[403]},
)
async def index_dir(
    path: str = Form(...),
    enable_tiles: bool = Form(default=True),
):
    resolved = resolve_allowed_path(path, must_be_dir=True)
    if resolved is None:
        return error_response("path not allowed or not a directory", status_code=403)
    if not os.path.isdir(resolved):
        return error_response("path is not a directory")
    jobs = []
    file_count = 0
    total_bytes = 0
    for root, _, files in os.walk(resolved):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in settings.VIDEO_EXTS:
                continue
            video_path = os.path.join(root, name)
            try:
                total_bytes += os.path.getsize(video_path)
            except OSError:
                continue
            file_count += 1
            limit_error = _check_dir_limits(root, resolved, file_count, total_bytes)
            if limit_error:
                return error_response(limit_error)
            video_id = uuid.uuid4().hex
            job_id = _enqueue_job({"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles})
            jobs.append({"video_id": video_id, "job_id": job_id})
    logger.info("Enqueued directory jobs=%s path=%s", len(jobs), resolved)
    return {"jobs": jobs}


@router.post("/index/precheck")
async def precheck(
    file: Optional[UploadFile] = File(default=None),
    path: Optional[str] = Form(default=None),
    url: Optional[str] = Form(default=None),
):
    if not file and not path and not url:
        return error_response("file, path, or url required")

    if file is not None:
        try:
            file_hash, _ = await hash_upload_limited(file, settings.MAX_UPLOAD_BYTES)
        except ValueError as exc:
            return error_response(str(exc), status_code=413)
        existing = get_by_hash(file_hash)
        if existing:
            return {"status": "duplicate", "reason": "hash", "existing": existing}
        return {"status": "new", "reason": "hash", "hash": file_hash}

    if path is not None:
        resolved = resolve_allowed_path(path, must_be_file=True)
        if resolved is None:
            return error_response("path not allowed or not a file", status_code=403)
        if not os.path.exists(resolved) or not os.path.isfile(resolved):
            return error_response("path not found")
        file_hash = file_sha256(resolved)
        existing = get_by_hash(file_hash)
        if existing:
            return {"status": "duplicate", "reason": "hash", "existing": existing}
        return {"status": "new", "reason": "hash", "hash": file_hash}

    try:
        validate_url(url)
    except ValueError as exc:
        return error_response(str(exc))

    existing = get_by_url(url)
    if existing:
        return {"status": "duplicate", "reason": "url", "existing": existing}
    size = None
    try:
        with safe_request("HEAD", url, timeout=settings.PRECHECK_URL_TIMEOUT) as head:
            if head.ok and head.headers.get("Content-Length"):
                size = int(head.headers["Content-Length"])
    except Exception:
        size = None
    if size is not None:
        by_size = get_by_size(size)
        if by_size:
            return {"status": "maybe", "reason": "size_match", "existing": by_size, "size_bytes": size}
    return {"status": "unknown", "reason": "url_unmatched", "size_bytes": size}


@router.post("/index/precheck_dir")
async def precheck_dir(
    path: str = Form(...),
    enqueue: bool = Form(default=False),
    enable_tiles: bool = Form(default=True),
):
    resolved = resolve_allowed_path(path, must_be_dir=True)
    if resolved is None:
        return error_response("path not allowed or not a directory", status_code=403)
    if not os.path.isdir(resolved):
        return error_response("path is not a directory")
    results = []
    jobs = []
    file_count = 0
    total_bytes = 0
    for root, _, files in os.walk(resolved):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in settings.VIDEO_EXTS:
                continue
            video_path = os.path.join(root, name)
            try:
                total_bytes += os.path.getsize(video_path)
            except OSError:
                results.append({"filename": os.path.basename(video_path), "status": "error", "reason": "stat_failed"})
                continue
            file_count += 1
            limit_error = _check_dir_limits(root, resolved, file_count, total_bytes)
            if limit_error:
                return error_response(limit_error)
            try:
                file_hash = file_sha256(video_path)
            except OSError as e:
                logger.debug("Hash failed for path=%s err=%s", video_path, e)
                results.append({"filename": os.path.basename(video_path), "status": "error", "reason": "hash_failed"})
                continue
            existing = get_by_hash(file_hash)
            if existing:
                results.append({"filename": os.path.basename(video_path), "status": "duplicate", "reason": "hash", "existing": existing})
            else:
                entry = {"filename": os.path.basename(video_path), "status": "new", "reason": "hash", "hash": file_hash}
                if enqueue:
                    video_id = uuid.uuid4().hex
                    job_id = _enqueue_job({"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles})
                    entry["enqueued"] = True
                    entry["video_id"] = video_id
                    entry["job_id"] = job_id
                    jobs.append({"video_id": video_id, "job_id": job_id})
                results.append(entry)
    logger.info("Precheck dir path=%s results=%s enqueue=%s", resolved, len(results), enqueue)
    return {"results": results, "jobs": jobs if enqueue else []}
