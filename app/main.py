import io
import os
import uuid
from typing import List, Optional

import requests
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from qdrant_client.http import models as qmodels

from app.schemas import ErrorResponse, QueryResponse, TextQuery
from models.dino_model import DINOEmbedder
from models.openclip_model import OpenCLIPEmbedder
from pipeline.config import settings, validate_settings
from pipeline.job_db import create_job, fetch_job, init_db
from pipeline.logging_utils import get_logger
from pipeline.processed_db import get_by_hash, get_by_size, get_by_url, init_db as init_processed_db
from pipeline.qdrant_utils import QdrantStore
from pipeline.utils import ensure_dir, file_sha256, file_sha256_bytes, resolve_allowed_path

app = FastAPI()
logger = get_logger(__name__)

init_db()
init_processed_db()
validate_settings()

clip_model = OpenCLIPEmbedder()
dino_model = None
if settings.MODEL_NAME in {"dinov2", "dinov3"}:
    try:
        dino_model = DINOEmbedder("dinov2_vitb14")
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("DINO model failed to load, disabling: %s", exc)
        dino_model = None

store = QdrantStore(
    clip_dim=clip_model.image_dim(),
    dino_dim=dino_model.image_dim() if dino_model else None,
)


def _error_response(message: str, status_code: int = 400, detail: Optional[str] = None) -> JSONResponse:
    """Return a consistent error response."""
    body = {"error": message}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=status_code)


def _enqueue_job(payload: dict) -> str:
    job_id = uuid.uuid4().hex
    create_job(job_id, payload)
    return job_id


def _payload_filter(search_type: str) -> Optional[qmodels.Filter]:
    if search_type == "both":
        return None
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="type",
                match=qmodels.MatchValue(value=search_type),
            )
        ]
    )


_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Bad request"},
    403: {"model": ErrorResponse, "description": "Forbidden"},
    404: {"model": ErrorResponse, "description": "Not found"},
    413: {"model": ErrorResponse, "description": "Payload too large"},
    503: {"model": ErrorResponse, "description": "Service unavailable"},
}


@app.get("/health", responses={503: {"model": ErrorResponse, "description": "Qdrant unreachable"}})
def health():
    """Health check for container orchestration. Verifies Qdrant connectivity."""
    try:
        store.client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return JSONResponse(
            {"error": str(exc)},
            status_code=503,
        )


@app.post(
    "/index/video",
    summary="Index a video from upload or allowed path",
    responses={400: _ERROR_RESPONSES[400], 403: _ERROR_RESPONSES[403], 413: _ERROR_RESPONSES[413]},
)
async def index_video(
    file: Optional[UploadFile] = File(default=None),
    path: Optional[str] = Form(default=None),
    enable_tiles: bool = Form(default=True),
):
    """Accept video file upload or a path (within ALLOWED_INDEX_PATHS). Returns video_id and job_id."""
    if file is None and path is None:
        return _error_response("file or path required")
    ensure_dir(settings.VIDEOS_DIR)
    video_id = uuid.uuid4().hex
    if file is not None:
        video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
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
                    return _error_response(
                        f"Upload exceeds max size {settings.MAX_UPLOAD_BYTES} bytes",
                        status_code=413,
                    )
                f.write(chunk)
    else:
        if path is None:
            return _error_response("path required")
        resolved = resolve_allowed_path(path, must_be_file=True)
        if resolved is None:
            return _error_response("path not allowed or not a file", status_code=403)
        video_path = resolved

    job_id = _enqueue_job({"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles})
    logger.info("Enqueued video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@app.post("/index/url", summary="Index a video from URL", responses={400: _ERROR_RESPONSES[400]})
async def index_url(
    url: str = Form(...),
    enable_tiles: bool = Form(default=True),
):
    video_id = uuid.uuid4().hex
    job_id = _enqueue_job({"video_id": video_id, "video_url": url, "enable_tiles": enable_tiles})
    logger.info("Enqueued url video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@app.post(
    "/index/dir",
    summary="Index all videos in an allowed directory",
    responses={403: _ERROR_RESPONSES[403]},
)
async def index_dir(
    path: str = Form(...),
    enable_tiles: bool = Form(default=True),
):
    resolved = resolve_allowed_path(path, must_be_dir=True)
    if resolved is None:
        return _error_response("path not allowed or not a directory", status_code=403)
    if not os.path.isdir(resolved):
        return _error_response("path is not a directory")
    jobs = []
    for root, _, files in os.walk(resolved):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in settings.VIDEO_EXTS:
                continue
            video_path = os.path.join(root, name)
            video_id = uuid.uuid4().hex
            job_id = _enqueue_job(
                {"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles}
            )
            jobs.append({"video_id": video_id, "job_id": job_id, "video_path": video_path})
    logger.info("Enqueued directory jobs=%s path=%s", len(jobs), resolved)
    return {"jobs": jobs}


@app.post("/index/precheck")
async def precheck(
    file: Optional[UploadFile] = File(default=None),
    path: Optional[str] = Form(default=None),
    url: Optional[str] = Form(default=None),
):
    if not file and not path and not url:
        return _error_response("file, path, or url required")

    if file is not None:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_BYTES:
            return _error_response(
                f"File exceeds max size {settings.MAX_UPLOAD_BYTES} bytes",
                status_code=413,
            )
        file_hash = file_sha256_bytes(content)
        existing = get_by_hash(file_hash)
        if existing:
            return {"status": "duplicate", "reason": "hash", "existing": existing}
        return {"status": "new", "reason": "hash", "hash": file_hash}

    if path is not None:
        resolved = resolve_allowed_path(path, must_be_file=True)
        if resolved is None:
            return _error_response("path not allowed or not a file", status_code=403)
        if not os.path.exists(resolved) or not os.path.isfile(resolved):
            return _error_response("path not found")
        file_hash = file_sha256(resolved)
        existing = get_by_hash(file_hash)
        if existing:
            return {"status": "duplicate", "reason": "hash", "existing": existing}
        return {"status": "new", "reason": "hash", "hash": file_hash}

    # URL heuristic: try exact URL match, then content-length match (weak)
    existing = get_by_url(url)
    if existing:
        return {"status": "duplicate", "reason": "url", "existing": existing}
    size = None
    try:
        head = requests.head(url, timeout=settings.PRECHECK_URL_TIMEOUT, allow_redirects=True)
        if head.ok and head.headers.get("Content-Length"):
            size = int(head.headers["Content-Length"])
    except Exception:
        size = None
    if size is not None:
        by_size = get_by_size(size)
        if by_size:
            return {"status": "maybe", "reason": "size_match", "existing": by_size, "size_bytes": size}
    return {"status": "unknown", "reason": "url_unmatched", "size_bytes": size}


@app.post("/index/precheck_dir")
async def precheck_dir(
    path: str = Form(...),
    enqueue: bool = Form(default=False),
    enable_tiles: bool = Form(default=True),
):
    resolved = resolve_allowed_path(path, must_be_dir=True)
    if resolved is None:
        return _error_response("path not allowed or not a directory", status_code=403)
    if not os.path.isdir(resolved):
        return _error_response("path is not a directory")
    results = []
    jobs = []
    for root, _, files in os.walk(resolved):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in settings.VIDEO_EXTS:
                continue
            video_path = os.path.join(root, name)
            try:
                file_hash = file_sha256(video_path)
            except OSError as e:
                logger.debug("Hash failed for path=%s err=%s", video_path, e)
                results.append({"path": video_path, "status": "error", "reason": "hash_failed"})
                continue
            existing = get_by_hash(file_hash)
            if existing:
                results.append(
                    {
                        "path": video_path,
                        "status": "duplicate",
                        "reason": "hash",
                        "existing": existing,
                    }
                )
            else:
                entry = {
                    "path": video_path,
                    "status": "new",
                    "reason": "hash",
                    "hash": file_hash,
                }
                if enqueue:
                    video_id = uuid.uuid4().hex
                    job_id = _enqueue_job(
                        {"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles}
                    )
                    entry["enqueued"] = True
                    entry["video_id"] = video_id
                    entry["job_id"] = job_id
                    jobs.append({"video_id": video_id, "job_id": job_id, "video_path": video_path})
                results.append(entry)
    logger.info("Precheck dir path=%s results=%s enqueue=%s", resolved, len(results), enqueue)
    return {"results": results, "jobs": jobs if enqueue else []}


@app.get(
    "/jobs/{job_id}",
    summary="Get job status by id",
    responses={404: _ERROR_RESPONSES[404]},
)
def job_status(job_id: str):
    """Return status, progress, started_at, finished_at, and error for a job."""
    job = fetch_job(job_id)
    if not job:
        return _error_response("job not found", status_code=404)
    return {
        "status": job["status"],
        "progress": job["progress"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
    }


@app.post(
    "/query/image",
    response_model=QueryResponse,
    summary="Search by image",
    responses={400: _ERROR_RESPONSES[400], 413: _ERROR_RESPONSES[413]},
)
async def query_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=20, ge=1, le=100),
    search_type: str = Form(default="both"),
    vector_space: str = Form(default="clip"),
    enable_rerank: bool = Form(default=True),
):
    if search_type not in ("both", "frame", "tile"):
        return _error_response("search_type must be both, frame, or tile")
    if vector_space not in ("clip", "dino"):
        return _error_response("vector_space must be clip or dino")
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_BYTES:
        return _error_response(
            f"Image exceeds max size {settings.MAX_UPLOAD_BYTES} bytes",
            status_code=413,
        )
    image = Image.open(io.BytesIO(content))
    image = image.convert("RGB")
    clip_vec = clip_model.encode_images([image], batch_size=1)[0]
    if vector_space == "dino" and dino_model:
        query_vec = dino_model.encode_images([image], batch_size=1)[0]
        vs = "dino"
    else:
        query_vec = clip_vec
        vs = "clip"

    results = _search_vectors(
        vector_space=vs,
        query_vec=query_vec,
        search_type=search_type,
        top_k=top_k,
        enable_rerank=enable_rerank,
        image_query=image,
    )
    return QueryResponse(results=results)


@app.post(
    "/query/text",
    response_model=QueryResponse,
    summary="Search by text",
    responses={400: _ERROR_RESPONSES[400]},
)
async def query_text(
    payload: TextQuery,
    top_k: int = Query(default=20, ge=1, le=100),
    search_type: str = Query(default="both"),
    enable_rerank: bool = Query(default=True),
):
    if search_type not in ("both", "frame", "tile"):
        return _error_response("search_type must be both, frame, or tile")
    text = payload.text
    clip_vec = clip_model.encode_texts([text], batch_size=1)[0]
    results = _search_vectors(
        vector_space="clip",
        query_vec=clip_vec,
        search_type=search_type,
        top_k=top_k,
        enable_rerank=enable_rerank,
        image_query=None,
    )
    return QueryResponse(results=results)


def _search_vectors(
    vector_space: str,
    query_vec,
    search_type: str,
    top_k: int,
    enable_rerank: bool,
    image_query: Optional[Image.Image],
) -> List[dict]:
    k_retrieve = max(top_k, settings.K_RETRIEVE)
    filter_obj = _payload_filter(search_type)

    scored = store.search(vector_space, query_vec, k_retrieve, filter_obj)
    results = _format_results(scored)

    if enable_rerank and image_query is not None and vector_space == "clip" and dino_model:
        dino_vec = dino_model.encode_images([image_query], batch_size=1)[0]
        dino_scored = store.search("dino", dino_vec, k_retrieve, filter_obj)
        dino_map = {p.id: p.score for p in dino_scored}
        for r in results:
            if r["id"] in dino_map:
                r["score"] = 0.7 * r["score"] + 0.3 * dino_map[r["id"]]
        results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def _format_results(scored: List[qmodels.ScoredPoint]) -> List[dict]:
    results = []
    for p in scored:
        payload = p.payload or {}
        result_type = payload.get("type") or "frame"
        result = {
            "id": p.id,
            "score": float(p.score),
            "type": result_type,
            "video_id": payload.get("video_id") or "",
            "segment_id": payload.get("segment_id") or 0,
            "t_sec": payload.get("t_sec") or 0.0,
            "thumbnail_path": payload.get("tile_path") or payload.get("frame_path") or "",
            "frame_path": payload.get("frame_path"),
            "tile_path": payload.get("tile_path"),
            "bbox": None,
        }
        if result_type == "tile":
            result["bbox"] = {
                "x": payload.get("x"),
                "y": payload.get("y"),
                "w": payload.get("w"),
                "h": payload.get("h"),
            }
        results.append(result)
    return results
