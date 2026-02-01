import os
import uuid
import io
import requests
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from qdrant_client.http import models as qmodels
from PIL import Image

from pipeline.config import settings
from pipeline.job_db import init_db, create_job, fetch_job
from pipeline.processed_db import init_db as init_processed_db, get_by_hash, get_by_url, get_by_size
from pipeline.utils import file_sha256
from models.openclip_model import OpenCLIPEmbedder
from models.dino_model import DINOEmbedder
from pipeline.qdrant_utils import QdrantStore
from pipeline.utils import ensure_dir, file_sha256_bytes
from pipeline.logging_utils import get_logger

app = FastAPI()
logger = get_logger(__name__)

init_db()
init_processed_db()

clip_model = OpenCLIPEmbedder()
try:
    dino_model = DINOEmbedder("dinov2_vitb14") if settings.MODEL_NAME in {"dinov2", "dinov3"} else None
except Exception:
    dino_model = None

store = QdrantStore(clip_dim=clip_model.image_dim(), dino_dim=dino_model.image_dim() if dino_model else None)


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def _enqueue_job(payload: dict):
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


@app.post("/index/video")
async def index_video(
    file: Optional[UploadFile] = File(default=None),
    path: Optional[str] = Form(default=None),
    enable_tiles: bool = Form(default=True),
):
    if file is None and path is None:
        return JSONResponse({"error": "file or path required"}, status_code=400)
    ensure_dir(settings.VIDEOS_DIR)
    video_id = uuid.uuid4().hex
    if file is not None:
        video_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
    else:
        video_path = path

    job_id = _enqueue_job({"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles})
    logger.info("Enqueued video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@app.post("/index/url")
async def index_url(
    url: str = Form(...),
    enable_tiles: bool = Form(default=True),
):
    video_id = uuid.uuid4().hex
    job_id = _enqueue_job({"video_id": video_id, "video_url": url, "enable_tiles": enable_tiles})
    logger.info("Enqueued url video_id=%s job_id=%s", video_id, job_id)
    return {"video_id": video_id, "job_id": job_id}


@app.post("/index/dir")
async def index_dir(
    path: str = Form(...),
    enable_tiles: bool = Form(default=True),
):
    if not os.path.isdir(path):
        return JSONResponse({"error": "path is not a directory"}, status_code=400)
    jobs = []
    for root, _, files in os.walk(path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            video_path = os.path.join(root, name)
            video_id = uuid.uuid4().hex
            job_id = _enqueue_job({"video_id": video_id, "video_path": video_path, "enable_tiles": enable_tiles})
            jobs.append({"video_id": video_id, "job_id": job_id, "video_path": video_path})
    logger.info("Enqueued directory jobs=%s path=%s", len(jobs), path)
    return {"jobs": jobs}


@app.post("/index/precheck")
async def precheck(
    file: Optional[UploadFile] = File(default=None),
    path: Optional[str] = Form(default=None),
    url: Optional[str] = Form(default=None),
):
    if not file and not path and not url:
        return JSONResponse({"error": "file, path, or url required"}, status_code=400)

    if file is not None:
        content = await file.read()
        file_hash = file_sha256_bytes(content)
        existing = get_by_hash(file_hash)
        if existing:
            return {"status": "duplicate", "reason": "hash", "existing": existing}
        return {"status": "new", "reason": "hash", "hash": file_hash}

    if path is not None:
        if not os.path.exists(path) or not os.path.isfile(path):
            return JSONResponse({"error": "path not found"}, status_code=400)
        file_hash = file_sha256(path)
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
        head = requests.head(url, timeout=20, allow_redirects=True)
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
    if not os.path.isdir(path):
        return JSONResponse({"error": "path is not a directory"}, status_code=400)
    results = []
    jobs = []
    for root, _, files in os.walk(path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            video_path = os.path.join(root, name)
            try:
                file_hash = file_sha256(video_path)
            except Exception:
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
    logger.info("Precheck dir path=%s results=%s enqueue=%s", path, len(results), enqueue)
    return {"results": results, "jobs": jobs if enqueue else []}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = fetch_job(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return {
        "status": job["status"],
        "progress": job["progress"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
    }


@app.post("/query/image")
async def query_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=20),
    search_type: str = Form(default="both"),
    vector_space: str = Form(default="clip"),
    enable_rerank: bool = Form(default=True),
):
    content = await file.read()
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
    return {"results": results}


@app.post("/query/text")
async def query_text(
    payload: dict,
    top_k: int = 20,
    search_type: str = "both",
    enable_rerank: bool = True,
):
    text = payload.get("text") if payload else None
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    clip_vec = clip_model.encode_texts([text], batch_size=1)[0]
    results = _search_vectors(
        vector_space="clip",
        query_vec=clip_vec,
        search_type=search_type,
        top_k=top_k,
        enable_rerank=enable_rerank,
        image_query=None,
    )
    return {"results": results}


def _search_vectors(
    vector_space: str,
    query_vec,
    search_type: str,
    top_k: int,
    enable_rerank: bool,
    image_query: Optional[Image.Image],
):
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


def _format_results(scored: List[qmodels.ScoredPoint]):
    results = []
    for p in scored:
        payload = p.payload or {}
        result = {
            "id": p.id,
            "score": p.score,
            "type": payload.get("type"),
            "video_id": payload.get("video_id"),
            "segment_id": payload.get("segment_id"),
            "t_sec": payload.get("t_sec"),
            "thumbnail_path": payload.get("tile_path") or payload.get("frame_path"),
            "frame_path": payload.get("frame_path"),
            "tile_path": payload.get("tile_path"),
            "bbox": None,
        }
        if payload.get("type") == "tile":
            result["bbox"] = {
                "x": payload.get("x"),
                "y": payload.get("y"),
                "w": payload.get("w"),
                "h": payload.get("h"),
            }
        results.append(result)
    return results
