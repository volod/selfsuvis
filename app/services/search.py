import logging
import time
from typing import Any, List, Optional

import asyncpg
from PIL import Image
from qdrant_client.http import models as qmodels

from app.state import dino_model, store
from pipeline.config import settings

logger = logging.getLogger(__name__)

_REEMBED_STATUS_CACHE_TTL_SEC = 2.0
_reembed_status_cache = {"value": False, "checked_at": 0.0}


def payload_filter(search_type: str) -> Optional[qmodels.Filter]:
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


async def _reembed_is_active(db_pool: Optional[Any] = None) -> bool:
    """Return True if a reembed job is currently running.

    Uses a short TTL cache to avoid querying DB for every image query.
    Fails fast on DB errors (returns False) so search is never blocked.
    """
    now = time.monotonic()
    if now - _reembed_status_cache["checked_at"] < _REEMBED_STATUS_CACHE_TTL_SEC:
        return bool(_reembed_status_cache["value"])

    db_url = settings.DATABASE_URL
    if db_pool is None and not db_url:
        _reembed_status_cache["value"] = False
        _reembed_status_cache["checked_at"] = now
        return False
    try:
        if db_pool is not None:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT 1 FROM jobs WHERE type = 'reembed' AND status = 'running' LIMIT 1"
                )
                active = row is not None
        else:
            conn = await asyncpg.connect(db_url, timeout=3)
            try:
                row = await conn.fetchrow(
                    "SELECT 1 FROM jobs WHERE type = 'reembed' AND status = 'running' LIMIT 1"
                )
                active = row is not None
            finally:
                await conn.close()
        _reembed_status_cache["value"] = active
        _reembed_status_cache["checked_at"] = now
        return active
    except Exception:
        _reembed_status_cache["value"] = False
        _reembed_status_cache["checked_at"] = now
        return False


async def search_vectors(
    vector_space: str,
    query_vec,
    search_type: str,
    top_k: int,
    enable_rerank: bool,
    image_query: Optional[Image.Image],
    db_pool: Optional[Any] = None,
) -> List[dict]:
    k_retrieve = max(top_k, settings.K_RETRIEVE)
    filter_obj = payload_filter(search_type)

    scored = store.search(vector_space, query_vec, k_retrieve, filter_obj)
    results = format_results(scored)

    if enable_rerank and image_query is not None and vector_space == "clip" and dino_model:
        # Suppress dino reranking while a reembed sweep is active: Qdrant holds a
        # mix of old-model and new-model dino vectors during the sweep, so cosine
        # similarity between them is meaningless and would corrupt result ranking.
        if await _reembed_is_active(db_pool=db_pool):
            logger.info(
                "Dino reranking suppressed: reembed sweep is active (falling back to clip)"
            )
        else:
            dino_vec = dino_model.encode_images([image_query], batch_size=1)[0]
            dino_scored = store.search("dino", dino_vec, k_retrieve, filter_obj)
            dino_map = {p.id: p.score for p in dino_scored}
            for r in results:
                if r["id"] in dino_map:
                    r["score"] = 0.7 * r["score"] + 0.3 * dino_map[r["id"]]
            results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def format_results(scored: List[qmodels.ScoredPoint]) -> List[dict]:
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
