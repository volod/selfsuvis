from typing import List, Optional

from PIL import Image
from qdrant_client.http import models as qmodels

from app.state import dino_model, store
from pipeline.config import settings


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


def search_vectors(
    vector_space: str,
    query_vec,
    search_type: str,
    top_k: int,
    enable_rerank: bool,
    image_query: Optional[Image.Image],
) -> List[dict]:
    k_retrieve = max(top_k, settings.K_RETRIEVE)
    filter_obj = payload_filter(search_type)

    scored = store.search(vector_space, query_vec, k_retrieve, filter_obj)
    results = format_results(scored)

    if enable_rerank and image_query is not None and vector_space == "clip" and dino_model:
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
