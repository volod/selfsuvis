"""Embedding and search steps for the local full-analysis pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline.media import extract_frames
from models.openclip_model import OpenCLIPEmbedder
from ._common import _log

# _log_vram_snapshot lives in steps_caption; import lazily inside functions to
# avoid circular imports at module level.

from datetime import datetime


def step_extract_frames(
    video_path: Path,
    video_id: str,
    video_dir: Path,
    fps: float,
) -> Dict[str, Any]:
    """Step A: extract frames via ffmpeg, write metadata JSON."""
    import json
    _log.info("Extracting frames from %s at %.1f fps …", video_path.name, fps)
    t0 = time.time()
    frame_list = extract_frames(str(video_path), video_id)
    elapsed = time.time() - t0
    from .steps_caption import _log_vram_snapshot as _lvs
    _lvs("after Gemma analysis step")
    meta = {
        "video": str(video_path),
        "video_id": video_id,
        "fps": fps,
        "frame_count": len(frame_list),
        "duration_sec": frame_list[-1][1] if frame_list else 0.0,
        "frames": [{"path": p, "t_sec": t} for p, t in frame_list],
        "extracted_at": datetime.now().isoformat(),
    }
    meta_path = video_dir / "frames_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log.info("  ✓ %d frames extracted in %.1fs → %s", len(frame_list), elapsed, meta_path)
    return {"frame_list": frame_list, "elapsed_sec": elapsed, "meta": meta}


def _embed_and_flush(
    batch_pil: List[Image.Image],
    batch_meta: List[Tuple[str, float]],
    video_id: str,
    clip_model: OpenCLIPEmbedder,
    dino_model: Any,
    store: Any,
    is_qdrant: bool,
) -> int:
    if not batch_pil:
        return 0
    clip_embeds = clip_model.encode_images(batch_pil)
    dino_embeds = dino_model.encode_images(batch_pil) if dino_model else None
    if is_qdrant:
        from qdrant_client.http import models as qmodels
        from pipeline.core.utils import stable_point_id
        points = []
        for i, (fp, t_sec) in enumerate(batch_meta):
            vectors: Dict[str, Any] = {"clip": clip_embeds[i].tolist()}
            if dino_embeds is not None:
                vectors["dino"] = dino_embeds[i].tolist()
            points.append(qmodels.PointStruct(
                id=stable_point_id(video_id, fp),
                vector=vectors,
                payload={"frame_path": fp, "t_sec": t_sec, "video_id": video_id},
            ))
        store.upsert_points(points)
    else:
        for i, (fp, t_sec) in enumerate(batch_meta):
            vec = dino_embeds[i] if dino_embeds is not None else clip_embeds[i]
            store.add(vec, {"frame_path": fp, "t_sec": t_sec, "video_id": video_id})
    return len(batch_pil)


def step_index_to_store(
    video_path: Path,
    video_id: str,
    store: Any,
    is_qdrant: bool,
    models: Dict[str, Any],
    frame_list: List[Tuple[str, float]],
) -> Dict[str, Any]:
    """Step B: embed frames and upsert into Qdrant or InMemoryStore."""
    t0   = time.time()
    dest = "Qdrant" if is_qdrant else "in-memory store"
    _log.info("Embedding %d frames into %s …", len(frame_list), dest)
    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")
    batch_pil: List[Image.Image]         = []
    batch_meta: List[Tuple[str, float]]  = []
    indexed = 0
    for fp, t_sec in frame_list:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        batch_pil.append(img); batch_meta.append((fp, t_sec))
        if len(batch_pil) >= 32:
            indexed += _embed_and_flush(batch_pil, batch_meta, video_id,
                                        clip_model, dino_model, store, is_qdrant)
            batch_pil, batch_meta = [], []
    indexed += _embed_and_flush(batch_pil, batch_meta, video_id,
                                clip_model, dino_model, store, is_qdrant)
    elapsed = time.time() - t0
    _log.info("  ✓ %d frames indexed into %s in %.1fs", indexed, dest, elapsed)
    return {"indexed": indexed, "elapsed_sec": elapsed}


def _pick_query_frame(frame_list: List[Tuple[str, float]]) -> Tuple[str, float]:
    return frame_list[len(frame_list) // 2]


def _embed_query(frame_path: str, models: Dict[str, Any], use_dino: bool = True) -> np.ndarray:
    img = Image.open(frame_path).convert("RGB")
    if use_dino and models.get("dino"):
        return models["dino"].encode_images([img])[0]
    return models["clip"].encode_images([img])[0]


def _search(
    query_vec: np.ndarray,
    store: Any,
    is_qdrant: bool,
    top_k: int,
    video_id: str,
    vector_name: str = "clip",
) -> List[Dict[str, Any]]:
    if is_qdrant:
        from qdrant_client.http import models as qmodels
        filt = qmodels.Filter(must=[qmodels.FieldCondition(
            key="video_id", match=qmodels.MatchValue(value=video_id),
        )])
        raw = store.search(vector_name, query_vec, limit=top_k, payload_filter=filt)
        return [{"score": p.score, "payload": p.payload} for p in raw]
    return store.search(query_vec, limit=top_k)


def step_base_model_search_test(
    frame_list: List[Tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: Dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Step C: embed query with base model, search, write base_search.md."""
    from .steps_report import write_search_md
    out_md = video_dir / "base_search.md"
    qfp, qt = _pick_query_frame(frame_list)
    _log.info("Query frame: %s (t=%.2fs)", Path(qfp).name, qt)
    use_dino = models.get("dino") is not None
    t0       = time.time()
    query_vec = _embed_query(qfp, models, use_dino=use_dino)
    results   = _search(query_vec, store, is_qdrant, top_k, video_id,
                        vector_name="dino" if use_dino else "clip")
    elapsed   = time.time() - t0
    label = "Base DINOv3 (pretrained)" if use_dino else "Base CLIP (pretrained)"
    write_search_md(out_md, video_name, label, qfp, results, qt)
    _log.info("  ✓ Search in %.2fs → top score %.4f", elapsed,
              results[0]["score"] if results else 0)
    return {"results": results, "query_frame": qfp, "query_t_sec": qt}


def step_finetuned_model_search_test(
    frame_list: List[Tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: Dict[str, Any],
    query_frame: str,
    query_t_sec: float,
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Step G: search with fine-tuned DINO, write finetuned_search.md."""
    from .steps_report import write_search_md
    out_md   = video_dir / "finetuned_search.md"
    use_dino = models.get("dino") is not None
    t0       = time.time()
    query_vec = _embed_query(query_frame, models, use_dino=use_dino)
    results   = _search(query_vec, store, is_qdrant, top_k, video_id,
                        vector_name="dino" if use_dino else "clip")
    ft_infer_ms = (time.time() - t0) * 1000 / max(len(frame_list), 1)
    write_search_md(out_md, video_name, "Fine-tuned DINOv3 (SSL adapted)",
                    query_frame, results, query_t_sec)
    return {"results": results, "infer_ms": ft_infer_ms}
