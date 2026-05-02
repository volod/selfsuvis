"""Pre-flight robot map cache export.

Exports all indexed frames from Qdrant into a single compressed NPZ file the
robot can carry onboard for local nearest-neighbour search without a network
round-trip.

NPZ layout:
    clip_vectors : float32 (N, D)   — CLIP embedding per frame
    gps          : float32 (N, 3)   — [lat, lon, alt],  NaN if missing
    enu          : float32 (N, 3)   — [tx, ty, tz] (m), NaN if missing
    t_sec        : float32 (N,)     — timestamp within source video
    meta_json    : uint8  (M,)      — JSON bytes; list of N dicts:
                                       {mission_id, frame_path, robot_id}

Usage (on robot):
    import numpy as np, json
    cache = np.load("map_cache.npz", allow_pickle=False)
    vecs   = cache["clip_vectors"]          # (N, D)
    gps    = cache["gps"]                   # (N, 3)
    enu    = cache["enu"]                   # (N, 3)
    meta   = json.loads(bytes(cache["meta_json"]).decode())  # list of N dicts
"""
import json
import math
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

# Maximum number of frames to scroll per Qdrant page
_PAGE_SIZE = 1000


def _append_bbox_filters(
    must: List[Any],
    qmodels: Any,
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
) -> None:
    if lat_min is not None and lat_max is not None:
        must.append(
            qmodels.FieldCondition(key="gps.lat", range=qmodels.Range(gte=lat_min, lte=lat_max))
        )
    if lon_min is not None and lon_max is not None:
        must.append(
            qmodels.FieldCondition(key="gps.lon", range=qmodels.Range(gte=lon_min, lte=lon_max))
        )


def _point_clip_vector(point: Any) -> Optional[List[float]]:
    vector = point.vector
    if isinstance(vector, dict):
        clip = vector.get("clip")
        return clip if clip is not None else None
    return vector if isinstance(vector, list) else None


def _payload_xyz(payload: Dict[str, Any], key: str, fields: List[str]) -> List[float]:
    values = payload.get(key) or {}
    return [values.get(field, math.nan) for field in fields]


def _point_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mission_id": payload.get("mission_id"),
        "frame_path": payload.get("frame_path"),
        "robot_id": payload.get("robot_id"),
    }


def build_map_cache(
    qdrant_store,
    mission_ids: Optional[List[str]] = None,
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lon_min: Optional[float] = None,
    lon_max: Optional[float] = None,
) -> bytes:
    """Build a compressed NPZ cache of all indexed frames.

    Args:
        qdrant_store:  QdrantStore instance (from app.state or passed directly).
        mission_ids:   If set, include only frames whose mission_id matches.
        lat_min/max, lon_min/max: Optional GPS bounding box filter.

    Returns:
        NPZ file as raw bytes (ready to stream as HTTP response or write to disk).

    Raises:
        RuntimeError: if Qdrant is unreachable.
    """
    from qdrant_client.http import models as qmodels  # type: ignore

    must = [
        qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="frame")),
    ]

    if mission_ids:
        must.append(
            qmodels.FieldCondition(
                key="mission_id",
                match=qmodels.MatchAny(any=mission_ids),
            )
        )

    _append_bbox_filters(must, qmodels, lat_min, lat_max, lon_min, lon_max)

    scroll_filter = qmodels.Filter(must=must)

    clip_vecs: List[List[float]] = []
    gps_rows: List[List[float]] = []
    enu_rows: List[List[float]] = []
    t_secs: List[float] = []
    metas: List[Dict[str, Any]] = []

    offset = None
    page = 0
    while True:
        kwargs: Dict[str, Any] = dict(
            collection_name=qdrant_store.collection_name,
            scroll_filter=scroll_filter,
            limit=_PAGE_SIZE,
            with_payload=True,
            with_vectors=True,
        )
        if offset is not None:
            kwargs["offset"] = offset

        results, next_offset = qdrant_store.client.scroll(**kwargs)
        page += 1
        logger.debug("Map cache scroll page=%d points=%d", page, len(results))

        for pt in results:
            payload = pt.payload or {}
            clip = _point_clip_vector(pt)
            if clip is None:
                continue

            clip_vecs.append(clip)
            gps_rows.append(_payload_xyz(payload, "gps", ["lat", "lon", "alt"]))
            enu_rows.append(_payload_xyz(payload, "enu", ["tx", "ty", "tz"]))

            t_secs.append(float(payload.get("t_sec", 0.0)))
            metas.append(_point_meta(payload))

        if next_offset is None or not results:
            break
        offset = next_offset

    n = len(clip_vecs)
    logger.info("Map cache: collected %d frame vectors", n)

    if n == 0:
        clip_arr = np.empty((0, 0), dtype=np.float32)
        gps_arr = np.empty((0, 3), dtype=np.float32)
        enu_arr = np.empty((0, 3), dtype=np.float32)
        t_arr = np.empty(0, dtype=np.float32)
    else:
        clip_arr = np.array(clip_vecs, dtype=np.float32)
        gps_arr = np.array(gps_rows, dtype=np.float32)
        enu_arr = np.array(enu_rows, dtype=np.float32)
        t_arr = np.array(t_secs, dtype=np.float32)

    meta_bytes = json.dumps(metas, separators=(",", ":")).encode("utf-8")
    meta_arr = np.frombuffer(meta_bytes, dtype=np.uint8)

    buf = BytesIO()
    np.savez_compressed(
        buf,
        clip_vectors=clip_arr,
        gps=gps_arr,
        enu=enu_arr,
        t_sec=t_arr,
        meta_json=meta_arr,
    )
    return buf.getvalue()
