"""Change detection across GPS-overlapping missions.

For each frame in a new mission, query Qdrant for frames from earlier missions
within a GPS bounding box. If the closest matching frame's cosine distance
exceeds the threshold, a change event is recorded.

The query_fn abstraction lets unit tests inject a mock without a live Qdrant.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import settings
from pipeline.logging_utils import get_logger

logger = get_logger(__name__)

# Approximate metres per degree of latitude at the equator
_M_PER_DEG_LAT = 111_320.0

QueryFn = Callable[[np.ndarray, Tuple[float, float, float, float]], List[Dict[str, Any]]]


def latlon_bbox(
    lat: float, lon: float, radius_m: float
) -> Tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) bounding box for a GPS circle.

    Uses a flat-earth approximation valid for radius_m << Earth radius.
    """
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / (_M_PER_DEG_LAT * (abs(np.cos(np.radians(lat))) + 1e-9))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 − cosine_similarity) between two vectors.

    Returns 1.0 if either vector is a zero vector.
    """
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-9 or b_norm < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))


def threshold_for_model() -> float:
    """Return the change-detection threshold appropriate for the configured model."""
    if settings.MODEL_NAME in {"dinov2", "dinov3"}:
        return settings.CHANGE_DETECTION_THRESHOLD_DINO
    return settings.CHANGE_DETECTION_THRESHOLD_CLIP


def detect_changes(
    new_frames: List[Dict[str, Any]],
    query_fn: QueryFn,
    threshold: Optional[float] = None,
    radius_m: float = 50.0,
) -> List[Dict[str, Any]]:
    """Detect visual changes between new_frames and historically indexed frames.

    Args:
        new_frames: Frames from the current mission. Each dict must have:
            frame_id (str), mission_id (str), embedding (list[float]),
            gps (dict with lat/lon keys, or None).
        query_fn: Callable(embedding, bbox) → list of candidate dicts.
            Each candidate must have: frame_id, mission_id, embedding.
        threshold: Cosine-distance threshold; defaults to model-appropriate value.
        radius_m: GPS search radius in metres (default 50 m).

    Returns:
        List of change-event dicts:
            frame_id, mission_id, ref_frame_id, ref_mission_id,
            change_score (cosine dist), threshold.
    """
    if threshold is None:
        threshold = threshold_for_model()

    changes: List[Dict[str, Any]] = []
    for frame in new_frames:
        gps = frame.get("gps")
        if not gps or gps.get("lat") is None or gps.get("lon") is None:
            continue

        bbox = latlon_bbox(gps["lat"], gps["lon"], radius_m)
        embedding = np.asarray(frame["embedding"], dtype=np.float32)
        candidates = query_fn(embedding, bbox)

        best_dist: Optional[float] = None
        best_ref: Optional[Dict[str, Any]] = None
        for cand in candidates:
            if cand.get("mission_id") == frame["mission_id"]:
                continue  # skip same-mission frames
            ref_emb = np.asarray(cand["embedding"], dtype=np.float32)
            dist = cosine_distance(embedding, ref_emb)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_ref = cand

        if best_ref is not None and best_dist is not None and best_dist >= threshold:
            changes.append(
                {
                    "frame_id": frame["frame_id"],
                    "mission_id": frame["mission_id"],
                    "ref_frame_id": best_ref["frame_id"],
                    "ref_mission_id": best_ref["mission_id"],
                    "change_score": best_dist,
                    "threshold": threshold,
                }
            )
            logger.debug(
                "Change detected: frame=%s ref=%s dist=%.3f",
                frame["frame_id"],
                best_ref["frame_id"],
                best_dist,
            )

    return changes
