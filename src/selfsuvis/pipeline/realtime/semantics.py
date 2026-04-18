"""Realtime semantic-observation helpers."""


from typing import Any, Dict

def normalize_semantic_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    class_name = str(observation.get("class_name", "")).strip().lower()
    if not class_name:
        raise ValueError("class_name is required")
    confidence = float(observation.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be within [0, 1]")
    return {
        "frame_id": observation.get("frame_id"),
        "class_name": class_name,
        "confidence": confidence,
        "position_enu": dict(observation.get("position_enu") or {}) or None,
        "bbox": dict(observation.get("bbox") or {}) or None,
        "mask_ref": observation.get("mask_ref"),
        "track_id": observation.get("track_id"),
        "facts": dict(observation.get("facts") or {}),
    }
