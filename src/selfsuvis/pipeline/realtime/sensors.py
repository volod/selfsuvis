"""Shared sensor registry helpers for realtime ingest."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

_SUPPORTED_SENSOR_TYPES: Set[str] = {
    "barometer",
    "camera",
    "gps",
    "imu",
    "lidar",
    "magnetometer",
}

_SENSOR_CAPABILITIES: Dict[str, List[str]] = {
    "barometer": ["altitude"],
    "camera": ["imagery"],
    "gps": ["position", "velocity", "global_reference"],
    "imu": ["orientation", "acceleration", "angular_velocity", "velocity"],
    "lidar": ["depth", "geometry"],
    "magnetometer": ["heading", "orientation_hint"],
}


def normalize_sensor_type(sensor_type: Any) -> str:
    return str(sensor_type or "").strip().lower()


def require_supported_sensor_type(sensor_type: Any) -> str:
    normalized = normalize_sensor_type(sensor_type)
    if normalized not in _SUPPORTED_SENSOR_TYPES:
        raise ValueError(f"unsupported sensor_type: {normalized or '<empty>'}")
    return normalized


def supported_sensor_types() -> Set[str]:
    return set(_SUPPORTED_SENSOR_TYPES)


def packet_sensor_summary(sensor_types: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sensor_type in sensor_types:
        key = normalize_sensor_type(sensor_type)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_sensor_profile(sensors: Iterable[Any]) -> Dict[str, Any]:
    counts = packet_sensor_summary(sensor for sensor in sensors if normalize_sensor_type(sensor))
    deduped = sorted(counts)
    return {
        "sensors": deduped,
        "sensor_count": len(deduped),
        "capabilities": {sensor: list(_SENSOR_CAPABILITIES.get(sensor, [])) for sensor in deduped},
    }
