"""Shared access helpers for normalized realtime event envelopes."""

from typing import Any, Dict

from .sensors import normalize_sensor_type


def event_kind(event: Dict[str, Any]) -> str:
    return str(event.get("event_kind", "")).strip().lower()


def event_sensor_type(event: Dict[str, Any]) -> str:
    return normalize_sensor_type(event.get("sensor_type", ""))


def event_node_id(event: Dict[str, Any], *, default: str = "unknown") -> str:
    text = str(event.get("node_id", default) or "").strip()
    return text or default


def event_sector_id(event: Dict[str, Any], *, default: str = "unknown") -> str:
    text = str(event.get("sector_id", default) or "").strip()
    return text or default


def event_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    return dict(event.get("payload") or {})


def event_freshness_sec(event: Dict[str, Any]) -> float:
    return float(event.get("freshness_sec", 0.0) or 0.0)


def payload_float(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(payload.get(key, default) or default)


def payload_text(payload: Dict[str, Any], key: str, default: str = "") -> str:
    return str(payload.get(key, default) or default)
