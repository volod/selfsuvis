"""Packet normalization for realtime sensor ingest."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

_VALID_SENSOR_TYPES = {"camera", "imu", "gps", "lidar", "barometer"}


def normalize_packets(packets: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for packet in packets:
        sensor_type = str(packet.get("sensor_type", "")).strip().lower()
        if sensor_type not in _VALID_SENSOR_TYPES:
            raise ValueError(f"unsupported sensor_type: {sensor_type or '<empty>'}")
        if "t_device" not in packet:
            raise ValueError("packet missing t_device")
        normalized.append(
            {
                "sensor_type": sensor_type,
                "t_device": float(packet["t_device"]),
                "seq": int(packet["seq"]) if packet.get("seq") is not None else None,
                "payload": dict(packet.get("payload") or {}),
            }
        )
    return normalized
