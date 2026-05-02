"""Packet normalization for realtime sensor ingest."""


from typing import Any, Dict, Iterable, List
from .sensors import require_supported_sensor_type


def normalize_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    sensor_type = require_supported_sensor_type(packet.get("sensor_type", ""))
    if "t_device" not in packet:
        raise ValueError("packet missing t_device")
    return {
        "sensor_type": sensor_type,
        "t_device": float(packet["t_device"]),
        "seq": int(packet["seq"]) if packet.get("seq") is not None else None,
        "payload": dict(packet.get("payload") or {}),
    }


def normalize_packets(packets: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_packet(packet) for packet in packets]
