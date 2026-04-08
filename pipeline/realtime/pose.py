"""Minimal realtime pose helpers.

The current milestone stores externally generated poses and also supports a
stub pose path for GPS packets so the API can be exercised before a SLAM
backend is integrated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def build_stub_pose_from_packet(packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if packet.get("sensor_type") != "gps":
        return None
    payload = packet.get("payload") or {}
    if payload.get("east") is None or payload.get("north") is None:
        return None
    return {
        "source": "gps_fallback",
        "t_sec": float(packet["t_device"]),
        "position_enu": {
            "x": float(payload["east"]),
            "y": float(payload["north"]),
            "z": float(payload.get("up", 0.0)),
        },
        "orientation_quat": None,
        "velocity_enu": None,
        "covariance": None,
        "tracking_status": "degraded",
        "global_map_id": payload.get("global_map_id"),
    }


def pose_freshness_ms(created_at: Any, now: Optional[datetime] = None) -> Optional[int]:
    if created_at is None:
        return None
    if isinstance(created_at, str):
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    else:
        created_dt = created_at
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=timezone.utc)
    now_dt = now or datetime.now(timezone.utc)
    return max(0, int((now_dt - created_dt).total_seconds() * 1000))
