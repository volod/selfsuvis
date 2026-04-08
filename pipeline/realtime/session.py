"""Session helpers for realtime drone ingest."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable


def new_session_id() -> str:
    return str(uuid.uuid4())


def build_sensor_profile(sensors: Iterable[str]) -> Dict[str, Any]:
    deduped = sorted({sensor.strip().lower() for sensor in sensors if sensor and sensor.strip()})
    return {
        "sensors": deduped,
        "sensor_count": len(deduped),
    }
