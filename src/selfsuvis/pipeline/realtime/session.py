"""Session helpers for realtime drone ingest."""


import uuid
from typing import Any, Dict, Iterable

from .sensors import build_sensor_profile as _build_sensor_profile

def new_session_id() -> str:
    return str(uuid.uuid4())


def build_sensor_profile(sensors: Iterable[str]) -> Dict[str, Any]:
    return _build_sensor_profile(sensors)
