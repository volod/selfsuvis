"""Lightweight sensor summaries for realtime ingest."""


from typing import Dict, Iterable
from .sensors import packet_sensor_summary as _packet_sensor_summary


def packet_sensor_summary(sensor_types: Iterable[str]) -> Dict[str, int]:
    return _packet_sensor_summary(sensor_types)
