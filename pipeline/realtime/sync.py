"""Lightweight sensor summaries for realtime ingest."""

from __future__ import annotations

from typing import Dict, Iterable


def packet_sensor_summary(sensor_types: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sensor_type in sensor_types:
        key = str(sensor_type).strip().lower()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts
