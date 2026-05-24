"""Bridge coop sensor readings into the selfsuvis realtime threat pipeline.

Converts ``SensorReading`` (LoRaWAN) and ``CameraEvent`` (Frigate) objects into
``SensorEvent`` / ``ThreatEvent`` envelopes consumed by ``RealtimeThreatAggregator``.

Sector IDs are derived from a coarse GPS grid so that field sensors that share
the same physical area naturally roll up into the same threat sector.
"""

import math
from datetime import datetime, timezone
from typing import Any

from .events import SensorEvent, ThreatEvent

# Sector grid resolution in degrees (~110 m per 0.001°)
_GRID_DEG = 0.001


def _payload_from_attrs(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields:
        value = getattr(obj, field, None)
        if value is not None:
            payload[field] = value
    return payload


def _sector_from_gps(lat: float | None, lon: float | None) -> str:
    """Map a GPS coordinate to a coarse grid sector ID."""
    if lat is None or lon is None:
        return "unknown"
    lat_cell = math.floor(lat / _GRID_DEG)
    lon_cell = math.floor(lon / _GRID_DEG)
    return f"grid:{lat_cell}:{lon_cell}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sensor_reading_to_event(reading: Any) -> SensorEvent:
    """Convert a ``SensorReading`` to a ``SensorEvent`` envelope.

    The ``payload`` carries all numeric measurements so downstream aggregators
    can compute environmental baselines without knowing LoRaWAN internals.
    """
    payload = _payload_from_attrs(
        reading,
        ("temperature_c", "humidity_pct", "co2_ppm", "pressure_hpa", "battery_v"),
    )
    if getattr(reading, "motion", None) is not None:
        payload["motion"] = reading.motion
    if getattr(reading, "rssi", None) is not None:
        payload["rssi"] = reading.rssi
    if getattr(reading, "snr", None) is not None:
        payload["snr"] = reading.snr

    sector = _sector_from_gps(
        getattr(reading, "gps_lat", None),
        getattr(reading, "gps_lon", None),
    )

    return SensorEvent(
        event_time=getattr(reading, "received_at", None) or _now_iso(),
        ingest_time=_now_iso(),
        node_id=getattr(reading, "dev_eui", "lorawan-unknown"),
        sensor_type="lorawan",
        sector_id=sector,
        payload=payload,
    )


def camera_event_to_threat(event: Any) -> ThreatEvent | None:
    """Convert a ``CameraEvent`` to a ``ThreatEvent`` if it carries detections.

    Returns ``None`` for low-confidence events so the caller can skip them.
    """
    label: str = getattr(event, "label", "") or ""
    score: float = float(getattr(event, "score", 0.0) or 0.0)

    if not label and score < 0.05:
        return None

    threat_score = min(1.0, score)

    payload = {
        **_payload_from_attrs(event, ("region",)),
        "threat_type": "camera_detection",
        "score": threat_score,
        "label": label,
        "camera": getattr(event, "camera", "unknown"),
    }

    return ThreatEvent(
        event_time=getattr(event, "started_at", None) or _now_iso(),
        ingest_time=_now_iso(),
        node_id=f"frigate:{getattr(event, 'camera', 'unknown')}",
        sensor_type="camera",
        sector_id="unknown",  # Frigate cameras have no GPS; caller may override
        payload=payload,
    )


class CoopRealtimeIngestor:
    """Feed coop observations into a ``RealtimeThreatAggregator``.

    Typically wired at app startup:

        ingestor = CoopRealtimeIngestor(aggregator)
        mqtt_subscriber = MqttSubscriber(
            on_sensor=ingestor.on_sensor_reading,
            on_camera=ingestor.on_camera_event,
        )

    Args:
        threat_aggregator: ``RealtimeThreatAggregator`` instance to receive events.
        camera_sector_map: Optional mapping from camera name to sector_id so
                           Frigate cameras can be placed on the threat grid.
    """

    def __init__(
        self,
        threat_aggregator: Any,
        camera_sector_map: dict[str, str] | None = None,
    ) -> None:
        self._agg = threat_aggregator
        self._cam_sectors = camera_sector_map or {}

    async def on_sensor_reading(self, reading: Any) -> None:
        try:
            event = sensor_reading_to_event(reading)
            self._agg.consume(event.to_dict())
        except Exception:
            pass

    async def on_camera_event(self, event: Any) -> None:
        try:
            threat = camera_event_to_threat(event)
            if threat is None:
                return
            cam = getattr(event, "camera", "")
            if cam in self._cam_sectors:
                threat.sector_id = self._cam_sectors[cam]
            self._agg.consume(threat.to_dict())
        except Exception:
            pass
