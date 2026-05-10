"""Real-time site state model and aggregator.

SiteStateAggregator maintains rolling windows of LoRaWAN sensor readings and
Frigate camera events.  It is the single source of truth for GET /site/state
and the WebSocket live stream.

Thread-safe: all mutations go through an asyncio.Lock.  Callers that run in a
different thread must use asyncio.run_coroutine_threadsafe.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..config import settings
from ..sensors.frigate_events import CameraEvent
from ..sensors.lorawan_decoder import SensorReading

# ── Public Pydantic models (used in API responses) ────────────────────────────


class SensorSummary(BaseModel):
    """Aggregated summary of recent readings from one LoRaWAN device."""

    dev_eui: str
    last_seen: datetime
    reading_count: int
    temperature_c: float | None = None
    humidity_pct: float | None = None
    co2_ppm: float | None = None
    pressure_hpa: float | None = None
    battery_v: float | None = None
    motion: bool | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_alt_m: float | None = None
    rssi: float | None = None
    snr: float | None = None


class CameraEventSummary(BaseModel):
    """Recent detections from one Frigate camera."""

    camera: str
    last_seen: datetime
    recent_detections: list[dict[str, Any]] = Field(default_factory=list)
    active_labels: list[str] = Field(default_factory=list)
    total_events: int = 0


class SiteState(BaseModel):
    """Snapshot of the current site state across all sensor modalities."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sensors: list[SensorSummary] = Field(default_factory=list)
    cameras: list[CameraEventSummary] = Field(default_factory=list)
    sensor_count: int = 0
    camera_count: int = 0
    active_motion: bool = False


# ── Aggregator ────────────────────────────────────────────────────────────────


class SiteStateAggregator:
    """Rolling-window collector for sensor readings and camera events.

    Call ingest_sensor_reading() and ingest_camera_event() from the MQTT
    subscriber callbacks.  Call get_state() to obtain a point-in-time
    SiteState snapshot.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # dev_eui → deque of (timestamp, SensorReading)
        self._sensors: dict[str, deque[tuple[datetime, SensorReading]]] = {}
        # camera → deque of CameraEvent
        self._cameras: dict[str, deque[CameraEvent]] = {}

    async def ingest_sensor_reading(self, reading: SensorReading) -> None:
        async with self._lock:
            if reading.dev_eui not in self._sensors:
                self._sensors[reading.dev_eui] = deque()
            self._sensors[reading.dev_eui].append((reading.received_at, reading))
            self._evict_old_sensor(reading.dev_eui)

    async def ingest_camera_event(self, event: CameraEvent) -> None:
        async with self._lock:
            if event.camera not in self._cameras:
                self._cameras[event.camera] = deque()
            self._cameras[event.camera].append(event)
            self._evict_old_camera(event.camera)

    def _evict_old_sensor(self, dev_eui: str) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.sensor_window_sec)
        q = self._sensors[dev_eui]
        while q and q[0][0] < cutoff:
            q.popleft()

    def _evict_old_camera(self, camera: str) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.camera_event_window_sec)
        q = self._cameras[camera]
        while q and q[0].started_at < cutoff:
            q.popleft()

    async def get_state(self) -> SiteState:
        async with self._lock:
            sensors = [
                self._summarize_sensor(dev_eui, q) for dev_eui, q in self._sensors.items() if q
            ]
            cameras = [
                self._summarize_camera(camera, q) for camera, q in self._cameras.items() if q
            ]

        active_motion = any(s.motion for s in sensors if s.motion)
        return SiteState(
            sensors=sensors,
            cameras=cameras,
            sensor_count=len(sensors),
            camera_count=len(cameras),
            active_motion=active_motion,
        )

    @staticmethod
    def _summarize_sensor(
        dev_eui: str, q: "deque[tuple[datetime, SensorReading]]"
    ) -> SensorSummary:
        # Most recent reading wins for scalar values
        latest_ts, latest = q[-1]
        return SensorSummary(
            dev_eui=dev_eui,
            last_seen=latest_ts,
            reading_count=len(q),
            temperature_c=latest.temperature_c,
            humidity_pct=latest.humidity_pct,
            co2_ppm=latest.co2_ppm,
            pressure_hpa=latest.pressure_hpa,
            battery_v=latest.battery_v,
            motion=latest.motion,
            gps_lat=latest.gps_lat,
            gps_lon=latest.gps_lon,
            gps_alt_m=latest.gps_alt_m,
            rssi=latest.rssi,
            snr=latest.snr,
        )

    @staticmethod
    def _summarize_camera(camera: str, q: "deque[CameraEvent]") -> CameraEventSummary:
        events = list(q)
        labels = list({e.label for e in events})
        recent = [
            {
                "event_id": e.event_id,
                "label": e.label,
                "score": e.score,
                "started_at": e.started_at.isoformat(),
                "has_snapshot": e.has_snapshot,
            }
            for e in sorted(events, key=lambda x: x.started_at, reverse=True)[:10]
        ]
        return CameraEventSummary(
            camera=camera,
            last_seen=max(e.started_at for e in events),
            recent_detections=recent,
            active_labels=labels,
            total_events=len(events),
        )
