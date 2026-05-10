"""Sensor mesh fusion — combines LoRaWAN, NVR, and selfsuvis visual observations
into a spatially-aware site mesh.

Each node in the mesh is anchored by GPS coordinates (lat/lon) and carries the
latest sensor readings, camera detections, and (when available) visual embeddings
from the selfsuvis indexing pipeline.  Nodes without GPS are keyed by device EUI
or camera name and treated as "unpositioned" but still tracked.

The SensorMeshFusion class is intentionally stateless: it accepts a
SiteStateAggregator and a Qdrant store reference and produces a mesh snapshot on
demand, suitable for serving via the /site/mesh API endpoint or feeding into the
selfsuvis change detection pipeline.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..sensors.frigate_events import CameraEvent
from ..sensors.lorawan_decoder import SensorReading
from .site_state import SiteState, SiteStateAggregator


class MeshNode(BaseModel):
    """Single node in the site sensor mesh."""

    node_id: str
    node_type: str  # "sensor" | "camera" | "visual"
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    last_updated: datetime
    attributes: dict[str, Any] = Field(default_factory=dict)
    neighbor_ids: list[str] = Field(default_factory=list)


class SiteMesh(BaseModel):
    """Point-in-time snapshot of the complete site sensor mesh."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nodes: list[MeshNode] = Field(default_factory=list)
    edge_count: int = 0


class SensorMeshFusion:
    """Build a spatial mesh from heterogeneous sensor streams.

    Args:
        aggregator: SiteStateAggregator that holds current sensor readings and
                    camera events.
        proximity_radius_m: Maximum distance (metres) between two GPS-positioned
                            nodes for them to be considered neighbours.
    """

    def __init__(
        self,
        aggregator: SiteStateAggregator,
        proximity_radius_m: float = 100.0,
    ) -> None:
        self._aggregator = aggregator
        self._proximity_m = proximity_radius_m

    async def get_mesh(self) -> SiteMesh:
        """Produce a current snapshot of the site mesh."""
        state: SiteState = await self._aggregator.get_state()
        nodes: list[MeshNode] = []

        for sensor in state.sensors:
            nodes.append(
                MeshNode(
                    node_id=f"sensor:{sensor.dev_eui}",
                    node_type="sensor",
                    lat=sensor.gps_lat,
                    lon=sensor.gps_lon,
                    alt_m=sensor.gps_alt_m,
                    last_updated=sensor.last_seen,
                    attributes=_sensor_attrs(sensor),
                )
            )

        for cam in state.cameras:
            nodes.append(
                MeshNode(
                    node_id=f"camera:{cam.camera}",
                    node_type="camera",
                    last_updated=cam.last_seen,
                    attributes={
                        "active_labels": cam.active_labels,
                        "total_events": cam.total_events,
                        "recent_detections": cam.recent_detections[:3],
                    },
                )
            )

        _link_neighbours(nodes, self._proximity_m)
        edge_count = sum(len(n.neighbor_ids) for n in nodes) // 2

        return SiteMesh(nodes=nodes, edge_count=edge_count)

    def node_from_sensor_reading(self, reading: SensorReading) -> MeshNode:
        return MeshNode(
            node_id=f"sensor:{reading.dev_eui}",
            node_type="sensor",
            lat=reading.gps_lat,
            lon=reading.gps_lon,
            alt_m=reading.gps_alt_m,
            last_updated=reading.received_at,
            attributes={
                "temperature_c": reading.temperature_c,
                "humidity_pct": reading.humidity_pct,
                "co2_ppm": reading.co2_ppm,
                "motion": reading.motion,
                "rssi": reading.rssi,
            },
        )

    def node_from_camera_event(self, event: CameraEvent) -> MeshNode:
        return MeshNode(
            node_id=f"camera:{event.camera}",
            node_type="camera",
            last_updated=event.started_at,
            attributes={
                "label": event.label,
                "score": event.score,
                "has_snapshot": event.has_snapshot,
            },
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sensor_attrs(sensor: Any) -> dict[str, Any]:
    return {
        k: v
        for k, v in {
            "temperature_c": sensor.temperature_c,
            "humidity_pct": sensor.humidity_pct,
            "co2_ppm": sensor.co2_ppm,
            "pressure_hpa": sensor.pressure_hpa,
            "battery_v": sensor.battery_v,
            "motion": sensor.motion,
            "rssi": sensor.rssi,
            "snr": sensor.snr,
            "reading_count": sensor.reading_count,
        }.items()
        if v is not None
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _link_neighbours(nodes: list[MeshNode], radius_m: float) -> None:
    positioned = [n for n in nodes if n.lat is not None and n.lon is not None]
    for i, a in enumerate(positioned):
        for b in positioned[i + 1 :]:
            dist = _haversine_m(a.lat, a.lon, b.lat, b.lon)  # type: ignore[arg-type]
            if dist <= radius_m:
                a.neighbor_ids.append(b.node_id)
                b.neighbor_ids.append(a.node_id)
