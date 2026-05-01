"""Site state API — real-time multi-sensor site awareness.

Endpoints:
  GET  /site/state      — current aggregated site state (sensors + cameras)
  GET  /site/mesh       — spatial sensor mesh with neighbour links
  GET  /site/sensors    — all active LoRaWAN device summaries
  GET  /site/cameras    — all active Frigate camera summaries
  GET  /site/synthesis  — LLM-fused scene narrative (cached, ~10 s TTL)
  GET  /site/threat     — realtime threat snapshot from RealtimeThreatAggregator
  WS   /site/stream     — push site state updates via WebSocket

The SiteStateAggregator is stored on app.state and fed by the background
MqttSubscriber task started at lifespan. If the broker is unreachable, the
subscriber reconnects in the background and endpoints return empty-but-valid
responses until events arrive.
"""

import asyncio

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/site", tags=["site"])


def _get_aggregator(request: Request):
    return getattr(request.app.state, "site_state_aggregator", None)


def _get_fusion(request: Request):
    return getattr(request.app.state, "sensor_mesh_fusion", None)


def _get_synthesizer(request: Request):
    return getattr(request.app.state, "scene_synthesizer", None)


def _get_threat_aggregator(request: Request):
    return getattr(request.app.state, "coop_threat_aggregator", None)


def _get_stream_service(request: Request):
    return getattr(request.app.state, "coop_streams", None)


@router.get("/state")
async def get_site_state(request: Request) -> JSONResponse:
    """Return a snapshot of the current site state across all sensor modalities."""
    aggregator = _get_aggregator(request)
    if aggregator is None:
        return JSONResponse({"status": "unavailable", "reason": "coop_pilot not configured"})

    from selfsuvis.coop_pilot.mesh.site_state import SiteState
    state: SiteState = await aggregator.get_state()
    return JSONResponse(state.model_dump(mode="json"))


@router.get("/mesh")
async def get_site_mesh(request: Request) -> JSONResponse:
    """Return the spatial sensor mesh with GPS-proximity neighbour links."""
    fusion = _get_fusion(request)
    if fusion is None:
        return JSONResponse({"status": "unavailable", "reason": "coop_pilot not configured"})

    from selfsuvis.coop_pilot.mesh.fusion import SiteMesh
    mesh: SiteMesh = await fusion.get_mesh()
    return JSONResponse(mesh.model_dump(mode="json"))


@router.get("/sensors")
async def get_sensors(request: Request) -> JSONResponse:
    """Return summaries of all active LoRaWAN devices in the rolling window."""
    aggregator = _get_aggregator(request)
    if aggregator is None:
        return JSONResponse({"sensors": []})
    state = await aggregator.get_state()
    return JSONResponse({"sensors": [s.model_dump(mode="json") for s in state.sensors]})


@router.get("/cameras")
async def get_cameras(request: Request) -> JSONResponse:
    """Return Frigate camera event summaries plus active RTSP bridge sessions."""
    aggregator = _get_aggregator(request)
    cameras: dict[str, dict] = {}
    if aggregator is not None:
        state = await aggregator.get_state()
        cameras.update({c.camera: c.model_dump(mode="json") for c in state.cameras})

    stream_service = _get_stream_service(request)
    if stream_service is not None:
        for session in stream_service.active_cameras():
            camera = str(session.get("camera", "unknown"))
            row = cameras.setdefault(
                camera,
                {
                    "camera": camera,
                    "last_seen": session.get("started_at"),
                    "recent_detections": [],
                    "active_labels": [],
                    "total_events": 0,
                },
            )
            row.update(
                {
                    "session_id": session.get("session_id"),
                    "rtsp_url": session.get("rtsp_url"),
                    "stream_started_at": session.get("started_at"),
                }
            )

    return JSONResponse({"cameras": list(cameras.values())})


@router.get("/synthesis")
async def get_scene_synthesis(request: Request, force: bool = False) -> JSONResponse:
    """Return an LLM-fused scene narrative for the monitored site.

    The result is cached for ~10 seconds (configurable in SceneSynthesizer).
    Pass ``?force=true`` to bypass the cache and request a fresh synthesis.
    """
    synthesizer = _get_synthesizer(request)
    if synthesizer is None:
        return JSONResponse({"status": "unavailable", "reason": "scene synthesizer not configured"})

    from selfsuvis.coop_pilot.mesh.scene_synthesis import SceneSynthesis
    synthesis: SceneSynthesis = await synthesizer.synthesize(force=force)
    return JSONResponse(synthesis.model_dump(mode="json"))


@router.get("/threat")
async def get_threat_snapshot(request: Request) -> JSONResponse:
    """Return the current realtime threat snapshot aggregated from coop sensors.

    Combines LoRaWAN sensor anomalies and Frigate camera detections into a
    sector-level threat map compatible with the robot advisory API.
    """
    threat_agg = _get_threat_aggregator(request)
    if threat_agg is None:
        return JSONResponse({"status": "unavailable", "reason": "threat aggregator not configured"})
    return JSONResponse(threat_agg.snapshot())


@router.websocket("/stream")
async def site_state_stream(websocket: WebSocket, interval: float = 2.0) -> None:
    """WebSocket endpoint that pushes the site state every `interval` seconds.

    The client receives JSON-serialised SiteState objects.  Send any message
    to change the push interval (seconds as a float string, clamped 0.5–60).
    """
    await websocket.accept()
    aggregator = _get_aggregator(websocket)
    if aggregator is None:
        await websocket.send_json({"status": "unavailable", "reason": "coop_pilot not configured"})
        await websocket.close()
        return

    push_interval = max(0.5, min(60.0, interval))

    try:
        while True:
            state = await aggregator.get_state()
            await websocket.send_text(state.model_dump_json())

            # Check for client messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=push_interval)
                try:
                    push_interval = max(0.5, min(60.0, float(msg)))
                except ValueError:
                    pass
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.debug("site_state_stream client disconnected")
    except Exception:
        logger.exception("site_state_stream error")
