"""Realtime session and pose endpoints for autonomous-drone integration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.db import get_db_pool
from app.deps import rate_limit, require_api_key
from app.services.realtime import (
    fetch_map_tiles,
    fetch_semantic_observations,
    finalize_realtime_session,
    ingest_realtime_packets,
    publish_map_tile,
    publish_semantic_observation,
    start_realtime_session,
)
from pipeline.core import get_logger, settings
from pipeline.realtime import pose_freshness_ms
from pipeline.storage.realtime import fetch_realtime_state, stop_robot_session

logger = get_logger(__name__)

router = APIRouter(
    prefix="/realtime",
    tags=["realtime"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


class SessionStartRequest(BaseModel):
    robot_id: str = Field(min_length=1, default="robot_0")
    mission_id: Optional[str] = None
    sensors: List[str] = Field(default_factory=lambda: ["camera", "imu", "gps"])


class SessionStartResponse(BaseModel):
    session_id: str
    robot_id: str
    mission_id: Optional[str]
    sensor_profile: Dict[str, Any]
    status: str


class SensorPacket(BaseModel):
    sensor_type: str
    t_device: float
    seq: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class PacketIngestRequest(BaseModel):
    packets: List[SensorPacket]


class PacketIngestResponse(BaseModel):
    session_id: str
    accepted_packets: int
    packet_summary: Dict[str, int]
    pose_updated: bool


class PoseResponse(BaseModel):
    session_id: str
    source: str
    t_sec: float
    position_enu: Dict[str, float]
    orientation_quat: Optional[Dict[str, float]]
    velocity_enu: Optional[Dict[str, float]]
    tracking_status: str
    global_map_id: Optional[int]
    freshness_ms: Optional[int]


class RealtimeStateResponse(BaseModel):
    session_id: str
    robot_id: str
    mission_id: Optional[str]
    status: str
    packet_counts: Dict[str, int]
    latest_pose: Optional[PoseResponse]


class SessionStopResponse(BaseModel):
    session_id: str
    status: str
    mission_id: Optional[str] = None
    job_id: Optional[str] = None
    enqueued_index_job: bool = False


class MapTileIn(BaseModel):
    tile_key: str
    map_type: str = "occupancy"
    storage_path: str
    resolution_m: float = 0.2
    bounds: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)
    global_map_id: Optional[int] = None


class MapTileOut(BaseModel):
    tile_key: str
    map_type: str
    storage_path: str
    resolution_m: float
    bounds: Dict[str, Any]
    stats: Dict[str, Any]
    global_map_id: Optional[int]


class MapTilesResponse(BaseModel):
    session_id: str
    tiles: List[MapTileOut]


class SemanticObservationIn(BaseModel):
    frame_id: Optional[str] = None
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    position_enu: Optional[Dict[str, Any]] = None
    bbox: Optional[Dict[str, Any]] = None
    mask_ref: Optional[str] = None
    track_id: Optional[str] = None
    facts: Dict[str, Any] = Field(default_factory=dict)


class SemanticObservationOut(BaseModel):
    frame_id: Optional[str]
    class_name: str
    confidence: float
    position_enu: Optional[Dict[str, Any]]
    bbox: Optional[Dict[str, Any]]
    mask_ref: Optional[str]
    track_id: Optional[str]
    facts: Dict[str, Any]


class SemanticObservationsResponse(BaseModel):
    session_id: str
    observations: List[SemanticObservationOut]


class SessionFinalizeRequest(BaseModel):
    recording_path: Optional[str] = None
    enqueue_index_job: bool = False


def _pose_response(session_id: str, row: Dict[str, Any]) -> PoseResponse:
    return PoseResponse(
        session_id=session_id,
        source=row["source"],
        t_sec=float(row["t_sec"]),
        position_enu=dict(row["position_enu_json"]),
        orientation_quat=dict(row["orientation_quat_json"]) if row.get("orientation_quat_json") else None,
        velocity_enu=dict(row["velocity_enu_json"]) if row.get("velocity_enu_json") else None,
        tracking_status=row["tracking_status"],
        global_map_id=row.get("global_map_id"),
        freshness_ms=pose_freshness_ms(row.get("created_at")),
    )


@router.post("/session/start", response_model=SessionStartResponse)
async def start_session(body: SessionStartRequest, request: Request) -> SessionStartResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        started = await start_realtime_session(
            conn,
            robot_id=body.robot_id,
            mission_id=body.mission_id,
            sensors=body.sensors,
        )
    return SessionStartResponse(**started)


@router.post("/session/{session_id}/packet", response_model=PacketIngestResponse)
async def ingest_packets(
    session_id: str,
    body: PacketIngestRequest,
    request: Request,
) -> PacketIngestResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            result = await ingest_realtime_packets(
                conn,
                session_id=session_id,
                packets=[packet.model_dump() for packet in body.packets],
            )
        except ValueError as exc:
            detail = str(exc)
            status = 413 if detail.startswith("too many packets") else 422
            raise HTTPException(status_code=status, detail=detail) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PacketIngestResponse(**result)


@router.get("/session/{session_id}/state", response_model=RealtimeStateResponse)
async def get_state(session_id: str, request: Request) -> RealtimeStateResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    latest_pose = state["latest_pose"]
    return RealtimeStateResponse(
        session_id=session_id,
        robot_id=state["session"]["robot_id"],
        mission_id=state["session"].get("mission_id"),
        status=state["session"]["status"],
        packet_counts=state["packet_counts"],
        latest_pose=_pose_response(session_id, latest_pose) if latest_pose else None,
    )


@router.get("/session/{session_id}/pose/latest", response_model=PoseResponse)
async def get_latest_pose(session_id: str, request: Request) -> PoseResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    latest_pose = state["latest_pose"]
    if latest_pose is None:
        raise HTTPException(status_code=404, detail="pose not available")
    return _pose_response(session_id, latest_pose)


@router.post("/session/{session_id}/map/tile", response_model=MapTilesResponse)
async def publish_tile(session_id: str, body: MapTileIn, request: Request) -> MapTilesResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            await publish_map_tile(conn, session_id=session_id, tile=body.model_dump())
            rows = await fetch_map_tiles(conn, session_id=session_id, map_type=body.map_type, limit=50)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MapTilesResponse(
        session_id=session_id,
        tiles=[
            MapTileOut(
                tile_key=row["tile_key"],
                map_type=row["map_type"],
                storage_path=row["storage_path"],
                resolution_m=float(row["resolution_m"]),
                bounds=dict(row.get("bounds_json") or {}),
                stats=dict(row.get("stats_json") or {}),
                global_map_id=row.get("global_map_id"),
            )
            for row in rows
        ],
    )


@router.get("/session/{session_id}/map/latest", response_model=MapTilesResponse)
async def get_latest_map(
    session_id: str,
    request: Request,
    map_type: Optional[str] = None,
    limit: int = 20,
) -> MapTilesResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            rows = await fetch_map_tiles(conn, session_id=session_id, map_type=map_type, limit=limit)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MapTilesResponse(
        session_id=session_id,
        tiles=[
            MapTileOut(
                tile_key=row["tile_key"],
                map_type=row["map_type"],
                storage_path=row["storage_path"],
                resolution_m=float(row["resolution_m"]),
                bounds=dict(row.get("bounds_json") or {}),
                stats=dict(row.get("stats_json") or {}),
                global_map_id=row.get("global_map_id"),
            )
            for row in rows
        ],
    )


@router.post("/session/{session_id}/semantic", response_model=SemanticObservationsResponse)
async def publish_semantic(
    session_id: str,
    body: SemanticObservationIn,
    request: Request,
) -> SemanticObservationsResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            await publish_semantic_observation(conn, session_id=session_id, observation=body.model_dump())
            rows = await fetch_semantic_observations(
                conn,
                session_id=session_id,
                class_name=body.class_name,
                limit=50,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SemanticObservationsResponse(
        session_id=session_id,
        observations=[
            SemanticObservationOut(
                frame_id=row.get("frame_id"),
                class_name=row["class_name"],
                confidence=float(row["confidence"]),
                position_enu=dict(row["position_enu_json"]) if row.get("position_enu_json") else None,
                bbox=dict(row["bbox_json"]) if row.get("bbox_json") else None,
                mask_ref=row.get("mask_ref"),
                track_id=row.get("track_id"),
                facts=dict(row.get("facts_json") or {}),
            )
            for row in rows
        ],
    )


@router.get("/session/{session_id}/semantic-nearby", response_model=SemanticObservationsResponse)
async def get_semantic_nearby(
    session_id: str,
    request: Request,
    class_name: Optional[str] = None,
    limit: int = 20,
) -> SemanticObservationsResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            rows = await fetch_semantic_observations(
                conn,
                session_id=session_id,
                class_name=class_name,
                limit=limit,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SemanticObservationsResponse(
        session_id=session_id,
        observations=[
            SemanticObservationOut(
                frame_id=row.get("frame_id"),
                class_name=row["class_name"],
                confidence=float(row["confidence"]),
                position_enu=dict(row["position_enu_json"]) if row.get("position_enu_json") else None,
                bbox=dict(row["bbox_json"]) if row.get("bbox_json") else None,
                mask_ref=row.get("mask_ref"),
                track_id=row.get("track_id"),
                facts=dict(row.get("facts_json") or {}),
            )
            for row in rows
        ],
    )


@router.post("/session/{session_id}/stop", response_model=SessionStopResponse)
async def stop_session(session_id: str, request: Request) -> SessionStopResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        state = await fetch_realtime_state(conn, session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="session not found")
        await stop_robot_session(conn, session_id)
    logger.info("Realtime session stopped: %s", session_id)
    return SessionStopResponse(session_id=session_id, status="stopped")


@router.post("/session/{session_id}/finalize", response_model=SessionStopResponse)
async def finalize_session(
    session_id: str,
    body: SessionFinalizeRequest,
    request: Request,
) -> SessionStopResponse:
    db_pool = get_db_pool(request)
    async with db_pool.acquire() as conn:
        try:
            result = await finalize_realtime_session(
                conn,
                session_id=session_id,
                recording_path=body.recording_path,
                enqueue_index_job=body.enqueue_index_job,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info("Realtime session finalized: %s mission=%s", session_id, result["mission_id"])
    return SessionStopResponse(**result)
