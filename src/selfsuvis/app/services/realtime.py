"""Service layer for realtime session, pose, tile, and semantic workflows."""


import uuid
from typing import Any, Dict, List, Optional

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.realtime import (
    build_sensor_profile,
    build_fused_pose_from_packets,
    new_session_id,
    normalize_map_tile,
    normalize_packets,
    normalize_semantic_observation,
    packet_sensor_summary,
)
from selfsuvis.pipeline.storage.jobs import create_job
from selfsuvis.pipeline.storage.missions import upsert_mission
from selfsuvis.pipeline.storage.realtime import (
    create_robot_session,
    fetch_realtime_state,
    insert_realtime_pose,
    insert_sensor_packets,
    insert_semantic_observation,
    list_map_tiles,
    list_semantic_observations,
    stop_robot_session,
    upsert_map_tile,
)


async def start_realtime_session(
    conn,
    *,
    robot_id: str,
    mission_id: Optional[str],
    sensors: List[str],
) -> Dict[str, Any]:
    session_id = new_session_id()
    sensor_profile = build_sensor_profile(sensors)
    await create_robot_session(
        conn,
        session_id=session_id,
        robot_id=robot_id,
        mission_id=mission_id,
        sensor_profile=sensor_profile,
    )
    return {
        "session_id": session_id,
        "robot_id": robot_id,
        "mission_id": mission_id,
        "sensor_profile": sensor_profile,
        "status": "active",
    }


async def ingest_realtime_packets(conn, *, session_id: str, packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = normalize_packets(packets)
    if len(normalized) > settings.REALTIME_PACKET_BATCH_SIZE:
        raise ValueError(f"too many packets: max {settings.REALTIME_PACKET_BATCH_SIZE}")
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    await insert_sensor_packets(conn, session_id, normalized)

    packet_summary = packet_sensor_summary(packet["sensor_type"] for packet in normalized)
    pose_updated = False
    pose = build_fused_pose_from_packets(
        normalized,
        max_lag_ms=settings.REALTIME_MAX_SENSOR_LAG_MS,
    )
    if pose is not None:
        await insert_realtime_pose(
            conn,
            session_id=session_id,
            source=pose["source"],
            t_sec=pose["t_sec"],
            position_enu=pose["position_enu"],
            orientation_quat=pose["orientation_quat"],
            velocity_enu=pose["velocity_enu"],
            covariance=pose["covariance"],
            tracking_status=pose["tracking_status"],
            global_map_id=pose["global_map_id"],
        )
        pose_updated = True
    return {
        "session_id": session_id,
        "accepted_packets": len(normalized),
        "packet_summary": packet_summary,
        "pose_updated": pose_updated,
    }


async def publish_map_tile(conn, *, session_id: str, tile: Dict[str, Any]) -> None:
    normalized = normalize_map_tile(tile)
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    await upsert_map_tile(conn, session_id=session_id, **normalized)


async def fetch_map_tiles(
    conn,
    *,
    session_id: str,
    map_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    return await list_map_tiles(conn, session_id, map_type=map_type, limit=limit)


async def publish_semantic_observation(conn, *, session_id: str, observation: Dict[str, Any]) -> None:
    normalized = normalize_semantic_observation(observation)
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    await insert_semantic_observation(conn, session_id=session_id, **normalized)


async def fetch_semantic_observations(
    conn,
    *,
    session_id: str,
    class_name: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    return await list_semantic_observations(conn, session_id, class_name=class_name, limit=limit)


async def finalize_realtime_session(
    conn,
    *,
    session_id: str,
    recording_path: Optional[str] = None,
    enqueue_index_job: bool = False,
) -> Dict[str, Any]:
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    session = state["session"]
    mission_id = session.get("mission_id") or f"realtime-{session_id}"
    video_id = mission_id

    await upsert_mission(
        conn,
        mission_id=mission_id,
        video_id=video_id,
        video_path=recording_path or "",
        job_id="",
        robot_id=session["robot_id"],
        status="pending",
        frame_count=0,
        duration_sec=None,
        gps_origin=None,
    )

    job_id: Optional[str] = None
    if enqueue_index_job and recording_path:
        job_id = uuid.uuid4().hex
        await create_job(
            conn,
            job_id,
            {
                "video_id": video_id,
                "video_path": recording_path,
                "mission_id": mission_id,
                "enable_tiles": True,
                "realtime_session_id": session_id,
            },
            job_type="index",
        )
        await upsert_mission(
            conn,
            mission_id=mission_id,
            video_id=video_id,
            video_path=recording_path,
            job_id=job_id,
            robot_id=session["robot_id"],
            status="pending",
            frame_count=0,
            duration_sec=None,
            gps_origin=None,
        )

    await stop_robot_session(conn, session_id)
    return {
        "session_id": session_id,
        "mission_id": mission_id,
        "job_id": job_id,
        "status": "stopped",
        "enqueued_index_job": bool(job_id),
    }
