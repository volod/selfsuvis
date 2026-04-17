"""PostgreSQL helpers for realtime drone sessions, poses, tiles, and semantics."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from selfsuvis.pipeline.core import utcnow


async def create_robot_session(
    conn,
    *,
    session_id: str,
    robot_id: str,
    mission_id: Optional[str] = None,
    sensor_profile: Optional[Dict[str, Any]] = None,
    status: str = "active",
) -> None:
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO robot_sessions
            (id, robot_id, mission_id, sensor_profile_json, status, started_at, updated_at)
        VALUES
            ($1, $2, $3, $4::jsonb, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE SET
            robot_id = EXCLUDED.robot_id,
            mission_id = EXCLUDED.mission_id,
            sensor_profile_json = EXCLUDED.sensor_profile_json,
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at
        """,
        session_id,
        robot_id,
        mission_id,
        json.dumps(sensor_profile or {}),
        status,
        now,
        now,
    )


async def stop_robot_session(conn, session_id: str, status: str = "stopped") -> None:
    now = utcnow()
    await conn.execute(
        """
        UPDATE robot_sessions
        SET status = $1, ended_at = $2, updated_at = $2
        WHERE id = $3
        """,
        status,
        now,
        session_id,
    )


async def fetch_robot_session(conn, session_id: str) -> Optional[Dict[str, Any]]:
    row = await conn.fetchrow(
        """
        SELECT id, robot_id, mission_id, sensor_profile_json, status,
               started_at, ended_at, updated_at
        FROM robot_sessions
        WHERE id = $1
        """,
        session_id,
    )
    return dict(row) if row else None


async def insert_sensor_packets(
    conn,
    session_id: str,
    packets: Iterable[Dict[str, Any]],
) -> int:
    rows = list(packets)
    if not rows:
        return 0
    now = utcnow()
    await conn.executemany(
        """
        INSERT INTO sensor_packets
            (session_id, sensor_type, t_device, t_server, seq, payload_json)
        VALUES
            ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        [
            (
                session_id,
                row["sensor_type"],
                row["t_device"],
                now,
                row.get("seq"),
                json.dumps(row.get("payload") or {}),
            )
            for row in rows
        ],
    )
    return len(rows)


async def insert_realtime_pose(
    conn,
    *,
    session_id: str,
    source: str,
    t_sec: float,
    position_enu: Dict[str, float],
    orientation_quat: Optional[Dict[str, float]] = None,
    velocity_enu: Optional[Dict[str, float]] = None,
    covariance: Optional[Dict[str, Any]] = None,
    tracking_status: str = "ok",
    global_map_id: Optional[int] = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO realtime_poses
            (session_id, source, t_sec, position_enu_json, orientation_quat_json,
             velocity_enu_json, covariance_json, tracking_status, global_map_id)
        VALUES
            ($1, $2, $3, $4::jsonb, $5::jsonb,
             $6::jsonb, $7::jsonb, $8, $9)
        """,
        session_id,
        source,
        t_sec,
        json.dumps(position_enu),
        json.dumps(orientation_quat) if orientation_quat is not None else None,
        json.dumps(velocity_enu) if velocity_enu is not None else None,
        json.dumps(covariance) if covariance is not None else None,
        tracking_status,
        global_map_id,
    )


async def fetch_latest_realtime_pose(conn, session_id: str) -> Optional[Dict[str, Any]]:
    row = await conn.fetchrow(
        """
        SELECT id, session_id, source, t_sec, position_enu_json, orientation_quat_json,
               velocity_enu_json, covariance_json, tracking_status, global_map_id, created_at
        FROM realtime_poses
        WHERE session_id = $1
        ORDER BY t_sec DESC, id DESC
        LIMIT 1
        """,
        session_id,
    )
    return dict(row) if row else None


async def fetch_realtime_state(conn, session_id: str) -> Optional[Dict[str, Any]]:
    session = await fetch_robot_session(conn, session_id)
    if session is None:
        return None
    latest_pose = await fetch_latest_realtime_pose(conn, session_id)
    packet_counts = await conn.fetch(
        """
        SELECT sensor_type, COUNT(*)::BIGINT AS n
        FROM sensor_packets
        WHERE session_id = $1
        GROUP BY sensor_type
        ORDER BY sensor_type
        """,
        session_id,
    )
    return {
        "session": session,
        "latest_pose": latest_pose,
        "packet_counts": {row["sensor_type"]: int(row["n"]) for row in packet_counts},
    }


async def upsert_map_tile(
    conn,
    *,
    session_id: str,
    tile_key: str,
    map_type: str,
    storage_path: str,
    resolution_m: float,
    bounds: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
    global_map_id: Optional[int] = None,
) -> None:
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO map_tiles
            (session_id, global_map_id, tile_key, map_type, storage_path,
             resolution_m, bounds_json, stats_json, updated_at)
        VALUES
            ($1, $2, $3, $4, $5,
             $6, $7::jsonb, $8::jsonb, $9)
        ON CONFLICT (session_id, tile_key, map_type) DO UPDATE SET
            global_map_id = EXCLUDED.global_map_id,
            storage_path = EXCLUDED.storage_path,
            resolution_m = EXCLUDED.resolution_m,
            bounds_json = EXCLUDED.bounds_json,
            stats_json = EXCLUDED.stats_json,
            updated_at = EXCLUDED.updated_at
        """,
        session_id,
        global_map_id,
        tile_key,
        map_type,
        storage_path,
        resolution_m,
        json.dumps(bounds or {}),
        json.dumps(stats or {}),
        now,
    )


async def list_map_tiles(
    conn,
    session_id: str,
    map_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if map_type:
        rows = await conn.fetch(
            """
            SELECT id, session_id, global_map_id, tile_key, map_type, storage_path,
                   resolution_m, bounds_json, stats_json, updated_at
            FROM map_tiles
            WHERE session_id = $1 AND map_type = $2
            ORDER BY updated_at DESC, id DESC
            LIMIT $3
            """,
            session_id,
            map_type,
            limit,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id, session_id, global_map_id, tile_key, map_type, storage_path,
                   resolution_m, bounds_json, stats_json, updated_at
            FROM map_tiles
            WHERE session_id = $1
            ORDER BY updated_at DESC, id DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return [dict(row) for row in rows]


async def insert_semantic_observation(
    conn,
    *,
    session_id: str,
    class_name: str,
    confidence: float,
    position_enu: Optional[Dict[str, Any]] = None,
    bbox: Optional[Dict[str, Any]] = None,
    frame_id: Optional[str] = None,
    mask_ref: Optional[str] = None,
    track_id: Optional[str] = None,
    facts: Optional[Dict[str, Any]] = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO semantic_observations
            (session_id, frame_id, class_name, confidence, position_enu_json,
             bbox_json, mask_ref, track_id, facts_json)
        VALUES
            ($1, $2, $3, $4, $5::jsonb,
             $6::jsonb, $7, $8, $9::jsonb)
        """,
        session_id,
        frame_id,
        class_name,
        confidence,
        json.dumps(position_enu) if position_enu is not None else None,
        json.dumps(bbox) if bbox is not None else None,
        mask_ref,
        track_id,
        json.dumps(facts or {}),
    )


async def list_semantic_observations(
    conn,
    session_id: str,
    class_name: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    if class_name:
        rows = await conn.fetch(
            """
            SELECT id, session_id, frame_id, class_name, confidence, position_enu_json,
                   bbox_json, mask_ref, track_id, facts_json, created_at
            FROM semantic_observations
            WHERE session_id = $1 AND class_name = $2
            ORDER BY created_at DESC, id DESC
            LIMIT $3
            """,
            session_id,
            class_name,
            limit,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id, session_id, frame_id, class_name, confidence, position_enu_json,
                   bbox_json, mask_ref, track_id, facts_json, created_at
            FROM semantic_observations
            WHERE session_id = $1
            ORDER BY created_at DESC, id DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return [dict(row) for row in rows]
