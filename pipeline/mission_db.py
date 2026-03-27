"""PostgreSQL helpers for mission/frame metadata persistence."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from pipeline.utils import to_utc_datetime, utcnow


async def upsert_mission(
    conn,
    *,
    mission_id: str,
    video_id: str,
    video_path: str,
    job_id: str,
    robot_id: str,
    status: str,
    frame_count: int,
    duration_sec: Optional[float],
    gps_origin: Optional[Dict[str, Any]],
    pose_status: str = "pending",
    map_status: str = "pending",
    error: Optional[str] = None,
) -> None:
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO missions
            (id, video_id, video_path, job_id, robot_id, status, pose_status, map_status,
             frame_count, duration_sec, gps_origin_json, created_at, updated_at, error)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8,
             $9, $10, $11::jsonb, $12, $13, $14)
        ON CONFLICT (id) DO UPDATE SET
            video_id = EXCLUDED.video_id,
            video_path = EXCLUDED.video_path,
            job_id = EXCLUDED.job_id,
            robot_id = EXCLUDED.robot_id,
            status = EXCLUDED.status,
            pose_status = EXCLUDED.pose_status,
            map_status = EXCLUDED.map_status,
            frame_count = EXCLUDED.frame_count,
            duration_sec = EXCLUDED.duration_sec,
            gps_origin_json = EXCLUDED.gps_origin_json,
            updated_at = EXCLUDED.updated_at,
            error = EXCLUDED.error
        """,
        mission_id,
        video_id,
        video_path,
        job_id,
        robot_id,
        status,
        pose_status,
        map_status,
        frame_count,
        duration_sec,
        json.dumps(gps_origin) if gps_origin is not None else None,
        now,
        now,
        error,
    )


async def replace_frames(conn, mission_id: str, frames: Iterable[Dict[str, Any]]) -> None:
    rows = list(frames)
    await conn.execute("DELETE FROM frames WHERE mission_id = $1", mission_id)
    if not rows:
        return

    now = utcnow()
    await conn.executemany(
        """
        INSERT INTO frames
            (id, mission_id, frame_path, t_sec, segment_id, caption, caption_confidence,
             al_score, al_tag, cvat_label, pose_status, pose_json, gps_json,
             global_pose_json, qdrant_id, created_at, updated_at)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7,
             $8, $9, $10, $11, $12::jsonb, $13::jsonb,
             $14::jsonb, $15, $16, $17)
        """,
        [
            (
                row["id"],
                mission_id,
                row["frame_path"],
                row["t_sec"],
                row.get("segment_id"),
                row.get("caption"),
                row.get("caption_confidence"),
                row.get("al_score"),
                row.get("al_tag", "none"),
                row.get("cvat_label"),
                row.get("pose_status", "pending"),
                json.dumps(row.get("pose_json")) if row.get("pose_json") is not None else None,
                json.dumps(row.get("gps_json")) if row.get("gps_json") is not None else None,
                json.dumps(row.get("global_pose_json")) if row.get("global_pose_json") is not None else None,
                row.get("qdrant_id"),
                now,
                now,
            )
            for row in rows
        ],
    )


async def mark_mission_finished(
    conn,
    mission_id: str,
    *,
    status: str,
    pose_status: Optional[str] = None,
    map_status: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    assignments = ["status = $1", "updated_at = $2", "error = $3"]
    values: List[Any] = [status, utcnow(), error]
    idx = 4
    if pose_status is not None:
        assignments.append(f"pose_status = ${idx}")
        values.append(pose_status)
        idx += 1
    if map_status is not None:
        assignments.append(f"map_status = ${idx}")
        values.append(map_status)
        idx += 1
    values.append(mission_id)
    await conn.execute(
        f"UPDATE missions SET {', '.join(assignments)} WHERE id = ${idx}",
        *values,
    )


async def apply_gps_registration(
    conn,
    mission_id: str,
    enu_origin: Optional[Dict[str, Any]],
    global_poses: Dict[str, Any],
) -> None:
    now = utcnow()
    if enu_origin is not None:
        await conn.execute(
            "UPDATE missions SET gps_origin_json = $1::jsonb, updated_at = $2 WHERE id = $3",
            json.dumps(enu_origin),
            now,
            mission_id,
        )
    if not global_poses or not isinstance(global_poses, dict):
        return
    await conn.executemany(
        """
        UPDATE frames
        SET global_pose_json = $1::jsonb,
            updated_at = $2
        WHERE id = $3
        """,
        [(json.dumps(pose), now, frame_id) for frame_id, pose in global_poses.items()],
    )


async def list_frames_after(conn, cursor: Optional[tuple], limit: int) -> List[Dict[str, Any]]:
    if cursor is None:
        rows = await conn.fetch(
            """
            SELECT id, qdrant_id, frame_path, mission_id, created_at
            FROM frames
            ORDER BY created_at ASC, id ASC
            LIMIT $1
            """,
            limit,
        )
    else:
        created_at, frame_id = cursor
        rows = await conn.fetch(
            """
            SELECT id, qdrant_id, frame_path, mission_id, created_at
            FROM frames
            WHERE (created_at, id) > ($1, $2)
            ORDER BY created_at ASC, id ASC
            LIMIT $3
            """,
            to_utc_datetime(created_at),
            frame_id,
            limit,
        )
    return [dict(r) for r in rows]
