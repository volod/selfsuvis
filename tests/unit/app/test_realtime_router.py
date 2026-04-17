"""Unit tests for app/routers/realtime.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.routers.realtime import router


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCtx(self._conn)


class FakeConn:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.packets: List[Dict[str, Any]] = []
        self.poses: List[Dict[str, Any]] = []
        self.tiles: List[Dict[str, Any]] = []
        self.semantic: List[Dict[str, Any]] = []
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.missions: Dict[str, Dict[str, Any]] = {}

    async def execute(self, query: str, *args):
        if "INSERT INTO robot_sessions" in query:
            self.sessions[args[0]] = {
                "id": args[0],
                "robot_id": args[1],
                "mission_id": args[2],
                "sensor_profile_json": json.loads(args[3]),
                "status": args[4],
                "started_at": args[5],
                "ended_at": None,
                "updated_at": args[6],
            }
            return "INSERT 0 1"
        if "UPDATE robot_sessions" in query:
            session = self.sessions[args[2]]
            session["status"] = args[0]
            session["ended_at"] = args[1]
            session["updated_at"] = args[1]
            return "UPDATE 1"
        if "INSERT INTO realtime_poses" in query:
            self.poses.append(
                {
                    "id": len(self.poses) + 1,
                    "session_id": args[0],
                    "source": args[1],
                    "t_sec": args[2],
                    "position_enu_json": json.loads(args[3]),
                    "orientation_quat_json": json.loads(args[4]) if args[4] else None,
                    "velocity_enu_json": json.loads(args[5]) if args[5] else None,
                    "covariance_json": json.loads(args[6]) if args[6] else None,
                    "tracking_status": args[7],
                    "global_map_id": args[8],
                    "created_at": datetime.now(timezone.utc),
                }
            )
            return "INSERT 0 1"
        if "INSERT INTO map_tiles" in query:
            row = {
                "session_id": args[0],
                "global_map_id": args[1],
                "tile_key": args[2],
                "map_type": args[3],
                "storage_path": args[4],
                "resolution_m": args[5],
                "bounds_json": json.loads(args[6]),
                "stats_json": json.loads(args[7]),
                "updated_at": args[8],
            }
            self.tiles = [
                existing
                for existing in self.tiles
                if not (
                    existing["session_id"] == row["session_id"]
                    and existing["tile_key"] == row["tile_key"]
                    and existing["map_type"] == row["map_type"]
                )
            ]
            self.tiles.append(row)
            return "INSERT 0 1"
        if "INSERT INTO semantic_observations" in query:
            self.semantic.append(
                {
                    "id": len(self.semantic) + 1,
                    "session_id": args[0],
                    "frame_id": args[1],
                    "class_name": args[2],
                    "confidence": args[3],
                    "position_enu_json": json.loads(args[4]) if args[4] else None,
                    "bbox_json": json.loads(args[5]) if args[5] else None,
                    "mask_ref": args[6],
                    "track_id": args[7],
                    "facts_json": json.loads(args[8]),
                    "created_at": datetime.now(timezone.utc),
                }
            )
            return "INSERT 0 1"
        if "INSERT INTO jobs" in query:
            self.jobs[args[0]] = {"id": args[0], "type": args[2], "payload_json": json.loads(args[4])}
            return "INSERT 0 1"
        if "INSERT INTO missions" in query:
            self.missions[args[0]] = {
                "id": args[0],
                "video_id": args[1],
                "video_path": args[2],
                "job_id": args[3],
                "robot_id": args[4],
                "status": args[5],
            }
            return "INSERT 0 1"
        raise AssertionError(f"unexpected execute query: {query}")

    async def executemany(self, query: str, rows):
        rows = list(rows)
        if "INSERT INTO sensor_packets" in query:
            for row in rows:
                self.packets.append(
                    {
                        "session_id": row[0],
                        "sensor_type": row[1],
                        "t_device": row[2],
                        "t_server": row[3],
                        "seq": row[4],
                        "payload_json": json.loads(row[5]),
                    }
                )
            return
        raise AssertionError(f"unexpected executemany query: {query}")

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        if "FROM robot_sessions" in query:
            return self.sessions.get(args[0])
        if "FROM realtime_poses" in query:
            matches = [row for row in self.poses if row["session_id"] == args[0]]
            if not matches:
                return None
            return sorted(matches, key=lambda row: (row["t_sec"], row["id"]), reverse=True)[0]
        raise AssertionError(f"unexpected fetchrow query: {query}")

    async def fetch(self, query: str, *args):
        if "FROM sensor_packets" in query and "GROUP BY sensor_type" in query:
            session_id = args[0]
            counts: Dict[str, int] = {}
            for row in self.packets:
                if row["session_id"] != session_id:
                    continue
                counts[row["sensor_type"]] = counts.get(row["sensor_type"], 0) + 1
            return [{"sensor_type": key, "n": value} for key, value in sorted(counts.items())]
        if "FROM map_tiles" in query:
            session_id = args[0]
            if len(args) == 3:
                map_type = args[1]
                limit = args[2]
                rows = [row for row in self.tiles if row["session_id"] == session_id and row["map_type"] == map_type]
            else:
                limit = args[1]
                rows = [row for row in self.tiles if row["session_id"] == session_id]
            return list(reversed(rows))[:limit]
        if "FROM semantic_observations" in query:
            session_id = args[0]
            if len(args) == 3:
                class_name = args[1]
                limit = args[2]
                rows = [row for row in self.semantic if row["session_id"] == session_id and row["class_name"] == class_name]
            else:
                limit = args[1]
                rows = [row for row in self.semantic if row["session_id"] == session_id]
            return list(reversed(rows))[:limit]
        raise AssertionError(f"unexpected fetch query: {query}")


def _app():
    app = FastAPI()

    async def _no_auth():
        return "test-key"

    async def _no_rate():
        return None

    app.dependency_overrides[require_api_key] = _no_auth
    app.dependency_overrides[rate_limit] = _no_rate
    app.include_router(router)
    return app


@pytest.mark.anyio
async def test_start_state_stop_session():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post(
                "/realtime/session/start",
                json={"robot_id": "drone_a", "mission_id": "mission_1", "sensors": ["camera", "imu", "gps"]},
            )
            assert start.status_code == 200
            session_id = start.json()["session_id"]

            state = await client.get(f"/realtime/session/{session_id}/state")
            assert state.status_code == 200
            assert state.json()["status"] == "active"
            assert state.json()["packet_counts"] == {}
            assert state.json()["latest_pose"] is None

            stop = await client.post(f"/realtime/session/{session_id}/stop")
            assert stop.status_code == 200
            assert stop.json()["status"] == "stopped"


@pytest.mark.anyio
async def test_packet_ingest_creates_stub_pose_from_gps():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "drone_b"})
            session_id = start.json()["session_id"]

            ingest = await client.post(
                f"/realtime/session/{session_id}/packet",
                json={
                    "packets": [
                        {
                            "sensor_type": "gps",
                            "t_device": 12.5,
                            "seq": 7,
                            "payload": {"east": 1.25, "north": -0.5, "up": 10.0, "global_map_id": 3},
                        },
                        {
                            "sensor_type": "imu",
                            "t_device": 12.52,
                            "seq": 8,
                            "payload": {"yaw": 0.1, "vx": 0.2, "vy": 0.0, "vz": 0.0},
                        },
                    ]
                },
            )
            assert ingest.status_code == 200
            assert ingest.json()["accepted_packets"] == 2
            assert ingest.json()["pose_updated"] is True
            assert ingest.json()["packet_summary"] == {"gps": 1, "imu": 1}

            pose = await client.get(f"/realtime/session/{session_id}/pose/latest")
            assert pose.status_code == 200
            data = pose.json()
            assert data["source"] == "fused_gps_imu"
            assert data["position_enu"] == {"x": 1.25, "y": -0.5, "z": 10.0}
            assert data["tracking_status"] == "ok"
            assert data["global_map_id"] == 3

            state = await client.get(f"/realtime/session/{session_id}/state")
            assert state.status_code == 200
            assert state.json()["packet_counts"] == {"gps": 1, "imu": 1}


@pytest.mark.anyio
async def test_latest_pose_404_when_missing():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "drone_c"})
            session_id = start.json()["session_id"]
            pose = await client.get(f"/realtime/session/{session_id}/pose/latest")
            assert pose.status_code == 404


@pytest.mark.anyio
async def test_publish_map_tile_and_semantic_observation():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post("/realtime/session/start", json={"robot_id": "drone_d"})
            session_id = start.json()["session_id"]

            tile = await client.post(
                f"/realtime/session/{session_id}/map/tile",
                json={
                    "tile_key": "tile-001",
                    "map_type": "occupancy",
                    "storage_path": "/tmp/tile-001.bin",
                    "resolution_m": 0.2,
                    "bounds": {"min_x": 0, "max_x": 10},
                    "stats": {"occupied": 42},
                    "global_map_id": 9,
                },
            )
            assert tile.status_code == 200
            tile_data = tile.json()
            assert len(tile_data["tiles"]) == 1
            assert tile_data["tiles"][0]["tile_key"] == "tile-001"

            semantic = await client.post(
                f"/realtime/session/{session_id}/semantic",
                json={
                    "class_name": "tree",
                    "confidence": 0.91,
                    "position_enu": {"x": 3.0, "y": 4.0, "z": 5.0},
                    "facts": {"source": "yolo"},
                },
            )
            assert semantic.status_code == 200
            obs = semantic.json()["observations"][0]
            assert obs["class_name"] == "tree"
            assert obs["facts"]["source"] == "yolo"

            latest_map = await client.get(f"/realtime/session/{session_id}/map/latest")
            assert latest_map.status_code == 200
            assert latest_map.json()["tiles"][0]["stats"]["occupied"] == 42

            nearby = await client.get(f"/realtime/session/{session_id}/semantic-nearby?class_name=tree")
            assert nearby.status_code == 200
            assert nearby.json()["observations"][0]["class_name"] == "tree"


@pytest.mark.anyio
async def test_finalize_session_creates_mission_and_optional_job():
    conn = FakeConn()
    app = _app()

    with patch("selfsuvis.app.routers.realtime.get_db_pool", return_value=FakePool(conn)):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            start = await client.post(
                "/realtime/session/start",
                json={"robot_id": "drone_e", "mission_id": "mission_live"},
            )
            session_id = start.json()["session_id"]

            finalize = await client.post(
                f"/realtime/session/{session_id}/finalize",
                json={"recording_path": "/data/videos/live.mp4", "enqueue_index_job": True},
            )
            assert finalize.status_code == 200
            data = finalize.json()
            assert data["mission_id"] == "mission_live"
            assert data["enqueued_index_job"] is True
            assert data["job_id"]
            assert "mission_live" in conn.missions
            assert data["job_id"] in conn.jobs
