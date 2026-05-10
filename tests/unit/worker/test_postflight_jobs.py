from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from selfsuvis.worker import main as worker_main


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
    class _TxCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def transaction(self):
        return self._TxCtx()


@pytest.mark.anyio
async def test_enqueue_postflight_jobs_creates_first_stage_only():
    created = []

    async def _fake_create_job(conn_arg, job_id, payload, job_type=None):
        created.append({"id": job_id, "payload": payload, "type": job_type})

    with patch.object(worker_main, "create_job", side_effect=_fake_create_job):
        result = await worker_main._enqueue_postflight_jobs(
            object(),
            {
                "video_id": "v1",
                "video_path": "/tmp/v1.mp4",
                "mission_id": "m1",
                "postflight_jobs": [
                    "postflight_mapping",
                    "postflight_semantic_graph",
                ],
            },
            MagicMock(),
        )

    assert len(result) == 1
    assert created[0]["type"] == "postflight_mapping"
    assert created[0]["payload"]["next_postflight_jobs"] == ["postflight_semantic_graph"]


def test_handle_postflight_mapping_job_marks_mission_done_without_followup():
    pool = FakePool(FakeConn())
    logger = MagicMock()

    with (
        patch.object(worker_main.os.path, "exists", return_value=True),
        patch.object(worker_main, "_resolve_site_origin", return_value=(7, (0.0, 0.0, 0.0))),
        patch.object(worker_main, "_run_pass_a") as run_pass_a,
        patch.object(
            worker_main, "list_mission_frames", new_callable=AsyncMock, return_value=[{"id": "f1"}]
        ),
        patch.object(worker_main, "update_job", new_callable=AsyncMock),
        patch.object(
            worker_main, "_enqueue_postflight_jobs", new_callable=AsyncMock, return_value=[]
        ),
        patch.object(worker_main, "mark_mission_finished", new_callable=AsyncMock) as mark_finished,
    ):
        worker_main.handle_postflight_mapping_job(
            "job-map",
            {
                "video_id": "v1",
                "video_path": "/tmp/v1.mp4",
                "mission_id": "m1",
                "next_postflight_jobs": [],
            },
            pool,
            logger,
        )

    run_pass_a.assert_called_once()
    assert mark_finished.await_args.kwargs["status"] == "done"


def test_handle_postflight_semantic_graph_job_builds_graph_and_marks_done():
    pool = FakePool(FakeConn())
    logger = MagicMock()
    mission = {"id": "m1", "video_id": "v1"}
    frames = [
        {
            "id": "f1",
            "frame_path": "/tmp/f1.jpg",
            "t_sec": 1.0,
            "frame_facts_json": {"yolo_detections": [{"label": "tree", "confidence": 0.9}]},
            "global_pose_json": {"tx": 1.0, "ty": 2.0, "tz": 3.0},
        }
    ]

    with (
        patch.object(worker_main, "fetch_mission", new_callable=AsyncMock, return_value=mission),
        patch.object(
            worker_main, "list_mission_frames", new_callable=AsyncMock, return_value=frames
        ),
        patch(
            "selfsuvis.pipeline.mapping.build_semantic_environment_graph",
            return_value={"summary": {"node_count": 1, "edge_count": 0}},
        ),
        patch(
            "selfsuvis.pipeline.mapping.write_semantic_graph_markdown", return_value="/tmp/graph.md"
        ),
        patch.object(worker_main, "update_job", new_callable=AsyncMock),
        patch.object(
            worker_main, "_enqueue_postflight_jobs", new_callable=AsyncMock, return_value=[]
        ),
        patch.object(worker_main, "mark_mission_finished", new_callable=AsyncMock) as mark_finished,
    ):
        worker_main.handle_postflight_semantic_graph_job(
            "job-graph",
            {"mission_id": "m1", "next_postflight_jobs": []},
            pool,
            logger,
        )

    assert mark_finished.await_args.kwargs["status"] == "done"
