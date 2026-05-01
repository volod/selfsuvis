import httpx
import pytest
from fastapi import FastAPI

from selfsuvis.app.routers.site_state import router


class _FakeStreamService:
    def active_cameras(self):
        return [
            {
                "camera": "entrance",
                "rtsp_url": "rtsp://frigate:8554/entrance",
                "session_id": "coop-entrance",
                "started_at": "2026-05-01T12:00:00+00:00",
            }
        ]


@pytest.mark.anyio
async def test_site_cameras_includes_rtsp_bridge_sessions() -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.coop_streams = _FakeStreamService()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/site/cameras")

    assert response.status_code == 200
    assert response.json() == {
        "cameras": [
            {
                "camera": "entrance",
                "last_seen": "2026-05-01T12:00:00+00:00",
                "recent_detections": [],
                "active_labels": [],
                "total_events": 0,
                "session_id": "coop-entrance",
                "rtsp_url": "rtsp://frigate:8554/entrance",
                "stream_started_at": "2026-05-01T12:00:00+00:00",
            }
        ]
    }
