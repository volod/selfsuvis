"""Live MediaMTX stream control and realtime caption runtime management."""


import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner
from selfsuvis.pipeline.media.rtsp_ingest import validate_rtsp_url

logger = get_logger(__name__)


def validate_stream_path(path_name: str) -> str:
    path = (path_name or "").strip().strip("/")
    if not path:
        raise ValueError("path_name is required")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-")
    if any(ch not in allowed for ch in path):
        raise ValueError("path_name may contain only letters, digits, '/', '-', and '_'")
    if "//" in path:
        raise ValueError("path_name may not contain empty path segments")
    return path


def build_rtsp_stream_url(path_name: str, *, public: bool = False) -> str:
    base = settings.MEDIAMTX_PUBLIC_RTSP_BASE_URL if public else settings.MEDIAMTX_RTSP_BASE_URL
    return f"{base.rstrip('/')}/{validate_stream_path(path_name)}"


class MediaMtxClient:
    """Small async client for the MediaMTX control API."""

    def __init__(
        self,
        *,
        api_url: Optional[str] = None,
        api_user: Optional[str] = None,
        api_pass: Optional[str] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self._api_url = (api_url or settings.MEDIAMTX_API_URL).rstrip("/")
        self._auth = None
        user = api_user if api_user is not None else settings.MEDIAMTX_API_USER
        password = api_pass if api_pass is not None else settings.MEDIAMTX_API_PASS
        if user:
            self._auth = (user, password or "")
        self._timeout_sec = timeout_sec

    async def ensure_path(
        self,
        path_name: str,
        *,
        source_url: Optional[str] = None,
        source_on_demand: bool = False,
    ) -> bool:
        path = validate_stream_path(path_name)
        payload: Dict[str, Any] = {"source": source_url or "publisher"}
        if source_url:
            validate_rtsp_url(source_url)
            if source_on_demand:
                payload["sourceOnDemand"] = True
        endpoint = f"{self._api_url}/v3/config/paths/add/{quote(path, safe='')}"
        async with httpx.AsyncClient(timeout=self._timeout_sec, auth=self._auth) as client:
            response = await client.post(endpoint, json=payload)
        if response.is_success:
            return True
        body = response.text.lower()
        if response.status_code in {400, 409} and ("already" in body or "exists" in body):
            return False
        raise RuntimeError(
            f"MediaMTX path create failed for '{path}': HTTP {response.status_code} {response.text.strip()}"
        )

    async def delete_path(self, path_name: str) -> bool:
        path = validate_stream_path(path_name)
        endpoint = f"{self._api_url}/v3/config/paths/delete/{quote(path, safe='')}"
        async with httpx.AsyncClient(timeout=self._timeout_sec, auth=self._auth) as client:
            response = await client.delete(endpoint)
        if response.is_success:
            return True
        if response.status_code == 404:
            return False
        raise RuntimeError(
            f"MediaMTX path delete failed for '{path}': HTTP {response.status_code} {response.text.strip()}"
        )

    async def list_paths(self) -> List[Dict[str, Any]]:
        endpoint = f"{self._api_url}/v3/paths/list"
        async with httpx.AsyncClient(timeout=self._timeout_sec, auth=self._auth) as client:
            response = await client.get(endpoint)
        if not response.is_success:
            raise RuntimeError(
                f"MediaMTX path list failed: HTTP {response.status_code} {response.text.strip()}"
            )
        data = response.json()
        items = data.get("items") if isinstance(data, dict) else None
        if isinstance(items, list):
            return items
        if isinstance(data, list):
            return data
        return []


@dataclass
class _LiveRuntime:
    session_id: str
    mission_id: str
    robot_id: str
    path_name: str
    rtsp_url: str
    caption_fps: float
    stop_event: asyncio.Event
    task: asyncio.Task
    started_at: datetime
    status: str = "starting"
    error: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mission_id": self.mission_id,
            "robot_id": self.robot_id,
            "path_name": self.path_name,
            "rtsp_url": self.rtsp_url,
            "caption_fps": self.caption_fps,
            "started_at": self.started_at.isoformat(),
            "status": self.status,
            "error": self.error,
        }


class RealtimeStreamManager:
    """Tracks background RTSP caption tasks tied to realtime sessions."""

    def __init__(self, db_pool) -> None:
        self._db_pool = db_pool
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, _LiveRuntime] = {}

    async def start(
        self,
        *,
        session_id: str,
        mission_id: str,
        robot_id: str,
        path_name: str,
        caption_fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._db_pool is None:
            raise RuntimeError("database pool is not available for realtime stream analysis")
        rtsp_url = build_rtsp_stream_url(path_name)
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None and not existing.task.done():
                raise RuntimeError(f"realtime analysis already running for session {session_id}")
            stop_event = asyncio.Event()
            caption_rate = float(caption_fps if caption_fps is not None else settings.RTSP_CAPTION_FPS)
            runtime = _LiveRuntime(
                session_id=session_id,
                mission_id=mission_id,
                robot_id=robot_id,
                path_name=validate_stream_path(path_name),
                rtsp_url=rtsp_url,
                caption_fps=caption_rate,
                stop_event=stop_event,
                task=asyncio.create_task(asyncio.sleep(0)),
                started_at=datetime.now(timezone.utc),
            )
            runtime.task = asyncio.create_task(self._run(runtime))
            self._sessions[session_id] = runtime
            return runtime.snapshot()

    async def _run(self, runtime: _LiveRuntime) -> None:
        runtime.status = "running"
        try:
            captioner = RtspCaptioner(
                rtsp_url=runtime.rtsp_url,
                mission_id=runtime.mission_id,
                db_pool=self._db_pool,
                caption_fps=runtime.caption_fps,
            )
            await captioner.run(stop_event=runtime.stop_event)
        except asyncio.CancelledError:
            runtime.status = "stopped"
            raise
        except Exception as exc:
            runtime.status = "error"
            runtime.error = str(exc)
            logger.exception("Realtime stream analysis failed for %s: %s", runtime.session_id, exc)
        else:
            runtime.status = "stopped" if runtime.stop_event.is_set() else "completed"

    async def stop(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            runtime = self._sessions.get(session_id)
        if runtime is None:
            raise LookupError("live stream not found")
        runtime.stop_event.set()
        try:
            await asyncio.wait_for(runtime.task, timeout=5.0)
        except asyncio.TimeoutError:
            runtime.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runtime.task
            runtime.status = "stopped"
        return runtime.snapshot()

    async def shutdown(self) -> None:
        async with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            with contextlib.suppress(Exception):
                await self.stop(session_id)

    async def get(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                raise LookupError("live stream not found")
            return runtime.snapshot()

    async def list(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [runtime.snapshot() for runtime in self._sessions.values()]
