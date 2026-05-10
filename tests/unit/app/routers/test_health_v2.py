"""Unit tests for enhanced /health endpoint."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_app(pool=None):
    from fastapi import FastAPI

    from selfsuvis.app.routers.health import router

    app = FastAPI()
    app.include_router(router)
    app.state.db_pool = pool
    app.state.sse_subscribers = {}
    return app


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.ping = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.llen = AsyncMock(return_value=0)
    r.aclose = AsyncMock()
    return r


def test_health_postgres_ok(mock_redis):
    pool = AsyncMock()
    from contextlib import asynccontextmanager

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)

    @asynccontextmanager
    async def _acq():
        yield conn

    pool.acquire = _acq

    with (
        patch("selfsuvis.app.routers.health.settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
        patch("selfsuvis.app.state.app_state") as mock_state,
    ):
        ms.REDIS_URL = "redis://localhost"
        ms.CORRELATOR_ENABLED = False
        ms.DRONE_AUDIO_MODEL_PATH = ""
        ms.DRONE_AUDIO_WATCH_DIR = ""
        mock_state.store.client.get_collections = MagicMock()

        app = _make_app(pool)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["postgres"] == "ok"


def test_health_correlator_stale():
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=35)).isoformat()

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.get = AsyncMock(return_value=stale_ts.encode())
    mock_redis.llen = AsyncMock(return_value=0)
    mock_redis.aclose = AsyncMock()

    pool = AsyncMock()
    from contextlib import asynccontextmanager

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)

    @asynccontextmanager
    async def _acq():
        yield conn

    pool.acquire = _acq

    with (
        patch("selfsuvis.app.routers.health.settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
        patch("selfsuvis.app.state.app_state") as mock_state,
    ):
        ms.REDIS_URL = "redis://localhost"
        ms.CORRELATOR_ENABLED = True
        ms.DRONE_AUDIO_MODEL_PATH = ""
        ms.DRONE_AUDIO_WATCH_DIR = ""
        mock_state.store.client.get_collections = MagicMock()

        app = _make_app(pool)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["status"] == "degraded"


def test_health_dlq_depth_nonzero():
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.llen = AsyncMock(return_value=3)
    mock_redis.aclose = AsyncMock()

    pool = AsyncMock()
    from contextlib import asynccontextmanager

    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)

    @asynccontextmanager
    async def _acq():
        yield conn

    pool.acquire = _acq

    with (
        patch("selfsuvis.app.routers.health.settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
        patch("selfsuvis.app.state.app_state") as mock_state,
    ):
        ms.REDIS_URL = "redis://localhost"
        ms.CORRELATOR_ENABLED = False
        ms.DRONE_AUDIO_MODEL_PATH = ""
        ms.DRONE_AUDIO_WATCH_DIR = ""
        mock_state.store.client.get_collections = MagicMock()

        app = _make_app(pool)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["status"] == "degraded"
        assert resp.json()["dlq_depth"] == 3


def test_health_redis_unreachable():

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=ConnectionError("refused"))

    with (
        patch("selfsuvis.app.routers.health.settings") as ms,
        patch("redis.asyncio.from_url", return_value=mock_redis),
        patch("selfsuvis.app.state.app_state") as mock_state,
    ):
        ms.REDIS_URL = "redis://localhost"
        ms.CORRELATOR_ENABLED = False
        ms.DRONE_AUDIO_MODEL_PATH = ""
        ms.DRONE_AUDIO_WATCH_DIR = ""
        mock_state.store.client.get_collections = MagicMock()

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["redis"] == "error"
        assert resp.json()["status"] == "down"
        assert resp.status_code == 503
