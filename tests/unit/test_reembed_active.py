"""Unit tests for app.services.search._reembed_is_active and
GET /admin/reembed-status.

All DB calls are mocked — no live PostgreSQL required.
asyncpg is not installed in the test environment; it is stubbed in sys.modules.
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Stub asyncpg before any imports that use it ───────────────────────────────
if "asyncpg" not in sys.modules:
    _asyncpg = types.ModuleType("asyncpg")
    _asyncpg.connect = MagicMock()   # patch.object requires attribute to pre-exist
    sys.modules["asyncpg"] = _asyncpg

# ── Stub app.state (heavy deps) before importing app modules ──────────────────
if "app.state" not in sys.modules:
    _state_stub = MagicMock()
    _state_stub.dino_model = None
    _state_stub.store = MagicMock()
    sys.modules["app.state"] = _state_stub


# ── _reembed_is_active ─────────────────────────────────────────────────────────

class TestReembedIsActive:

    def _call(self, fetchrow_return, db_url="postgresql://fake/db"):
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=fetchrow_return)
        mock_conn.close = AsyncMock()

        asyncpg_mock = MagicMock()
        asyncpg_mock.connect = AsyncMock(return_value=mock_conn)

        from app.services import search as search_mod
        with patch.object(sys.modules["asyncpg"], "connect", asyncpg_mock.connect), \
             patch.object(search_mod.settings, "DATABASE_URL", db_url):
            return search_mod._reembed_is_active()

    def test_returns_true_when_reembed_running(self):
        result = self._call(fetchrow_return=MagicMock())  # truthy row
        assert result is True

    def test_returns_false_when_no_reembed_running(self):
        result = self._call(fetchrow_return=None)
        assert result is False

    def test_returns_false_when_no_database_url(self):
        result = self._call(fetchrow_return=MagicMock(), db_url="")
        assert result is False

    def test_returns_false_on_db_connect_error(self):
        """DB connection failure → fail-open (False), never raises."""
        from app.services import search as search_mod
        with patch.object(sys.modules["asyncpg"], "connect",
                          AsyncMock(side_effect=OSError("refused"))), \
             patch.object(search_mod.settings, "DATABASE_URL", "postgresql://fake/db"):
            result = search_mod._reembed_is_active()
        assert result is False

    def test_returns_false_on_fetchrow_error(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=RuntimeError("query failed"))
        mock_conn.close = AsyncMock()

        from app.services import search as search_mod
        with patch.object(sys.modules["asyncpg"], "connect",
                          AsyncMock(return_value=mock_conn)), \
             patch.object(search_mod.settings, "DATABASE_URL", "postgresql://fake/db"):
            result = search_mod._reembed_is_active()
        assert result is False

    def test_connection_closed_on_success(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.close = AsyncMock()

        from app.services import search as search_mod
        with patch.object(sys.modules["asyncpg"], "connect",
                          AsyncMock(return_value=mock_conn)), \
             patch.object(search_mod.settings, "DATABASE_URL", "postgresql://fake/db"):
            search_mod._reembed_is_active()

        mock_conn.close.assert_called_once()


# ── GET /admin/reembed-status ──────────────────────────────────────────────────

def _make_admin_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers.admin import router as admin_router
    app = FastAPI()
    app.include_router(admin_router, dependencies=[])
    return TestClient(app, raise_server_exceptions=True)


class TestReembedStatusEndpoint:

    def _get(self, mock_conn, db_url="postgresql://fake/db"):
        from app.routers import admin as admin_mod
        with patch.object(sys.modules["asyncpg"], "connect",
                          AsyncMock(return_value=mock_conn)), \
             patch.object(admin_mod.settings, "DATABASE_URL", db_url):
            client = _make_admin_client()
            return client.get("/admin/reembed-status")

    def test_returns_active_true_when_job_running(self):
        import json
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={
            "id": "job-abc",
            "progress_json": json.dumps({"frames_reembedded": 512}),
        })
        mock_conn.close = AsyncMock()

        resp = self._get(mock_conn)
        assert resp.status_code == 200
        body = resp.json()
        assert body["active"] is True
        assert body["job_id"] == "job-abc"
        assert body["frames_reembedded"] == 512

    def test_returns_active_false_when_no_job(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.close = AsyncMock()

        resp = self._get(mock_conn)
        assert resp.status_code == 200
        body = resp.json()
        assert body["active"] is False
        assert body["job_id"] is None
        assert body["frames_reembedded"] is None

    def test_returns_false_when_no_database_url(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_conn.close = AsyncMock()

        resp = self._get(mock_conn, db_url="")
        assert resp.status_code == 200
        assert resp.json()["active"] is False

    def test_returns_false_on_db_error(self):
        from app.routers import admin as admin_mod
        with patch.object(sys.modules["asyncpg"], "connect",
                          AsyncMock(side_effect=OSError("refused"))), \
             patch.object(admin_mod.settings, "DATABASE_URL", "postgresql://fake/db"):
            client = _make_admin_client()
            resp = client.get("/admin/reembed-status")

        assert resp.status_code == 200
        assert resp.json()["active"] is False
