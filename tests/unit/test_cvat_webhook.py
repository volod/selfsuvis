"""Unit tests for app.routers.cvat — webhook receiver and admin endpoints."""
import hashlib
import hmac
import json
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.cvat import (
    _verify_cvat_signature,
    cvat_admin_router,
    webhook_router,
)

_app = FastAPI()
_app.include_router(webhook_router)
_app.include_router(cvat_admin_router)
_client = TestClient(_app)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _webhook_payload(event: str, state: str, task_id: int = 1, job_id: int = 10) -> Dict[str, Any]:
    if event == "update:job":
        return {"event": event, "job": {"id": job_id, "task_id": task_id, "state": state}}
    if event == "update:task":
        return {"event": event, "task": {"id": task_id, "state": state}}
    return {"event": event}


# ── _verify_cvat_signature ────────────────────────────────────────────────────

def test_verify_signature_no_secret_always_passes():
    with patch("app.routers.cvat.settings") as m:
        m.CVAT_WEBHOOK_SECRET = ""
        assert _verify_cvat_signature(b"body", "anything") is True


def test_verify_signature_correct():
    secret = "mysecret"
    body = b'{"event":"update:job"}'
    sig = _sign(body, secret)
    with patch("app.routers.cvat.settings") as m:
        m.CVAT_WEBHOOK_SECRET = secret
        assert _verify_cvat_signature(body, sig) is True


def test_verify_signature_wrong():
    secret = "mysecret"
    body = b'{"event":"update:job"}'
    with patch("app.routers.cvat.settings") as m:
        m.CVAT_WEBHOOK_SECRET = secret
        assert _verify_cvat_signature(body, "badsignature") is False


# ── POST /webhook/cvat ────────────────────────────────────────────────────────

@patch("app.routers.cvat._mark_frames_annotated", new_callable=AsyncMock)
@patch("app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("app.routers.cvat.settings")
def test_webhook_job_completed_marks_frames(mock_settings, mock_frames, mock_mark):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_frames.return_value = ["f1", "f2", "f3"]
    mock_mark.return_value = 3

    body = json.dumps(_webhook_payload("update:job", "completed", task_id=5)).encode()
    resp = _client.post("/webhook/cvat", content=body, headers={"Content-Type": "application/json"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["annotated"] == 3
    mock_frames.assert_awaited_once_with(5)
    mock_mark.assert_awaited_once_with(["f1", "f2", "f3"])


@patch("app.routers.cvat._mark_frames_annotated", new_callable=AsyncMock)
@patch("app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("app.routers.cvat.settings")
def test_webhook_task_completed_marks_frames(mock_settings, mock_frames, mock_mark):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_frames.return_value = ["fa", "fb"]
    mock_mark.return_value = 2

    body = json.dumps(_webhook_payload("update:task", "completed", task_id=7)).encode()
    resp = _client.post("/webhook/cvat", content=body, headers={"Content-Type": "application/json"})

    assert resp.status_code == 200
    assert resp.json()["annotated"] == 2
    mock_frames.assert_awaited_once_with(7)


@patch("app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("app.routers.cvat.settings")
def test_webhook_job_not_completed_returns_zero(mock_settings, mock_frames):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_frames.return_value = []

    body = json.dumps(_webhook_payload("update:job", "in_progress", task_id=3)).encode()
    resp = _client.post("/webhook/cvat", content=body, headers={"Content-Type": "application/json"})

    assert resp.status_code == 200
    assert resp.json()["annotated"] == 0
    mock_frames.assert_not_awaited()


@patch("app.routers.cvat.settings")
def test_webhook_unknown_event_returns_ok(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    body = json.dumps({"event": "create:project"}).encode()
    resp = _client.post("/webhook/cvat", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["annotated"] == 0


@patch("app.routers.cvat.settings")
def test_webhook_invalid_signature_returns_400(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = "secret123"
    body = json.dumps(_webhook_payload("update:job", "completed")).encode()
    resp = _client.post(
        "/webhook/cvat",
        content=body,
        headers={"Content-Type": "application/json", "X-Hook-Secret": "wrongsig"},
    )
    assert resp.status_code == 400


@patch("app.routers.cvat._mark_frames_annotated", new_callable=AsyncMock)
@patch("app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("app.routers.cvat.settings")
def test_webhook_valid_signature_passes(mock_settings, mock_frames, mock_mark):
    secret = "supersecret"
    mock_settings.CVAT_WEBHOOK_SECRET = secret
    mock_frames.return_value = ["f1"]
    mock_mark.return_value = 1

    body = json.dumps(_webhook_payload("update:job", "completed", task_id=2)).encode()
    sig = _sign(body, secret)
    resp = _client.post(
        "/webhook/cvat",
        content=body,
        headers={"Content-Type": "application/json", "X-Hook-Secret": sig},
    )
    assert resp.status_code == 200
    assert resp.json()["annotated"] == 1


@patch("app.routers.cvat.settings")
def test_webhook_invalid_json_returns_400(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    resp = _client.post(
        "/webhook/cvat",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


@patch("app.routers.cvat._frames_for_cvat_task", new_callable=AsyncMock)
@patch("app.routers.cvat.settings")
def test_webhook_no_mapping_returns_zero(mock_settings, mock_frames):
    """Job completed but no task mapping registered — returns 0 annotated."""
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_frames.return_value = []

    body = json.dumps(_webhook_payload("update:job", "completed", task_id=99)).encode()
    resp = _client.post("/webhook/cvat", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["annotated"] == 0


# ── POST /admin/cvat/task ─────────────────────────────────────────────────────

@patch("app.routers.cvat.settings")
def test_register_task_empty_frame_ids_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    resp = _client.post(
        "/admin/cvat/task",
        json={"cvat_task_id": 1, "frame_ids": []},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 422


@patch("app.routers.cvat.settings")
def test_register_task_too_many_frames_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    resp = _client.post(
        "/admin/cvat/task",
        json={"cvat_task_id": 1, "frame_ids": [f"f{i}" for i in range(5001)]},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 422


@patch("app.routers.cvat.settings")
def test_register_task_no_db_returns_503(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = ""
    resp = _client.post(
        "/admin/cvat/task",
        json={"cvat_task_id": 1, "frame_ids": ["f1", "f2"]},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 503


# ── GET /admin/cvat/frames ────────────────────────────────────────────────────

@patch("app.routers.cvat.settings")
def test_frames_no_db_returns_empty(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = ""
    resp = _client.get("/admin/cvat/frames", headers={"X-API-Key": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["frames"] == []


@patch("app.routers.cvat.settings")
def test_frames_invalid_al_tag_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = "postgresql://x"
    resp = _client.get(
        "/admin/cvat/frames?al_tag=invalid",
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 422


@patch("app.routers.cvat.settings")
def test_frames_invalid_limit_returns_422(mock_settings):
    mock_settings.CVAT_WEBHOOK_SECRET = ""
    mock_settings.API_KEY = ""
    mock_settings.DATABASE_URL = "postgresql://x"
    resp = _client.get(
        "/admin/cvat/frames?limit=0",
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 422
