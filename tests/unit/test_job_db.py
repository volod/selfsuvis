"""Unit tests for pipeline.job_db."""

import tempfile
from pathlib import Path

import pytest

from pipeline import config
from pipeline.job_db import create_job, fetch_and_claim_next_pending, fetch_job, init_db, update_job


@pytest.fixture
def temp_job_db(monkeypatch):
    """Use a temporary database file for job_db tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    monkeypatch.setattr(config.settings, "JOB_DB_PATH", path)
    # Reset connection so new path is used
    import pipeline.job_db as job_db_mod
    if hasattr(job_db_mod._conn_local, "conn"):
        job_db_mod._conn_local.conn = None
    init_db()
    yield path
    Path(path).unlink(missing_ok=True)


def test_update_job_whitelist_only_allowed_columns(temp_job_db):
    """update_job only updates whitelisted columns; unknown columns are ignored."""
    create_job("job1", {"video_id": "v1", "video_path": "/tmp/x.mp4"})
    # Try to update with a non-whitelisted column (e.g. malicious "id" or arbitrary key)
    update_job("job1", status="running", started_at=123.0, malicious_col="ignored")
    job = fetch_job("job1")
    assert job["status"] == "running"
    assert job["started_at"] == 123.0
    assert "malicious_col" not in job


def test_update_job_progress_and_payload(temp_job_db):
    """update_job correctly serializes progress and payload JSON."""
    create_job("job2", {"video_id": "v2"})
    update_job("job2", progress={"frames": 10, "tiles": 5}, finished_at=456.0)
    job = fetch_job("job2")
    assert job["progress"] == {"frames": 10, "tiles": 5}
    assert job["finished_at"] == 456.0


def test_update_job_empty_kwargs_no_op(temp_job_db):
    """update_job with no whitelisted kwargs does nothing."""
    create_job("job3", {"video_id": "v3"})
    update_job("job3", unknown_key="value")
    job = fetch_job("job3")
    assert job["status"] == "pending"


def test_fetch_and_claim_next_pending(temp_job_db):
    """fetch_and_claim_next_pending atomically claims the next pending job."""
    create_job("a", {"video_id": "v1"})
    create_job("b", {"video_id": "v2"})
    job1 = fetch_and_claim_next_pending()
    assert job1 is not None
    assert job1["id"] == "a"
    assert job1["status"] == "running"
    assert job1["started_at"] is not None
    assert fetch_job("a")["status"] == "running"
    job2 = fetch_and_claim_next_pending()
    assert job2 is not None
    assert job2["id"] == "b"
    assert job2["status"] == "running"
    none_job = fetch_and_claim_next_pending()
    assert none_job is None
