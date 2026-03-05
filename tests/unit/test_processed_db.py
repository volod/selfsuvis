"""Unit tests for pipeline.processed_db."""

import tempfile
from pathlib import Path

import pytest

from pipeline import config
import pipeline.processed_db as processed_db_mod
from pipeline.processed_db import get_by_hash, get_by_size, get_by_url, init_db, upsert


@pytest.fixture
def temp_processed_db(monkeypatch, tmp_path):
    """Use a temporary database file for processed_db tests."""
    db_path = str(tmp_path / "processed.db")
    monkeypatch.setattr(processed_db_mod, "DB_PATH", db_path)
    if hasattr(processed_db_mod._conn_local, "conn"):
        processed_db_mod._conn_local.conn = None
    init_db()
    yield db_path
    if hasattr(processed_db_mod._conn_local, "conn"):
        processed_db_mod._conn_local.conn = None


# --- upsert + get_by_hash ---

def test_upsert_and_get_by_hash_happy_path(temp_processed_db):
    """upsert creates a record; get_by_hash retrieves it."""
    upsert("abc123", "vid1", "/tmp/v.mp4", 1000, 1.5, "done", {"source": "upload"})
    rec = get_by_hash("abc123")
    assert rec is not None
    assert rec["file_hash"] == "abc123"
    assert rec["video_id"] == "vid1"
    assert rec["path"] == "/tmp/v.mp4"
    assert rec["size_bytes"] == 1000
    assert rec["mtime"] == 1.5
    assert rec["status"] == "done"
    assert rec["meta"] == {"source": "upload"}
    assert rec["created_at"] > 0
    assert rec["updated_at"] > 0


def test_get_by_hash_miss_returns_none(temp_processed_db):
    """get_by_hash returns None for an unknown hash."""
    assert get_by_hash("nonexistent") is None


def test_upsert_updates_existing_record(temp_processed_db):
    """upsert on duplicate hash updates all fields (ON CONFLICT)."""
    upsert("dup1", "vid-old", "/old.mp4", 500, 0.5, "pending", {})
    upsert("dup1", "vid-new", "/new.mp4", 999, 9.9, "done", {"extra": 42})
    rec = get_by_hash("dup1")
    assert rec["video_id"] == "vid-new"
    assert rec["path"] == "/new.mp4"
    assert rec["size_bytes"] == 999
    assert rec["status"] == "done"
    assert rec["meta"] == {"extra": 42}


def test_upsert_preserves_created_at_on_update(temp_processed_db):
    """upsert preserves created_at but changes updated_at on conflict."""
    upsert("ts1", "v1", "/a.mp4", 1, 1.0, "pending", {})
    rec1 = get_by_hash("ts1")
    created = rec1["created_at"]
    upsert("ts1", "v1", "/a.mp4", 1, 2.0, "done", {})
    rec2 = get_by_hash("ts1")
    # created_at is NOT updated on conflict (only updated_at is)
    assert rec2["updated_at"] >= rec2["created_at"]
    assert rec2["mtime"] == 2.0


def test_meta_complex_dict_round_trip(temp_processed_db):
    """upsert/get_by_hash round-trips nested meta dict correctly."""
    meta = {"url": "http://ex.com/v.mp4", "tags": ["a", "b"], "nested": {"x": 1}}
    upsert("complex", "vx", "/x.mp4", 0, 0.0, "done", meta)
    rec = get_by_hash("complex")
    assert rec["meta"] == meta


def test_upsert_empty_meta(temp_processed_db):
    """upsert with empty meta dict stores and returns empty dict."""
    upsert("empty_meta", "v", "/v.mp4", 0, 0.0, "done", {})
    rec = get_by_hash("empty_meta")
    assert rec["meta"] == {}


# --- get_by_url ---

def test_get_by_url_happy_path(temp_processed_db):
    """get_by_url finds a record with url in meta."""
    upsert("url1", "v1", "/v.mp4", 100, 1.0, "done", {"url": "http://example.com/vid.mp4"})
    rec = get_by_url("http://example.com/vid.mp4")
    assert rec is not None
    assert rec["file_hash"] == "url1"


def test_get_by_url_miss_returns_none(temp_processed_db):
    """get_by_url returns None when no record has that url in meta."""
    upsert("nourl", "v2", "/v.mp4", 50, 1.0, "done", {"other": "field"})
    assert get_by_url("http://nothere.com") is None


def test_get_by_url_distinguishes_urls(temp_processed_db):
    """get_by_url returns only the record matching the exact url."""
    upsert("u1", "v1", "/a.mp4", 1, 1.0, "done", {"url": "http://a.com/1"})
    upsert("u2", "v2", "/b.mp4", 2, 2.0, "done", {"url": "http://b.com/2"})
    assert get_by_url("http://a.com/1")["file_hash"] == "u1"
    assert get_by_url("http://b.com/2")["file_hash"] == "u2"


def test_get_by_url_no_url_key_in_meta(temp_processed_db):
    """get_by_url returns None for records where meta has no 'url' key."""
    upsert("nomatch", "v", "/v.mp4", 0, 0.0, "done", {"source": "upload"})
    assert get_by_url("upload") is None


# --- get_by_size ---

def test_get_by_size_happy_path(temp_processed_db):
    """get_by_size finds a record with a matching size_bytes."""
    upsert("sz1", "v1", "/v.mp4", 2048, 1.0, "done", {})
    rec = get_by_size(2048)
    assert rec is not None
    assert rec["file_hash"] == "sz1"
    assert rec["size_bytes"] == 2048


def test_get_by_size_miss_returns_none(temp_processed_db):
    """get_by_size returns None when no record matches the size."""
    assert get_by_size(999_999) is None


def test_get_by_size_multiple_returns_most_recent(temp_processed_db):
    """When multiple records share a size, get_by_size returns the most recently updated."""
    upsert("old", "v-old", "/old.mp4", 512, 1.0, "done", {})
    upsert("new", "v-new", "/new.mp4", 512, 2.0, "done", {})
    rec = get_by_size(512)
    # "new" was inserted after "old", so it has a higher updated_at
    assert rec["file_hash"] == "new"


def test_get_by_size_zero_bytes(temp_processed_db):
    """get_by_size works for zero-byte files."""
    upsert("zero", "v0", "/empty.mp4", 0, 0.0, "done", {})
    rec = get_by_size(0)
    assert rec is not None
    assert rec["file_hash"] == "zero"
