"""Unit tests for pipeline.report_generator."""
import os

import pytest

from pipeline.workflows.reporting import generate_summary_html, write_mission_report


@pytest.fixture
def sample_frames():
    return [
        {
            "frame_path": "/data/frames/f1.jpg",
            "caption": "A rocky terrain",
            "al_tag": "needs_annotation",
            "al_score": 0.85,
            "t_sec": 10.0,
        },
        {
            "frame_path": "/data/frames/f2.jpg",
            "caption": "Clear sky",
            "al_tag": "none",
            "al_score": 0.1,
            "t_sec": 20.0,
        },
        {
            "frame_path": "/data/frames/f3.jpg",
            "caption": "Unusual structure",
            "al_tag": "novel",
            "al_score": 0.6,
            "t_sec": 30.0,
        },
    ]


# ── generate_summary_html ─────────────────────────────────────────────────────

def test_generate_html_contains_mission_id(sample_frames):
    out = generate_summary_html("mission-001", sample_frames)
    assert "mission-001" in out


def test_generate_html_contains_frame_count(sample_frames):
    out = generate_summary_html("m1", sample_frames)
    assert "Frames: 3" in out


def test_generate_html_contains_duration(sample_frames):
    out = generate_summary_html("m1", sample_frames)
    assert "30.0s" in out  # max t_sec is 30.0


def test_generate_html_al_tag_distribution(sample_frames):
    out = generate_summary_html("m1", sample_frames)
    assert "Needs annotation: 1" in out
    assert "Novel: 1" in out
    assert "None: 1" in out


def test_generate_html_badge_labels(sample_frames):
    out = generate_summary_html("m1", sample_frames)
    assert "ANNOTATE" in out
    assert "NOVEL" in out


def test_generate_html_contains_caption(sample_frames):
    out = generate_summary_html("m1", sample_frames)
    assert "rocky terrain" in out


def test_generate_html_escapes_xss():
    """HTML-special characters in captions are escaped — no raw <script> tags."""
    frames = [
        {
            "frame_path": "x.jpg",
            "caption": "<script>alert(1)</script>",
            "al_tag": "none",
            "al_score": 0.0,
            "t_sec": 1.0,
        }
    ]
    out = generate_summary_html("m1", frames)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_generate_html_escapes_mission_id():
    """mission_id with HTML chars is escaped."""
    out = generate_summary_html("<bad>", [])
    assert "<bad>" not in out
    assert "&lt;bad&gt;" in out


def test_generate_html_empty_frames():
    out = generate_summary_html("empty-mission", [])
    assert "empty-mission" in out
    assert "Frames: 0" in out
    assert "Duration: 0.0s" in out


def test_generate_html_sorted_by_al_score_descending(sample_frames):
    """Highest al_score frame (f1, score=0.85) appears before lowest (f2, score=0.1)."""
    out = generate_summary_html("m1", sample_frames)
    pos_f1 = out.find("rocky terrain")
    pos_f2 = out.find("Clear sky")
    assert pos_f1 < pos_f2, "Higher-score frame should appear first in the HTML"


def test_generate_html_caption_truncated():
    """Captions longer than 80 chars are truncated."""
    long_caption = "x" * 200
    frames = [
        {"frame_path": "f.jpg", "caption": long_caption, "al_tag": "none", "al_score": 0.0, "t_sec": 0.0}
    ]
    out = generate_summary_html("m1", frames)
    # The first 80 chars should appear, not the full 200
    assert "x" * 80 in out
    assert "x" * 81 not in out


def test_generate_html_none_al_tag_no_badge():
    """Frames with al_tag=none show no ANNOTATE or NOVEL badge."""
    frames = [
        {"frame_path": "f.jpg", "caption": "hello", "al_tag": "none", "al_score": 0.0, "t_sec": 0.0}
    ]
    out = generate_summary_html("m1", frames)
    assert "ANNOTATE" not in out
    assert "NOVEL" not in out


# ── write_mission_report ──────────────────────────────────────────────────────

def test_write_mission_report_creates_file(sample_frames, monkeypatch, tmp_path):
    from pipeline.core import config
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    path = write_mission_report("test-mission", sample_frames)
    assert os.path.isfile(path)
    assert path.endswith("summary.html")


def test_write_mission_report_returns_absolute_path(sample_frames, monkeypatch, tmp_path):
    from pipeline.core import config
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    path = write_mission_report("m1", sample_frames)
    assert os.path.isabs(path)


def test_write_mission_report_content(sample_frames, monkeypatch, tmp_path):
    from pipeline.core import config
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    path = write_mission_report("m99", sample_frames)
    content = open(path).read()
    assert "m99" in content
    assert "rocky terrain" in content


def test_write_mission_report_creates_nested_dirs(monkeypatch, tmp_path):
    """write_mission_report creates reports/{mission_id}/ if it doesn't exist."""
    from pipeline.core import config
    monkeypatch.setattr(config.settings, "DATA_DIR", str(tmp_path))
    path = write_mission_report("brand-new-mission", [])
    assert os.path.isdir(os.path.dirname(path))
