"""Unit tests for pipeline.frame_extractor.

Requires a working cv2 installation.  Skipped automatically when cv2 is not
importable (e.g. NumPy 2.x / OpenCV binary incompatibility in the test venv).
Run the full suite including these tests with: make test-unit  (Docker stack)
or after installing a compatible OpenCV wheel.
"""
import os

import pytest

cv2 = pytest.importorskip("cv2", reason="cv2 not available (NumPy 2.x incompatibility)")

from pipeline.media.frames import extract_frames_fixed  # noqa: E402


def test_extract_frames_fixed_from_video_test(tmp_path):
    video_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "video_test",
        "mixkit-curved-highway-going-down-a-hill-40848-4k.mp4",
    )
    if not os.path.exists(video_path):
        pytest.skip("test video file not present")

    out_dir = tmp_path / "frames"
    frames = extract_frames_fixed(video_path, str(out_dir), interval_sec=1.0, max_frames=2)

    assert len(frames) == 2
    for rec in frames:
        assert os.path.exists(rec.path)
        assert rec.path.endswith(".png")
