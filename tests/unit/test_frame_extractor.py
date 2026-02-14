import os

from pipeline.frame_extractor import extract_frames_fixed


def test_extract_frames_fixed_from_video_test(tmp_path):
    video_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "video_test",
        "mixkit-curved-highway-going-down-a-hill-40848-4k.mp4",
    )
    if not os.path.exists(video_path):
        return

    out_dir = tmp_path / "frames"
    frames = extract_frames_fixed(video_path, str(out_dir), interval_sec=1.0, max_frames=2)

    assert len(frames) == 2
    for rec in frames:
        assert os.path.exists(rec.path)
        assert rec.path.endswith(".png")
