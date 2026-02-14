import os
from dataclasses import dataclass
from typing import List, Optional

import cv2

from pipeline.heuristics import downsample_gray, mean_abs_diff, histogram_diff, ssim_diff
from pipeline.logging_utils import get_logger
from pipeline.utils import ensure_dir


@dataclass
class FrameRecord:
    path: str
    t_sec: float
    index: int
    width: int
    height: int


def _save_png(frame_bgr, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def extract_frames_fixed(
    video_path: str,
    out_dir: str,
    interval_sec: float = 1.0,
    max_frames: Optional[int] = None,
) -> List[FrameRecord]:
    logger = get_logger(__name__)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    frames: List[FrameRecord] = []
    t_sec = 0.0
    idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h, w = frame.shape[:2]
        out_path = os.path.join(out_dir, f"frame_{idx:08d}_{int(t_sec * 1000):010d}ms.png")
        _save_png(frame, out_path)
        frames.append(FrameRecord(path=out_path, t_sec=t_sec, index=idx, width=w, height=h))
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
        t_sec += interval_sec

    cap.release()
    logger.info("Extracted %s frames from %s", len(frames), video_path)
    return frames


def extract_frames_adaptive(
    video_path: str,
    out_dir: str,
    min_interval_sec: float = 1.0,
    max_gap_sec: float = 10.0,
    diff_threshold: float = 0.12,
    probe_fps: float = 5.0,
) -> List[FrameRecord]:
    logger = get_logger(__name__)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1
    if fps > 0 and probe_fps > 0:
        step = max(1, int(round(fps / probe_fps)))

    frames: List[FrameRecord] = []
    idx = 0
    frame_idx = 0
    last_kept_small = None
    last_kept_t = -1e9

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_sec <= 0.0 and fps > 0:
            t_sec = frame_idx / fps

        small = downsample_gray(frame, 64)
        keep = False
        if last_kept_small is None:
            keep = True
        else:
            diff = mean_abs_diff(last_kept_small, small)
            try:
                hist_d = histogram_diff(last_kept_small, small)
                ssim_d = ssim_diff(last_kept_small, small)
            except Exception:
                hist_d = 0.0
                ssim_d = 0.0
            time_gap = t_sec - last_kept_t
            if time_gap >= max_gap_sec:
                keep = True
            elif time_gap >= min_interval_sec and (diff >= diff_threshold or hist_d >= diff_threshold or ssim_d >= diff_threshold):
                keep = True

        if keep:
            h, w = frame.shape[:2]
            out_path = os.path.join(out_dir, f"frame_{idx:08d}_{int(t_sec * 1000):010d}ms.png")
            _save_png(frame, out_path)
            frames.append(FrameRecord(path=out_path, t_sec=t_sec, index=idx, width=w, height=h))
            last_kept_small = small
            last_kept_t = t_sec
            idx += 1

        frame_idx += 1

    cap.release()
    logger.info("Extracted %s adaptive frames from %s", len(frames), video_path)
    return frames


def extract_stream_frames(
    source,
    out_dir: str,
    min_interval_sec: float = 1.0,
    max_gap_sec: float = 10.0,
    diff_threshold: float = 0.12,
    probe_fps: float = 5.0,
    max_frames: Optional[int] = None,
) -> List[FrameRecord]:
    logger = get_logger(__name__)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open stream source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1
    if fps > 0 and probe_fps > 0:
        step = max(1, int(round(fps / probe_fps)))

    frames: List[FrameRecord] = []
    idx = 0
    frame_idx = 0
    last_kept_small = None
    last_kept_t = 0.0
    t_sec = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        if fps > 0:
            t_sec = frame_idx / fps
        else:
            t_sec += 1.0 / max(probe_fps, 1.0)

        small = downsample_gray(frame, 64)
        keep = False
        if last_kept_small is None:
            keep = True
        else:
            diff = mean_abs_diff(last_kept_small, small)
            try:
                hist_d = histogram_diff(last_kept_small, small)
                ssim_d = ssim_diff(last_kept_small, small)
            except Exception:
                hist_d = 0.0
                ssim_d = 0.0
            time_gap = t_sec - last_kept_t
            if time_gap >= max_gap_sec:
                keep = True
            elif time_gap >= min_interval_sec and (diff >= diff_threshold or hist_d >= diff_threshold or ssim_d >= diff_threshold):
                keep = True

        if keep:
            h, w = frame.shape[:2]
            out_path = os.path.join(out_dir, f"frame_{idx:08d}_{int(t_sec * 1000):010d}ms.png")
            _save_png(frame, out_path)
            frames.append(FrameRecord(path=out_path, t_sec=t_sec, index=idx, width=w, height=h))
            last_kept_small = small
            last_kept_t = t_sec
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info("Extracted %s stream frames", len(frames))
    return frames
