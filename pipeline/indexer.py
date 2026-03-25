import dataclasses
import itertools
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from qdrant_client.http import models as qmodels

from models.openclip_model import OpenCLIPEmbedder
from models.dino_model import DINOEmbedder
from pipeline.config import settings
from pipeline.ffmpeg_utils import extract_frames
from pipeline.heuristics import (
    downsample_gray,
    histogram_diff,
    mean_abs_diff,
    ssim_diff,
    phase_corr_align,
    frame_quality_ok,
    tile_quality_ok,
    edge_density,
)
from pipeline.dedup import dhash, PhashLRU
from pipeline.qdrant_utils import QdrantStore
from pipeline.recent_index import RecentEmbeddingIndex
from pipeline.utils import ensure_dir, stable_point_id, RateTimer
from pipeline.logging_utils import get_logger


@dataclass
class _IndexFrameState:
    """Mutable state carried across frame processing iterations."""

    prev_small: Optional[np.ndarray] = None
    last_kept_embed: Optional[np.ndarray] = None
    last_kept_time: float = -1.0
    last_kept_frame: Optional[np.ndarray] = None
    last_kept_small: Optional[np.ndarray] = None
    eff_fps: float = 0.0
    skip_step: int = 1
    segment_count: int = 0


class VideoIndexer:
    def __init__(self, enable_tiles: bool = True):
        self.logger = get_logger(__name__)
        self.enable_tiles = enable_tiles
        self.clip_model = OpenCLIPEmbedder()
        self.dino_model = None
        if settings.MODEL_NAME in {"dinov2", "dinov3"}:
            name = "dinov2_vitb14" if settings.MODEL_NAME == "dinov2" else "dinov3_vitb14"
            self.dino_model = DINOEmbedder(model_name=name)
        self.store = QdrantStore(clip_dim=self.clip_model.image_dim(), dino_dim=self._dino_dim())
        self.phash_lru = PhashLRU(settings.PHASH_LRU_SIZE, settings.PHASH_HAMMING_MAX)
        self.recent_index = RecentEmbeddingIndex(
            dim=self.clip_model.image_dim(),
            max_size=settings.DEDUP_RECENT_TILES,
            ttl_sec=settings.DEDUP_TTL_SEC,
        )

    def _dino_dim(self) -> Optional[int]:
        if self.dino_model is None:
            return None
        return self.dino_model.encode_images([Image.new("RGB", (224, 224))]).shape[1]

    # ── per-frame helpers ─────────────────────────────────────────────────────

    def _stabilize(self, prev_small: np.ndarray, small: np.ndarray) -> np.ndarray:
        """Return `small` aligned to `prev_small` via phase correlation, or `small` unchanged."""
        if not settings.STAB_ENABLE:
            return small
        aligned, dx, dy, resp = phase_corr_align(
            prev_small.astype(np.float32), small.astype(np.float32)
        )
        if (
            resp >= settings.PHASECORR_MIN_RESPONSE
            and abs(dx) <= settings.STAB_MAX_SHIFT
            and abs(dy) <= settings.STAB_MAX_SHIFT
        ):
            return aligned.astype(np.uint8)
        return small

    def _update_adaptive_fps(self, eff_fps: float, motion: float) -> Tuple[float, int]:
        """Adjust effective FPS based on motion level. Returns (new_eff_fps, skip_step)."""
        if motion < settings.MOTION_LOW:
            eff_fps = max(settings.SAMPLE_FPS_MIN, eff_fps * 0.5)
        elif motion > settings.MOTION_HIGH:
            eff_fps = min(settings.SAMPLE_FPS_MAX, eff_fps * 1.5)
        skip_step = max(1, int(round(settings.SAMPLE_FPS_MAX / eff_fps)))
        return eff_fps, skip_step

    def _visual_diffs(
        self, last_kept_small: Optional[np.ndarray], small_for_diff: np.ndarray
    ) -> Tuple[float, float]:
        """Return (histogram_diff, ssim_diff) vs last kept frame, or (0, 0) if none."""
        if last_kept_small is None:
            return 0.0, 0.0
        hist_d = histogram_diff(last_kept_small, small_for_diff)
        try:
            ssim_d = ssim_diff(last_kept_small, small_for_diff)
        except (ValueError, TypeError):
            ssim_d = 0.0
        return hist_d, ssim_d

    def _should_keep(self, hist_d: float, ssim_d: float, drift: float, time_gap: float) -> bool:
        """Return True if the frame is sufficiently different from the last kept frame."""
        return (
            hist_d > settings.HIST_THRESH
            or ssim_d > settings.HIST_THRESH
            or drift > settings.EMBED_DRIFT_THRESH
            or time_gap > settings.MAX_GAP_SEC
        )

    def _build_frame_point(
        self,
        video_id: str,
        segment_id: int,
        t_sec: float,
        frame_path: str,
        frame_pil: Image.Image,
        clip_embed: np.ndarray,
        mission_id: Optional[str] = None,
        gps: Optional[Dict[str, float]] = None,
        enu: Optional[Dict[str, float]] = None,
        robot_id: Optional[str] = None,
        global_map_id: Optional[int] = None,
    ) -> qmodels.PointStruct:
        """Build a Qdrant point for one keyframe."""
        vectors: Dict[str, Any] = {"clip": clip_embed.tolist()}
        if self.dino_model:
            vectors["dino"] = self.dino_model.encode_images([frame_pil], batch_size=1)[0].tolist()
        payload: Dict[str, Any] = {
            "type": "frame",
            "video_id": video_id,
            "segment_id": segment_id,
            "t_sec": t_sec,
            "frame_path": frame_path,
        }
        if mission_id is not None:
            payload["mission_id"] = mission_id
        if robot_id is not None:
            payload["robot_id"] = robot_id
        if gps is not None:
            payload["gps"] = gps
        if enu is not None:
            payload["enu"] = enu
        if global_map_id is not None:
            payload["global_map_id"] = global_map_id
        # Model version provenance: lets queries be traced back to the model that
        # produced the embedding.  Stored as-is; "base" means the pretrained backbone.
        payload["model_version_id"] = settings.MODEL_VERSION_ID
        return qmodels.PointStruct(
            id=stable_point_id(video_id, segment_id, int(t_sec * 1000), "frame"),
            vector=vectors,
            payload=payload,
        )

    # ── main frame processing ─────────────────────────────────────────────────

    def _process_frame(
        self,
        video_id: str,
        frame_path: str,
        t_sec: float,
        frame: np.ndarray,
        state: "_IndexFrameState",
        cell_state: Dict[Tuple[int, int], Tuple[float, float]],
        mission_id: Optional[str] = None,
        gps: Optional[Dict[str, float]] = None,
        enu: Optional[Dict[str, float]] = None,
        robot_id: Optional[str] = None,
        global_map_id: Optional[int] = None,
    ) -> Tuple[Optional[Tuple[Dict[str, Any], List[qmodels.PointStruct], int]], "_IndexFrameState"]:
        """Process one frame. Returns (result, new_state); result is None if frame is skipped."""
        small = downsample_gray(frame, settings.STAB_SIZE)
        prev_small = state.prev_small if state.prev_small is not None else small
        small_for_diff = self._stabilize(prev_small, small)

        motion = mean_abs_diff(prev_small, small_for_diff)
        eff_fps, skip_step = self._update_adaptive_fps(state.eff_fps, motion)
        new_state = dataclasses.replace(state, prev_small=small, eff_fps=eff_fps, skip_step=skip_step)

        if not frame_quality_ok(frame):
            return None, new_state

        hist_d, ssim_d = self._visual_diffs(state.last_kept_small, small_for_diff)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embed = self.clip_model.encode_images([frame_pil], batch_size=1)[0]
        drift = float(1.0 - np.dot(state.last_kept_embed, embed)) if state.last_kept_embed is not None else 1.0
        time_gap = t_sec - state.last_kept_time

        if not self._should_keep(hist_d, ssim_d, drift, time_gap):
            return None, new_state

        segment_id = state.segment_count
        segment_dict = {
            "segment_id": segment_id,
            "start_t": t_sec,
            "end_t": t_sec,
            "rep_frame_t": t_sec,
            "rep_frame_path": frame_path,
        }
        frame_point = self._build_frame_point(
            video_id, segment_id, t_sec, frame_path, frame_pil, embed,
            mission_id=mission_id, gps=gps, enu=enu, robot_id=robot_id, global_map_id=global_map_id,
        )

        tile_points: List[qmodels.PointStruct] = []
        tile_count = 0
        if self.enable_tiles or drift > settings.TILE_INDEX_IF_EMBED_DRIFT_GT:
            tile_points, tile_count = self._index_tiles(
                frame, frame_path, video_id, segment_id, t_sec, embed, cell_state,
                mission_id=mission_id, robot_id=robot_id, global_map_id=global_map_id,
            )

        new_state = dataclasses.replace(
            new_state,
            last_kept_embed=embed,
            last_kept_time=t_sec,
            last_kept_frame=frame,
            last_kept_small=downsample_gray(frame, settings.STAB_SIZE),
            segment_count=segment_id + 1,
        )
        return (segment_dict, [frame_point] + tile_points, 1 + tile_count), new_state

    def index_video(
        self,
        video_path: str,
        video_id: str,
        mission_id: Optional[str] = None,
        robot_id: Optional[str] = None,
        site_enu_origin=None,
        global_map_id: Optional[int] = None,
        progress_cb=None,
    ) -> Dict[str, Any]:
        ensure_dir(settings.VIDEOS_DIR)
        self.logger.info("Indexing video_id=%s path=%s", video_id, video_path)
        dst_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
        if os.path.abspath(video_path) != os.path.abspath(dst_path):
            shutil.copy(video_path, dst_path)

        effective_mission_id = mission_id or video_id
        effective_robot_id = robot_id or settings.ROBOT_ID

        frames = extract_frames(dst_path, video_id)
        if not frames:
            return {"segments": 0, "tiles": 0, "frames": 0}

        # Pre-compute GPS + ENU for every extracted frame
        gps_lookup: Dict[float, Optional[Dict[str, float]]] = {}
        enu_lookup: Dict[float, Optional[Dict[str, float]]] = {}
        try:
            from pipeline.gps_extractor import extract_gps
            from pipeline.gps_registration import gps_to_enu
            timestamps_ms = [t * 1000.0 for _, t in frames]
            gps_list = extract_gps(dst_path, timestamps_ms)
            # Use the site's canonical ENU origin when provided (multi-site ENU support).
            # Fall back to the mission's own first-frame origin for backwards compatibility.
            enu_origin = site_enu_origin
            if enu_origin is None:
                for g in gps_list:
                    if g is not None:
                        enu_origin = (g["lat"], g["lon"], g["alt"])
                        break
            for (_, t_sec), g in zip(frames, gps_list):
                if g is not None:
                    gps_lookup[t_sec] = {"lat": g["lat"], "lon": g["lon"], "alt": g["alt"]}
                    if enu_origin is not None:
                        tx, ty, tz = gps_to_enu(g["lat"], g["lon"], g["alt"], *enu_origin)
                        enu_lookup[t_sec] = {"tx": float(tx), "ty": float(ty), "tz": float(tz)}
        except Exception:
            self.logger.debug("GPS extraction unavailable or failed for video_id=%s", video_id)

        eff_fps = settings.SAMPLE_FPS_BASE
        state = _IndexFrameState(
            eff_fps=eff_fps,
            skip_step=max(1, int(round(settings.SAMPLE_FPS_MAX / eff_fps))),
        )
        segments: List[Dict[str, Any]] = []
        points: List[qmodels.PointStruct] = []
        cell_state: Dict[Tuple[int, int], Tuple[float, float]] = {}
        tiles_indexed = 0
        frames_indexed = 0
        frame_timer = RateTimer()
        embed_timer = RateTimer()

        idx = 0
        while idx < len(frames):
            frame_path, t_sec = frames[idx]
            idx += state.skip_step

            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame_timer.tick()

            result, state = self._process_frame(
                video_id, frame_path, t_sec, frame, state, cell_state,
                mission_id=effective_mission_id,
                gps=gps_lookup.get(t_sec),
                enu=enu_lookup.get(t_sec),
                robot_id=effective_robot_id,
                global_map_id=global_map_id,
            )
            embed_timer.tick()

            if result is None:
                continue

            segment_dict, frame_and_tile_points, count = result
            segments.append(segment_dict)
            points.extend(frame_and_tile_points)
            frames_indexed += 1
            tiles_indexed += count - 1

            if len(points) >= 128:
                self.store.upsert_points(points)
                points = []

            if progress_cb:
                progress_cb({
                    "frames_processed": frame_timer.count,
                    "segments_found": len(segments),
                    "frames_indexed": frames_indexed,
                    "tiles_indexed": tiles_indexed,
                    "frame_fps": frame_timer.rate(),
                    "embed_fps": embed_timer.rate(),
                })

        if points:
            self.store.upsert_points(points)

        self.logger.info(
            "Indexing complete video_id=%s segments=%s frames=%s tiles=%s",
            video_id, len(segments), frames_indexed, tiles_indexed,
        )
        return {"segments": len(segments), "tiles": tiles_indexed, "frames": frames_indexed}

    def _index_tiles(
        self,
        frame: np.ndarray,
        frame_path: str,
        video_id: str,
        segment_id: int,
        t_sec: float,
        segment_embed: np.ndarray,
        cell_state: Dict[Tuple[int, int], Tuple[float, float]],
        mission_id: Optional[str] = None,
        robot_id: Optional[str] = None,
        global_map_id: Optional[int] = None,
    ) -> Tuple[List[qmodels.PointStruct], int]:
        h, w, _ = frame.shape
        tile_points: List[qmodels.PointStruct] = []
        count = 0

        ys = range(0, h - settings.TILE_SIZE + 1, settings.STRIDE)
        xs = range(0, w - settings.TILE_SIZE + 1, settings.STRIDE)
        for y, x in itertools.product(ys, xs):
            if count >= settings.MAX_TILES_PER_SEGMENT:
                break

            tile = frame[y : y + settings.TILE_SIZE, x : x + settings.TILE_SIZE]
            if not tile_quality_ok(tile):
                continue

            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            ph = dhash(gray)
            if self.phash_lru.near_duplicate(ph):
                continue

            cell_key = (x // settings.CELL_SIZE, y // settings.CELL_SIZE)
            score = float(edge_density(gray) * 1000.0 + gray.std())
            last = cell_state.get(cell_key)
            if last and (t_sec - last[0]) <= settings.CELL_WINDOW_SEC and score <= last[1]:
                continue
            cell_state[cell_key] = (t_sec, score)

            tile_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            clip_vec = self.clip_model.encode_images([tile_pil], batch_size=1)[0]
            if self.recent_index.max_cosine(clip_vec) > settings.DEDUP_COS_SIM_THRESH:
                continue

            # Passed all filters — commit to dedup indices and write to disk
            self.phash_lru.add(ph)
            self.recent_index.add(np.expand_dims(clip_vec, axis=0))

            tile_path = os.path.join(
                settings.TILES_DIR, video_id, str(segment_id), f"tile_{int(t_sec*1000)}_{x}_{y}.jpg"
            )
            ensure_dir(os.path.dirname(tile_path))
            cv2.imwrite(tile_path, tile)

            vectors: Dict[str, Any] = {"clip": clip_vec.tolist()}
            if self.dino_model:
                vectors["dino"] = self.dino_model.encode_images([tile_pil], batch_size=1)[0].tolist()

            tile_payload: Dict[str, Any] = {
                "type": "tile",
                "video_id": video_id,
                "segment_id": segment_id,
                "t_sec": t_sec,
                "frame_path": frame_path,
                "tile_path": tile_path,
                "x": x, "y": y,
                "w": settings.TILE_SIZE,
                "h": settings.TILE_SIZE,
            }
            if mission_id is not None:
                tile_payload["mission_id"] = mission_id
            if robot_id is not None:
                tile_payload["robot_id"] = robot_id
            if global_map_id is not None:
                tile_payload["global_map_id"] = global_map_id
            tile_points.append(qmodels.PointStruct(
                id=stable_point_id(video_id, segment_id, int(t_sec * 1000), "tile", x, y),
                vector=vectors,
                payload=tile_payload,
            ))
            count += 1

        return tile_points, count
