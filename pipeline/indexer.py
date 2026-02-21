import os
import shutil
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

    def index_video(self, video_path: str, video_id: str, progress_cb=None) -> Dict[str, Any]:
        ensure_dir(settings.VIDEOS_DIR)
        self.logger.info("Indexing video_id=%s path=%s", video_id, video_path)
        dst_path = os.path.join(settings.VIDEOS_DIR, f"{video_id}.mp4")
        if os.path.abspath(video_path) != os.path.abspath(dst_path):
            shutil.copy(video_path, dst_path)

        frames = extract_frames(dst_path, video_id)
        if not frames:
            return {"segments": 0, "tiles": 0, "frames": 0}

        fps = settings.SAMPLE_FPS_MAX
        eff_fps = settings.SAMPLE_FPS_BASE
        skip_step = max(1, int(round(fps / eff_fps)))

        segments = []
        points: List[qmodels.PointStruct] = []

        last_kept_embed = None
        last_kept_time = -1.0
        last_kept_frame = None
        last_kept_small = None

        prev_small = None

        frame_timer = RateTimer()
        embed_timer = RateTimer()

        tiles_indexed = 0
        frames_indexed = 0

        cell_state: Dict[Tuple[int, int], Tuple[float, float]] = {}

        idx = 0
        while idx < len(frames):
            frame_path, t_sec = frames[idx]
            idx += skip_step

            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            frame_timer.tick()

            small = downsample_gray(frame, settings.STAB_SIZE)
            if prev_small is None:
                prev_small = small

            if settings.STAB_ENABLE:
                aligned, dx, dy, resp = phase_corr_align(prev_small.astype(np.float32), small.astype(np.float32))
                if resp >= settings.PHASECORR_MIN_RESPONSE and abs(dx) <= settings.STAB_MAX_SHIFT and abs(dy) <= settings.STAB_MAX_SHIFT:
                    small_for_diff = aligned.astype(np.uint8)
                else:
                    small_for_diff = small
            else:
                small_for_diff = small

            motion = mean_abs_diff(prev_small, small_for_diff)
            if motion < settings.MOTION_LOW:
                eff_fps = max(settings.SAMPLE_FPS_MIN, eff_fps * 0.5)
            elif motion > settings.MOTION_HIGH:
                eff_fps = min(settings.SAMPLE_FPS_MAX, eff_fps * 1.5)
            skip_step = max(1, int(round(fps / eff_fps)))
            prev_small = small

            if not frame_quality_ok(frame):
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist_d = 0.0
            ssim_d = 0.0
            if last_kept_small is not None:
                hist_d = histogram_diff(last_kept_small, small_for_diff)
                try:
                    ssim_d = ssim_diff(last_kept_small, small_for_diff)
                except (ValueError, TypeError):
                    ssim_d = 0.0

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            embed = self.clip_model.encode_images([frame_pil], batch_size=1)[0]
            embed_timer.tick()

            drift = 1.0
            if last_kept_embed is not None:
                drift = float(1.0 - np.dot(last_kept_embed, embed))

            time_gap = t_sec - last_kept_time
            keep = (
                hist_d > settings.HIST_THRESH
                or ssim_d > settings.HIST_THRESH
                or drift > settings.EMBED_DRIFT_THRESH
                or time_gap > settings.MAX_GAP_SEC
            )

            if not keep:
                continue

            segment_id = len(segments)
            segments.append(
                {
                    "segment_id": segment_id,
                    "start_t": t_sec,
                    "end_t": t_sec,
                    "rep_frame_t": t_sec,
                    "rep_frame_path": frame_path,
                }
            )

            last_kept_embed = embed
            last_kept_time = t_sec
            last_kept_frame = frame
            last_kept_small = downsample_gray(frame, settings.STAB_SIZE)

            frame_id = stable_point_id(video_id, segment_id, int(t_sec * 1000), "frame")
            payload = {
                "type": "frame",
                "video_id": video_id,
                "segment_id": segment_id,
                "t_sec": t_sec,
                "frame_path": frame_path,
            }
            vectors = {"clip": embed.tolist()}
            if self.dino_model:
                dino_vec = self.dino_model.encode_images([frame_pil], batch_size=1)[0]
                vectors["dino"] = dino_vec.tolist()
            points.append(qmodels.PointStruct(id=frame_id, vector=vectors, payload=payload))
            frames_indexed += 1

            index_tiles = self.enable_tiles or drift > settings.TILE_INDEX_IF_EMBED_DRIFT_GT
            if index_tiles:
                tile_points, tile_count = self._index_tiles(
                    frame,
                    frame_path,
                    video_id,
                    segment_id,
                    t_sec,
                    embed,
                    cell_state,
                )
                points.extend(tile_points)
                tiles_indexed += tile_count

            if len(points) >= 128:
                self.store.upsert_points(points)
                points = []

            if progress_cb:
                progress_cb(
                    {
                        "frames_processed": frame_timer.count,
                        "segments_found": len(segments),
                        "frames_indexed": frames_indexed,
                        "tiles_indexed": tiles_indexed,
                        "frame_fps": frame_timer.rate(),
                        "embed_fps": embed_timer.rate(),
                    }
                )

        if points:
            self.store.upsert_points(points)

        self.logger.info(
            "Indexing complete video_id=%s segments=%s frames=%s tiles=%s",
            video_id,
            len(segments),
            frames_indexed,
            tiles_indexed,
        )
        return {
            "segments": len(segments),
            "tiles": tiles_indexed,
            "frames": frames_indexed,
        }

    def _index_tiles(
        self,
        frame: np.ndarray,
        frame_path: str,
        video_id: str,
        segment_id: int,
        t_sec: float,
        segment_embed: np.ndarray,
        cell_state: Dict[Tuple[int, int], Tuple[float, float]],
    ) -> Tuple[List[qmodels.PointStruct], int]:
        tile_points: List[qmodels.PointStruct] = []
        tiles_kept = 0

        h, w, _ = frame.shape
        max_tiles = settings.MAX_TILES_PER_SEGMENT
        count = 0

        for y in range(0, h - settings.TILE_SIZE + 1, settings.STRIDE):
            for x in range(0, w - settings.TILE_SIZE + 1, settings.STRIDE):
                if count >= max_tiles:
                    break
                tile = frame[y : y + settings.TILE_SIZE, x : x + settings.TILE_SIZE]
                if not tile_quality_ok(tile):
                    continue
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                ph = dhash(gray)
                if self.phash_lru.near_duplicate(ph):
                    continue

                cell_x = x // settings.CELL_SIZE
                cell_y = y // settings.CELL_SIZE
                cell_key = (cell_x, cell_y)
                score = float(edge_density(gray) * 1000.0 + gray.std())
                last = cell_state.get(cell_key)
                if last and (t_sec - last[0]) <= settings.CELL_WINDOW_SEC and score <= last[1]:
                    continue
                cell_state[cell_key] = (t_sec, score)

                tile_path = os.path.join(
                    settings.TILES_DIR, video_id, str(segment_id), f"tile_{int(t_sec*1000)}_{x}_{y}.jpg"
                )
                ensure_dir(os.path.dirname(tile_path))
                cv2.imwrite(tile_path, tile)

                tile_pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                clip_vec = self.clip_model.encode_images([tile_pil], batch_size=1)[0]
                if self.recent_index.max_cosine(clip_vec) > settings.DEDUP_COS_SIM_THRESH:
                    continue

                self.phash_lru.add(ph)
                self.recent_index.add(np.expand_dims(clip_vec, axis=0))

                vectors = {"clip": clip_vec.tolist()}
                if self.dino_model:
                    dino_vec = self.dino_model.encode_images([tile_pil], batch_size=1)[0]
                    vectors["dino"] = dino_vec.tolist()

                point_id = stable_point_id(video_id, segment_id, int(t_sec * 1000), "tile", x, y)
                payload = {
                    "type": "tile",
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "t_sec": t_sec,
                    "frame_path": frame_path,
                    "tile_path": tile_path,
                    "x": x,
                    "y": y,
                    "w": settings.TILE_SIZE,
                    "h": settings.TILE_SIZE,
                }
                tile_points.append(qmodels.PointStruct(id=point_id, vector=vectors, payload=payload))
                tiles_kept += 1
                count += 1
            if count >= max_tiles:
                break

        return tile_points, tiles_kept
