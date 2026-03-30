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
from pipeline.config import get_dino_model_name, settings
from pipeline.florence_model import FlorenceModel
from pipeline.qwen_model import QwenModel
from pipeline.asr_model import ASRModel
from pipeline.ocr_model import OCRModel
from pipeline.depth_model import DepthModel
from pipeline.detection_model import DetectionModel
from pipeline.world_model import WorldModel
from pipeline.audio_extractor import extract_audio, map_subtitles_to_frames
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
            name = get_dino_model_name(settings.MODEL_NAME)
            if name is None:
                raise ValueError(f"Unsupported DINO model family: {settings.MODEL_NAME}")
            self.dino_model = DINOEmbedder(model_name=name)
        self._florence_model: Optional[FlorenceModel] = None
        self.store = QdrantStore(clip_dim=self.clip_model.image_dim(), dino_dim=self._dino_dim())
        self.qwen_model = QwenModel(clip_prescreen_fn=self._make_vehicle_prescreen()) if settings.QWEN_API_URL else None
        # Optional enrichment models — lazily loaded on first use (enabled via env vars).
        self.asr_model = ASRModel() if settings.ASR_ENABLED else None
        self.ocr_model = OCRModel() if settings.OCR_ENABLED else None
        self.depth_model = DepthModel() if settings.DEPTH_ENABLED else None
        self.detection_model = DetectionModel() if settings.DETECTION_ENABLED else None
        self.world_model = WorldModel() if settings.WORLD_MODEL_ENABLED else None
        self.phash_lru = PhashLRU(settings.PHASH_LRU_SIZE, settings.PHASH_HAMMING_MAX)
        self.recent_index = RecentEmbeddingIndex(
            dim=self.clip_model.image_dim(),
            max_size=settings.DEDUP_RECENT_TILES,
            ttl_sec=settings.DEDUP_TTL_SEC,
        )

    def _dino_dim(self) -> Optional[int]:
        if self.dino_model is None:
            return None
        return self.dino_model.image_dim()

    @property
    def florence_model(self) -> FlorenceModel:
        if self._florence_model is None:
            self._florence_model = FlorenceModel()
        return self._florence_model

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
    ) -> Tuple[Optional[Tuple[Dict[str, Any], Dict[str, Any], List[qmodels.PointStruct], int]], "_IndexFrameState"]:
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
        qdrant_id = stable_point_id(video_id, segment_id, int(t_sec * 1000), "frame")
        frame_point = self._build_frame_point(
            video_id, segment_id, t_sec, frame_path, frame_pil, embed,
            mission_id=mission_id, gps=gps, enu=enu, robot_id=robot_id, global_map_id=global_map_id,
        )
        frame_record = {
            "id": f"{mission_id}:{segment_id}:{int(t_sec * 1000)}",
            "frame_path": frame_path,
            "t_sec": t_sec,
            "segment_id": segment_id,
            "caption": None,
            "caption_confidence": None,
            "caption_model": None,
            "subtitle_text": None,   # filled by ASR pass
            "ocr_text": None,        # filled by OCR pass
            "al_score": None,
            "al_tag": "none",
            "cvat_label": None,
            "pose_status": "pending",
            "pose_json": None,
            "gps_json": gps,
            "global_pose_json": enu,
            "qdrant_id": qdrant_id,
        }

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
        return (segment_dict, frame_record, [frame_point] + tile_points, 1 + tile_count), new_state

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
        frame_records: List[Dict[str, Any]] = []
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

            segment_dict, frame_record, frame_and_tile_points, count = result
            segments.append(segment_dict)
            frame_records.append(frame_record)
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

        # ── ASR pass — audio transcription (before captioning so Qwen can use it) ──
        if frame_records and self.asr_model and self.asr_model.is_enabled():
            self._run_asr_pass(dst_path, frame_records)

        # ── Florence captioning pass (post-loop) ──────────────────────────────
        # Runs after all Qdrant upserts are complete. Re-loads frames from disk
        # to avoid keeping ~250MB of PIL images in RAM during the main loop.
        if frame_records:
            self._run_florence_pass(frame_records)

        # ── OCR pass — visible text extraction ───────────────────────────────
        if frame_records and self.ocr_model and self.ocr_model.is_enabled():
            self._run_ocr_pass(frame_records)

        # ── Qwen2.5-VL Phase 2 structured extraction pass ─────────────────────
        if frame_records and self.qwen_model:
            self._run_qwen_pass(frame_records)

        # ── Depth estimation pass ─────────────────────────────────────────────
        if frame_records and self.depth_model and self.depth_model.is_enabled():
            self._run_depth_pass(frame_records)

        # ── Object detection pass ─────────────────────────────────────────────
        if frame_records and self.detection_model and self.detection_model.is_enabled():
            self._run_detection_pass(frame_records)

        # ── World model pass (video clip embeddings) ──────────────────────────
        if frame_records and self.world_model and self.world_model.is_enabled():
            self._run_world_model_pass(frame_records)

        self.logger.info(
            "Indexing complete video_id=%s segments=%s frames=%s tiles=%s",
            video_id, len(segments), frames_indexed, tiles_indexed,
        )
        duration_sec = max((record["t_sec"] for record in frame_records), default=0.0)
        gps_origin = next((record["gps_json"] for record in frame_records if record.get("gps_json")), None)
        return {
            "segments": len(segments),
            "tiles": tiles_indexed,
            "frames": frames_indexed,
            "duration_sec": duration_sec,
            "gps_origin": gps_origin,
            "frame_records": frame_records,
        }

    def _make_vehicle_prescreen(self):
        """Build a vehicle pre-screen function reusing self.clip_model (avoids second CLIP load)."""
        import numpy as np
        from pipeline.qwen_model import _VEHICLE_LABELS
        threshold = settings.QWEN_CLIP_THRESHOLD
        labels = list(_VEHICLE_LABELS)
        prompts = [f"a photo of {l}" for l in labels]
        text_embeds = self.clip_model.encode_texts(prompts)  # (N, dim)

        def prescreen(image: Image.Image) -> bool:
            img_emb = self.clip_model.encode_images([image], batch_size=1)[0]
            sims = float(np.dot(text_embeds, img_emb).max())
            return sims >= threshold

        return prescreen

    def _run_asr_pass(
        self, video_path: str, frame_records: List[Dict[str, Any]]
    ) -> None:
        """Transcribe video audio and map subtitle segments to frame timestamps."""
        import os, tempfile
        self.logger.info("ASR pass: transcribing audio from %s", video_path)
        audio_dir = settings.ASR_AUDIO_DIR
        from pipeline.utils import ensure_dir
        ensure_dir(audio_dir)
        wav_path = extract_audio(video_path, audio_dir)
        if not wav_path:
            self.logger.info("ASR pass: no audio track — skipping")
            return
        segments = self.asr_model.transcribe(wav_path)
        if not segments:
            self.logger.info("ASR pass: no transcript segments produced")
            return
        timestamps = [rec["t_sec"] for rec in frame_records]
        subtitle_map = map_subtitles_to_frames(
            segments, timestamps, window_sec=settings.ASR_SUBTITLE_WINDOW_SEC
        )
        for rec in frame_records:
            text = subtitle_map.get(rec["t_sec"])
            if text:
                rec["subtitle_text"] = text
        subtitled = sum(1 for r in frame_records if r.get("subtitle_text"))
        self.logger.info("ASR pass complete: %d/%d frames have subtitle text", subtitled, len(frame_records))

    def _run_ocr_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Run OCR on kept frames and store text in frame_facts_json + ocr_text."""
        self.logger.info("OCR pass: %d frames", len(frame_records))
        for batch_start in range(0, len(frame_records), settings.OCR_BATCH_SIZE):
            batch = frame_records[batch_start: batch_start + settings.OCR_BATCH_SIZE]
            images = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
            results = self.ocr_model.extract_text_batch(images)
            for rec, res in zip(batch, results):
                text = res.get("ocr_text", "") or ""
                rec["ocr_text"] = text if text else None
                # Also merge into frame_facts_json for Qwen context
                if text:
                    fj = rec.get("frame_facts_json") or {}
                    if isinstance(fj, dict):
                        fj["ocr_text"] = text
                        rec["frame_facts_json"] = fj
        ocr_found = sum(1 for r in frame_records if r.get("ocr_text"))
        self.logger.info("OCR pass complete: %d/%d frames contain text", ocr_found, len(frame_records))

    def _run_qwen_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Run Qwen2.5-VL structured extraction, enriched with subtitle+OCR context."""
        if not self.qwen_model or not self.qwen_model.is_enabled():
            return
        self.logger.info("Qwen2.5-VL Phase 2 pass: %d frames", len(frame_records))
        for rec in frame_records:
            try:
                img = Image.open(rec["frame_path"]).convert("RGB")
            except Exception:
                rec["frame_facts_json"] = {"file_error": True}
                continue
            subtitle = rec.get("subtitle_text")
            ocr = rec.get("ocr_text")
            qwen_result = self.qwen_model.extract_frame_facts(img, subtitle_text=subtitle, ocr_text=ocr)
            # Merge Qwen result with any existing keys (e.g. ocr_text from OCR pass)
            existing = rec.get("frame_facts_json")
            if isinstance(existing, dict) and isinstance(qwen_result, dict):
                merged = {**existing, **qwen_result}
                rec["frame_facts_json"] = merged
            else:
                rec["frame_facts_json"] = qwen_result
        self.logger.info("Qwen2.5-VL pass complete")

    def _run_depth_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Estimate monocular depth and store percentiles in frame_facts_json."""
        self.logger.info("Depth estimation pass: %d frames", len(frame_records))
        for rec in frame_records:
            try:
                img = Image.open(rec["frame_path"]).convert("RGB")
            except Exception:
                continue
            depth_result = self.depth_model.estimate(img)
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj.update(depth_result)
                rec["frame_facts_json"] = fj
        self.logger.info("Depth pass complete")

    def _run_detection_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Run object detection and store bounding boxes in frame_facts_json."""
        self.logger.info("Detection pass: %d frames", len(frame_records))
        for batch_start in range(0, len(frame_records), 8):
            batch = frame_records[batch_start: batch_start + 8]
            images = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
            results = self.detection_model.detect_batch(images)
            for rec, res in zip(batch, results):
                fj = rec.get("frame_facts_json") or {}
                if isinstance(fj, dict):
                    fj.update(res)
                    rec["frame_facts_json"] = fj
        self.logger.info("Detection pass complete")

    def _run_world_model_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Run world model on sliding windows of consecutive kept frames."""
        clip_size = settings.WORLD_MODEL_CLIP_FRAMES
        self.logger.info(
            "World model pass: %d frames, clip_size=%d", len(frame_records), clip_size,
        )
        for batch_start in range(0, len(frame_records), clip_size):
            batch = frame_records[batch_start: batch_start + clip_size]
            images = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))
            result = self.world_model.process_clip(images)
            # Assign world model result to the middle frame of the clip
            mid = len(batch) // 2
            rec = batch[mid]
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj.update(result)
                rec["frame_facts_json"] = fj
        self.logger.info("World model pass complete")

    def _run_florence_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Caption all kept frames with Florence-2 and update Qdrant payloads.

        Loads each image from disk, runs caption_batch() in FLORENCE_BATCH_SIZE
        chunks, updates frame_records in-place, then pushes captions to Qdrant
        via set_payload in 128-frame batches.
        """
        self.logger.info(
            "Florence captioning pass: %d frames (batch_size=%d)",
            len(frame_records),
            settings.FLORENCE_BATCH_SIZE,
        )

        caption_model_tag = self.florence_model.model_tag

        # Process in FLORENCE_BATCH_SIZE chunks
        for batch_start in range(0, len(frame_records), settings.FLORENCE_BATCH_SIZE):
            batch = frame_records[batch_start : batch_start + settings.FLORENCE_BATCH_SIZE]

            # Load PIL images from disk
            pil_images: List = []
            for rec in batch:
                try:
                    pil_images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    self.logger.warning(
                        "Florence: could not open %s; using blank image", rec["frame_path"]
                    )
                    pil_images.append(Image.new("RGB", (224, 224)))

            # caption_batch already handles OOM fallback internally
            try:
                captions_and_confs = self.florence_model.caption_batch(
                    pil_images, batch_size=settings.FLORENCE_BATCH_SIZE
                )
            except Exception:
                self.logger.warning(
                    "Florence batch failed for frames %d–%d; using empty captions",
                    batch_start,
                    batch_start + len(batch) - 1,
                    exc_info=True,
                )
                captions_and_confs = [("", 0.5)] * len(batch)

            for rec, (caption, confidence) in zip(batch, captions_and_confs):
                rec["caption"] = caption
                rec["caption_confidence"] = confidence
                rec["caption_model"] = caption_model_tag

        # Push captions to Qdrant in 128-frame batches
        self._set_caption_payload(frame_records)

    def _set_caption_payload(self, frame_records: List[Dict[str, Any]]) -> None:
        """Write caption into Qdrant point payloads (display-only in Phase 1).

        Qdrant set_payload applies a single payload dict to all listed points,
        so each distinct caption requires its own call. We bound the outer loop
        at 128-frame chunks for consistent log granularity and error reporting.
        """
        for batch_start in range(0, len(frame_records), 128):
            batch = frame_records[batch_start : batch_start + 128]
            failed = 0
            for rec in batch:
                qdrant_id = rec.get("qdrant_id")
                if qdrant_id is None:
                    continue
                try:
                    self.store.client.set_payload(
                        collection_name=self.store.collection,
                        payload={"caption": rec.get("caption", "")},
                        points=[qdrant_id],
                    )
                except Exception:
                    failed += 1
            if failed:
                self.logger.warning(
                    "Florence: %d/%d Qdrant set_payload calls failed in batch "
                    "starting at frame %d; DB has captions, backfill will sync Qdrant.",
                    failed,
                    len(batch),
                    batch_start,
                )

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
