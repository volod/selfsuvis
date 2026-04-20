import dataclasses
import itertools
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from qdrant_client.http import models as qmodels

from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.models.dino_model import DINOEmbedder
from selfsuvis.models.rssm_model import RSSMEmbedder
from selfsuvis.pipeline.core import RateTimer, ensure_dir, get_dino_model_name, get_logger, settings, stable_point_id
from selfsuvis.pipeline.media import extract_audio, extract_frames, map_subtitles_to_frames
from selfsuvis.pipeline.media.heuristics import (
    downsample_gray,
    histogram_diff,
    mean_abs_diff,
    ssim_diff,
    phase_corr_align,
    frame_quality_ok,
    tile_quality_ok,
    edge_density,
)
from selfsuvis.pipeline.media.dedup import dhash, PhashLRU
from selfsuvis.pipeline.fusion import run_platform_state_fusion
from selfsuvis.pipeline.mapping import build_semantic_environment_graph
from selfsuvis.pipeline.storage import QdrantStore, RecentEmbeddingIndex
from selfsuvis.pipeline.vision import (
    ASRModel,
    DetectionModel,
    DepthModel,
    FlorenceModel,
    OCRModel,
    QwenModel,
    RFSignalAnalyzer,
    SAMPredictor,
    UniDriveVLAModel,
    WorldModel,
    YOLODetector,
)
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
        self.unidrive_model = UniDriveVLAModel() if settings.UNIDRIVE_ENABLED and settings.UNIDRIVE_API_URL else None
        # Optional enrichment models — lazily loaded on first use (enabled via env vars).
        self.asr_model = ASRModel() if settings.ASR_ENABLED else None
        self.ocr_model = OCRModel() if settings.OCR_ENABLED else None
        self.depth_model = DepthModel() if settings.DEPTH_ENABLED else None
        self.detection_model = DetectionModel() if settings.DETECTION_ENABLED else None
        self.world_model = WorldModel() if settings.WORLD_MODEL_ENABLED else None
        self.yolo_detector = YOLODetector() if settings.YOLO_ENABLED else None
        self.sam_predictor = SAMPredictor() if settings.SAM_ENABLED else None
        self.rf_analyzer = RFSignalAnalyzer() if settings.RF_ENABLED else None
        # RF-DETR tracker is initialised lazily inside _run_gemma_directed_tracking_pass
        self.rfdetr_tracker = None
        # RSSM temporal world model (DreamerV3-inspired, CPU-friendly)
        self.rssm_embedder = RSSMEmbedder(
            hidden_dim=settings.DREAMER_HIDDEN_DIM,
            latent_dim=settings.DREAMER_LATENT_DIM,
            train_steps=settings.DREAMER_TRAIN_STEPS,
        ) if settings.DREAMER_ENABLED else None
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
            "_clip_embed": embed,    # temporary — used by RSSM/AL pass, stripped before DB write
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
            from selfsuvis.pipeline.media.gps import extract_gps
            from selfsuvis.pipeline.mapping.gps_registration import gps_to_enu
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

        # ── Captioning pass (post-loop) ───────────────────────────────────────
        # When GEMMA_API_URL is set, the top-N highest-quality frames (ranked by
        # histogram-diff score) are captioned via Gemma in async chunks; the
        # remainder fall back to Florence.  Without GEMMA_API_URL, all frames
        # use Florence.
        if frame_records:
            if settings.GEMMA_API_URL:
                self._run_gemma_caption_pass(frame_records)
            else:
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

        # ── RF signal analysis pass (TorchSig) ───────────────────────────────
        if frame_records and self.rf_analyzer and self.rf_analyzer.is_enabled():
            self._run_rf_analysis_pass(dst_path, frame_records)

        # ── YOLO11 + SAM3/SAM2 pass (priority-ranked detections + masks) ─────
        if frame_records and self.yolo_detector and self.yolo_detector.is_enabled():
            self._run_yolo_sam_pass(frame_records)

        semantic_graph_summary = None
        if frame_records and settings.YOLO_SSG_ENABLED:
            semantic_graph_summary = self._run_yolo_ssg_pass(
                video_id=video_id,
                mission_id=effective_mission_id,
                frame_records=frame_records,
            )

        # ── Gemma directed tracking pass (step P3) ────────────────────────────
        if frame_records and settings.RFDETR_ENABLED and settings.GEMMA_API_URL:
            self._run_gemma_directed_tracking_pass(frame_records)

        # ── World model pass (video clip embeddings) ──────────────────────────
        if frame_records and self.world_model and self.world_model.is_enabled():
            self._run_world_model_pass(frame_records)

        # ── UniDriveVLA expert pass (understanding/perception/planning) ──────
        if frame_records and self.unidrive_model and self.unidrive_model.is_enabled():
            self._run_unidrive_pass(frame_records)

        state_fusion_summary: Dict[str, Any] = {
            "enabled": False,
            "status": "skipped",
            "reason": "no frame records",
        }
        if frame_records:
            state_fusion = run_platform_state_fusion(
                video_path=video_path,
                frame_times_sec=[record["t_sec"] for record in frame_records],
                gps_samples=[record.get("gps_json") for record in frame_records],
            )
            state_fusion_summary = state_fusion.summary()
            samples_by_t = {
                round(sample.t_sec, 3): sample
                for sample in state_fusion.posterior_samples
            }
            for record in frame_records:
                sample = samples_by_t.get(round(record["t_sec"], 3))
                if sample is None:
                    continue
                frame_facts = record.get("frame_facts_json") or {}
                frame_facts["state_fusion"] = {
                    "source": state_fusion.source,
                    "position_enu_m": dict(sample.position_enu_m),
                    "velocity_enu_mps": dict(sample.velocity_enu_mps),
                    "covariance_trace": sample.covariance_trace,
                    "quality": sample.quality,
                    "measurement_kinds": list(sample.measurement_kinds),
                }
                record["frame_facts_json"] = frame_facts

        # ── RSSM temporal surprise + active learning tagging ─────────────────
        # Runs after all enrichment passes so caption_confidence is available.
        # Populates al_score and al_tag in frame_records; strips _clip_embed.
        if frame_records:
            self._run_al_rssm_pass(frame_records)

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
            "semantic_graph": semantic_graph_summary,
            "state_fusion_summary": state_fusion_summary,
            "unidrive_summary": self._summarize_unidrive_records(frame_records),
        }

    def _make_vehicle_prescreen(self):
        """Build a vehicle pre-screen function reusing self.clip_model (avoids second CLIP load)."""
        import numpy as np
        from selfsuvis.pipeline.vision.qwen import _VEHICLE_LABELS
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
        from selfsuvis.pipeline.core.utils import ensure_dir
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
                fj = rec.get("frame_facts_json") or {}
                fj["file_error"] = True
                rec["frame_facts_json"] = fj
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

    def _run_unidrive_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Run UniDriveVLA expert analysis on a sparse sample and store results."""
        if not self.unidrive_model or not self.unidrive_model.is_enabled():
            return
        max_frames = max(1, int(getattr(settings, "UNIDRIVE_MAX_FRAMES", 24) or 24))
        sample_step = max(1, len(frame_records) // max_frames)
        sampled = frame_records[::sample_step][:max_frames]
        self.logger.info("UniDriveVLA pass: %d sampled frames", len(sampled))
        for rec in sampled:
            try:
                img = Image.open(rec["frame_path"]).convert("RGB")
            except Exception:
                fj = rec.get("frame_facts_json") or {}
                if isinstance(fj, dict):
                    fj["unidrive_vla"] = {"file_error": True}
                    rec["frame_facts_json"] = fj
                continue
            existing = rec.get("frame_facts_json") or {}
            extra_context = ""
            if isinstance(existing, dict) and existing:
                try:
                    extra_context = json.dumps(existing, ensure_ascii=True, sort_keys=True)[:2000]
                except Exception:
                    extra_context = ""
            result = self.unidrive_model.analyze_frame(
                img,
                subtitle_text=rec.get("subtitle_text"),
                ocr_text=rec.get("ocr_text"),
                extra_context=extra_context,
            )
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj["unidrive_vla"] = result
                rec["frame_facts_json"] = fj
        self.logger.info("UniDriveVLA pass complete")

    def _summarize_unidrive_records(self, frame_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise UniDriveVLA outputs for worker/job status reporting."""
        analysed = 0
        high_risk = 0
        agreement_counts: Dict[str, int] = {}
        for rec in frame_records:
            facts = rec.get("frame_facts_json") or {}
            if not isinstance(facts, dict):
                continue
            uv = facts.get("unidrive_vla")
            if not isinstance(uv, dict) or uv.get("service_unavailable") or uv.get("parse_error"):
                continue
            analysed += 1
            risk = ((uv.get("understanding") or {}).get("risk_level", "unknown"))
            if risk == "high":
                high_risk += 1
            agreement = ((uv.get("mixture_of_experts") or {}).get("expert_agreement", "unknown"))
            agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1
        return {
            "analysed_frames": analysed,
            "high_risk_frames": high_risk,
            "expert_agreement": agreement_counts,
        }

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
        for batch_start in range(0, len(frame_records), settings.DETECTION_BATCH_SIZE):
            batch = frame_records[batch_start: batch_start + settings.DETECTION_BATCH_SIZE]
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

    def _run_rf_analysis_pass(
        self, video_path: str, frame_records: List[Dict[str, Any]]
    ) -> None:
        """Analyze IQ sidecar (or audio proxy) and write RF metrics to frame_facts_json.

        Stores ``frame_facts_json["rf_signal"]`` with SNR, spectral flatness,
        occupied bandwidth, peak frequency ratio, and optionally modulation class.
        If no IQ sidecar is present the analyzer falls back to the audio track
        extracted by the ASR pass (reuses the WAV in ASR_AUDIO_DIR if present).
        """
        import os
        base = os.path.splitext(os.path.basename(video_path))[0]
        audio_dir = settings.ASR_AUDIO_DIR
        audio_wav = os.path.join(audio_dir, f"{base}.wav")
        audio_wav = audio_wav if os.path.isfile(audio_wav) else None

        timestamps = [rec["t_sec"] for rec in frame_records]
        results = self.rf_analyzer.analyze_video(video_path, timestamps, audio_wav_path=audio_wav)

        for rec, res in zip(frame_records, results):
            if not res:
                continue
            fj = rec.get("frame_facts_json") or {}
            if isinstance(fj, dict):
                fj.update(res)
                rec["frame_facts_json"] = fj

    def _run_yolo_sam_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Run YOLO11 detection (+ optional SAM3/SAM2 masks) and store results in frame_facts_json.

        Detections are priority-sorted (human=1 → vehicle=2 → artificial=3 → other=4).
        Stored under ``frame_facts_json["yolo_detections"]`` as a list of dicts:
        ``{label, confidence, bbox_norm, priority, priority_label, mask_area_norm}``.

        When ``SAM_ENABLED`` is true and a SAM backend is available, each bounding
        box is refined with a segmentation mask and ``mask_area_norm`` is populated.
        """
        use_sam = self.sam_predictor is not None and self.sam_predictor.is_available()
        self.logger.info(
            "YOLO+SAM pass: %d frames  model=%s  sam=%s",
            len(frame_records),
            self.yolo_detector.model_id,
            "enabled" if use_sam else "disabled",
        )
        batch_size = getattr(settings, "DETECTION_BATCH_SIZE", 8)
        total_dets = 0
        for batch_start in range(0, len(frame_records), batch_size):
            batch = frame_records[batch_start: batch_start + batch_size]
            images: List[Image.Image] = []
            for rec in batch:
                try:
                    images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (224, 224)))

            # YOLO detection — returns priority-sorted list per image
            yolo_results = self.yolo_detector.detect_batch(images)

            for rec, img, detections in zip(batch, images, yolo_results):
                if use_sam and detections:
                    bboxes = [d["bbox_norm"] for d in detections]
                    try:
                        sam_masks = self.sam_predictor.predict_boxes(img, bboxes)
                        for det, mask_info in zip(detections, sam_masks):
                            det["mask_area_norm"] = mask_info.get("area_norm")
                    except Exception as exc:
                        self.logger.debug("SAM prediction failed for frame %s: %s", rec.get("frame_id"), exc)

                fj = rec.get("frame_facts_json") or {}
                if isinstance(fj, dict):
                    fj["yolo_detections"] = detections
                    rec["frame_facts_json"] = fj
                total_dets += len(detections)

        self.logger.info(
            "YOLO+SAM pass complete: %d detections across %d frames",
            total_dets, len(frame_records),
        )

    def _run_gemma_directed_tracking_pass(
        self,
        frame_records: List[Dict[str, Any]],
    ) -> None:
        """Gemma directed tracking pass: Gemma scene understanding → SAM segmentation
        → RF-DETR tracking. Stores results in ``frame_facts_json["gemma_tracking"]``.

        Tracking results per frame:
            {
                "scene_type":        str,
                "tracking_priority": List[str],
                "detections":        List[detection_dict],
                "sam_masks":         [{"category": str, "area_norm": float, "source": str}],
            }
        """
        from selfsuvis.pipeline.workflows.local.steps_gemma_tracking import (
            _gemma_structured_scene_analysis,
            _sam_directed_by_gemma,
        )
        from selfsuvis.pipeline.vision.rfdetr import RFDETRTracker

        frame_list = [
            (r["frame_path"], float(r.get("t_sec", 0.0)))
            for r in frame_records
            if r.get("frame_path")
        ]
        if not frame_list:
            return

        # Structured Gemma scene analysis on a sparse sample
        gemma_scene = _gemma_structured_scene_analysis(
            frame_list,
            api_url=settings.GEMMA_API_URL,
            model=settings.GEMMA_API_MODEL,
            timeout=float(settings.GEMMA_API_TIMEOUT_SEC),
            clip_model=self.clip_model,
        )
        tracking_priority = gemma_scene.get("tracking_priority", [])
        gemma_objects = gemma_scene.get("dominant_objects", [])
        scene_type = gemma_scene.get("scene_type", "other")
        self.logger.info(
            "Gemma directed tracking: scene_type=%s priority=%s objects=%d",
            scene_type, tracking_priority, len(gemma_objects),
        )

        # RF-DETR tracking pass across all frame records
        if self.rfdetr_tracker is None:
            self.rfdetr_tracker = RFDETRTracker()
        tracking_results = self.rfdetr_tracker.track_sequence(
            frame_list,
            target_labels=tracking_priority if tracking_priority else None,
        )
        path_to_dets = {r["frame_path"]: r.get("detections", []) for r in tracking_results}

        # SAM segmentation pass (only when SAM is available and objects were identified)
        use_sam = (
            self.sam_predictor is not None
            and self.sam_predictor.is_available()
            and bool(gemma_objects)
        )

        # Write results into frame_records
        for rec in frame_records:
            fp = rec.get("frame_path", "")
            fj = rec.get("frame_facts_json") or {}
            if not isinstance(fj, dict):
                fj = {}
            tracking_dets = path_to_dets.get(fp, [])
            sam_masks_summary: List[Dict] = []
            if use_sam and fp:
                try:
                    img = Image.open(fp).convert("RGB")
                    masks = _sam_directed_by_gemma(
                        img, gemma_objects, self.sam_predictor, self.clip_model,
                    )
                    w_img, h_img = img.size
                    sam_masks_summary = [
                        {
                            "category":  m.get("category", "unknown"),
                            "area_norm": round(
                                float(m["mask"].sum()) / (w_img * h_img), 6
                            ) if m.get("mask") is not None else 0.0,
                            "source":    m.get("source", "unknown"),
                        }
                        for m in masks
                    ]
                except Exception as exc:
                    self.logger.debug(
                        "Gemma directed tracking SAM pass failed for %s: %s",
                        rec.get("frame_id", fp), exc,
                    )
            fj["gemma_tracking"] = {
                "scene_type":        scene_type,
                "tracking_priority": tracking_priority,
                "detections":        tracking_dets,
                "sam_masks":         sam_masks_summary,
            }
            rec["frame_facts_json"] = fj

        self.logger.info(
            "Gemma directed tracking pass complete: %d frames, scene=%s",
            len(frame_records), scene_type,
        )

    def _run_yolo_ssg_pass(
        self,
        *,
        video_id: str,
        mission_id: str,
        frame_records: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build a YOLO semantic scene graph and attach node ids back to frames."""
        graph_dir = os.path.join(settings.MAPS_DIR, mission_id)
        ensure_dir(graph_dir)
        graph_path = os.path.join(graph_dir, "semantic_environment_graph.json")
        graph = build_semantic_environment_graph(
            frame_records,
            graph_id=mission_id or video_id,
            output_path=graph_path,
        )
        assignments = graph.get("frame_assignments", {})
        for record in frame_records:
            frame_key = str(record.get("id") or record.get("frame_path") or "")
            node_ids = assignments.get(frame_key, [])
            if not node_ids:
                continue
            facts = record.get("frame_facts_json") or {}
            if isinstance(facts, dict):
                facts["semantic_graph_node_ids"] = node_ids
                record["frame_facts_json"] = facts

        summary = {
            **graph.get("summary", {}),
            "output_path": graph.get("output_path", graph_path),
            "anchor_source": graph.get("anchor_source", "unknown"),
            "coordinate_frame": graph.get("coordinate_frame", "unknown"),
        }
        self.logger.info(
            "YOLO SSG pass complete: %d nodes, %d edges → %s",
            summary.get("node_count", 0),
            summary.get("edge_count", 0),
            summary["output_path"],
        )
        return summary

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

    def _run_al_rssm_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Compute active learning scores and assign al_tags.

        Integrates DreamerV3-inspired RSSM temporal surprise scoring
        (Romero et al., ICRA 2026) with the existing DINOv3-dist + caption
        confidence signal.

        Step 1 — collect per-frame data:
            - CLIP embeddings (stored temporarily in frame_records["_clip_embed"])
            - caption confidences (from Florence/Gemma captioning pass)

        Step 2 — RSSM temporal surprise (when DREAMER_ENABLED=true):
            Train a lightweight RSSM online on the mission's CLIP sequence,
            then compute surprise_k = cosine_distance(predicted_z̃_k, actual_z_k).
            Stores rssm_surprise in frame_facts_json["rssm"].

        Step 3 — active learning scoring:
            With RSSM:    al_score = 0.35*dino + 0.25*(1-conf) + 0.40*surprise
            Without RSSM: al_score = 0.60*dino + 0.40*(1-conf)

        Step 4 — strip temporary _clip_embed fields before DB write.
        """
        import numpy as np
        from selfsuvis.pipeline.analysis.active_learning import assign_al_tags, fit_kmeans, dino_distances_from_centroids

        n = len(frame_records)
        self.logger.info("AL+RSSM pass: %d frames", n)

        # ── Collect CLIP embeds (written temporarily during _process_frame) ──
        clip_embeds_list = []
        valid_indices = []
        for i, rec in enumerate(frame_records):
            emb = rec.pop("_clip_embed", None)
            if emb is not None:
                clip_embeds_list.append(emb.astype(np.float32))
                valid_indices.append(i)

        caption_confidences = [
            float(rec.get("caption_confidence") or 0.5)
            for rec in frame_records
        ]

        # ── Compute dino-proxy distance via CLIP k-means centroid distance ──
        # Uses CLIP embeddings as proxy when DINOv3 is not separately embedded.
        # The k-means distance captures per-frame novelty relative to the
        # mission's overall embedding distribution.
        dino_dists = [0.5] * n  # fallback
        if clip_embeds_list:
            try:
                all_embeds = np.stack(clip_embeds_list)
                kmeans = fit_kmeans(all_embeds, n_clusters=min(20, len(clip_embeds_list)))
                centroid_dists = dino_distances_from_centroids(all_embeds, kmeans.cluster_centers_)
                for rank, idx in enumerate(valid_indices):
                    dino_dists[idx] = float(centroid_dists[rank])
            except Exception as exc:
                self.logger.debug("AL k-means failed (%s) — using uniform dino_dists", exc)

        # ── RSSM temporal surprise ────────────────────────────────────────────
        rssm_surprises: Optional[List[float]] = None
        if self.rssm_embedder is not None and clip_embeds_list:
            try:
                import time
                t0 = time.time()
                all_embeds = np.stack(clip_embeds_list)
                rssm_result = self.rssm_embedder.encode_sequence(all_embeds)
                surprises_arr = rssm_result["surprise_scores"]
                method = rssm_result.get("method", "unknown")
                elapsed = time.time() - t0
                self.logger.info(
                    "RSSM pass complete: method=%s hidden=%d latent=%d elapsed=%.2fs",
                    method, rssm_result["hidden_dim"], rssm_result["latent_dim"], elapsed,
                )
                # Map back from valid_indices to full frame_records
                rssm_surprises = [0.5] * n
                for rank, idx in enumerate(valid_indices):
                    rssm_surprises[idx] = float(surprises_arr[rank])
                # Store RSSM metadata in frame_facts_json
                for rank, idx in enumerate(valid_indices):
                    rec = frame_records[idx]
                    fj = rec.get("frame_facts_json") or {}
                    if not isinstance(fj, dict):
                        fj = {}
                    rssm_entry: Dict[str, Any] = {
                        "surprise_score": float(surprises_arr[rank]),
                        "method": method,
                        "model": rssm_result.get("model", "RSSMEmbedder"),
                    }
                    if settings.DREAMER_STORE_TEMPORAL and "recurrent_states" in rssm_result:
                        rssm_entry["recurrent_state"] = rssm_result["recurrent_states"][rank].tolist()
                    fj["rssm"] = rssm_entry
                    rec["frame_facts_json"] = fj
            except Exception as exc:
                self.logger.warning("RSSM temporal surprise failed (%s) — skipping", exc)

        # ── Assign AL scores and tags ─────────────────────────────────────────
        scores, tags = assign_al_tags(
            dino_dists,
            caption_confidences,
            rssm_surprises=rssm_surprises,
        )
        for rec, score, tag in zip(frame_records, scores, tags):
            rec["al_score"] = float(score)
            rec["al_tag"] = tag

        needs = tags.count("needs_annotation")
        novel = tags.count("novel")
        formula = "rssm+dino+caption" if rssm_surprises is not None else "dino+caption"
        self.logger.info(
            "AL tagging complete: needs_annotation=%d novel=%d none=%d formula=%s",
            needs, novel, n - needs - novel, formula,
        )

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

    # ------------------------------------------------------------------
    # Gemma caption pass (production captioner when GEMMA_API_URL is set)
    # ------------------------------------------------------------------

    def _run_gemma_caption_pass(self, frame_records: List[Dict[str, Any]]) -> None:
        """Caption frames via the Gemma sidecar API with Florence fallback.

        Strategy:
        - Rank all frames by absolute histogram-diff score (higher = more
          informative / more diverse).  Take the top GEMMA_MAX_CAPTION_FRAMES
          for Gemma; caption the rest with Florence.
        - Gemma frames are sent in chunks of GEMMA_CAPTION_CHUNK_SIZE with a
          50-second timeout and GEMMA_CAPTION_RETRIES retry.  On second failure
          the chunk falls back to Florence.
        - Every frame record gets caption, caption_confidence, caption_model set.
        """
        import asyncio as _asyncio
        import httpx as _httpx

        max_gemma = settings.GEMMA_MAX_CAPTION_FRAMES
        chunk_size = settings.GEMMA_CAPTION_CHUNK_SIZE
        retries = settings.GEMMA_CAPTION_RETRIES
        timeout = 50.0
        api_url = settings.GEMMA_API_URL.rstrip("/")
        model = settings.GEMMA_API_MODEL
        endpoint = f"{api_url}/chat/completions"
        florence_tag = self.florence_model.model_tag
        gemma_tag = f"gemma-api:{model}"

        total = len(frame_records)
        # Rank by histogram-diff quality score stored during frame extraction.
        # Fall back to positional index when score is unavailable.
        scored = sorted(
            enumerate(frame_records),
            key=lambda iv: float(iv[1].get("hist_diff", 0.0) or 0.0),
            reverse=True,
        )
        gemma_indices = {idx for idx, _ in scored[:max_gemma]} if max_gemma > 0 else set(range(total))
        gemma_recs = [r for i, r in enumerate(frame_records) if i in gemma_indices]
        florence_recs = [r for i, r in enumerate(frame_records) if i not in gemma_indices]

        self.logger.info(
            "Gemma captioning pass: %d frames via Gemma API, %d via Florence fallback",
            len(gemma_recs), len(florence_recs),
        )

        def _caption_image_gemma(frame_path: str) -> tuple:
            """Return (caption, confidence) for one frame via Gemma API, or raise."""
            import base64 as _b64
            try:
                with open(frame_path, "rb") as _f:
                    img_b64 = _b64.b64encode(_f.read()).decode()
            except OSError:
                raise

            prompt = (
                "Describe this image in one concise sentence suitable for outdoor "
                "robotics scene understanding. Focus on terrain, objects, and activities."
            )
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    }
                ],
                "max_tokens": 128,
                "temperature": 0.2,
            }
            resp = _httpx.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text, 0.85  # Gemma has no explicit confidence; use a fixed prior

        def _caption_chunk_with_retry(chunk: List[Dict[str, Any]]) -> None:
            """Caption a chunk of records in-place, retrying once then falling back to Florence."""
            for attempt in range(retries + 1):
                try:
                    for rec in chunk:
                        caption, conf = _caption_image_gemma(rec["frame_path"])
                        rec["caption"] = caption
                        rec["caption_confidence"] = conf
                        rec["caption_model"] = gemma_tag
                    return  # success
                except Exception as exc:
                    if attempt < retries:
                        self.logger.debug(
                            "Gemma caption chunk attempt %d failed (%s) — retrying",
                            attempt + 1, exc,
                        )
                    else:
                        self.logger.warning(
                            "Gemma caption chunk failed after %d attempt(s) (%s) — falling back to Florence",
                            retries + 1, exc,
                        )
                        self._caption_records_with_florence(chunk, florence_tag)

        # Process Gemma frames in chunks
        for chunk_start in range(0, len(gemma_recs), chunk_size):
            chunk = gemma_recs[chunk_start : chunk_start + chunk_size]
            _caption_chunk_with_retry(chunk)

        # Caption remaining frames with Florence
        if florence_recs:
            self._caption_records_with_florence(florence_recs, florence_tag)

        self._set_caption_payload(frame_records)

    def _caption_records_with_florence(
        self, records: List[Dict[str, Any]], model_tag: str
    ) -> None:
        """Caption the given records in-place using the Florence model."""
        for batch_start in range(0, len(records), settings.FLORENCE_BATCH_SIZE):
            batch = records[batch_start : batch_start + settings.FLORENCE_BATCH_SIZE]
            pil_images: List = []
            for rec in batch:
                try:
                    pil_images.append(Image.open(rec["frame_path"]).convert("RGB"))
                except Exception:
                    pil_images.append(Image.new("RGB", (224, 224)))
            try:
                captions_and_confs = self.florence_model.caption_batch(
                    pil_images, batch_size=settings.FLORENCE_BATCH_SIZE
                )
            except Exception:
                self.logger.warning(
                    "Florence fallback batch failed for %d frames; using empty captions",
                    len(batch), exc_info=True,
                )
                captions_and_confs = [("", 0.5)] * len(batch)
            for rec, (caption, confidence) in zip(batch, captions_and_confs):
                rec["caption"] = caption
                rec["caption_confidence"] = confidence
                rec["caption_model"] = model_tag

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
