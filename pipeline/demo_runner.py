"""selfsuvis end-to-end demo pipeline runner.

All demo logic lives here.  Entry point: :func:`run_demo`.

Called via::

    python main.py --mode demo [options]

The caller is responsible for setting the necessary env vars **before**
importing this module (so ``pipeline.config.settings`` picks them up).
``main.py`` calls :func:`pipeline.demo_env.apply_demo_env` for this.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── Pipeline imports — safe because caller sets env vars before importing us ──
from pipeline.config import settings
from pipeline.ffmpeg_utils import extract_frames
from pipeline.ssl_finetune import FinetuneConfig, run_finetune
from pipeline.distill import DistillConfig, run_distillation
from pipeline.edge_inference import build_gallery
from pipeline.vector_store import InMemoryStore
from pipeline.map_builder import build_sparse_map
from pipeline.viewer import view_npz, _HAS_MPL
from models.openclip_model import OpenCLIPEmbedder

try:
    from models.dino_model import DINOEmbedder
    _HAS_DINO = True
except Exception:
    _HAS_DINO = False

try:
    from models.gemma_model import GemmaEmbedder
    _HAS_GEMMA = True
except Exception:
    _HAS_GEMMA = False

logger = logging.getLogger(__name__)

# ── Logging helpers ────────────────────────────────────────────────────────────

_LOG_FMT  = "%(asctime)s  %(levelname)-7s  %(message)s"
_DATE_FMT = "%H:%M:%S"

_NOISY_LOGGERS = ("urllib3", "PIL", "filelock", "torch", "timm")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_DATE_FMT)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _configure_warnings() -> None:
    warnings.filterwarnings("ignore", message="xFormers is available",          category=UserWarning)
    warnings.filterwarnings("ignore", message="xFormers is not available",       category=UserWarning)
    warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated",
                            category=FutureWarning)
    warnings.filterwarnings("ignore", message="The image_processor_class argument is deprecated",
                            category=FutureWarning)


_log = logging.getLogger("demo")


def _banner(msg: str) -> None:
    width = 72
    _log.info("=" * width)
    _log.info("  %s", msg)
    _log.info("=" * width)


def _step(n: int, total: int, name: str) -> None:
    _log.info("─── Step %d/%d: %s", n, total, name)


class _Timer:
    """Context manager that records elapsed seconds into a dict under *key*."""
    def __init__(self, store: Dict[str, float], key: str) -> None:
        self._store = store
        self._key   = key
        self._t0    = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.time()
        return self

    def __exit__(self, *_: Any) -> None:
        self._store[self._key] = time.time() - self._t0


def _open_frame_image(frame_path: str) -> Image.Image:
    try:
        return Image.open(frame_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))


def _open_frame_batch(batch: List[Tuple[str, float]]) -> List[Image.Image]:
    return [_open_frame_image(fp) for fp, _t in batch]


def _run_batched_frame_inference(
    frame_list: List[Tuple[str, float]],
    *,
    batch_size: int,
    batch_fn,
    warning_label: str,
    error_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        imgs = _open_frame_batch(batch)
        try:
            batch_out = batch_fn(batch, imgs)
        except Exception as exc:
            _log.warning("  %s batch %d failed: %s", warning_label, batch_start, exc)
            batch_out = [dict(error_result) for _ in batch]
        if len(batch_out) != len(batch):
            _log.warning(
                "  %s batch %d returned %d results for %d frames; padding/truncating",
                warning_label,
                batch_start,
                len(batch_out),
                len(batch),
            )
            padded = list(batch_out[: len(batch)])
            while len(padded) < len(batch):
                padded.append(dict(error_result))
            batch_out = padded
        for (fp, t_sec), r in zip(batch, batch_out):
            results.append({"frame_path": fp, "t_sec": t_sec, **r})
    return results


# ── Text prompts for CLIP video-to-text description ───────────────────────────

_TEXT_PROMPTS: List[str] = [
    "aerial footage of a road or highway",
    "outdoor terrain with green vegetation",
    "urban environment with buildings and streets",
    "industrial site or construction area",
    "rural landscape viewed from above",
    "dense forest or woodland area",
    "coastal area or water body",
    "agricultural field or farmland",
    "mountain or rocky terrain",
    "open desert or arid landscape",
    "parking lot or vehicle depot",
    "residential neighbourhood from above",
    "radar antenna or rotating radar dish on a rooftop or tower",
    "military radar installation in open terrain",
    "phased array radar or sensor array on a vehicle or structure",
    "surveillance radar dome or radome on a building",
    "weather radar tower in a field",
    "radar site with large parabolic antenna",
    "electronic warfare sensor mast on a ship or vehicle",
    "panoramic wide-angle view of vehicles on a road",
    "multiple cars and trucks visible in a wide scene",
    "convoy of military vehicles on a road viewed from above",
    "vehicles moving along a highway in a panoramic shot",
    "armoured vehicles or tanks in an open field",
    "trucks and heavy transport vehicles at an industrial site",
    "emergency vehicles with lights visible from aerial view",
    "vehicles parked in an open area viewed from a drone",
    "mobile radar unit mounted on a truck in a field",
    "radar vehicle or electronic warfare truck in a convoy",
    "surveillance vehicle with antenna array on a road",
    "small vehicles weaving in a serpentine pattern along a road",
    "tiny cars following a zigzag slalom course on a wide road",
    "overhead view of vehicles navigating obstacles in a serpentine layout",
    "small objects moving in curved paths on a straight road from above",
    "miniature vehicles visible as small dots arranged in a winding line",
    "drone view of traffic slowing and weaving around road obstacles",
    "serpentine convoy of small vehicles on an open road from altitude",
    "simple portable radar unit on a tripod in a field",
    "small ground surveillance radar deployed on the roadside",
    "handheld or man-portable radar device in open terrain",
    "compact radar sensor on a pole or mast near a road",
    "short-range radar unit with small dish antenna on the ground",
    "mobile radar system on a lightweight trailer or cart",
    "radar detector or traffic speed radar on a road",
]

# ── Gemma analysis constants ──────────────────────────────────────────────────

_GEMMA_ANALYSIS_SAMPLE_N = 30    # max frames sampled per video for Gemma analysis
_SCENE_CHANGE_THRESH     = 0.25  # cosine distance threshold for scene change detection

# Text probes for cross-modal search and zero-shot classification
_GEMMA_TEXT_PROBES: List[str] = [
    "aerial view of open terrain",
    "military vehicle or equipment",
    "road or highway from above",
    "buildings and urban infrastructure",
    "natural landscape or vegetation",
    "radar or surveillance equipment",
    "convoy or vehicle formation",
    "industrial or construction site",
    "open field or farmland",
    "coastal or water feature",
]

# ── VideoKnowledge — agentic knowledge accumulator ────────────────────────────


class VideoKnowledge:
    """Accumulates structured observations across pipeline steps.

    Each step deposits its output here via ``add_*`` methods.  Later steps
    query ``context_for_frame(t_sec)`` or ``domain_hint()`` to receive all
    prior knowledge relevant to that moment in the video.

    Design goals:
    - Per-frame context: Florence caption + depth profile + detections +
      ASR + OCR + scene segment — all aligned by timestamp.
    - Domain summary: Gemma scene type + entity inventory → enriches Qwen
      system prompt and Florence domain hint.
    - Continuity: previous Qwen structured results feed back as "prior state"
      for the next Qwen call, letting the model track what changed.
    """

    def __init__(self, video_name: str, duration_sec: float, frame_count: int) -> None:
        self.video_name   = video_name
        self.duration_sec = duration_sec
        self.frame_count  = frame_count

        # Gemma-derived domain knowledge (step J)
        self.scene_type: str       = ""   # dominant zero-shot category
        self.n_transitions: int    = 0
        self.n_clusters: int       = 0
        self.gemma_mnn_dino: float = 0.0

        # Per-frame outputs keyed by t_sec (steps L, M, N, O, P)
        self._captions:   Dict[float, str]        = {}  # Florence caption text
        self._asr:        Dict[float, str]         = {}  # ASR subtitle text
        self._ocr:        Dict[float, str]         = {}  # OCR visible text
        self._depth:      Dict[float, Dict]        = {}  # depth summary dict
        self._detections: Dict[float, List[str]]  = {}  # detected labels at t

        # Sorted timestamp index for nearest-frame lookups
        self._ts_captions:   List[float] = []
        self._ts_depth:      List[float] = []
        self._ts_detections: List[float] = []

        # Scene segments from caption analysis (step L enrichment)
        self._segments: List[Dict[str, Any]] = []

        # Entity inventory: all distinct labels seen across all frames
        self.known_entities: List[str] = []

        # Last Qwen result: feeds into next Qwen call as "previous state"
        self._last_qwen: Dict[str, Any] = {}

    # ── Deposit methods ───────────────────────────────────────────────────────

    def add_gemma(self, task_results: Dict[str, Any], mnn_dino: float = 0.0) -> None:
        """Deposit Gemma analysis results (step J)."""
        clf = task_results.get("scene_classification", {})
        if clf.get("category_distribution"):
            self.scene_type = next(iter(clf["category_distribution"]), "")
        sc = task_results.get("scene_change_detection", {})
        self.n_transitions = sc.get("n_changes", 0)
        cl = task_results.get("scene_clustering", {})
        self.n_clusters   = cl.get("n_clusters", 0)
        self.gemma_mnn_dino = mnn_dino

    def add_captions(self, caption_results: List[Dict[str, Any]]) -> None:
        """Deposit Florence per-frame captions (step L) and derive segments."""
        self._captions   = {r["t_sec"]: r.get("caption") or "" for r in caption_results if "t_sec" in r}
        self._ts_captions = sorted(self._captions)
        # Re-use existing segment analysis
        enriched = _analyze_caption_sequence(caption_results)
        seg_map: Dict[int, Dict[str, Any]] = {}
        for r in enriched:
            sid = r["segment_id"]
            if sid not in seg_map:
                seg_map[sid] = {"segment_id": sid, "start_t": r["t_sec"],
                                "end_t": r["t_sec"], "caption": r.get("caption") or ""}
            else:
                seg_map[sid]["end_t"] = r["t_sec"]
        self._segments = [seg_map[k] for k in sorted(seg_map)]

    def add_asr(self, subtitle_map: Dict[float, str]) -> None:
        """Deposit ASR subtitle map (step M)."""
        self._asr = {float(k): v for k, v in subtitle_map.items() if v}

    def add_ocr(self, ocr_results: List[Dict[str, Any]]) -> None:
        """Deposit OCR per-frame results (step N)."""
        self._ocr = {r["t_sec"]: r["ocr_text"] for r in ocr_results
                     if r.get("ocr_text") and "t_sec" in r}

    def add_depth(self, depth_results: List[Dict[str, Any]]) -> None:
        """Deposit depth estimation per-frame results (step O)."""
        self._depth = {r["t_sec"]: r for r in depth_results if "t_sec" in r}
        self._ts_depth = sorted(self._depth)

    def add_detections(self, detection_results: List[Dict[str, Any]]) -> None:
        """Deposit object detection per-frame results (step P)."""
        entity_set: set = set()
        for r in detection_results:
            t = r.get("t_sec")
            if t is None:
                continue
            labels = [d["label"] for d in r.get("detections", []) if d.get("label")]
            self._detections[float(t)] = labels
            entity_set.update(labels)
        self._ts_detections = sorted(self._detections)
        # Keep top entities by frequency
        counts: Dict[str, int] = {}
        for labels in self._detections.values():
            for lbl in labels:
                counts[lbl] = counts.get(lbl, 0) + 1
        self.known_entities = [k for k, _ in sorted(counts.items(), key=lambda x: -x[1])[:15]]

    def update_qwen_state(self, result: Dict[str, Any]) -> None:
        """Record the most recent Qwen output for use as prior state context."""
        if not result.get("service_unavailable") and not result.get("parse_error"):
            self._last_qwen = result

    # ── Query methods ─────────────────────────────────────────────────────────

    def domain_hint(self) -> str:
        """Short domain summary for use as a model prompt prefix."""
        parts: List[str] = []
        if self.scene_type:
            parts.append(f"Dominant scene: {self.scene_type}")
        if self.known_entities:
            parts.append(f"Known objects: {', '.join(self.known_entities[:6])}")
        if self.n_transitions:
            parts.append(f"Visual transitions: {self.n_transitions}")
        return " | ".join(parts)

    def context_for_frame(self, t_sec: float, asr_window: float = 2.0) -> str:
        """Build a multi-line context string for *t_sec* from all deposited knowledge.

        Returned string is injected into Qwen's user prompt so it can
        reason with full situational awareness, not just the raw image.
        """
        lines: List[str] = []

        # Florence caption for this frame
        cap = self._nearest(self._ts_captions, self._captions, t_sec, max_gap=2.0)
        if cap:
            lines.append(f"[Prior scene description]: {cap[:150]}")

        # Current scene segment
        seg = self._segment_at(t_sec)
        if seg and seg.get("caption"):
            lines.append(
                f"[Scene segment {seg['segment_id']+1}, "
                f"{seg['start_t']:.1f}s–{seg['end_t']:.1f}s]: "
                f"{seg['caption'][:120]}"
            )

        # ASR in window
        asr_parts = [txt for ts, txt in self._asr.items()
                     if abs(ts - t_sec) <= asr_window]
        if asr_parts:
            lines.append(f"[Audio context]: {' '.join(asr_parts)[:120]}")

        # OCR exact or ±1 s
        ocr_parts = [txt for ts, txt in self._ocr.items() if abs(ts - t_sec) <= 1.0]
        if ocr_parts:
            lines.append(f"[Visible text]: {' '.join(ocr_parts)[:100]}")

        # Depth profile
        dep = self._nearest(self._ts_depth, self._depth, t_sec, max_gap=2.0)
        if dep:
            nr  = dep.get("near_ratio",  dep.get("near_frac",  0.0))
            mn  = dep.get("mean_depth",  dep.get("median",     0.0))
            if nr or mn:
                lines.append(f"[Depth profile]: near_ratio={nr:.2f}  mean={mn:.2f}")

        # Detected objects at this timestamp
        dets = self._nearest(self._ts_detections, self._detections, t_sec, max_gap=2.0)
        if dets:
            lines.append(f"[Detected objects]: {', '.join(dets[:8])}")

        # Prior Qwen state (what the model extracted from the previous frame)
        if self._last_qwen:
            prev_vg = self._last_qwen.get("vehicle_groups", [])
            prev_road = self._last_qwen.get("road_surface", "")
            prev_cond = self._last_qwen.get("road_condition", "")
            if prev_vg or prev_road:
                vg_str = "; ".join(
                    f"{g.get('count', 1)}×{g.get('type', '?')}" for g in prev_vg
                ) if prev_vg else "none"
                lines.append(
                    f"[Prior frame state]: vehicles={vg_str}  "
                    f"road={prev_road}  condition={prev_cond}"
                )

        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _nearest(ts_index: List[float], data: Dict, t: float, max_gap: float = 5.0):
        """Return the value in *data* whose key is closest to *t*, within *max_gap*."""
        if not ts_index:
            return None
        idx = min(range(len(ts_index)), key=lambda i: abs(ts_index[i] - t))
        if abs(ts_index[idx] - t) <= max_gap:
            return data.get(ts_index[idx])
        return None

    def _segment_at(self, t: float) -> Optional[Dict[str, Any]]:
        """Return the scene segment that contains timestamp *t*."""
        for seg in self._segments:
            if seg["start_t"] <= t <= seg["end_t"] + 0.5:
                return seg
        return self._segments[-1] if self._segments else None


# ── Markdown writers ───────────────────────────────────────────────────────────

_RUNNER_LABEL = "demo pipeline (`main.py --mode demo`)"


def _md_image(rel_path: str, alt: str = "frame") -> str:
    return f"![{alt}]({rel_path})"


def write_search_md(
    output_path: Path,
    video_name: str,
    model_label: str,
    query_frame: str,
    results: List[Dict[str, Any]],
    query_t_sec: float,
) -> None:
    if output_path.exists():
        _log.info("  Skipping %s (already exists)", output_path.name)
        return
    lines = [
        f"# {model_label} Transformation Test — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Model: {model_label}",
        f"",
        f"## Query Frame",
        f"",
        f"**Timestamp:** {query_t_sec:.2f}s",
        f"",
        _md_image(os.path.relpath(query_frame, output_path.parent), "Query frame"),
        f"",
        f"## Top {len(results)} Similar Frames",
        f"",
        f"| Rank | Score | Timestamp | Frame |",
        f"|------|-------|-----------|-------|",
    ]
    for i, r in enumerate(results, 1):
        payload = r.get("payload", r)
        fp = payload.get("frame_path", "")
        t  = payload.get("t_sec", 0.0)
        score = r.get("score", 0.0)
        rel = os.path.relpath(fp, output_path.parent) if fp else ""
        lines.append(f"| {i} | {score:.4f} | {t:.2f}s | {_md_image(rel, f'match {i}')} |")
    lines += ["", "---", f"*Artifact produced by {_RUNNER_LABEL}.*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def _jaccard(a: str, b: str) -> float:
    """Token-overlap similarity between two caption strings (0=different, 1=identical)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _analyze_caption_sequence(
    caption_results: List[Dict[str, Any]],
    new_segment_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """Annotate caption results with temporal segment info.

    Adds to each result:
        segment_id       — integer, increments when caption content changes
        is_new_segment   — True for first frame of each segment
        similarity       — Jaccard similarity to previous frame's caption (None for first)
        segment_start_t  — t_sec of the first frame in this segment
    """
    enriched: List[Dict[str, Any]] = []
    seg_id = 0
    prev_caption = ""
    seg_start_t = 0.0

    for i, r in enumerate(caption_results):
        cap = (r.get("caption") or "").strip()
        if i == 0:
            sim = None
            is_new = True
            seg_start_t = r.get("t_sec", 0.0)
        else:
            sim = _jaccard(prev_caption, cap)
            if sim < new_segment_threshold:
                seg_id += 1
                is_new = True
                seg_start_t = r.get("t_sec", 0.0)
            else:
                is_new = False

        enriched.append({
            **r,
            "segment_id": seg_id,
            "is_new_segment": is_new,
            "similarity": sim,
            "segment_start_t": seg_start_t,
        })
        if cap:
            prev_caption = cap

    return enriched


def _diff_structured_caption(prev: Dict[str, Any], curr: Dict[str, Any]) -> str:
    """Return a short string describing what changed between two Qwen structured dicts."""
    changes: List[str] = []

    prev_surface = prev.get("road_surface", "unknown")
    curr_surface = curr.get("road_surface", "unknown")
    if prev_surface != curr_surface:
        changes.append(f"road: {prev_surface}→{curr_surface}")

    prev_cond = prev.get("road_condition", "unknown")
    curr_cond = curr.get("road_condition", "unknown")
    if prev_cond != curr_cond:
        changes.append(f"condition: {prev_cond}→{curr_cond}")

    def _vehicle_signature(groups: list) -> Dict[str, int]:
        sig: Dict[str, int] = {}
        for g in (groups or []):
            vtype = g.get("type", "other")
            sig[vtype] = sig.get(vtype, 0) + int(g.get("count") or 1)
        return sig

    prev_sig = _vehicle_signature(prev.get("vehicle_groups", []))
    curr_sig = _vehicle_signature(curr.get("vehicle_groups", []))
    if prev_sig != curr_sig:
        if not prev_sig and curr_sig:
            changes.append("vehicles appeared")
        elif prev_sig and not curr_sig:
            changes.append("vehicles left")
        else:
            all_types = set(prev_sig) | set(curr_sig)
            for vt in sorted(all_types):
                p = prev_sig.get(vt, 0)
                c = curr_sig.get(vt, 0)
                if p != c:
                    changes.append(f"{vt}: {p}→{c}")

    return "; ".join(changes) if changes else ""


def write_scene_captions_md(
    output_path: Path,
    video_name: str,
    caption_results: List[Dict[str, Any]],
    elapsed_sec: float,
) -> None:
    enriched = _analyze_caption_sequence(caption_results)

    # Build segment-level summary
    segments: List[Dict[str, Any]] = []
    for r in enriched:
        if r["is_new_segment"]:
            segments.append({
                "segment_id": r["segment_id"],
                "start_t": r["t_sec"],
                "end_t": r["t_sec"],
                "caption": r.get("caption") or "",
                "frame_count": 1,
            })
        elif segments:
            segments[-1]["end_t"] = r["t_sec"]
            segments[-1]["frame_count"] += 1

    n_segments = len(segments)
    n_unchanged = sum(1 for r in enriched if not r["is_new_segment"])

    lines = [
        f"# Scene Captions — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: Florence-2-large (MORE_DETAILED_CAPTION)",
        f"Frames captioned: {len(caption_results)}  |  Unique scenes: {n_segments}"
        f"  |  Repeated frames: {n_unchanged}",
        f"Elapsed: {elapsed_sec:.1f}s",
        f"",
        f"## Scene Timeline",
        f"",
        f"| # | Start (s) | End (s) | Frames | Caption |",
        f"|---|-----------|---------|--------|---------|",
    ]
    for seg in segments:
        cap = seg["caption"].replace("|", "\\|")[:200]
        lines.append(
            f"| {seg['segment_id'] + 1} | {seg['start_t']:.1f}"
            f" | {seg['end_t']:.1f} | {seg['frame_count']} | {cap} |"
        )

    lines += [
        f"",
        f"## Per-Frame Captions",
        f"",
        f"Frames with similarity ≥ 0.45 to the previous caption are marked *same scene*.",
        f"",
        f"| Frame | t (s) | Seg | Sim | Confidence | Caption |",
        f"|-------|-------|-----|-----|------------|---------|",
    ]
    for r in enriched:
        fp   = r.get("frame_path", "")
        name = Path(fp).name if fp else "—"
        t    = r.get("t_sec", 0.0)
        conf = r.get("caption_confidence", 0.0) or 0.0
        cap  = (r.get("caption") or "").replace("|", "\\|")
        seg  = r["segment_id"] + 1
        sim  = r["similarity"]
        sim_str = f"{sim:.2f}" if sim is not None else "—"
        if not r["is_new_segment"]:
            cap = f"*same scene* {cap}"
        lines.append(f"| `{name}` | {t:.1f} | {seg} | {sim_str} | {conf:.3f} | {cap} |")

    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · Florence-2-large · phase1 captioning*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_gemma_analysis_md(
    output_path: Path,
    video_name: str,
    model_id: str,
    sample_n: int,
    analysis: Dict[str, Any],
    dino_comparison: Dict[str, Any],
    text_query_results: List[Dict[str, Any]],
    elapsed_sec: float,
    clip_comparison: Optional[Dict[str, Any]] = None,
) -> None:
    """Write Gemma multimodal analysis report to *output_path*."""
    lines = [
        f"# Gemma Open-Weight Analysis — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{model_id}`  |  Frames sampled: {sample_n}  |  Elapsed: {elapsed_sec:.1f}s",
        f"",
        f"## Analyses Performed",
        f"",
        f"| Analysis | Status |",
        f"|----------|--------|",
    ]
    for key, res in analysis.items():
        label = key.replace("_", " ").title()
        if res.get("error"):
            status = f"✗ {res['error'][:60]}"
        else:
            status = "✓"
        lines.append(f"| {label} | {status} |")
    lines += [""]

    # DINOv3 comparison
    if dino_comparison.get("available"):
        mnn = dino_comparison.get("mnn_rate", 0.0)
        k   = dino_comparison.get("k", 5)
        cg  = dino_comparison.get("mean_cossim_gemma", 0.0)
        cd  = dino_comparison.get("mean_cossim_dino", 0.0)
        lines += [
            f"## Gemma vs DINOv3 Embedding Comparison",
            f"",
            f"Both models embedded the same {dino_comparison.get('n_frames', sample_n)} frames.",
            f"Gemma model: `{model_id}`.  DINOv3 model: `dinov3_vitb14`.",
            f"",
            f"| Metric | Gemma | DINOv3 |",
            f"|--------|-------|--------|",
            f"| Mean pairwise cosine similarity | {cg:.4f} | {cd:.4f} |",
            f"| Mutual nearest-neighbor overlap (k={k}) | {mnn:.3f} | — |",
            f"",
            f"**Mean pairwise cosine similarity**: lower = more discriminative embedding space.",
            f"",
            f"**MNN@{k}** ({mnn:.1%}): fraction of frames whose top-{k} visual neighbours agree",
            f"between Gemma and DINOv3.",
            f"",
        ]
    else:
        lines += [
            f"## Gemma vs DINOv3 Embedding Comparison",
            f"",
            f"Skipped: {dino_comparison.get('reason', 'DINOv3 not available')}",
            f"",
        ]

    # CLIP comparison
    cc = clip_comparison or {}
    if cc.get("available"):
        mnn_c = cc.get("mnn_rate", 0.0)
        k_c   = cc.get("k", 5)
        cg_c  = cc.get("mean_cossim_gemma", 0.0)
        cl_c  = cc.get("mean_cossim_clip", 0.0)
        lines += [
            f"## Gemma vs CLIP Embedding Comparison",
            f"",
            f"Both models embedded the same {cc.get('n_frames', sample_n)} frames.",
            f"Gemma model: `{model_id}`.  CLIP model: `ViT-B-16/openai`.",
            f"",
            f"| Metric | Gemma | CLIP |",
            f"|--------|-------|------|",
            f"| Mean pairwise cosine similarity | {cg_c:.4f} | {cl_c:.4f} |",
            f"| Mutual nearest-neighbor overlap (k={k_c}) | {mnn_c:.3f} | — |",
            f"",
            f"**MNN@{k_c}** ({mnn_c:.1%}): fraction of frames whose top-{k_c} visual neighbours agree",
            f"between Gemma and CLIP.",
            f"",
        ]
    elif cc:
        lines += [
            f"## Gemma vs CLIP Embedding Comparison",
            f"",
            f"Skipped: {cc.get('reason', 'CLIP not available')}",
            f"",
        ]

    # Scene change detection
    sc = analysis.get("scene_change_detection", {})
    if not sc.get("error") and sc.get("changes") is not None:
        changes = sc.get("changes", [])
        lines += [
            f"## Scene Change Detection",
            f"",
            f"Cosine distance > {_SCENE_CHANGE_THRESH} between consecutive sampled frames.",
            f"Detected {sc.get('n_changes', 0)} transition(s).",
            f"",
        ]
        if changes:
            lines += [f"| # | t (s) | Cosine Distance |", f"|---|-------|-----------------|"]
            for i, ch in enumerate(changes[:15], 1):
                lines.append(f"| {i} | {ch['t_sec']:.1f} | {ch['distance']:.4f} |")
            lines += [""]

    # Zero-shot classification
    clf = analysis.get("scene_classification", {})
    if not clf.get("error") and clf.get("category_distribution"):
        lines += [
            f"## Zero-Shot Scene Classification",
            f"",
            f"Top predicted scene categories across {sample_n} frames:",
            f"",
            f"| Category | Frame Count |",
            f"|----------|-------------|",
        ]
        for cat, cnt in clf["category_distribution"].items():
            lines.append(f"| {cat} | {cnt} |")
        lines += [""]

    # Cross-modal text queries
    if text_query_results:
        lines += [
            f"## Cross-Modal Text → Frame Retrieval",
            f"",
            f"Text probes (mean-pooled text embeddings) vs frame embeddings (cosine similarity):",
            f"",
            f"| Query | Best Frame (t) | Score |",
            f"|-------|---------------|-------|",
        ]
        for qr in text_query_results:
            q   = qr.get("query", "—")
            top = qr.get("top_results", [])
            if top:
                fp    = Path(top[0].get("frame_path", "")).name
                t_s   = top[0].get("t_sec", 0.0)
                score = top[0].get("score", 0.0)
                lines.append(f"| {q} | `{fp}` ({t_s:.1f}s) | {score:.4f} |")
            else:
                lines.append(f"| {q} | — | — |")
        lines += [""]

    # Temporal video embedding
    te = analysis.get("temporal_embedding", {})
    if not te.get("error"):
        lines += [
            f"## Temporal Video Embedding",
            f"",
            f"Mean-pool of all {sample_n} frame embeddings → single video-level vector",
            f"(dim={te.get('dim', 0)}).  Can be used for video-level retrieval or comparison.",
            f"",
        ]

    # Clustering
    cl = analysis.get("scene_clustering", {})
    if not cl.get("error") and cl.get("n_clusters"):
        lines += [
            f"## Scene Clustering",
            f"",
            f"{cl['n_clusters']} semantic clusters from {sample_n} frames",
            f"(mean cluster size: {cl.get('mean_cluster_size', 0):.1f} frames).",
            f"",
        ]

    # ── Analysis interpretation ───────────────────────────────────────────────
    lines += ["## Findings & Interpretation", ""]

    # Embedding discrimination
    dino_avail = dino_comparison.get("available", False)
    cc = clip_comparison or {}
    clip_avail = cc.get("available", False)

    if dino_avail:
        cg = dino_comparison.get("mean_cossim_gemma", 0.0)
        cd = dino_comparison.get("mean_cossim_dino", 0.0)
        mnn = dino_comparison.get("mnn_rate", 0.0)
        if cg < cd:
            lines.append(
                f"- **Gemma is more discriminative than DINOv3** for this video "
                f"(mean cosine {cg:.4f} < {cd:.4f}). Gemma's language-grounded embeddings "
                f"spread frames further apart in embedding space — useful for precise retrieval."
            )
        elif abs(cg - cd) < 0.05:
            lines.append(
                f"- **Gemma and DINOv3 have similar discrimination** (cosine {cg:.4f} vs {cd:.4f}). "
                f"Both models capture similar visual structure for this mission content."
            )
        else:
            lines.append(
                f"- **DINOv3 is more discriminative than Gemma** for this video "
                f"(cosine {cd:.4f} < {cg:.4f}). DINOv3's self-supervised visual features "
                f"give finer-grained distinctions. Gemma remains valuable for language-grounded queries."
            )
        if mnn >= 0.8:
            lines.append(
                f"- **High DINOv3↔Gemma agreement (MNN={mnn:.1%})**: both models agree on which "
                f"frames are visually similar. Gemma embeddings can safely substitute DINOv3 for "
                f"retrieval with additional benefit of text-query compatibility."
            )
        elif mnn >= 0.5:
            lines.append(
                f"- **Moderate DINOv3↔Gemma agreement (MNN={mnn:.1%})**: the models partially "
                f"disagree on visual neighbourhoods. Gemma captures semantic similarity; DINOv3 "
                f"captures low-level visual similarity. Both are complementary — use Gemma for "
                f"text queries, DINOv3 for image-to-image search."
            )
        else:
            lines.append(
                f"- **Low DINOv3↔Gemma agreement (MNN={mnn:.1%})**: the models assign very "
                f"different neighbourhoods. Likely cause: 30 fps near-duplicate frames collapse "
                f"to the same DINOv3 cluster while Gemma's language bias separates them differently. "
                f"This is expected and not a failure — the two spaces serve different query types."
            )
        lines.append("")

    if clip_avail:
        mnn_c = cc.get("mnn_rate", 0.0)
        if mnn_c >= 0.8:
            lines.append(
                f"- **High CLIP↔Gemma agreement (MNN={mnn_c:.1%})**: Gemma embeddings are "
                f"strongly aligned with CLIP's image-text space. Gemma can replace CLIP for "
                f"cross-modal retrieval while also supporting image-to-image search."
            )
        elif mnn_c >= 0.5:
            lines.append(
                f"- **Moderate CLIP↔Gemma agreement (MNN={mnn_c:.1%})**: Gemma and CLIP agree "
                f"on roughly half of visual neighbourhoods. Use CLIP for image-text matching "
                f"and Gemma for richer structured reasoning."
            )
        else:
            lines.append(
                f"- **Low CLIP↔Gemma agreement (MNN={mnn_c:.1%})**: Gemma organises this "
                f"visual content differently from CLIP. Gemma may be using scene-level semantics "
                f"while CLIP relies on global appearance statistics."
            )
        lines.append("")

    # Scene change detection
    sc = analysis.get("scene_change_detection", {})
    n_changes = sc.get("n_changes", 0)
    if not sc.get("error") and sc.get("changes") is not None:
        if n_changes == 0:
            lines.append(
                f"- **No scene transitions detected**: all {sample_n} sampled frames are "
                f"visually continuous. This is typical of 30 fps missions where scenes evolve slowly. "
                f"Use the Scene Timeline in `scene_captions.md` for segment-level analysis."
            )
        elif n_changes <= 3:
            lines.append(
                f"- **{n_changes} scene transition(s)**: the video has a small number of "
                f"distinct visual states. Gemma embedding distances reliably flag these transitions "
                f"as higher-priority frames for annotation (`al_tag=needs_annotation`)."
            )
        else:
            lines.append(
                f"- **{n_changes} scene transitions**: high visual variability in this mission. "
                f"Frames at transition boundaries carry the most novel information and should be "
                f"prioritised for SSL training data."
            )
        lines.append("")

    # Clustering
    cl = analysis.get("scene_clustering", {})
    n_clusters = cl.get("n_clusters", 0)
    mean_sz = cl.get("mean_cluster_size", 0)
    if not cl.get("error") and n_clusters:
        if mean_sz > sample_n * 0.3:
            lines.append(
                f"- **Few, large clusters ({n_clusters} clusters, ~{mean_sz:.0f} frames each)**: "
                f"the mission covers a small set of visually distinct scenes. "
                f"SSL temporal pairs will be highly informative — nearby frames share the same cluster."
            )
        else:
            lines.append(
                f"- **Many small clusters ({n_clusters} clusters, ~{mean_sz:.0f} frames each)**: "
                f"high scene diversity. More SSL epochs may be needed to cover all visual states."
            )
        lines.append("")

    # Distillation recommendation
    if dino_avail:
        mnn_d = dino_comparison.get("mnn_rate", 0.0)
        if mnn_d >= 0.7:
            lines.append(
                "- **Distillation recommendation**: Gemma embeddings are a strong teacher signal. "
                "Set `gemma_embedder` in `step_distill` (done automatically when `MODEL_NAME=gemma`) "
                "for maximum-hydration distillation — the student inherits both visual and language-grounded structure."
            )
        else:
            lines.append(
                "- **Distillation recommendation**: Gemma and DINOv3 neighbourhoods diverge for "
                "this content. Run both distillation chains and compare Recall@1: "
                "DINOv3-teacher for image retrieval, Gemma-teacher for text-query tasks."
            )
        lines.append("")

    lines += ["---", f"*Produced by {_RUNNER_LABEL} — Gemma open-weight multimodal analysis.*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def _loss_sparkline(history: List[float], width: int = 40) -> str:
    """Return a fixed-width ASCII sparkline for a loss curve.

    Uses Unicode block elements ▁▂▃▄▅▆▇█ to represent relative height.
    Values are normalised to [0, 1] then mapped to 8 levels.
    """
    if not history:
        return "(no data)"
    blocks = " ▁▂▃▄▅▆▇█"
    lo, hi = min(history), max(history)
    span = hi - lo if hi > lo else 1.0
    # Sample evenly if more epochs than width
    if len(history) > width:
        step = len(history) / width
        sampled = [history[int(i * step)] for i in range(width)]
    else:
        sampled = list(history)
    chars = [blocks[min(8, int(((v - lo) / span) * 8) + 1)] for v in sampled]
    return "".join(chars)


def _analyze_loss_curve(history: List[float]) -> Dict[str, Any]:
    """Compute summary statistics for a training loss curve."""
    if not history:
        return {}
    n = len(history)
    first, last = history[0], history[-1]
    best = min(history)
    best_epoch = int(np.argmin(history)) + 1

    # Total relative drop
    drop_pct = (first - best) / first * 100 if first > 0 else 0.0

    # Convergence epoch: first epoch within 5 % of best loss
    threshold = best * 1.05
    convergence_epoch = next(
        (i + 1 for i, v in enumerate(history) if v <= threshold), best_epoch
    )

    # Monotone check: count epochs where loss increased
    increases = sum(1 for a, b in zip(history, history[1:]) if b > a)

    # Plateau: last 20 % of epochs — std relative to mean
    tail = history[max(0, n - max(2, n // 5)):]
    tail_mean = float(np.mean(tail)) if tail else float("nan")
    tail_std  = float(np.std(tail))  if tail else float("nan")
    plateau_cv = (tail_std / tail_mean) if tail_mean > 0 else float("nan")

    # Epoch-over-epoch deltas
    deltas = [b - a for a, b in zip(history, history[1:])]
    avg_drop_per_epoch = float(np.mean(deltas)) if deltas else 0.0

    return {
        "n_epochs": n,
        "first_loss": first,
        "last_loss": last,
        "best_loss": best,
        "best_epoch": best_epoch,
        "drop_pct": drop_pct,
        "convergence_epoch": convergence_epoch,
        "n_increases": increases,
        "plateau_cv": plateau_cv,
        "avg_drop_per_epoch": avg_drop_per_epoch,
        "deltas": deltas,
    }


def _interpret_finetune_results(
    cfg: "FinetuneConfig",
    stats: Dict[str, Any],
    elapsed_sec: float,
) -> List[str]:
    """Return a list of Markdown bullet-point strings interpreting the training run."""
    if not stats:
        return ["*No training data — stats unavailable.*"]

    bullets: List[str] = []
    drop   = stats["drop_pct"]
    best   = stats["best_loss"]
    best_e = stats["best_epoch"]
    n      = stats["n_epochs"]
    cv     = stats["plateau_cv"]
    incr   = stats["n_increases"]
    conv_e = stats["convergence_epoch"]

    # --- Approach explanation ---
    if cfg.approach == "temporal":
        bullets.append(
            "**Approach (temporal pairs):** Consecutive frames from the mission video "
            "form *positive pairs* under the assumption that nearby frames show the same "
            "scene. The model learns to pull these embeddings together and push apart "
            "embeddings from different timesteps (negatives within the same batch). "
            "This is the preferred approach when enough frames are available (≥ 2 × batch size)."
        )
    else:
        bullets.append(
            "**Approach (augmentation pairs):** Each frame is augmented twice with random "
            "crops, flips, colour jitter, and Gaussian blur to produce a positive pair. "
            "The model learns viewpoint- and appearance-invariant representations. "
            "This approach is used when the frame count is too low for temporal pairing."
        )

    # --- Loss magnitude ---
    if best < 0.5:
        loss_comment = "excellent — the model has learned tight, well-separated embeddings"
    elif best < 1.5:
        loss_comment = "good — further epochs or a lower LR could improve it"
    elif best < 3.0:
        loss_comment = "moderate — consider more epochs, a lower temperature, or more frames"
    else:
        loss_comment = "high — the model may not have converged; try more epochs or check data quality"
    bullets.append(f"**Best loss ({best:.4f}):** {loss_comment}.")

    # --- Drop ---
    if drop > 40:
        bullets.append(
            f"**Loss drop ({drop:.1f} %):** Large improvement over training — "
            "the backbone adapted meaningfully to this mission's visual domain."
        )
    elif drop > 15:
        bullets.append(
            f"**Loss drop ({drop:.1f} %):** Moderate improvement — "
            "the model captured some domain-specific structure."
        )
    else:
        bullets.append(
            f"**Loss drop ({drop:.1f} %):** Small improvement — "
            "the pre-trained weights already generalise well, or training was too short."
        )

    # --- Convergence ---
    if conv_e <= n // 3:
        bullets.append(
            f"**Convergence (epoch {conv_e}/{n}):** Loss converged early. "
            "Remaining epochs did not help much — future runs can use fewer epochs."
        )
    elif conv_e >= int(n * 0.85):
        bullets.append(
            f"**Convergence (epoch {conv_e}/{n}):** Loss was still improving near the end. "
            "Training more epochs would likely yield a lower loss."
        )
    else:
        bullets.append(
            f"**Convergence (epoch {conv_e}/{n}):** Loss stabilised in the middle of training — "
            "epoch budget looks appropriate."
        )

    # --- Best epoch position ---
    if best_e < n:
        bullets.append(
            f"**Best checkpoint (epoch {best_e}/{n}):** Loss increased in later epochs, "
            "suggesting slight overfitting or LR too high at the end. "
            "The saved checkpoint is from epoch {best_e}."
        )
    else:
        bullets.append(
            f"**Best checkpoint (epoch {best_e}/{n}):** Best loss was at the final epoch — "
            "the run had not overfit."
        )

    # --- Plateau ---
    if not math.isnan(cv):
        if cv < 0.01:
            bullets.append(
                f"**Plateau (CV={cv:.4f}):** Loss is flat in the final epochs — "
                "the model has converged and additional epochs are unlikely to help."
            )
        elif cv < 0.05:
            bullets.append(
                f"**Plateau (CV={cv:.4f}):** Minor oscillation in the final epochs — "
                "training is mostly converged."
            )
        else:
            bullets.append(
                f"**Plateau (CV={cv:.4f}):** Noisy loss in the final epochs — "
                "try reducing LR or increasing batch size for more stable convergence."
            )

    # --- Non-monotone ---
    if incr > n // 4:
        bullets.append(
            f"**Instability ({incr} loss increases out of {n - 1} steps):** "
            "Loss oscillated frequently — consider a lower learning rate or larger batch size."
        )

    # --- Speed ---
    secs_per_epoch = elapsed_sec / n if n else 0
    bullets.append(
        f"**Training speed:** {elapsed_sec:.1f}s total, "
        f"~{secs_per_epoch:.1f}s/epoch on `{cfg.device}`."
    )

    return bullets


def write_finetune_stats_md(
    output_path: Path,
    video_name: str,
    cfg: "FinetuneConfig",
    best_loss: float,
    checkpoint_path: str,
    elapsed_sec: float,
    loss_history: List[float],
) -> None:
    ckpt_mb    = os.path.getsize(checkpoint_path) / 1e6 if os.path.exists(checkpoint_path) else 0
    best_epoch = int(np.argmin(loss_history)) + 1 if loss_history else 0
    stats      = _analyze_loss_curve(loss_history)
    sparkline  = _loss_sparkline(loss_history)
    deltas     = stats.get("deltas", [])
    bullets    = _interpret_finetune_results(cfg, stats, elapsed_sec)

    lines = [
        f"# SSL Fine-Tuning Statistics — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## What We Do",
        f"",
        f"**Self-Supervised Learning (SSL)** adapts a pre-trained vision backbone to the "
        f"specific visual domain of this mission without any labelled annotations.",
        f"",
        f"### Method: NT-Xent Contrastive Loss",
        f"",
        f"We use **NT-Xent** (Normalised Temperature-scaled Cross Entropy, a.k.a. InfoNCE) "
        f"contrastive learning:",
        f"",
        f"1. Each training step produces a batch of *positive pairs* (two views of the same scene).",
        f"2. The model encodes both views through the DINOv3 backbone + a small projection head "
        f"   (embed_dim={cfg.embed_dim} → proj_dim={cfg.proj_out_dim}).",
        f"3. The loss pushes the two views of the same scene together in embedding space "
        f"   and pushes all other pairs in the batch apart.",
        f"4. Temperature τ={cfg.temperature} controls the sharpness of the distribution "
        f"   (lower = harder negatives, more informative but less stable).",
        f"",
        f"The backbone is **partially frozen**: the first {cfg.freeze_blocks} transformer blocks "
        f"are kept fixed (preserving generic low-level features), and only the top "
        f"{12 - cfg.freeze_blocks} blocks + projection head are trained. "
        f"This prevents catastrophic forgetting on a small video dataset.",
        f"",
        f"### Pair Construction Strategy: `{cfg.approach}`",
        f"",
    ]

    if cfg.approach == "temporal":
        lines += [
            f"**Temporal pairs** — consecutive frames within ±{cfg.max_gap} positions "
            f"in the frame sequence form positive pairs.",
            f"Rationale: adjacent frames in a 30 fps outdoor video show nearly the same scene, "
            f"so pulling their embeddings together teaches the model scene-level consistency "
            f"while naturally using real mission content (no synthetic augmentation needed).",
        ]
    else:
        lines += [
            f"**Augmentation pairs** — each frame is augmented twice with random crops, "
            f"horizontal flips, colour jitter, and Gaussian blur.",
            f"Rationale: fewer than {cfg.batch_size * 2} frames are available, so temporal "
            f"pairing would produce too few unique positive pairs. "
            f"Augmentation-based SSL is used as a fallback.",
        ]

    lines += [
        f"",
        f"### Optimiser",
        f"",
        f"| Component | Setting |",
        f"|-----------|---------|",
        f"| Optimiser | AdamW |",
        f"| Learning rate | {cfg.lr} |",
        f"| Weight decay | {cfg.weight_decay} |",
        f"| LR schedule | Cosine annealing over {cfg.epochs} epochs |",
        f"| Batch size | {cfg.batch_size} pairs |",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Model | `{cfg.model_name}` |",
        f"| Approach | `{cfg.approach}` |",
        f"| Epochs | {cfg.epochs} |",
        f"| Batch size | {cfg.batch_size} |",
        f"| Learning rate | {cfg.lr} |",
        f"| Temperature | {cfg.temperature} |",
        f"| Frozen blocks | {cfg.freeze_blocks} / 12 |",
        f"| Embed dim | {cfg.embed_dim} → proj {cfg.proj_out_dim} |",
        f"| Device | `{cfg.device}` |",
        f"",
        f"## Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best loss | {best_loss:.4f} |",
        f"| Best epoch | {best_epoch}/{cfg.epochs} |",
        f"| First loss | {stats.get('first_loss', float('nan')):.4f} |",
        f"| Last loss | {stats.get('last_loss', float('nan')):.4f} |",
        f"| Total drop | {stats.get('drop_pct', 0):.1f} % |",
        f"| Convergence epoch | {stats.get('convergence_epoch', '—')} |",
        f"| Training time | {elapsed_sec:.1f}s |",
        f"| Checkpoint size | {ckpt_mb:.1f} MB |",
        f"| Checkpoint path | `{checkpoint_path}` |",
        f"",
        f"## Result Analysis",
        f"",
    ]
    for b in bullets:
        lines.append(f"- {b}")
        lines.append(f"")

    lines += [
        f"## Loss Curve",
        f"",
        f"```",
        f"high │{sparkline}│",
        f" low │{'─' * len(sparkline)}│",
        f"      epoch 1{'':>{max(0, len(sparkline) - 9)}}epoch {len(loss_history)}",
        f"```",
        f"",
        f"*Each character represents {'one epoch' if len(loss_history) <= 40 else 'a range of epochs'}. "
        f"Higher bar = higher loss.*",
        f"",
        f"| Epoch | Loss | Δ vs prev | Trend |",
        f"|-------|------|-----------|-------|",
    ]
    for ep, loss in enumerate(loss_history, 1):
        if ep == 1:
            delta_str = "—"
            trend = "—"
        else:
            d = deltas[ep - 2]
            delta_str = f"{d:+.4f}"
            if d < -0.01:
                trend = "↓ improving"
            elif d > 0.01:
                trend = "↑ worsening"
            else:
                trend = "→ stable"
        marker = " ← best" if ep == best_epoch else ""
        lines.append(f"| {ep} | {loss:.4f} | {delta_str} | {trend}{marker} |")

    lines += [
        f"",
        f"## How to Use This Checkpoint",
        f"",
        f"```bash",
        f"export DINO_CHECKPOINT={checkpoint_path}",
        f"python main.py --mode demo --videos-dir data_test/videos",
        f"```",
        f"",
        f"---",
        f"*Artifact produced by {_RUNNER_LABEL}. See `edge_models/` for ONNX export.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_distill_stats_md(
    output_path: Path,
    video_name: str,
    stats: Dict[str, Any],
) -> None:
    loss_history    = stats.get("loss_history", [])
    recall_history  = stats.get("recall_history", [])
    loss_components = stats.get("loss_components", {})
    compression     = stats.get("compression_ratio", 0.0)
    t_params        = stats.get("teacher_params", 0)
    s_params        = stats.get("student_params", 0)
    best_recall     = stats.get("best_recall", float("nan"))

    lines = [
        f"# Knowledge Distillation — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Teacher | DINOv3 ViT-B/14 (fine-tuned SSL) — dim={stats.get('teacher_dim', 768)}, {t_params // 1_000_000}M params |",
        f"| Student | {stats.get('student_model', 'dinov2_vits14')} — dim={stats.get('student_dim', 384)}, {s_params // 1_000_000}M params |",
        f"| Method | RKD-DA (distance + angle) + KoLeo spread regulariser + cosine anchor |",
        f"| Loss weights | λ_D=25  λ_A=50  λ_kd=1.0  λ_koleo=0.1 |",
        f"| Epochs | {len(loss_history)} |",
        f"| Elapsed | {stats.get('elapsed', 0):.1f}s |",
        f"",
        f"## Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best total loss | {stats.get('best_loss', float('nan')):.4f} |",
        f"| Best Recall@1 (student vs teacher) | {best_recall:.3f} |",
        f"| Compression ratio | {compression:.1f}× ({t_params // 1_000_000}M → {s_params // 1_000_000}M params) |",
        f"| Student dim | {stats.get('student_dim', 384)} (vs teacher {stats.get('teacher_dim', 768)}) |",
        f"| Best checkpoint | `{Path(stats.get('best_path', '')).name}` |",
        f"",
        f"## Per-Epoch Metrics",
        f"",
        f"| Epoch | Total | RKD-D | RKD-A | Cosine | KoLeo | Recall@1 |",
        f"|-------|-------|-------|-------|--------|-------|----------|",
    ]
    n = len(loss_history)
    for i in range(n):
        r1  = recall_history[i] if i < len(recall_history) else float("nan")
        rd  = loss_components.get("rkd_d", [])[i]   if i < len(loss_components.get("rkd_d",   [])) else float("nan")
        ra  = loss_components.get("rkd_a", [])[i]   if i < len(loss_components.get("rkd_a",   [])) else float("nan")
        cos = loss_components.get("cosine", [])[i]  if i < len(loss_components.get("cosine",  [])) else float("nan")
        kol = loss_components.get("koleo", [])[i]   if i < len(loss_components.get("koleo",   [])) else float("nan")
        lines.append(f"| {i+1} | {loss_history[i]:.4f} | {rd:.4f} | {ra:.4f} | {cos:.4f} | {kol:.4f} | {r1:.3f} |")

    lines += [
        f"",
        f"## Architecture",
        f"",
        f"```",
        f"Teacher (frozen):  DINOv3 ViT-B/14  →  768-dim embedding",
        f"                         ↓ RKD-DA (distance + angle) + cosine anchor",
        f"Proj head (temp):  Linear(384 → 768, orthogonal init)  [discarded after training]",
        f"                         ↑",
        f"Student (trained): DINOv2 ViT-S/14  →  384-dim embedding",
        f"                         ↑",
        f"                    KoLeo spread regulariser (prevents collapse)",
        f"```",
        f"",
        f"**RKD-DA** (Relational Knowledge Distillation) preserves pairwise neighbourhood",
        f"topology in the student embedding space, directly optimising retrieval Recall@K.",
        f"The student is {compression:.1f}× smaller and ~2× faster at inference.",
        f"The projection head is used only during training to align embedding spaces.",
        f"The saved checkpoint contains **only the student backbone weights**.",
        f"",
        f"---",
        f"*Artifact produced by {_RUNNER_LABEL}. Student exported to `edge_models/dino_demo.onnx`.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_comparison_md(
    output_path: Path,
    video_name: str,
    base_results: List[Dict],
    ft_results: List[Dict],
    base_infer_ms: float,
    ft_infer_ms: float,
    ckpt_mb: float,
    onnx_mb: float,
    text_descriptions: List[Tuple[str, float]],
) -> None:
    base_paths = {r.get("payload", r).get("frame_path", "") for r in base_results}
    ft_paths   = {r.get("payload", r).get("frame_path", "") for r in ft_results}
    overlap    = len(base_paths & ft_paths)
    base_scores = [r.get("score", 0) for r in base_results]
    ft_scores   = [r.get("score", 0) for r in ft_results]
    avg_base = float(np.mean(base_scores)) if base_scores else 0.0
    avg_ft   = float(np.mean(ft_scores))   if ft_scores   else 0.0
    lines = [
        f"# Model Comparison — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Video-to-Text Description",
        f"",
        f"Top content descriptions (via CLIP text similarity):",
        f"",
    ]
    for desc, score in text_descriptions[:3]:
        lines.append(f"- **{desc}** (similarity: {score:.3f})")
    lines += [
        f"",
        f"## Search Quality Comparison",
        f"",
        f"| Metric | Base Model | Fine-tuned Model |",
        f"|--------|-----------|-----------------|",
        f"| Avg top-5 score | {avg_base:.4f} | {avg_ft:.4f} |",
        f"| Δ score | — | {avg_ft - avg_base:+.4f} |",
        f"| Result overlap | {overlap}/{len(base_results)} frames in common | |",
        f"",
        f"## Model Statistics",
        f"",
        f"| Metric | Base Model | Fine-tuned (PyTorch) | Fine-tuned (ONNX) |",
        f"|--------|-----------|---------------------|------------------|",
        f"| Checkpoint size | ~330 MB (hub) | {ckpt_mb:.1f} MB | {onnx_mb:.1f} MB |",
        f"| Inference time (GPU/CPU) | {base_infer_ms:.1f} ms/frame | {ft_infer_ms:.1f} ms/frame | — |",
        f"",
        f"## How to Use Artifacts",
        f"",
        f"- **`base_search.md`** — nearest-neighbour results with the pretrained DINOv3 backbone",
        f"- **`finetuned_search.md`** — same query with the mission-adapted backbone",
        f"- **`edge_models/dino_demo.onnx`** — ONNX model for on-device inference (Jetson, Hailo-8)",
        f"- **`edge_models/gallery.npz`** — embedding gallery for 1-NN classification",
        f"- **`3d_map/`** — sparse 3D point cloud from Structure-from-Motion",
        f"",
        f"```python",
        f"from pipeline.edge_inference import EdgeClassifier",
        f"clf = EdgeClassifier('edge_models/dino_demo.onnx', 'edge_models/gallery.npz')",
        f"labels = clf.classify(frame_pil)   # [(label, score), ...]",
        f"```",
        f"",
        f"---",
        f"*Artifact produced by {_RUNNER_LABEL}.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_description_md(
    output_path: Path,
    video_name: str,
    frame_list: List[Tuple[str, float]],
    text_descriptions: List[Tuple[str, float]],
    all_scored: List[Tuple[str, float]],
) -> None:
    lines = [
        f"# Image-to-Text Description — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Top Video Descriptions",
        f"",
        f"Ranked by cosine similarity between the average CLIP frame embedding and each text prompt:",
        f"",
        f"| Rank | Description | Similarity |",
        f"|------|-------------|-----------|",
    ]
    for rank, (desc, score) in enumerate(text_descriptions, 1):
        lines.append(f"| {rank} | {desc} | {score:.4f} |")
    lines += [
        f"",
        f"## All Prompts Scored",
        f"",
        f"| Description | Similarity |",
        f"|-------------|-----------|",
    ]
    for desc, score in all_scored:
        lines.append(f"| {desc} | {score:.4f} |")
    lines += [f"", f"## Sample Frames", f"", f"Frames used for description (evenly spaced, up to 32):", f""]
    step = max(1, len(frame_list) // 8)
    for fp, t_sec in frame_list[::step][:8]:
        lines.append(f"- `{Path(fp).name}` (t={t_sec:.1f}s)")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · model: OpenCLIP ViT-B/16 (openai)*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_final_stats_md(
    output_path: Path,
    per_video: List[Dict[str, Any]],
    total_elapsed: float,
) -> None:
    lines = [
        f"# Demo Pipeline — Final Statistics",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total elapsed: {total_elapsed:.1f}s",
        f"Videos processed: {len(per_video)}",
        f"",
        f"## Per-Video Summary",
        f"",
        f"| Video | Frames | Index (s) | Finetune loss | Distill loss | SfM poses | Ckpt (MB) |",
        f"|-------|--------|-----------|---------------|--------------|-----------|-----------|",
    ]
    for v in per_video:
        distill_loss = v.get("distill_loss", float("nan"))
        distill_str  = f"{distill_loss:.4f}" if not math.isnan(distill_loss) else "skipped"
        lines.append(
            f"| {v['name']} | {v.get('frames', 0)} | "
            f"{v.get('index_sec', 0):.1f} | "
            f"{v.get('best_loss', float('nan')):.4f} | "
            f"{distill_str} | "
            f"{v.get('sfm_poses', 0)} | "
            f"{v.get('ckpt_mb', 0):.1f} |"
        )
    lines += [
        f"",
        f"## Artifacts",
        f"",
        f"Each video produced these outputs under `{output_path.parent}/{{video_name}}/`:",
        f"",
        f"| File | Description |",
        f"|------|-------------|",
        f"| `frames_metadata.json` | Extracted frame paths, timestamps, fps |",
        f"| `base_search.md` | Nearest-neighbour results with base DINOv3 |",
        f"| `scene_captions.md` | Per-frame Florence-2 captions (confidence scores) |",
        f"| `finetune_stats.md` | SSL fine-tuning loss curve + config |",
        f"| `finetuned_search.md` | Nearest-neighbour results with fine-tuned DINOv3 |",
        f"| `comparison.md` | Base vs fine-tuned stats + video description |",
        f"| `checkpoints/dino_ssl_best.pt` | Fine-tuned teacher backbone (PyTorch) |",
        f"| `checkpoints/student_best.pt` | Distilled student backbone (PyTorch, ~22M params) |",
        f"| `distill_stats.md` | Distillation loss curve + architecture notes |",
        f"| `edge_models/dino_demo.onnx` | ONNX export (student when distilled, teacher otherwise) |",
        f"| `edge_models/gallery.npz` | Embedding gallery for 1-NN classification |",
        f"| `asr_subtitles.md` | Whisper ASR segments + per-frame subtitle coverage (step M) |",
        f"| `multimodal_features.md` | OCR text, depth percentiles, detections, world model (steps N–Q) |",
        f"| `detailed_captions.md` | Qwen VLM detailed per-frame scene captions with ASR context (step R) |",
        f"| `video_synthesis.md` | LLM video ontology + fine-grained narrative (step Z) |",
        f"| `agentic_flow.md` | Step-by-step agentic context trace, risk analysis, and context-propagation audit (step AA) |",
        f"| `video_ontology.json` | Structured ontology JSON (domain, environment, activities, objects) |",
        f"| `3d_map/sparse_map.npz` | 3D point cloud (from SfM or PCA fallback) |",
        f"| `3d_map/map_stats.json` | Point count, SfM pose count, scene count |",
        f"",
        f"---",
        f"*Run `python main.py --mode demo --help` for all options.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("✓ Final stats written to %s", output_path)


def write_multimodal_md(
    output_path: Path,
    video_name: str,
    asr_result: Dict[str, Any],
    ocr_result: Dict[str, Any],
    depth_result: Dict[str, Any],
    det_result: Dict[str, Any],
    world_result: Dict[str, Any],
    qwen_result: Dict[str, Any],
) -> None:
    lines = [
        f"# Multimodal Features — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Summary",
        f"",
        f"| Step | Status | Detail |",
        f"|------|--------|--------|",
        f"| ASR (Whisper) | {'✓' if not asr_result.get('skipped') else '—'} | "
        f"{asr_result.get('covered_frames', 0)} frames with subtitles |",
        f"| OCR | {'✓' if not ocr_result.get('skipped') else '—'} | "
        f"{ocr_result.get('non_empty', 0)} frames with text |",
        f"| Depth | {'✓' if not depth_result.get('skipped') else '—'} | "
        f"{depth_result.get('ok_count', 0)} frames estimated |",
        f"| Detection | {'✓' if not det_result.get('skipped') else '—'} | "
        f"{det_result.get('total_objects', 0)} objects detected |",
        f"| World Model | {'✓' if not world_result.get('skipped') else '—'} | "
        f"{world_result.get('ok_count', 0)} clips processed |",
        f"| Qwen VLM captioning | {'✓' if not qwen_result.get('skipped') else '—'} | "
        f"{qwen_result.get('ok_count', 0)} frames captioned |",
        f"",
    ]
    if not ocr_result.get("skipped"):
        lines += ["## OCR — Sample Text Extractions", ""]
        ocr_rows = [r for r in ocr_result.get("ocr_results", []) if r.get("ocr_text")][:10]
        if ocr_rows:
            lines += ["| t (s) | Extracted Text |", "|-------|----------------|"]
            for r in ocr_rows:
                txt = (r.get("ocr_text") or "").replace("|", "\\|")[:120]
                lines.append(f"| {r['t_sec']:.1f} | {txt} |")
        lines.append("")
    if not det_result.get("skipped"):
        lines += ["## Detection — Objects Found", ""]
        det_rows = [r for r in det_result.get("detection_results", []) if r.get("detections")][:10]
        if det_rows:
            lines += ["| t (s) | Detections |", "|-------|------------|"]
            for r in det_rows:
                objs = ", ".join(
                    f"{d['label']} ({d['confidence']:.2f})" for d in r["detections"][:5]
                )
                lines.append(f"| {r['t_sec']:.1f} | {objs} |")
        lines.append("")
    if not depth_result.get("skipped"):
        lines += ["## Depth — Percentile Summary (sample)", ""]
        depth_rows = [r for r in depth_result.get("depth_results", []) if r.get("depth")][:5]
        if depth_rows:
            lines += ["| t (s) | p10 | p25 | p50 | p75 | p90 |",
                      "|-------|-----|-----|-----|-----|-----|"]
            for r in depth_rows:
                p = r["depth"].get("percentiles", [0]*5)
                lines.append(f"| {r['t_sec']:.1f} | "
                              f"{p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {p[3]:.3f} | {p[4]:.3f} |")
        lines.append("")
    lines += ["---", f"*Produced by {_RUNNER_LABEL} · multimodal steps M–R*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_detailed_captions_md(
    output_path: Path,
    video_name: str,
    results: List[Dict[str, Any]],
    elapsed_sec: float,
    model_id: str,
) -> None:
    ok = sum(1 for r in results if not r.get("service_unavailable") and not r.get("skipped"))

    # Build text captions for scene-segment detection (use scene_summary from Qwen JSON)
    text_results: List[Dict[str, Any]] = []
    for r in results:
        summary = r.get("scene_summary") or r.get("caption") or r.get("scene_description") or ""
        text_results.append({**r, "caption": summary})
    enriched = _analyze_caption_sequence(text_results)

    # Segment-level summary
    segments: List[Dict[str, Any]] = []
    for r in enriched:
        if r["is_new_segment"]:
            segments.append({
                "segment_id": r["segment_id"],
                "start_t": r["t_sec"],
                "end_t": r["t_sec"],
                "frame_count": 1,
                "scene_summary": r.get("scene_summary") or r.get("caption") or "",
                "road_surface": r.get("road_surface", ""),
                "road_condition": r.get("road_condition", ""),
                "vehicle_groups": r.get("vehicle_groups", []),
            })
        elif segments:
            segments[-1]["end_t"] = r["t_sec"]
            segments[-1]["frame_count"] += 1

    n_unchanged = sum(1 for r in enriched if not r["is_new_segment"])

    lines = [
        f"# Detailed Scene Captions — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Frames processed: {ok}/{len(results)}"
        f"  |  Unique scenes: {len(segments)}  |  Repeated: {n_unchanged}",
        f"Elapsed: {elapsed_sec:.1f}s",
        f"",
        f"## Scene Timeline",
        f"",
        f"| # | Start (s) | End (s) | Frames | Road | Condition | Vehicles | Summary |",
        f"|---|-----------|---------|--------|------|-----------|----------|---------|",
    ]
    for seg in segments:
        vg = seg.get("vehicle_groups") or []
        v_str = "; ".join(
            f"{g.get('count', 1)}×{g.get('type', '?')}" for g in vg
        ) if vg else "none"
        summary = (seg.get("scene_summary") or "").replace("|", "\\|")[:120]
        lines.append(
            f"| {seg['segment_id'] + 1} | {seg['start_t']:.1f} | {seg['end_t']:.1f}"
            f" | {seg['frame_count']} | {seg.get('road_surface') or '—'}"
            f" | {seg.get('road_condition') or '—'} | {v_str} | {summary} |"
        )

    lines += [
        f"",
        f"## Per-Frame Analysis",
        f"",
        f"The **Δ Changes** column shows structured fields that differ from the previous frame.",
        f"Frames with no changes are marked *unchanged*.",
        f"",
        f"| Frame | t (s) | Seg | Δ Changes | Caption / Scene Facts | Audio Context |",
        f"|-------|-------|-----|-----------|----------------------|---------------|",
    ]

    prev_structured: Dict[str, Any] = {}
    for r in enriched:
        fp       = r.get("frame_path", "")
        name     = Path(fp).name if fp else "—"
        t        = r.get("t_sec", 0.0)
        subtitle = (r.get("subtitle_text") or "").replace("|", "\\|")[:60]
        seg      = r["segment_id"] + 1

        if r.get("service_unavailable"):
            caption  = "*sidecar unavailable*"
            delta    = "—"
        elif r.get("skipped"):
            caption  = "*skipped*"
            delta    = "—"
        else:
            # Structured diff against previous frame
            delta = _diff_structured_caption(prev_structured, r) if prev_structured else ""
            delta = delta.replace("|", "\\|") if delta else ("—" if prev_structured else "first")

            # Caption text: prefer scene_summary, then fallback keys
            facts = r.get("scene_summary") or r.get("caption") or r.get("scene_description") or ""
            if not facts:
                parts = []
                for k, v in r.items():
                    if k not in (
                        "frame_path", "t_sec", "subtitle_text", "ocr_text",
                        "segment_id", "is_new_segment", "similarity", "segment_start_t",
                        "caption",
                    ) and v:
                        parts.append(f"{k}: {v}")
                facts = "; ".join(parts[:4])
            caption = str(facts).replace("|", "\\|")[:200]
            if not r["is_new_segment"]:
                caption = f"*unchanged* {caption}"

            # Update structured state for next diff
            if not r.get("parse_error"):
                prev_structured = r

        lines.append(f"| `{name}` | {t:.1f} | {seg} | {delta} | {caption} | {subtitle} |")

    lines += [
        f"",
        f"---",
        f"*Produced by {_RUNNER_LABEL} · Qwen VLM step R · ASR subtitle context injected where available*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_video_synthesis_md(
    output_path: Path,
    video_name: str,
    ontology: Dict[str, Any],
    narrative: str,
    elapsed_sec: float,
    model_id: str,
) -> None:
    lines = [
        f"# Video Synthesis — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Elapsed: {elapsed_sec:.1f}s",
        f"",
    ]
    if ontology:
        lines += [
            f"## Video Ontology",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
        ]
        for k, v in ontology.items():
            val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            lines.append(f"| {k} | {val.replace('|', '&#124;')} |")
        lines.append("")
    if narrative:
        lines += [
            f"## Video Narrative",
            f"",
            narrative,
            f"",
        ]
    lines += ["---", f"*Produced by {_RUNNER_LABEL} · synthesis step Z · context from steps A–H*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_agentic_flow_md(
    output_path: Path,
    video_name: str,
    trace: List[Dict[str, Any]],
    elapsed_sec: float,
    model_id: str,
    llm_analysis: str,
) -> None:
    lines = [
        f"# Agentic Flow Trace — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Reasoning model: {model_id}  |  Elapsed: {elapsed_sec:.1f}s",
        f"",
        f"## Step Trace",
        f"",
        f"| Step | Status | Context Received | Context Produced | Key Risks |",
        f"|------|--------|------------------|------------------|-----------|",
    ]

    for item in trace:
        inputs = "; ".join(item.get("context_inputs", [])[:4]) or "—"
        outputs = "; ".join(item.get("context_outputs", [])[:4]) or "—"
        risks = "; ".join(item.get("risks", [])[:3]) or "—"
        lines.append(
            f"| {item.get('step_id', '?')} {item.get('title', '')} | "
            f"{item.get('status', 'unknown')} | "
            f"{inputs.replace('|', '&#124;')[:180]} | "
            f"{outputs.replace('|', '&#124;')[:180]} | "
            f"{risks.replace('|', '&#124;')[:180]} |"
        )

    lines += ["", "## Agentic Analysis", ""]
    if llm_analysis.strip():
        lines.append(llm_analysis.strip())
    else:
        lines.append("Reasoning analysis unavailable.")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · final agentic audit step*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


# ── Run statistics printer ────────────────────────────────────────────────────

_STEP_LABELS = [
    ("A_extract",    "A  Frame extraction"),
    ("B_index",      "B  Vector store indexing"),
    ("L_caption",    "L  Scene captioning (Florence-2)"),
    ("M_asr",        "M  ASR (Whisper)"),
    ("N_ocr",        "N  OCR (text extraction)"),
    ("O_depth",      "O  Depth estimation"),
    ("P_detection",  "P  Object detection"),
    ("Q_world",      "Q  World model"),
    ("R_qwen",       "R  Qwen detailed captioning"),
    ("C_base_search","C  Base search test"),
    ("D_finetune",   "D  SSL fine-tuning"),
    ("E_distill",    "E  Knowledge distillation"),
    ("F_export",     "F  ONNX export + gallery"),
    ("G_ft_search",  "G  Fine-tuned search test"),
    ("H_compare",    "H  Comparison + description"),
    ("I_3dmap",      "I  3D map creation"),
    ("Z_synthesis",  "Z  Video synthesis (ontology + narrative)"),
    ("AA_agentic",   "AA Agentic flow audit"),
]


def _fmt_sec(sec: float) -> str:
    if math.isnan(sec) or sec < 0:
        return "—"
    if sec >= 3600:
        h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
        return f"{h}h {m:02d}m {s:02d}s"
    if sec >= 60:
        m = int(sec // 60); s = sec % 60
        return f"{m}m {s:04.1f}s"
    return f"{sec:.1f}s"


def print_run_stats(
    per_video: List[Dict[str, Any]],
    total_elapsed: float,
    init_elapsed: float,
    device: str,
) -> None:
    W   = 72
    SEP = "─" * W

    def _row(label: str, *cols: str) -> str:
        col_w = max(1, (W - 28) // max(len(cols), 1))
        return "".join([f"  {label:<26}"] + [f"{c:>{col_w}}" for c in cols])

    _banner("RUN STATISTICS")
    _log.info("  Device       : %s", device.upper())
    _log.info("  Videos       : %d", len(per_video))
    total_frames   = sum(v.get("frames", 0) for v in per_video)
    total_duration = sum(v.get("duration_sec", 0.0) for v in per_video)
    _log.info("  Total frames : %d  (%.1f min of video)", total_frames, total_duration / 60)
    _log.info("  Total runtime: %s", _fmt_sec(total_elapsed))
    _log.info("")

    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]
    _log.info("  TIME BREAKDOWN")
    _log.info("  " + SEP[:W-2])
    _log.info(_row("Step", *(names + ["TOTAL"])))
    _log.info("  " + SEP[:W-2])
    for key, label in _STEP_LABELS:
        vals = [v.get("timings", {}).get(key, 0.0) for v in per_video]
        _log.info(_row(label, *[_fmt_sec(s) for s in vals], _fmt_sec(sum(vals))))
    _log.info("  " + SEP[:W-2])
    pipeline_per_video = [v.get("pipeline_sec", 0.0) for v in per_video]
    _log.info(_row("Pipeline (steps sum)",
                   *[_fmt_sec(s) for s in pipeline_per_video],
                   _fmt_sec(sum(pipeline_per_video))))
    overhead = total_elapsed - sum(pipeline_per_video) - init_elapsed
    _log.info(_row("Model init", _fmt_sec(init_elapsed), *([""] * (len(per_video) - 1)), ""))
    _log.info(_row("Overhead (I/O, viewer, etc.)",
                   *([""] * len(per_video)), _fmt_sec(max(0.0, overhead))))
    _log.info(_row("WALL CLOCK TOTAL",
                   *([""] * len(per_video)), _fmt_sec(total_elapsed)))
    _log.info("")
    _log.info("  THROUGHPUT")
    _log.info("  " + SEP[:W-2])
    for v in per_video:
        t_extract = v.get("timings", {}).get("A_extract", 0.0) or 1e-9
        t_index   = v.get("timings", {}).get("B_index",   0.0) or 1e-9
        frames    = v.get("frames", 0)
        _log.info("  %-26s  extract: %5.1f fr/s   index: %5.1f fr/s",
                  v.get("name", "?"), frames / t_extract, frames / t_index)
    _log.info("")
    _log.info("  MODEL METRICS")
    _log.info("  " + SEP[:W-2])
    _log.info(_row("Metric", *names))
    _log.info("  " + SEP[:W-2])
    _log.info(_row("SSL finetune loss",
                   *[f"{v.get('best_loss', float('nan')):.4f}" for v in per_video]))
    _log.info(_row("Distill loss",
                   *[f"{v.get('distill_loss', float('nan')):.4f}"
                     if not math.isnan(v.get("distill_loss", float("nan"))) else "skipped"
                     for v in per_video]))
    _log.info(_row("Teacher ckpt (MB)",
                   *[f"{v.get('ckpt_mb', 0.0):.1f}" for v in per_video]))
    _log.info(_row("Student ckpt (MB)",
                   *[f"{v.get('student_ckpt_mb', 0.0):.1f}" if v.get("student_ckpt_mb") else "—"
                     for v in per_video]))
    _log.info(_row("ONNX size (MB)",
                   *[f"{v.get('onnx_mb', 0.0):.1f}" if v.get("onnx_exported") else "—"
                     for v in per_video]))
    _log.info(_row("Compression ratio",
                   *[f"{v['teacher_dim']/v['student_dim']:.1f}×"
                     if v.get("student_dim") and v.get("teacher_dim") else "—"
                     for v in per_video]))
    _log.info(_row("Base infer (ms/fr)",
                   *[f"{v.get('base_infer_ms', 0.0):.1f}" for v in per_video]))
    _log.info(_row("Fine-tuned infer (ms/fr)",
                   *[f"{v.get('ft_infer_ms', 0.0):.1f}" for v in per_video]))
    _log.info("")
    _log.info("  SEARCH QUALITY  (top-1 cosine score, same query frame)")
    _log.info("  " + SEP[:W-2])
    _log.info(_row("Base model (pretrained)",
                   *[f"{v.get('base_top_score', 0.0):.4f}" for v in per_video]))
    _log.info(_row("Fine-tuned model",
                   *[f"{v.get('ft_top_score', 0.0):.4f}" for v in per_video]))
    _log.info("")
    _log.info("  3D MAP")
    _log.info("  " + SEP[:W-2])
    _log.info(_row("Method",    *[v.get("map_method", "—") for v in per_video]))
    _log.info(_row("Points",    *[str(v.get("map_points", 0)) for v in per_video]))
    _log.info(_row("SfM poses", *[str(v.get("sfm_poses", 0)) for v in per_video]))
    _log.info("")
    _log.info("  TOP VIDEO DESCRIPTION  (CLIP text similarity)")
    _log.info("  " + SEP[:W-2])
    for v in per_video:
        _log.info("  %-20s  %s", v.get("name", "?"), v.get("top_description", "—") or "—")
    _log.info("")
    _log.info("  " + "═" * (W-2))


# ── Model & store initialisation ──────────────────────────────────────────────

def _resolve_device(device_cfg: str) -> str:
    import torch
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def init_models(device: str) -> Dict[str, Any]:
    _banner("Initialising models")
    models: Dict[str, Any] = {"device": device, "uses_api_embedder": False}

    # The pre-flight check above may have left Ollama sidecars resident in VRAM.
    # Evict them now so local model loads (GemmaEmbedder / OpenCLIP / DINO) have
    # enough headroom.  We'll re-load the sidecar models on-demand in each step.
    if device == "cuda":
        import gc as _gc
        import torch as _torch_init
        _unload_known_sidecars([
            (settings.GEMMA_API_URL, settings.GEMMA_API_MODEL),
            (getattr(settings, "QWEN_API_URL", ""), getattr(settings, "QWEN_MODEL", "")),
            (getattr(settings, "REASONING_API_URL", ""), getattr(settings, "REASONING_MODEL", "")),
        ])
        _gc.collect()
        _torch_init.cuda.empty_cache()

    if settings.MODEL_NAME == "gemma":
        if not _HAS_GEMMA:
            raise ImportError(
                "models.gemma_model is unavailable — install transformers and accelerate."
            )
        hf_token = settings.HF_TOKEN
        if not hf_token:
            _log.warning(
                "HF_TOKEN is not set. Gemma is a gated model — set HF_TOKEN=hf_... in .env "
                "or run: huggingface-cli login"
            )
        else:
            from pipeline.config import mask_secret as _mask  # noqa: PLC0415
            _log.info("  HF_TOKEN: %s", _mask(hf_token))
        _log.info("Loading GemmaEmbedder (%s) …", settings.GEMMA_MODEL_ID)
        t0 = time.time()
        try:
            models["clip"] = GemmaEmbedder(
                model_id=settings.GEMMA_MODEL_ID,
                device=device,
                use_bf16=settings.GEMMA_USE_BF16,
                hf_token=hf_token,
            )
        except Exception as exc:
            if settings.GEMMA_API_URL:
                _log.warning(
                    "  GemmaEmbedder load failed (%s) — falling back to OpenCLIP for embeddings. "
                    "Sidecar (%s) will still handle generative analysis.",
                    exc, settings.GEMMA_API_URL,
                )
            else:
                raise RuntimeError(
                    f"GemmaEmbedder failed to load: {exc}\n"
                    "Fix: set HF_TOKEN=hf_... in .env (accept license at "
                    "huggingface.co/google/gemma-4-it-2b) or run: huggingface-cli login"
                ) from exc
        else:
            models["dino"] = None
            models["uses_api_embedder"] = False
            _log.info(
                "  ✓ GemmaEmbedder ready in %.1fs  (dim=%d)",
                time.time() - t0,
                models["clip"].image_dim(),
            )
            _log.info(
                "  ℹ  SSL fine-tuning and distillation steps are skipped for Gemma embedder."
            )
            return models
        # Fall through to load OpenCLIP when local Gemma failed but sidecar is set

    _log.info("Loading OpenCLIP ViT-B-16 …")
    t0 = time.time()
    _log_vram_snapshot("before Gemma analysis step")
    models["clip"] = OpenCLIPEmbedder()
    _log.info("  ✓ CLIP ready in %.1fs  (dim=%d)", time.time() - t0, models["clip"].image_dim())

    if _HAS_DINO:
        _log.info("Loading DINOv3 ViT-B/14 …  (first run downloads ~330 MB)")
        t0 = time.time()
        try:
            models["dino"] = DINOEmbedder("dinov3_vitb14")
            _log.info("  ✓ DINO ready in %.1fs  (dim=%d)",
                      time.time() - t0, models["dino"].image_dim())
        except Exception as exc:
            _log.warning("  ✗ DINOv3 load failed (%s) — using CLIP only", exc)
            models["dino"] = None
    else:
        _log.warning("  ✗ models.dino_model unavailable — using CLIP only")
        models["dino"] = None

    return models


def init_store(models: Dict[str, Any], use_qdrant: bool) -> Tuple[Any, bool]:
    if not use_qdrant:
        _log.info("Qdrant disabled (--no-qdrant) — using in-memory cosine store")
        return InMemoryStore(), False
    try:
        from pipeline.qdrant_utils import QdrantStore
        clip_dim = models["clip"].image_dim()
        dino_dim = models["dino"].image_dim() if models.get("dino") else None
        store    = QdrantStore(clip_dim=clip_dim, dino_dim=dino_dim)
        store.client.get_collections()
        _log.info("✓ Qdrant connected at %s:%s  collection=%s",
                  settings.QDRANT_HOST, settings.QDRANT_PORT, settings.QDRANT_COLLECTION)
        return store, True
    except Exception as exc:
        _log.warning("Qdrant unavailable (%s) — falling back to in-memory store", exc)
        _log.warning("  To enable: docker run -p 6333:6333 qdrant/qdrant")
        return InMemoryStore(), False


# ── Step implementations ───────────────────────────────────────────────────────

def step_extract_frames(
    video_path: Path,
    video_id: str,
    video_dir: Path,
    fps: float,
) -> Dict[str, Any]:
    """Step A: extract frames via ffmpeg, write metadata JSON."""
    _log.info("Extracting frames from %s at %.1f fps …", video_path.name, fps)
    t0 = time.time()
    frame_list = extract_frames(str(video_path), video_id)
    elapsed = time.time() - t0
    _log_vram_snapshot("after Gemma analysis step")
    meta = {
        "video": str(video_path),
        "video_id": video_id,
        "fps": fps,
        "frame_count": len(frame_list),
        "duration_sec": frame_list[-1][1] if frame_list else 0.0,
        "frames": [{"path": p, "t_sec": t} for p, t in frame_list],
        "extracted_at": datetime.now().isoformat(),
    }
    meta_path = video_dir / "frames_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log.info("  ✓ %d frames extracted in %.1fs → %s", len(frame_list), elapsed, meta_path)
    return {"frame_list": frame_list, "elapsed_sec": elapsed, "meta": meta}


def _embed_and_flush(
    batch_pil: List[Image.Image],
    batch_meta: List[Tuple[str, float]],
    video_id: str,
    clip_model: OpenCLIPEmbedder,
    dino_model: Any,
    store: Any,
    is_qdrant: bool,
) -> int:
    if not batch_pil:
        return 0
    clip_embeds = clip_model.encode_images(batch_pil)
    dino_embeds = dino_model.encode_images(batch_pil) if dino_model else None
    if is_qdrant:
        from qdrant_client.http import models as qmodels
        from pipeline.utils import stable_point_id
        points = []
        for i, (fp, t_sec) in enumerate(batch_meta):
            vectors: Dict[str, Any] = {"clip": clip_embeds[i].tolist()}
            if dino_embeds is not None:
                vectors["dino"] = dino_embeds[i].tolist()
            points.append(qmodels.PointStruct(
                id=stable_point_id(video_id, fp),
                vector=vectors,
                payload={"frame_path": fp, "t_sec": t_sec, "video_id": video_id},
            ))
        store.upsert_points(points)
    else:
        for i, (fp, t_sec) in enumerate(batch_meta):
            vec = dino_embeds[i] if dino_embeds is not None else clip_embeds[i]
            store.add(vec, {"frame_path": fp, "t_sec": t_sec, "video_id": video_id})
    return len(batch_pil)


def step_index_to_store(
    video_path: Path,
    video_id: str,
    store: Any,
    is_qdrant: bool,
    models: Dict[str, Any],
    frame_list: List[Tuple[str, float]],
) -> Dict[str, Any]:
    """Step B: embed frames and upsert into Qdrant or InMemoryStore."""
    t0   = time.time()
    dest = "Qdrant" if is_qdrant else "in-memory store"
    _log.info("Embedding %d frames into %s …", len(frame_list), dest)
    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")
    batch_pil: List[Image.Image]         = []
    batch_meta: List[Tuple[str, float]]  = []
    indexed = 0
    for fp, t_sec in frame_list:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        batch_pil.append(img); batch_meta.append((fp, t_sec))
        if len(batch_pil) >= 32:
            indexed += _embed_and_flush(batch_pil, batch_meta, video_id,
                                        clip_model, dino_model, store, is_qdrant)
            batch_pil, batch_meta = [], []
    indexed += _embed_and_flush(batch_pil, batch_meta, video_id,
                                clip_model, dino_model, store, is_qdrant)
    elapsed = time.time() - t0
    _log.info("  ✓ %d frames indexed into %s in %.1fs", indexed, dest, elapsed)
    return {"indexed": indexed, "elapsed_sec": elapsed}


def _gemma_analyse_frame_via_api(
    fp: str,
    api_url: str,
    model: str,
    timeout: float,
) -> str:
    """Send a single frame to a Gemma Ollama/vLLM sidecar and return its description."""
    import base64
    import io

    try:
        import httpx
    except ImportError:
        return ""

    try:
        img = Image.open(fp).convert("RGB")
        img.thumbnail((768, 768))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",
                     "text": (
                         "Analyse this frame from aerial/robotics mission video. "
                         "Describe in 2-3 sentences: scene type, visible objects, "
                         "terrain, any notable features or anomalies. "
                         "Be concise and factual."
                     )},
                ],
            }],
            # 600 tokens: thinking models (gemma4:e4b) consume ~300-400 on reasoning
            # before writing the final answer into content.
            "max_tokens": 600,
            "temperature": 0.1,
        }
        endpoint = f"{api_url.rstrip('/')}/chat/completions"
        resp = httpx.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content")
        # Thinking models (gemma4:e4b) place the answer in content and
        # the chain-of-thought in reasoning.  If content is still empty
        # (budget exhausted on reasoning), use the last sentence of reasoning.
        if not content:
            reasoning = msg.get("reasoning") or msg.get("thinking") or ""
            if reasoning:
                # Take last non-empty sentence as a best-effort summary
                sentences = [s.strip() for s in reasoning.replace("\n", " ").split(".") if s.strip()]
                content = sentences[-1] if sentences else reasoning[-200:]
        # content may be a list of parts (some backends)
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        return (content or "").strip()
    except Exception as exc:
        _log.debug("  [Gemma API] frame analysis failed for %s: %s", Path(fp).name, exc)
        return ""


def step_gemma_analysis(
    video_path: Path,
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
    gemma_api_url: str = "",
    gemma_api_model: str = "",
) -> Dict[str, Any]:
    """Step J: Gemma open-weight multimodal video analysis.

    Uses the local GemmaEmbedder for embedding-based analysis and, when a
    Gemma Ollama/vLLM sidecar is configured (GEMMA_API_URL), also runs
    generative per-frame scene description.

    Analyses performed:
    1. Generative frame descriptions via Gemma sidecar (if GEMMA_API_URL set)
    2. Scene change detection — consecutive-frame cosine distance
    3. Semantic scene clustering — greedy cosine-based grouping
    4. Zero-shot scene classification — text probe vs frame embedding scores
    5. Cross-modal text -> frame retrieval — nearest-neighbour in Gemma space
    6. Temporal video embedding — mean-pool of all frame embeddings
    7. Gemma vs CLIP embedding comparison — MNN@k and mean pairwise cosine sim
    8. Gemma vs DINOv3 embedding comparison — MNN@k and mean pairwise cosine sim

    Skipped gracefully when GemmaEmbedder is unavailable and no sidecar is set.
    Writes ``gemma_analysis.md`` to *video_dir*.
    """
    result: Dict[str, Any] = {"skipped": True, "reason": ""}

    effective_api_url   = gemma_api_url or settings.GEMMA_API_URL
    effective_api_model = gemma_api_model or settings.GEMMA_API_MODEL
    effective_timeout   = float(settings.GEMMA_API_TIMEOUT_SEC)

    # Use local GemmaEmbedder when available; otherwise fall back to whatever
    # embedder is loaded (OpenCLIP) — all embedding analyses still run, just
    # powered by a different backbone.
    _clip_model = models.get("clip")
    has_local  = _clip_model is not None
    _embedder_name = (
        "GemmaEmbedder" if (_HAS_GEMMA and isinstance(_clip_model, GemmaEmbedder))
        else type(_clip_model).__name__ if _clip_model is not None
        else "none"
    )
    has_sidecar = bool(effective_api_url)

    if not has_local and not has_sidecar:
        result["reason"] = "No embedder available and GEMMA_API_URL not set"
        _log.info("  Gemma analysis skipped: %s", result["reason"])
        return result

    t0 = time.time()

    # Sample frames evenly
    n_avail  = len(frame_list)
    n_sample = min(_GEMMA_ANALYSIS_SAMPLE_N, n_avail)
    step     = max(1, n_avail // n_sample)
    sample_frames = frame_list[::step][:n_sample]
    sample_images = [_open_frame_image(fp) for fp, _ in sample_frames]
    sample_paths  = [fp for fp, _ in sample_frames]
    sample_ts     = [t for _, t in sample_frames]
    n = len(sample_images)
    _log.info("  Gemma analysis: %d sampled frames (from %d total)", n, n_avail)

    task_results: Dict[str, Any] = {}

    # 1. Generative per-frame analysis via Ollama/vLLM sidecar
    gemma_captions: List[Dict[str, Any]] = []
    if has_sidecar:
        _log.info(
            "  [Gemma] Generative scene analysis via sidecar (url=%s  model=%s  frames=%d) ...",
            effective_api_url, effective_api_model, n,
        )
        for idx, (fp, t_sec) in enumerate(sample_frames):
            desc = _gemma_analyse_frame_via_api(
                fp, effective_api_url, effective_api_model, effective_timeout,
            )
            gemma_captions.append({"frame_path": fp, "t_sec": t_sec, "description": desc})
            if (idx + 1) % 10 == 0:
                _log.info("    ... %d/%d frames analysed via Gemma sidecar", idx + 1, n)
        described = sum(1 for c in gemma_captions if c.get("description"))
        _log.info("  [Gemma] Generative descriptions: %d/%d frames", described, n)
        task_results["generative_descriptions"] = {
            "description": "Per-frame scene description generated by Gemma sidecar",
            "n_frames": n,
            "described_count": described,
            "model": effective_api_model,
            "captions": gemma_captions,
        }
        _write_gemma_captions_md(
            video_dir / "gemma_captions.md",
            video_name, effective_api_model, gemma_captions,
        )
    else:
        task_results["generative_descriptions"] = {
            "description": "Skipped — GEMMA_API_URL not configured",
            "skipped": True,
        }

    # For the remaining embedding-based analyses we need a local embedder.
    if not has_local:
        _log.info("  [Gemma] Skipping embedding analyses (no embedder loaded)")
        text_query_results: List[Dict[str, Any]] = []
        dino_comparison: Dict[str, Any] = {"available": False, "reason": "no embedder loaded"}
        clip_comparison: Dict[str, Any] = {"available": False, "reason": "GemmaEmbedder not loaded"}
        elapsed = time.time() - t0
        write_gemma_analysis_md(
            video_dir / "gemma_analysis.md",
            video_name, effective_api_model or settings.GEMMA_MODEL_ID,
            n, task_results, dino_comparison, text_query_results, elapsed,
            clip_comparison=clip_comparison,
        )
        result.update({
            "skipped": False, "n_frames": n,
            "task_results": task_results,
            "dino_comparison": dino_comparison,
            "clip_comparison": clip_comparison,
            "elapsed_sec": elapsed,
        })
        return result

    # Use whichever embedder is loaded (GemmaEmbedder preferred, CLIP fallback).
    gemma: GemmaEmbedder = models["clip"]  # type: ignore[assignment]
    _log.info("  [Gemma] Embedding analyses using %s", _embedder_name)

    # 2. Scene change detection via consecutive-frame cosine distance
    gemma_embeds: Optional[np.ndarray] = None
    try:
        _log.info("  [Gemma] Scene change detection ...")
        gemma_embeds = gemma.encode_images(sample_images)
        changes = []
        for i in range(1, n):
            cos_sim  = float(np.dot(gemma_embeds[i - 1], gemma_embeds[i]))
            distance = 1.0 - cos_sim
            if distance >= _SCENE_CHANGE_THRESH:
                changes.append({"frame_idx": i, "t_sec": sample_ts[i], "distance": distance})
        task_results["scene_change_detection"] = {
            "description": "Consecutive-frame cosine distance > threshold",
            "n_changes": len(changes),
            "threshold": _SCENE_CHANGE_THRESH,
            "changes": changes,
        }
        _log.info("  [Gemma] Scene changes detected: %d", len(changes))
    except Exception as exc:
        task_results["scene_change_detection"] = {"error": str(exc)}
        _log.warning("  [Gemma] Scene change detection failed: %s", exc)

    # 3. Greedy cosine-based scene clustering
    try:
        _log.info("  [Gemma] Semantic scene clustering ...")
        cl_embeds = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        sim_mat   = np.dot(cl_embeds, cl_embeds.T)
        labels    = [-1] * n
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, n):
                if labels[j] == -1 and sim_mat[i, j] > (1.0 - _SCENE_CHANGE_THRESH):
                    labels[j] = cluster_id
            cluster_id += 1
        task_results["scene_clustering"] = {
            "description": "Greedy scene grouping by cosine similarity",
            "n_clusters": cluster_id,
            "n_frames": n,
            "mean_cluster_size": round(n / max(1, cluster_id), 2),
        }
        _log.info("  [Gemma] Scene clusters: %d from %d frames", cluster_id, n)
    except Exception as exc:
        task_results["scene_clustering"] = {"error": str(exc)}
        _log.warning("  [Gemma] Scene clustering failed: %s", exc)

    # 4. Zero-shot scene classification via text probe matching
    try:
        _log.info("  [Gemma] Zero-shot scene classification (%d probes) ...", len(_GEMMA_TEXT_PROBES))
        clf_frame  = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        clf_text   = gemma.encode_texts(_GEMMA_TEXT_PROBES)
        clf_scores = np.dot(clf_frame, clf_text.T)  # (n_frames, n_categories)
        from collections import Counter
        top_cats: List[str] = [
            _GEMMA_TEXT_PROBES[int(np.argmax(clf_scores[i]))] for i in range(n)
        ]
        cat_dist = dict(Counter(top_cats).most_common(5))
        task_results["scene_classification"] = {
            "description": "Zero-shot classification against %d scene categories" % len(_GEMMA_TEXT_PROBES),
            "n_frames": n,
            "category_distribution": cat_dist,
        }
        _log.info("  [Gemma] Top category: %s", next(iter(cat_dist)) if cat_dist else "---")
    except Exception as exc:
        task_results["scene_classification"] = {"error": str(exc)}
        _log.warning("  [Gemma] Zero-shot classification failed: %s", exc)

    # 5. Cross-modal text -> frame retrieval
    text_query_results = []
    try:
        _log.info("  [Gemma] Cross-modal text->frame retrieval (%d probes) ...", len(_GEMMA_TEXT_PROBES))
        doc_embeds   = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
        query_embeds = gemma.encode_texts(_GEMMA_TEXT_PROBES)
        tq_scores    = np.dot(query_embeds, doc_embeds.T)  # (n_queries, n_frames)
        for q_idx, query in enumerate(_GEMMA_TEXT_PROBES):
            top_idxs = list(np.argsort(-tq_scores[q_idx])[:3])
            text_query_results.append({
                "query": query,
                "top_results": [
                    {"frame_path": sample_paths[i], "t_sec": sample_ts[i],
                     "score": float(tq_scores[q_idx, i])}
                    for i in top_idxs
                ],
            })
        task_results["cross_modal_retrieval"] = {
            "description": "Text probes matched against Gemma frame embeddings",
            "n_queries": len(_GEMMA_TEXT_PROBES),
        }
    except Exception as exc:
        task_results["cross_modal_retrieval"] = {"error": str(exc)}
        _log.warning("  [Gemma] Cross-modal retrieval failed: %s", exc)

    # 6. Temporal video embedding (mean-pool all frames)
    try:
        _log.info("  [Gemma] Temporal video embedding ...")
        vid_embed = gemma.encode_images_temporal(sample_images)
        task_results["temporal_embedding"] = {
            "description": "Mean-pool of %d frame embeddings -> single video-level vector" % n,
            "dim": int(vid_embed.shape[1]),
            "n_frames": n,
        }
        _log.info("  [Gemma] Temporal embedding dim=%d", vid_embed.shape[1])
    except Exception as exc:
        task_results["temporal_embedding"] = {"error": str(exc)}
        _log.warning("  [Gemma] Temporal embedding failed: %s", exc)

    # 7. Gemma vs CLIP comparison — skip when the main embedder IS CLIP (trivial)
    clip_comparison: Dict[str, Any] = {"available": False}
    from models.openclip_model import OpenCLIPEmbedder as _CLIPModel
    _main_is_clip = isinstance(gemma, _CLIPModel)
    if _main_is_clip:
        clip_comparison = {"available": False, "reason": "main embedder is OpenCLIP — comparison skipped (self vs self)"}
        _log.info("  [Gemma vs CLIP] Skipped — main embedder is already OpenCLIP")
    else:
        try:
            _log.info("  [Gemma vs CLIP] Loading temporary OpenCLIP ViT-B-16 ...")
            temp_clip  = _CLIPModel()
            clip_frame = temp_clip.encode_images(sample_images)
            g_e = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
            g_sim_c = np.dot(g_e, g_e.T)
            c_sim   = np.dot(clip_frame, clip_frame.T)
            mask_c  = ~np.eye(n, dtype=bool)
            mean_cossim_gemma_c = float(np.mean(g_sim_c[mask_c]))
            mean_cossim_clip    = float(np.mean(c_sim[mask_c]))
            k_c = min(5, n - 1)
            mnn_c = 0
            for i in range(n):
                gr = g_sim_c[i].copy(); gr[i] = -2.0
                cr = c_sim[i].copy();   cr[i] = -2.0
                mnn_c += len(set(np.argsort(-gr)[:k_c].tolist()) & set(np.argsort(-cr)[:k_c].tolist()))
            mnn_rate_c = mnn_c / (n * k_c)
            clip_comparison = {
                "available": True,
                "n_frames": n,
                "k": k_c,
                "mnn_rate": mnn_rate_c,
                "mean_cossim_gemma": mean_cossim_gemma_c,
                "mean_cossim_clip": mean_cossim_clip,
            }
            _log.info(
                "  [Gemma vs CLIP] MNN@%d=%.3f  mean_cossim: Gemma=%.4f  CLIP=%.4f",
                k_c, mnn_rate_c, mean_cossim_gemma_c, mean_cossim_clip,
            )
            try:
                import torch as _t
                _bb = getattr(temp_clip, "model", None)
                if _bb is not None:
                    _bb.cpu()
            except Exception:
                pass
            del temp_clip, clip_frame
        except Exception as exc:
            clip_comparison = {"available": False, "reason": str(exc)}
            _log.warning("  [Gemma vs CLIP] comparison failed: %s", exc)

    # 8. Gemma vs DINOv3 comparison
    dino_comparison: Dict[str, Any] = {"available": False}
    if _HAS_DINO and n > 1:
        try:
            _log.info("  [Gemma vs DINOv3] Loading temporary DINOv3 ViT-B/14 ...")
            temp_dino   = DINOEmbedder("dinov3_vitb14")
            dino_embeds = temp_dino.encode_images(sample_images)
            g_e = gemma_embeds if gemma_embeds is not None else gemma.encode_images(sample_images)
            g_sim_d = np.dot(g_e, g_e.T)
            d_sim   = np.dot(dino_embeds, dino_embeds.T)
            mask_d  = ~np.eye(n, dtype=bool)
            mean_cossim_gemma_d = float(np.mean(g_sim_d[mask_d]))
            mean_cossim_dino    = float(np.mean(d_sim[mask_d]))
            k_d = min(5, n - 1)
            mnn_d = 0
            for i in range(n):
                gr = g_sim_d[i].copy(); gr[i] = -2.0
                dr = d_sim[i].copy();   dr[i] = -2.0
                mnn_d += len(set(np.argsort(-gr)[:k_d].tolist()) & set(np.argsort(-dr)[:k_d].tolist()))
            mnn_rate_d = mnn_d / (n * k_d)
            dino_comparison = {
                "available": True,
                "n_frames": n,
                "k": k_d,
                "mnn_rate": mnn_rate_d,
                "mean_cossim_gemma": mean_cossim_gemma_d,
                "mean_cossim_dino": mean_cossim_dino,
            }
            _log.info(
                "  [Gemma vs DINOv3] MNN@%d=%.3f  mean_cossim: Gemma=%.4f  DINO=%.4f",
                k_d, mnn_rate_d, mean_cossim_gemma_d, mean_cossim_dino,
            )
            try:
                import torch as _t
                _bb = getattr(temp_dino, "model", None)
                if _bb is not None:
                    _bb.cpu()
            except Exception:
                pass
            del temp_dino, dino_embeds
        except Exception as exc:
            dino_comparison = {"available": False, "reason": str(exc)}
            _log.warning("  [Gemma vs DINOv3] comparison failed: %s", exc)
    else:
        dino_comparison["reason"] = (
            "DINOv3 not available" if not _HAS_DINO else "too few frames for comparison"
        )

    elapsed = time.time() - t0

    write_gemma_analysis_md(
        video_dir / "gemma_analysis.md",
        video_name, effective_api_model or settings.GEMMA_MODEL_ID,
        n, task_results, dino_comparison, text_query_results, elapsed,
        clip_comparison=clip_comparison,
    )

    result.update({
        "skipped": False,
        "n_frames": n,
        "task_results": task_results,
        "dino_comparison": dino_comparison,
        "clip_comparison": clip_comparison,
        "elapsed_sec": elapsed,
    })
    return result


def _write_gemma_captions_md(
    output_path: Path,
    video_name: str,
    model_id: str,
    captions: List[Dict[str, Any]],
) -> None:
    """Write per-frame Gemma generative descriptions to a markdown file."""
    lines = [
        f"# Gemma Frame Descriptions -- {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{model_id}`  |  Frames: {len(captions)}",
        f"",
        f"| # | t (s) | Frame | Description |",
        f"|---|-------|-------|-------------|",
    ]
    for i, c in enumerate(captions, 1):
        fp   = Path(c.get("frame_path", "")).name
        t    = c.get("t_sec", 0.0)
        desc = c.get("description", "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {i} | {t:.1f} | `{fp}` | {desc} |")
    lines += ["", f"---", f"*Produced by {_RUNNER_LABEL}*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  Written %s", output_path)

# ── Memory helpers for GPU-constrained machines ───────────────────────────────

def _offload_models_to_cpu(models: Dict[str, Any]) -> None:
    """Move CLIP and DINO backbones to CPU and flush the CUDA allocator cache.

    Called before loading a large model (Florence-2, ASR) when VRAM is tight.
    The embedders keep their ``self.device`` attribute unchanged so they work
    correctly once the backbone is moved back by :func:`_restore_models_to_gpu`.
    """
    _log_vram_snapshot("before offload CLIP+DINO to CPU")
    import gc
    import torch as _torch
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is not None:
            try:
                backbone.cpu()
            except Exception:
                pass
    try:
        from models.dino_model import _set_dino_xformers_enabled
        _set_dino_xformers_enabled(False)
    except Exception:
        pass
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    free_mb = _torch.cuda.mem_get_info(0)[0] / 1024 ** 2 if _torch.cuda.is_available() else 0
    _log.info("  CLIP+DINO offloaded to CPU — %.0f MiB free on GPU", free_mb)
    _log_vram_snapshot("after offload CLIP+DINO to CPU")


def _prep_vram_for_step(
    models: Dict[str, Any],
    device: str,
    ollama_url: str = "",
    ollama_model: str = "",
) -> None:
    """Offload CLIP+DINOv3, evict any Ollama resident, and flush the CUDA allocator.

    Call this before loading any local inference model (OCR, depth, detection,
    world model) to maximise available VRAM and avoid OOM on 16 GiB class GPUs.
    """
    import gc
    import torch as _torch
    if device != "cuda":
        return
    _log_vram_snapshot("before prep VRAM for next step")
    _offload_models_to_cpu(models)
    _unload_known_sidecars(
        [
            (ollama_url, ollama_model),
            (settings.GEMMA_API_URL, settings.GEMMA_API_MODEL),
            (getattr(settings, "QWEN_API_URL", ""), getattr(settings, "QWEN_MODEL", "")),
            (getattr(settings, "REASONING_API_URL", ""), getattr(settings, "REASONING_MODEL", "")),
        ]
    )
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    free_mb = _torch.cuda.mem_get_info(0)[0] / 1024 ** 2 if _torch.cuda.is_available() else 0
    _log.info("  VRAM cleared for next step — %.0f MiB free", free_mb)
    _log_vram_snapshot("after prep VRAM for next step")


def _restore_models_to_gpu(models: Dict[str, Any], device: str) -> None:
    """Move CLIP and DINO backbones back to *device* after a large model releases."""
    _log_vram_snapshot(f"before restore models to {device}")
    import gc
    import torch as _torch
    # Free any GPU memory held by objects that were just released before trying to
    # restore the backbones — prevents partial moves caused by transient OOM.
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is not None:
            try:
                backbone.to(device)
            except RuntimeError as exc:
                # If OOM halfway through .to(), the model is in a mixed-device
                # state (some params on GPU, others on CPU).  Roll back to a
                # coherent CPU state and log clearly rather than silently failing.
                _log.warning(
                    "  Could not move %s backbone to %s (%s) — rolling back to CPU",
                    key, device, exc,
                )
                try:
                    backbone.cpu()
                except Exception:
                    pass
    try:
        from models.dino_model import _set_dino_xformers_enabled
        _set_dino_xformers_enabled(str(device).startswith("cuda"))
    except Exception:
        pass
    _log.info("  CLIP+DINO restored to %s", device)
    _log_vram_snapshot(f"after restore models to {device}")


def _models_on_device(models: Dict[str, Any], device: str) -> bool:
    import torch as _torch
    expected = _torch.device(device)
    for key in ("clip", "dino"):
        m = models.get(key)
        if m is None:
            continue
        backbone = getattr(m, "model", None)
        if backbone is None:
            continue
        try:
            actual = next(backbone.parameters()).device
        except StopIteration:
            continue
        if actual != expected:
            return False
    return True


def _list_ollama_models(api_url: str) -> List[str]:
    """Return model names available in the Ollama instance at *api_url*."""
    try:
        import httpx
        base = api_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        resp = httpx.get(f"{base}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


# Preferred Gemma model order: smallest usable first so we never pick a 26B/31B
# when a lighter option is available.
_GEMMA_PREFERENCE_ORDER = [
    "gemma4:e4b", "gemma4:4b", "gemma3:4b", "gemma3:1b",
    "gemma4:12b", "gemma3:12b",
    "gemma4:26b", "gemma4:31b", "gemma3:27b",
]


_REASONING_PREFERENCE_ORDER = [
    "deepseek-r1:32b", "qwen3:32b", "qwen3:30b",
    "deepseek-r1:14b", "qwen3:14b",
    "deepseek-r1:8b", "qwen3:8b",
    "gemma3:27b", "gemma3:12b", "gemma4:12b",
    "gemma3:4b", "gemma4:4b", "gemma4:e4b", "gemma3:1b",
]


def _recommend_gemma_sidecar_models(resources: Dict[str, float]) -> Tuple[str, str]:
    """Return recommended (analysis_model, reasoning_model) for current hardware.

    Analysis runs over sampled video frames and should stay relatively light.
    Reasoning runs once at the end and can use a larger long-thinking model.
    """
    vram = resources.get("vram_gb", 0.0)
    free_vram = resources.get("free_vram_gb", vram)
    ram = resources.get("ram_gb", 0.0)

    if free_vram >= 64 or vram >= 80:
        return "gemma4:26b", "deepseek-r1:32b"
    if free_vram >= 32 or vram >= 48:
        return "gemma4:12b", "qwen3:30b"
    if free_vram >= 18 or vram >= 24:
        return "gemma4:4b", "deepseek-r1:14b"
    if free_vram >= 10 or vram >= 16:
        return "gemma4:e4b", "deepseek-r1:14b"

    # CPU / mixed RAM-heavy fallback. Keep analysis lighter; spend RAM on the final audit.
    if ram >= 96:
        return "gemma4:12b", "deepseek-r1:32b"
    if ram >= 64:
        return "gemma4:4b", "deepseek-r1:14b"
    if ram >= 32:
        return "gemma4:e4b", "qwen3:8b"
    return "gemma3:1b", "gemma4:e4b"


def _resolve_ollama_model_with_preferences(
    api_url: str,
    configured_model: str,
    *,
    preference_order: List[str],
    family_prefixes: Tuple[str, ...],
    label: str,
) -> str:
    """Resolve a requested Ollama model against the instance model list."""
    available = _list_ollama_models(api_url)
    if not available:
        return configured_model
    if configured_model in available:
        return configured_model
    for preferred in preference_order:
        if preferred in available:
            _log.warning(
                "  %s model '%s' not found in Ollama; auto-selected '%s'. Pull the desired model with: ollama pull %s",
                label, configured_model, preferred, configured_model,
            )
            return preferred
    family_models = [m for m in available if m.startswith(family_prefixes)]
    if family_models:
        chosen = family_models[0]
        _log.warning(
            "  %s model '%s' not found; using first available family match: '%s'",
            label, configured_model, chosen,
        )
        return chosen
    return configured_model


def _resolve_ollama_gemma_model(api_url: str, configured_model: str) -> str:
    """Return the best available Gemma model for *api_url*.

    1. If *configured_model* is present in Ollama → use it.
    2. Otherwise scan available models and return the lightest Gemma by
       ``_GEMMA_PREFERENCE_ORDER``, or the first gemma* found.
    3. Falls back to *configured_model* (caller will get a 404 and fail clearly).
    """
    resolved = _resolve_ollama_model_with_preferences(
        api_url,
        configured_model,
        preference_order=_GEMMA_PREFERENCE_ORDER,
        family_prefixes=("gemma",),
        label="Gemma analysis",
    )
    if resolved == configured_model:
        available = _list_ollama_models(api_url)
        if available and not any(m.startswith("gemma") for m in available):
            _log.error(
                "No Gemma model found in Ollama. Pull one with: ollama pull gemma4:e4b\n"
                "Available models: %s", available,
            )
    return resolved


def _resolve_ollama_reasoning_model(api_url: str, configured_model: str) -> str:
    """Resolve the final reasoning model against Ollama availability."""
    return _resolve_ollama_model_with_preferences(
        api_url,
        configured_model,
        preference_order=_REASONING_PREFERENCE_ORDER,
        family_prefixes=("deepseek", "qwen", "llama", "gemma"),
        label="Reasoning",
    )


def _unload_ollama_model(api_url: str, model: str) -> bool:
    """Ask Ollama to evict *model* from VRAM by setting keep_alive=0.

    Only works when *api_url* points to an Ollama server (the /api/generate
    endpoint is Ollama-specific; vLLM will return 404 and we silently ignore
    that).  Returns True if the model was successfully unloaded.

    Typical VRAM freed: ~11–12 GiB for a 7B-param model, giving Florence-2
    (~1.5 GiB FP16) plenty of room to load locally.  Ollama auto-reloads the
    model on the next inference request (step R), so no explicit warmup needed.
    """
    try:
        import httpx
    except ImportError:
        return False
    base = api_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        resp = httpx.post(
            f"{base}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=15.0,
        )
        if resp.status_code == 200:
            _log.info("  Ollama: '%s' unloaded from VRAM", model)
            return True
        _log.debug("  Ollama unload returned HTTP %d — may be vLLM (ignored)", resp.status_code)
    except Exception as exc:
        _log.debug("  Could not contact Ollama for unload: %s", exc)
    return False


def _unload_known_sidecars(pairs: List[Tuple[str, str]]) -> None:
    """Unload all known Ollama sidecars from prior steps/runs when possible."""
    seen: set[Tuple[str, str]] = set()
    for url, model in pairs:
        if not url or not model:
            continue
        key = (url, model)
        if key in seen:
            continue
        seen.add(key)
        _unload_ollama_model(url, model)


def _log_vram_snapshot(label: str) -> None:
    """Best-effort VRAM snapshot for both local process and sidecar-heavy runs."""
    try:
        from pipeline.model_registry import detect_resources  # noqa: PLC0415

        resources = detect_resources()
        total = resources.get("vram_gb", 0.0)
        free = resources.get("free_vram_gb", 0.0)
        used = max(0.0, total - free) if total > 0 else 0.0
        _log.info(
            "  [VRAM] %s | total=%.1f GiB free=%.1f GiB used~=%.1f GiB ram=%.1f GiB",
            label,
            total,
            free,
            used,
            resources.get("ram_gb", 0.0),
        )
        return
    except Exception as exc:
        _log.debug("  [VRAM] %s | resource snapshot failed: %s", label, exc)


def _caption_via_florence_api(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    api_url: str,
    model: str,
    domain_hint: str = "",
) -> Dict[str, Any]:
    """Caption frames via a vLLM endpoint serving Florence-2-large.

    vLLM serves Florence-2 with ``--task generate --trust-remote-code``.
    The ``<MORE_DETAILED_CAPTION>`` task token is passed as a text message
    alongside the base64-encoded image; the response is the plain caption string.

    This path consumes zero local VRAM — all inference runs inside the vLLM
    process, which can be on a separate GPU or port from Ollama.
    """
    import base64
    import io

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — cannot use Florence API")
        return {"skipped": True, "reason": "httpx not installed", "captions": []}

    _log.info(
        "  Florence-2 via vLLM API (url=%s  model=%s  frames=%d)",
        api_url, model, len(frame_list),
    )
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    caption_results: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx, (fp, t_sec) in enumerate(frame_list):
        caption = ""
        try:
            img = Image.open(fp).convert("RGB")
            img.thumbnail((768, 768))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": (
                            f"[Context: {domain_hint}] <MORE_DETAILED_CAPTION>"
                            if domain_hint else "<MORE_DETAILED_CAPTION>"
                        )},
                    ],
                }],
                "max_tokens": 256,
                "temperature": 0.0,
            }
            resp = httpx.post(endpoint, json=payload, timeout=60.0)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            # Florence-2 sometimes echoes the task token; strip it
            if raw.startswith("<MORE_DETAILED_CAPTION>"):
                raw = raw[len("<MORE_DETAILED_CAPTION>"):].strip()
            caption = raw
        except Exception as exc:
            _log.debug("  Florence API error for %s: %s", Path(fp).name, exc)

        caption_results.append({
            "frame_path": fp, "t_sec": t_sec,
            "caption": caption,
            "caption_confidence": 0.75 if caption else 0.0,
        })
        if (idx + 1) % 20 == 0:
            _log.info("    ... %d/%d frames captioned via Florence API", idx + 1, len(frame_list))

    elapsed   = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  ✓ Florence API captions: %d/%d frames in %.1fs", captioned, len(frame_list), elapsed)
    out_md = video_dir / "scene_captions.md"
    write_scene_captions_md(out_md, video_name, caption_results, elapsed)
    return {
        "skipped": False, "captions": caption_results,
        "captioned_count": captioned, "elapsed_sec": elapsed, "backend": "florence_api",
    }


def _caption_via_qwen_api(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    api_url: str,
    model: str,
    domain_hint: str = "",
) -> Dict[str, Any]:
    """Caption frames via an OpenAI-compatible VLM endpoint (Ollama / vLLM).

    Used as a fallback when Florence-2 cannot load due to OOM.  Sends one
    ``/chat/completions`` request per frame with the image embedded as a base64
    data-URI.  Images are downscaled to 512 px on the longest side before
    encoding to keep latency reasonable.
    """
    import base64
    import io

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — cannot use Qwen API for captioning")
        return {"skipped": True, "reason": "httpx not installed", "captions": []}

    _log.info(
        "  Florence-2 OOM — falling back to Qwen API captioning "
        "(url=%s  model=%s  frames=%d)",
        api_url, model, len(frame_list),
    )
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    caption_results: List[Dict[str, Any]] = []
    t0 = time.time()
    _MAX_CONSECUTIVE_FAILURES = 3

    # Pre-flight: verify the endpoint is responsive before iterating all frames.
    try:
        probe = httpx.post(
            endpoint,
            json={"model": model, "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            timeout=15.0,
        )
        if probe.status_code >= 500:
            _log.warning(
                "  Qwen API pre-flight failed (HTTP %d) — skipping captioning",
                probe.status_code,
            )
            return {"skipped": True, "reason": f"Qwen API returned {probe.status_code}", "captions": []}
    except Exception as exc:
        _log.warning("  Qwen API pre-flight error (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    consecutive_failures = 0
    for idx, (fp, t_sec) in enumerate(frame_list):
        caption = ""
        try:
            img = Image.open(fp).convert("RGB")
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text",
                         "text": (
                             (f"[Context: {domain_hint}]\n" if domain_hint else "")
                             + "Describe this image in one or two sentences. "
                             "Focus on the scene type, visible objects, and environment."
                         )},
                    ],
                }],
                "max_tokens": 150,
                "temperature": 0.1,
            }
            resp = httpx.post(endpoint, json=payload, timeout=30.0)
            resp.raise_for_status()
            caption = resp.json()["choices"][0]["message"]["content"].strip()
            consecutive_failures = 0
        except Exception as exc:
            _log.debug("  Qwen caption error for %s: %s", Path(fp).name, exc)
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                _log.warning(
                    "  Qwen API: %d consecutive failures — aborting captioning early "
                    "(%d/%d frames done)",
                    consecutive_failures, idx + 1, len(frame_list),
                )
                # Fill remaining frames with empty captions and break
                for fp2, t2 in frame_list[idx + 1:]:
                    caption_results.append({"frame_path": fp2, "t_sec": t2,
                                            "caption": "", "caption_confidence": 0.0})
                break

        caption_results.append({
            "frame_path": fp, "t_sec": t_sec,
            "caption": caption,
            "caption_confidence": 0.7 if caption else 0.0,
        })
        if (idx + 1) % 50 == 0:
            _log.info("    ... %d/%d frames captioned via Qwen API", idx + 1, len(frame_list))

    elapsed   = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  ✓ Qwen API captions: %d/%d frames in %.1fs", captioned, len(frame_list), elapsed)
    out_md = video_dir / "scene_captions.md"
    write_scene_captions_md(out_md, video_name, caption_results, elapsed)
    return {
        "skipped": False, "captions": caption_results,
        "captioned_count": captioned, "elapsed_sec": elapsed, "backend": "qwen_api",
    }


def step_scene_captioning(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    models: Optional[Dict[str, Any]] = None,
    qwen_api_url: str = "",
    qwen_model: str = "",
    florence_api_url: str = "",
    florence_model: str = "",
    domain_hint: str = "",
) -> Dict[str, Any]:
    """Step L: Florence-2 scene captioning with memory management and API support.

    Memory strategy (CUDA only):
      1. If ``florence_api_url`` is set: call Florence-2 via vLLM API — no local
         weights loaded, zero VRAM consumed.  Use this when another process
         (e.g. Ollama) already occupies most of VRAM.
      2. Otherwise load Florence-2 locally:
         a. Offload CLIP+DINO to CPU to free ~1.7 GiB.
         b. If ``qwen_api_url`` looks like Ollama (port 11434): send keep_alive=0
            to evict the VLM (~11-12 GiB freed), giving Florence plenty of room.
            Ollama auto-reloads on the next request (step R).
         c. If Florence still OOMs and ``qwen_api_url`` + ``qwen_model`` are set:
            fall back to Qwen API captioning.
    """
    # ── API route: vLLM serving Florence-2 ────────────────────────────────────
    effective_florence_api_url = florence_api_url or settings.FLORENCE_API_URL
    effective_florence_model   = florence_model or settings.FLORENCE_MODEL
    if effective_florence_api_url:
        _log.info("  Florence-2 via vLLM API at %s", effective_florence_api_url)
        _log_vram_snapshot("before Florence API captioning")
        # Offload CLIP+DINO while API captions run (they aren't needed until step C)
        if models and device == "cuda":
            _offload_models_to_cpu(models)
        result = _caption_via_florence_api(
            frame_list, video_name, video_dir,
            effective_florence_api_url, effective_florence_model,
            domain_hint=domain_hint,
        )
        _log_vram_snapshot("after Florence API captioning")
        return result

    # ── Local route: load Florence-2 weights into this process ────────────────
    out_md = video_dir / "scene_captions.md"
    try:
        from pipeline.florence_model import FlorenceModel
    except ImportError as exc:
        _log.warning("  Florence-2 unavailable (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    # Step 1: offload CLIP+DINO to free ~1.7 GiB
    if models and device == "cuda":
        _offload_models_to_cpu(models)

    # Step 2: unload all Ollama models to free VRAM for Florence
    if device == "cuda":
        if qwen_api_url and qwen_model:
            _unload_ollama_model(qwen_api_url, qwen_model)
        # Also unload Gemma sidecar if configured (may still be resident from step J)
        _gemma_url_cap = settings.GEMMA_API_URL
        _gemma_model_cap = settings.GEMMA_API_MODEL
        if _gemma_url_cap and _gemma_model_cap and _gemma_model_cap != qwen_model:
            _unload_ollama_model(_gemma_url_cap, _gemma_model_cap)

    _log.info("Loading Florence-2-large on %s …", device)
    _log_vram_snapshot("before local Florence load")
    t0 = time.time()
    try:
        florence = FlorenceModel()
    except Exception as exc:
        if qwen_api_url and qwen_model:
            _log.warning("  Florence-2 load failed (%s) — using Qwen API fallback", exc)
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    _log.info("  CUDA cache cleared before Qwen fallback")
            except Exception:
                pass
            return _caption_via_qwen_api(frame_list, video_name, video_dir, qwen_api_url, qwen_model,
                                         domain_hint=domain_hint)
        _log.warning(
            "  Florence-2 load failed (%s) — skipping captioning "
            "(pass --qwen-api-url + --qwen to enable Qwen API fallback)",
            exc,
        )
        return {"skipped": True, "reason": str(exc), "captions": []}

    _log.info("  ✓ Florence-2-large loaded in %.1fs", time.time() - t0)
    _log.info("  Captioning %d frames …", len(frame_list))
    caption_results: List[Dict[str, Any]] = []
    batch_size = settings.FLORENCE_BATCH_SIZE
    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        pil_images = []
        for fp, _t in batch:
            try:
                pil_images.append(Image.open(fp).convert("RGB"))
            except Exception:
                pil_images.append(Image.new("RGB", (224, 224)))
        try:
            captions_and_confs = florence.caption_batch(pil_images)
        except Exception as exc:
            _log.warning("  Florence batch %d failed: %s", batch_start, exc)
            captions_and_confs = [("", 0.5)] * len(batch)
        for (fp, t_sec), (cap, conf) in zip(batch, captions_and_confs):
            caption_results.append({"frame_path": fp, "t_sec": t_sec,
                                    "caption": cap, "caption_confidence": conf})

    elapsed   = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    _log.info("  ✓ %d/%d frames captioned in %.1fs", captioned, len(frame_list), elapsed)
    _log_vram_snapshot("after local Florence captioning")
    write_scene_captions_md(out_md, video_name, caption_results, elapsed)
    florence.release()
    # VRAM freed — caller (_run_video_pipeline) decides when to restore CLIP+DINO

    return {"skipped": False, "captions": caption_results,
            "captioned_count": captioned, "elapsed_sec": elapsed}


def _pick_query_frame(frame_list: List[Tuple[str, float]]) -> Tuple[str, float]:
    return frame_list[len(frame_list) // 2]


def _embed_query(frame_path: str, models: Dict[str, Any], use_dino: bool = True) -> np.ndarray:
    img = Image.open(frame_path).convert("RGB")
    if use_dino and models.get("dino"):
        return models["dino"].encode_images([img])[0]
    return models["clip"].encode_images([img])[0]


def _search(
    query_vec: np.ndarray,
    store: Any,
    is_qdrant: bool,
    top_k: int,
    video_id: str,
    vector_name: str = "clip",
) -> List[Dict[str, Any]]:
    if is_qdrant:
        from qdrant_client.http import models as qmodels
        filt = qmodels.Filter(must=[qmodels.FieldCondition(
            key="video_id", match=qmodels.MatchValue(value=video_id),
        )])
        raw = store.search(vector_name, query_vec, limit=top_k, payload_filter=filt)
        return [{"score": p.score, "payload": p.payload} for p in raw]
    return store.search(query_vec, limit=top_k)


def step_base_model_search_test(
    frame_list: List[Tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: Dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Step C: embed query with base model, search, write base_search.md."""
    out_md = video_dir / "base_search.md"
    qfp, qt = _pick_query_frame(frame_list)
    _log.info("Query frame: %s (t=%.2fs)", Path(qfp).name, qt)
    use_dino = models.get("dino") is not None
    t0       = time.time()
    query_vec = _embed_query(qfp, models, use_dino=use_dino)
    results   = _search(query_vec, store, is_qdrant, top_k, video_id,
                        vector_name="dino" if use_dino else "clip")
    elapsed   = time.time() - t0
    label = "Base DINOv3 (pretrained)" if use_dino else "Base CLIP (pretrained)"
    write_search_md(out_md, video_name, label, qfp, results, qt)
    _log.info("  ✓ Search in %.2fs → top score %.4f", elapsed,
              results[0]["score"] if results else 0)
    return {"results": results, "query_frame": qfp, "query_t_sec": qt}


def step_ssl_finetune(
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    device: str,
    epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Step D: SSL DINOv3 fine-tuning, write finetune_stats.md."""
    out_md   = video_dir / "finetune_stats.md"
    ckpt_dir = video_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(frame_list)
    approach = "temporal" if n_frames >= batch_size * 2 else "augment"
    if approach == "augment":
        _log.info("  Only %d frames — using augment approach", n_frames)
    cfg = FinetuneConfig(
        frames_dir=settings.FRAMES_DIR,
        output_dir=str(ckpt_dir),
        model_name="dinov3_vitb14",
        approach=approach,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-5, weight_decay=0.04, temperature=0.07,
        freeze_blocks=10, embed_dim=768, proj_out_dim=128,
        num_workers=0, save_every=1, max_gap=3, device=device, seed=42,
    )
    _log.info("Starting SSL fine-tuning: %d epochs, approach=%s, device=%s",
              epochs, approach, device)
    t0 = time.time()
    loss_history: List[float] = []

    import pipeline.ssl_finetune as _ssl_mod

    def _run_capturing(c: FinetuneConfig) -> str:
        import torch, random
        random.seed(c.seed); torch.manual_seed(c.seed)
        os.makedirs(c.output_dir, exist_ok=True)
        from pipeline.ssl_finetune import (
            build_augment_transform, TemporalPairDataset, AugmentPairDataset,
            DINOFineTuner, NTXentLoss,
        )
        from torch.utils.data import DataLoader
        transform = build_augment_transform()
        dataset = (TemporalPairDataset(c.frames_dir, transform=transform, max_gap=c.max_gap)
                   if c.approach == "temporal"
                   else AugmentPairDataset(c.frames_dir, transform=transform))
        loader = DataLoader(dataset, batch_size=c.batch_size, shuffle=True,
                            num_workers=c.num_workers, pin_memory=(c.device != "cpu"),
                            drop_last=True)
        tuner     = DINOFineTuner(model_name=c.model_name, freeze_blocks=c.freeze_blocks,
                                  device=c.device, embed_dim=c.embed_dim, proj_out_dim=c.proj_out_dim)
        optimizer = torch.optim.AdamW(tuner.trainable_params(), lr=c.lr, weight_decay=c.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
        loss_fn   = NTXentLoss(temperature=c.temperature)
        best_loss = float("inf")
        best_path = os.path.join(c.output_dir, "dino_ssl_best.pt")
        for epoch in range(1, c.epochs + 1):
            tuner.train(); epoch_losses = []
            for v1, v2 in loader:
                v1, v2 = v1.to(c.device), v2.to(c.device)
                loss = loss_fn(tuner.forward(v1), tuner.forward(v2))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_losses.append(loss.item())
            scheduler.step()
            avg = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            loss_history.append(avg)
            _log.info("    Epoch %d/%d  loss=%.4f", epoch, c.epochs, avg)
            ckpt = os.path.join(c.output_dir, f"dino_ssl_{epoch:03d}.pt")
            tuner.save_checkpoint(ckpt)
            if avg < best_loss:
                best_loss = avg; tuner.save_checkpoint(best_path)
        return best_path

    best_path = _run_capturing(cfg)
    elapsed   = time.time() - t0
    best_loss = min(loss_history) if loss_history else float("nan")
    _log.info("  ✓ Fine-tuning complete in %.1fs | best loss=%.4f | checkpoint: %s",
              elapsed, best_loss, best_path)
    _log.info("  To use: export DINO_CHECKPOINT=%s", best_path)
    write_finetune_stats_md(out_md, video_name, cfg, best_loss, best_path, elapsed, loss_history)
    ckpt_mb = os.path.getsize(best_path) / 1e6 if os.path.exists(best_path) else 0
    return {"checkpoint": best_path, "best_loss": best_loss,
            "elapsed_sec": elapsed, "ckpt_mb": ckpt_mb, "cfg": cfg}


def step_distill(
    teacher_checkpoint: str,
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
    distill_epochs: int,
    batch_size: int,
    caption_embeddings: Optional[np.ndarray] = None,
    gemma_embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Step E: distil fine-tuned teacher → student with maximum hydration.

    Maximum-hydration distillation chain:
      1. Teacher backbone: fine-tuned DINOv3 ViT-B/14 (SSL checkpoint).
         When *gemma_embedder* is provided, uses GemmaVisionTeacher instead —
         richer multimodal embeddings as the distillation target.
      2. Caption anchor loss: when *caption_embeddings* are provided (CLIP text
         embeddings of per-frame captions), adds a λ=0.5 cosine term that pulls
         the student toward language-grounded targets, transferring Gemma's
         semantic understanding into the small student.
      3. Student: DINOv2 ViT-S/14 (22M params, 384-dim).
    """
    out_md = video_dir / "distill_stats.md"
    result: Dict[str, Any] = {
        "student_backbone": None, "best_path": "", "best_loss": float("nan"),
        "best_recall": float("nan"), "compression_ratio": 0.0,
        "student_dim": 384, "teacher_dim": 768,
        "student_model": "dinov2_vits14", "ckpt_mb": 0.0, "skipped": False,
    }
    if not _HAS_DINO:
        _log.warning("  DINO not available — skipping distillation")
        result["skipped"] = True; return result

    # ── Choose teacher ────────────────────────────────────────────────────────
    teacher_bb = None
    teacher_label = "DINOv3 ViT-B/14 (SSL)"

    if gemma_embedder is not None and _HAS_GEMMA:
        try:
            from pipeline.distill import GemmaVisionTeacher
            teacher_bb    = GemmaVisionTeacher(gemma_embedder)
            teacher_label = f"Gemma 4 vision encoder (dim={gemma_embedder.image_dim()})"
            result["teacher_dim"] = gemma_embedder.image_dim()
            _log.info("  Distillation teacher: %s", teacher_label)
        except Exception as exc:
            _log.warning("  GemmaVisionTeacher failed (%s) — falling back to DINOv3", exc)
            teacher_bb = None

    if teacher_bb is None:
        try:
            import torch
            from models.dino_model import hub_load_dino
            teacher_bb = hub_load_dino("dinov3_vitb14", pretrained=True).to(device)
            state = torch.load(teacher_checkpoint, map_location=device)
            teacher_bb.load_state_dict(state); teacher_bb.eval()
            _log.info("  Teacher loaded from checkpoint: %s", teacher_checkpoint)
        except Exception as exc:
            _log.warning("  Could not load teacher checkpoint (%s) — skipping distillation", exc)
            result["skipped"] = True; return result

    # ── Caption anchor ────────────────────────────────────────────────────────
    lambda_cap = 0.0
    cap_embs   = None
    if caption_embeddings is not None and len(caption_embeddings) > 0:
        lambda_cap = 0.5
        cap_embs   = caption_embeddings
        _log.info(
            "  Caption anchor loss enabled: λ=%.1f  anchors=%d  dim=%d",
            lambda_cap, len(cap_embs), cap_embs.shape[1],
        )

    cfg = DistillConfig(
        student_model="dinov2_vits14",
        epochs=distill_epochs,
        batch_size=batch_size,
        device=device,
        lambda_caption_anchor=lambda_cap,
        caption_embeddings=cap_embs,
    )
    frame_paths = [fp for fp, _ in frame_list]
    _log.info(
        "Starting distillation: %s → ViT-S/14  epochs=%d  frames=%d  caption_anchor=%s",
        teacher_label, cfg.epochs, len(frame_paths),
        f"λ={lambda_cap}" if lambda_cap > 0 else "off",
    )
    try:
        stats = run_distillation(teacher_bb, frame_paths, video_dir / "checkpoints", cfg)
    except Exception as exc:
        _log.warning("  Distillation failed (%s) — skipping", exc)
        result["skipped"] = True; return result
    distiller = stats.pop("distiller")
    best_path = stats.get("best_path", "")
    if not best_path or not os.path.exists(best_path) or not math.isfinite(stats.get("best_loss", float("nan"))):
        _log.warning("  Distillation produced no valid student checkpoint — skipping")
        result["skipped"] = True
        return result
    result.update(stats)
    result["student_backbone"] = distiller.student_backbone()
    result["ckpt_mb"]          = os.path.getsize(best_path) / 1e6
    result["teacher_label"]    = teacher_label
    result["caption_anchor_used"] = lambda_cap > 0
    _log.info(
        "  ✓ Distillation complete in %.1fs | best_loss=%.4f | best_R@1=%.3f | "
        "compression=%.1f× | student=%s (dim=%d)",
        stats["elapsed"], stats["best_loss"], stats.get("best_recall", float("nan")),
        stats.get("compression_ratio", 0.0), stats["student_model"], stats["student_dim"],
    )
    write_distill_stats_md(out_md, video_name, stats)
    return result


def step_export_model(
    checkpoint_path: str,
    frame_list: List[Tuple[str, float]],
    video_dir: Path,
    device: str,
    models: Dict[str, Any],
    no_onnx: bool,
    student_backbone: Optional[Any] = None,
    student_dim: int = 768,
) -> Dict[str, Any]:
    """Step F: export model to ONNX + build gallery.npz."""
    edge_dir     = video_dir / "edge_models"
    edge_dir.mkdir(parents=True, exist_ok=True)
    onnx_path    = str(edge_dir / "dino_demo.onnx")
    gallery_path = str(edge_dir / "gallery.npz")
    result: Dict[str, Any] = {"onnx_path": onnx_path, "gallery_path": gallery_path,
                               "onnx_mb": 0.0, "exported": False, "gallery_saved": False}
    backbone_to_export = None
    if student_backbone is not None:
        backbone_to_export = student_backbone
        model_label        = f"distilled student (ViT-S/14, dim={student_dim})"
    elif _HAS_DINO:
        dino = models.get("dino")
        if dino is None:
            _log.warning("  DINO not available — will use CLIP for gallery only")
        else:
            try:
                _log.info("Loading fine-tuned checkpoint: %s", checkpoint_path)
                dino.load_backbone_checkpoint(checkpoint_path)
                backbone_to_export = dino.model.eval()
                model_label        = "fine-tuned teacher (ViT-B/14)"
            except Exception as exc:
                _log.warning("  Could not load checkpoint (%s) — using base DINO", exc)
                backbone_to_export = dino.model.eval()
                model_label        = "base DINOv3 teacher (ViT-B/14)"
    else:
        _log.warning("  DINO not available — skipping ONNX export; will use CLIP for gallery")

    if backbone_to_export is not None and not no_onnx:
        try:
            import torch
            for _mod_name, _mod in sys.modules.items():
                if "dinov2" in _mod_name and hasattr(_mod, "XFORMERS_AVAILABLE"):
                    _mod.XFORMERS_AVAILABLE = False
            backbone_cpu = backbone_to_export.cpu().eval()
            # Wrap in a single-input module so ONNX never captures 'masks'
            # as a required input (DINOv2 forward(x, masks=None) leaks the
            # masks node into the graph under torch.onnx tracing).
            class _SingleInputWrapper(torch.nn.Module):
                def __init__(self, bb):
                    super().__init__()
                    self.bb = bb
                def forward(self, x):
                    return self.bb(x)
            export_model = _SingleInputWrapper(backbone_cpu).eval()
            if hasattr(export_model.bb, "interpolate_antialias"):
                export_model.bb.interpolate_antialias = False
            if hasattr(export_model.bb, "interpolate_offset"):
                export_model.bb.interpolate_offset = 0.0
            # 224 matches EdgeClassifier._preprocess_image default (224×224).
            # DINOv2 accepts any multiple of patch_size=14; 224=14×16 is valid.
            dummy = torch.zeros(1, 3, 224, 224)
            _log.info("Exporting ONNX (%s) to %s …", model_label, onnx_path)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                torch.onnx.export(export_model, dummy, onnx_path, opset_version=18,
                                  input_names=["pixel_values"], output_names=["embedding"],
                                  do_constant_folding=True, dynamo=False)
            if os.path.exists(onnx_path):
                onnx_mb = os.path.getsize(onnx_path) / 1e6
                result["onnx_mb"] = onnx_mb; result["exported"] = True
                _log.info("  ✓ ONNX export complete: %.1f MB → %s", onnx_mb, onnx_path)
            else:
                _log.warning("  ONNX export ran but file not found at %s", onnx_path)
            for _mod_name, _mod in sys.modules.items():
                if "dinov2" in _mod_name and hasattr(_mod, "XFORMERS_AVAILABLE"):
                    _mod.XFORMERS_AVAILABLE = True
            backbone_to_export = backbone_to_export.to(device).eval()
        except Exception as exc:
            _log.warning("  ONNX export failed (%s) — skipping", exc)
            for _mod_name, _mod in sys.modules.items():
                if "dinov2" in _mod_name and hasattr(_mod, "XFORMERS_AVAILABLE"):
                    _mod.XFORMERS_AVAILABLE = True
            if backbone_to_export is not None:
                try:
                    backbone_to_export = backbone_to_export.to(device).eval()
                except Exception:
                    pass
    elif backbone_to_export is not None:
        _log.info("  ONNX export skipped (--no-onnx)")

    _log.info("Building embedding gallery from %d frames …", len(frame_list))
    try:
        step     = max(1, len(frame_list) // 200)
        sampled  = [fp for fp, _ in frame_list[::step] if os.path.isfile(fp)]
        if not sampled:
            raise ValueError("No valid frame paths for gallery build")
        labels_map = {"scene": sampled}
        if result["exported"] and os.path.exists(onnx_path):
            build_gallery(labels_map=labels_map, output_path=gallery_path, onnx_path=onnx_path)
            _log.info("  Gallery built using ONNX model")
        elif backbone_to_export is not None:
            build_gallery(labels_map=labels_map, output_path=gallery_path,
                          backbone=backbone_to_export)
            _log.info("  Gallery built using PyTorch backbone")
        else:
            clip_model: OpenCLIPEmbedder = models["clip"]
            all_embeds = []
            for fp in sampled:
                img = Image.open(fp).convert("RGB")
                emb = clip_model.encode_images([img])[0]
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                all_embeds.append(emb.astype(np.float32))
            np.savez(gallery_path,
                     embeddings=np.stack(all_embeds, axis=0),
                     labels=np.array(["scene"] * len(all_embeds), dtype=object),
                     label_names=np.array(["scene"], dtype=object))
            _log.info("  Gallery built using CLIP fallback")
        if os.path.exists(gallery_path):
            result["gallery_saved"] = True
            _log.info("  ✓ Gallery saved: %d embeddings → %s (%.1f MB)",
                      len(sampled), gallery_path, os.path.getsize(gallery_path) / 1e6)
        else:
            _log.warning("  Gallery file not found after build: %s", gallery_path)
    except Exception as exc:
        _log.warning("  Gallery build failed (%s)", exc, exc_info=True)
    return result


def step_finetuned_model_search_test(
    frame_list: List[Tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: Dict[str, Any],
    query_frame: str,
    query_t_sec: float,
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Step G: search with fine-tuned DINO, write finetuned_search.md."""
    out_md   = video_dir / "finetuned_search.md"
    use_dino = models.get("dino") is not None
    t0       = time.time()
    query_vec = _embed_query(query_frame, models, use_dino=use_dino)
    results   = _search(query_vec, store, is_qdrant, top_k, video_id,
                        vector_name="dino" if use_dino else "clip")
    ft_infer_ms = (time.time() - t0) * 1000 / max(len(frame_list), 1)
    write_search_md(out_md, video_name, "Fine-tuned DINOv3 (SSL adapted)",
                    query_frame, results, query_t_sec)
    return {"results": results, "infer_ms": ft_infer_ms}


def step_asr_transcription(
    video_path: Path,
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step M: extract audio, run Whisper ASR."""
    out_md = video_dir / "asr_subtitles.md"
    result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    try:
        from pipeline.audio_extractor import extract_audio, map_subtitles_to_frames
        from pipeline.asr_model import ASRModel
    except ImportError as exc:
        _log.warning("  ASR unavailable (%s) — skipping", exc)
        return result
    asr = ASRModel()
    _log_vram_snapshot("before ASR model use")
    if not asr.is_enabled():
        _log.info("  ASR disabled (ASR_ENABLED=false) — skipping")
        return result
    audio_dir = video_dir / "audio"; audio_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Extracting audio from %s …", video_path.name)
    wav_path = extract_audio(str(video_path), str(audio_dir))
    if not wav_path:
        _log.warning("  No audio stream found in %s — ASR skipped", video_path.name)
        return result
    _log.info("Transcribing audio with %s …", asr.model_id)
    t0       = time.time()
    segments = asr.transcribe(wav_path)
    elapsed  = time.time() - t0
    if not segments:
        _log.warning("  ASR returned no segments for %s", video_path.name)
        return result
    frame_timestamps = [t for _, t in frame_list]
    subtitle_map     = map_subtitles_to_frames(segments, frame_timestamps,
                                               window_sec=settings.ASR_SUBTITLE_WINDOW_SEC)
    covered = sum(1 for t in frame_timestamps if t in subtitle_map)
    _log.info("  ✓ ASR: %d segments → %d/%d frames have subtitles (%.1fs, model=%s)",
              len(segments), covered, len(frame_list), elapsed, asr.model_id)
    _log_vram_snapshot("after ASR model use")
    lines = [
        f"# ASR Subtitles — {video_name}", f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{asr.model_id}`",
        f"Segments: {len(segments)}  |  Frames with subtitles: {covered}/{len(frame_list)}",
        f"Elapsed: {elapsed:.1f}s", f"",
        f"## Subtitle Segments", f"",
        f"| Start (s) | End (s) | Text |",
        f"|-----------|---------|------|",
    ]
    for seg in segments:
        ts = seg.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
        start = float(ts[0]) if len(ts) > 0 and ts[0] is not None else 0.0
        end = float(ts[1]) if len(ts) > 1 and ts[1] is not None else start
        text = seg.get("text", "").strip().replace("|", "\\|")
        lines.append(f"| {start:.2f} | {end:.2f} | {text} |")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · ASR step M*"]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    result.update({"skipped": False, "subtitle_map": subtitle_map,
                   "segments": segments, "elapsed_sec": elapsed, "covered_frames": covered})
    return result


def step_ocr_extraction(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    caption_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Step N: visible text extraction per frame."""
    result: Dict[str, Any] = {"skipped": True, "ocr_results": []}
    try:
        from pipeline.ocr_model import OCRModel
    except ImportError as exc:
        _log.warning("  OCR unavailable (%s) — skipping", exc)
        return result
    ocr = OCRModel()
    _log_vram_snapshot("before OCR model use")
    if not ocr.is_enabled():
        _log.info("  OCR disabled (OCR_ENABLED=false) — skipping"); return result
    _log.info("Running OCR on %d frames (model=%s) …", len(frame_list), ocr.model_id)
    t0 = time.time()
    threshold = settings.OCR_MIN_CAPTION_CONFIDENCE
    caption_conf_by_frame: Dict[str, float] = {}
    if caption_results:
        caption_conf_by_frame = {
            str(r.get("frame_path")): float(r.get("caption_confidence", 0.0) or 0.0)
            for r in caption_results
            if r.get("frame_path")
        }

    selected_frame_list: List[Tuple[str, float]] = []
    skipped_by_caption: Dict[str, Dict[str, Any]] = {}
    if threshold > 0.0 and caption_conf_by_frame:
        for fp, t_sec in frame_list:
            conf = caption_conf_by_frame.get(fp)
            if conf is not None and conf >= threshold:
                skipped_by_caption[fp] = {
                    "frame_path": fp,
                    "t_sec": t_sec,
                    "ocr_text": "",
                    "ocr_model": ocr.model_id,
                    "ocr_skipped_by_caption": True,
                }
            else:
                selected_frame_list.append((fp, t_sec))
        _log.info(
            "  OCR prescreen: %d/%d frames selected (caption_confidence < %.2f)",
            len(selected_frame_list),
            len(frame_list),
            threshold,
        )
    else:
        selected_frame_list = list(frame_list)

    processed_results = _run_batched_frame_inference(
        selected_frame_list,
        batch_size=settings.OCR_BATCH_SIZE,
        batch_fn=lambda _batch, imgs: ocr.extract_text_batch(imgs),
        warning_label="OCR",
        error_result={"ocr_text": "", "ocr_error": True},
    )
    processed_by_frame = {str(r["frame_path"]): r for r in processed_results}
    ocr_results: List[Dict[str, Any]] = []
    for fp, t_sec in frame_list:
        if fp in processed_by_frame:
            ocr_results.append(processed_by_frame[fp])
        else:
            ocr_results.append(
                skipped_by_caption.get(
                    fp,
                    {"frame_path": fp, "t_sec": t_sec, "ocr_text": "", "ocr_error": True},
                )
            )
    elapsed   = time.time() - t0
    non_empty = sum(1 for r in ocr_results if r.get("ocr_text"))
    _log.info("  ✓ OCR: %d/%d frames have text in %.1fs", non_empty, len(frame_list), elapsed)
    result.update({"skipped": False, "ocr_results": ocr_results,
                   "non_empty": non_empty, "elapsed_sec": elapsed})
    ocr.release()
    _log_vram_snapshot("after OCR model use")
    return result


def step_depth_estimation(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step O: depth estimation per frame."""
    result: Dict[str, Any] = {"skipped": True, "depth_results": []}
    try:
        from pipeline.depth_model import DepthModel
    except ImportError as exc:
        _log.warning("  Depth model unavailable (%s) — skipping", exc)
        return result
    depth_model = DepthModel()
    _log_vram_snapshot("before depth model use")
    if not depth_model.is_enabled():
        _log.info("  Depth disabled (DEPTH_ENABLED=false) — skipping"); return result
    _log.info("Running depth estimation on %d frames (model=%s) …",
              len(frame_list), depth_model.model_id)
    t0 = time.time()
    depth_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=lambda _batch, imgs: depth_model.estimate_batch(imgs),
        warning_label="Depth",
        error_result={"depth_error": True},
    )
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in depth_results
        if not r.get("depth_error")
        and not r.get("depth_unavailable")
        and not r.get("depth_disabled")
    )
    _log.info("  ✓ Depth: %d/%d frames estimated in %.1fs", ok, len(frame_list), elapsed)
    result.update({"skipped": False, "depth_results": depth_results,
                   "ok_count": ok, "elapsed_sec": elapsed})
    depth_model.release()
    _log_vram_snapshot("after depth model use")
    return result


def step_object_detection(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step P: object detection per frame."""
    result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    try:
        from pipeline.detection_model import DetectionModel
    except ImportError as exc:
        _log.warning("  Detection model unavailable (%s) — skipping", exc)
        return result
    det_model = DetectionModel()
    _log_vram_snapshot("before detection model use")
    if not det_model.is_enabled():
        _log.info("  Detection disabled (DETECTION_ENABLED=false) — skipping"); return result
    _log.info("Running object detection on %d frames (model=%s) …",
              len(frame_list), det_model.model_id)
    t0 = time.time()
    det_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=lambda _batch, imgs: det_model.detect_batch(imgs),
        warning_label="Detection",
        error_result={"detection_error": True},
    )
    elapsed     = time.time() - t0
    total_objs  = sum(len(r.get("detections", [])) for r in det_results)
    ok          = sum(
        1
        for r in det_results
        if not r.get("detection_error")
        and not r.get("detection_unavailable")
        and not r.get("detection_disabled")
    )
    _log.info("  ✓ Detection: %d objects across %d/%d frames in %.1fs",
              total_objs, ok, len(frame_list), elapsed)
    result.update({"skipped": False, "detection_results": det_results,
                   "total_objects": total_objs, "ok_count": ok, "elapsed_sec": elapsed})
    det_model.release()
    _log_vram_snapshot("after detection model use")
    return result


def step_world_model_pass(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step Q: world model video embeddings."""
    result: Dict[str, Any] = {"skipped": True, "world_results": []}
    try:
        from pipeline.world_model import WorldModel
    except ImportError as exc:
        _log.warning("  World model unavailable (%s) — skipping", exc)
        return result
    wm = WorldModel()
    _log_vram_snapshot("before world model use")
    if not wm.is_enabled():
        _log.info("  World model disabled (WORLD_MODEL_ENABLED=false) — skipping"); return result
    clip_frames = settings.WORLD_MODEL_CLIP_FRAMES
    _log.info("Running world model on %d frames in clips of %d (model=%s) …",
              len(frame_list), clip_frames, wm.model_id)
    t0 = time.time()
    world_results: List[Dict[str, Any]] = []
    for clip_start in range(0, len(frame_list), clip_frames):
        clip = frame_list[clip_start : clip_start + clip_frames]
        imgs = _open_frame_batch(clip)
        try:
            clip_out = wm.process_clip(imgs)
        except Exception as exc:
            _log.warning("  World model clip %d failed: %s", clip_start, exc)
            clip_out = {"world_model_error": True}
        mid = clip_start + len(clip) // 2
        fp, t_sec = frame_list[mid]
        world_results.append({"frame_path": fp, "t_sec": t_sec, **clip_out})
    elapsed = time.time() - t0
    ok = sum(
        1
        for r in world_results
        if not r.get("world_model_error")
        and not r.get("world_model_unavailable")
        and not r.get("world_model_disabled")
    )
    _log.info("  ✓ World model: %d clips processed in %.1fs", ok, elapsed)
    result.update({"skipped": False, "world_results": world_results,
                   "ok_count": ok, "elapsed_sec": elapsed})
    wm.release()
    _log_vram_snapshot("after world model use")
    return result


def step_qwen_captioning(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: Dict[float, str],
    ocr_results: List[Dict[str, Any]],
    clip_prescreen_fn=None,
    knowledge: Optional["VideoKnowledge"] = None,
) -> Dict[str, Any]:
    """Step R: Qwen VLM detailed scene captioning with full agentic context.

    When *knowledge* is provided, each frame's prompt is enriched with all
    prior observations: Florence caption, depth profile, detected objects,
    scene segment, ASR, OCR, and the previous frame's Qwen structured output.
    This lets Qwen reason about *what changed* rather than describing each
    frame in isolation.
    """
    out_md = video_dir / "detailed_captions.md"
    result: Dict[str, Any] = {"skipped": True, "results": []}
    try:
        from pipeline.qwen_model import QwenModel
    except ImportError as exc:
        _log.warning("  Qwen model unavailable (%s) — skipping", exc)
        return result
    qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn)
    _log_vram_snapshot("before Qwen sidecar use")
    if not qwen.is_enabled():
        _log.info("  Qwen disabled (QWEN_API_URL not set) — skipping detailed captioning")
        _log.info("  To enable: --qwen-api-url http://localhost:8010/v1  (or set QWEN_API_URL)")
        return result
    ocr_map: Dict[float, str] = {r["t_sec"]: r["ocr_text"]
                                  for r in ocr_results
                                  if r.get("t_sec") is not None and r.get("ocr_text")}

    domain = knowledge.domain_hint() if knowledge else ""
    if domain:
        _log.info("  Qwen domain hint: %s", domain)
    _log.info("Running Qwen detailed captioning on %d frames (model=%s  agentic=%s) …",
              len(frame_list), settings.QWEN_MODEL, "yes" if knowledge else "no")
    t0 = time.time()

    caption_results: List[Dict[str, Any]] = []

    def _batch_fn(batch: List[Tuple[str, float]], imgs: List) -> List[Dict[str, Any]]:
        extra_contexts = None
        if knowledge:
            extra_contexts = [knowledge.context_for_frame(t_sec) for _fp, t_sec in batch]
        results = qwen.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
            extra_contexts=extra_contexts,
            domain_hint=domain or None,
        )
        # Feed each successful result back into knowledge as prior state
        if knowledge:
            for r in results:
                knowledge.update_qwen_state(r)
        return results

    batch_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=_batch_fn,
        warning_label="Qwen",
        error_result={"service_unavailable": True},
    )
    for r in batch_results:
        t_sec = r.get("t_sec", 0.0)
        caption_results.append({**r, "subtitle_text": subtitle_map.get(t_sec) or ""})
    elapsed = time.time() - t0
    ok             = sum(1 for r in caption_results
                         if not r.get("service_unavailable") and not r.get("skipped"))
    subtitle_used  = sum(1 for r in caption_results if r.get("subtitle_text"))
    _log.info("  ✓ Qwen: %d/%d frames captioned in %.1fs (%d with ASR  agentic=%s)",
              ok, len(frame_list), elapsed, subtitle_used, "yes" if knowledge else "no")
    _log_vram_snapshot("after Qwen sidecar use")
    write_detailed_captions_md(out_md, video_name, caption_results, elapsed, settings.QWEN_MODEL)
    result.update({"skipped": False, "results": caption_results,
                   "ok_count": ok, "subtitle_used": subtitle_used, "elapsed_sec": elapsed})
    return result


def step_compare_and_describe(
    frame_list: List[Tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    base_results: List[Dict],
    ft_results: List[Dict],
    models: Dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
    ckpt_mb: float,
    onnx_mb: float,
) -> Dict[str, Any]:
    """Step H: compare results, caption video, write comparison.md."""
    out_md       = video_dir / "comparison.md"
    sample_paths = [fp for fp, _ in frame_list[:10]]
    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")
    t0 = time.time()
    clip_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
    base_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)
    ft_infer_ms   = base_infer_ms
    if dino_model:
        t0 = time.time()
        dino_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
        ft_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)
    _log.info("Computing video-to-text description …")
    try:
        step = max(1, len(frame_list) // 32)
        sampled_imgs  = [Image.open(fp).convert("RGB") for fp, _ in frame_list[::step]]
        frame_embeds  = clip_model.encode_images(sampled_imgs)
        avg_embed     = frame_embeds.mean(axis=0)
        text_embeds   = clip_model.encode_texts(_TEXT_PROMPTS)
        scores        = text_embeds @ avg_embed
        ranked        = sorted(zip(_TEXT_PROMPTS, scores.tolist()), key=lambda x: x[1], reverse=True)
        text_descriptions = ranked[:3]; all_scored = ranked
        for desc, score in text_descriptions:
            _log.info("  Video description: \"%s\" (sim=%.3f)", desc, score)
    except Exception as exc:
        _log.warning("  Video-to-text failed (%s)", exc)
        text_descriptions = [("description unavailable", 0.0)]; all_scored = text_descriptions
    write_comparison_md(out_md, video_name, base_results, ft_results,
                        base_infer_ms, ft_infer_ms, ckpt_mb, onnx_mb, text_descriptions)
    desc_md = video_dir / "description.md"
    write_description_md(desc_md, video_name, frame_list, text_descriptions, all_scored)
    return {"text_descriptions": text_descriptions, "base_infer_ms": base_infer_ms,
            "ft_infer_ms": ft_infer_ms,
            "top_description": text_descriptions[0][0] if text_descriptions else ""}


def step_create_3d_map(
    video_path: Path,
    video_id: str,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
    run_sfm_flag: bool,
    run_gsplat_flag: bool = True,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Step I: build sparse 3D map + 3D Gaussian Splat."""
    return build_sparse_map(
        video_path=str(video_path),
        video_id=video_id,
        map_dir=video_dir / "3d_map",
        frame_list=frame_list,
        models=models,
        run_sfm_flag=run_sfm_flag,
        run_gsplat_flag=run_gsplat_flag,
        device=device,
    )


# ── Agentic video synthesis ───────────────────────────────────────────────────


def _build_context_prompt(video_name: str, video_context: Dict[str, Any]) -> str:
    """Build a text prompt summarising accumulated observations for the LLM."""
    parts = [f"Video: {video_name}"]

    meta = video_context.get("meta", {})
    if meta:
        parts.append(
            f"Duration: {meta.get('duration_sec', 0):.1f}s | Frames: {meta.get('frame_count', 0)}"
        )

    gem_ctx = video_context.get("gemma_analysis", {})
    if gem_ctx:
        parts.append(
            f"\nGemma analysis ({gem_ctx.get('n_frames', 0)} frames, "
            f"{gem_ctx.get('n_tasks', 0)} analyses):"
        )
        task_res = gem_ctx.get("task_results", {})
        clf = task_res.get("scene_classification", {})
        if clf.get("category_distribution"):
            top_cat = next(iter(clf["category_distribution"]))
            parts.append(f"  - dominant scene type: {top_cat}")
        fv = task_res.get("fact_verification", {})
        if fv.get("claims"):
            top_claim = max(fv["claims"], key=lambda r: r.get("mean_score", 0.0))
            parts.append(
                f"  - strongest visual claim: {top_claim['claim']} "
                f"(score={top_claim['mean_score']:.3f})"
            )
        sc = task_res.get("scene_change_detection", {})
        if sc.get("n_changes") is not None:
            parts.append(f"  - scene transitions detected: {sc['n_changes']}")
        cl = task_res.get("scene_clustering", {})
        if cl.get("n_clusters"):
            parts.append(f"  - semantic clusters: {cl['n_clusters']}")
        mnn = gem_ctx.get("mnn_rate")
        if mnn is not None:
            parts.append(f"  - Gemma/DINOv3 MNN agreement: {mnn:.1%}")

    top_descs = video_context.get("top_descriptions", [])
    if top_descs:
        parts.append("\nTop scene descriptions (CLIP similarity):")
        for desc, score in top_descs[:5]:
            parts.append(f"  - {desc} (score={score:.3f})")

    captions = video_context.get("captions", [])
    if captions:
        step = max(1, len(captions) // 20)
        sampled = captions[::step][:20]
        parts.append(
            f"\nPer-frame captions ({len(sampled)} sampled from {len(captions)}):"
        )
        for r in sampled:
            cap = r.get("caption", "")
            if cap:
                parts.append(f"  [{r.get('t_sec', 0.0):.1f}s] {cap}")

    asr_segs = video_context.get("asr_segments", [])
    if asr_segs:
        parts.append(f"\nAudio transcript ({len(asr_segs)} segments):")
        for seg in asr_segs[:10]:
            ts = seg.get("timestamp") or (0.0, 0.0)
            text = seg.get("text", "").strip()
            if text:
                parts.append(f"  [{ts[0]:.1f}s–{ts[1]:.1f}s] {text}")

    ocr_list = video_context.get("ocr", [])
    if ocr_list:
        ocr_with_text = [r for r in ocr_list if r.get("ocr_text")][:10]
        if ocr_with_text:
            parts.append(
                f"\nVisible text (OCR, {len(ocr_with_text)} frames with text):"
            )
            for r in ocr_with_text[:5]:
                parts.append(f"  [{r['t_sec']:.1f}s] {r['ocr_text'][:100]}")

    obj_counts = video_context.get("detections", {})
    if obj_counts:
        parts.append("\nDetected objects (label: count):")
        for label, count in sorted(obj_counts.items(), key=lambda x: -x[1])[:10]:
            parts.append(f"  - {label}: {count}")

    qwen_caps = video_context.get("qwen_captions", [])
    if qwen_caps:
        step = max(1, len(qwen_caps) // 10)
        sampled = qwen_caps[::step][:10]
        parts.append(
            f"\nDetailed scene analysis ({len(sampled)} sampled from {len(qwen_caps)}):"
        )
        for r in sampled:
            cap = r.get("caption") or r.get("scene_description") or ""
            if cap:
                parts.append(f"  [{r.get('t_sec', 0.0):.1f}s] {str(cap)[:200]}")

    return "\n".join(parts)


def _append_agentic_step(
    trace: List[Dict[str, Any]],
    *,
    step_id: str,
    title: str,
    description: str,
    status: str,
    context_inputs: Optional[List[str]] = None,
    context_outputs: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
    artifacts: Optional[List[str]] = None,
) -> None:
    trace.append(
        {
            "step_id": step_id,
            "title": title,
            "description": description,
            "status": status,
            "context_inputs": context_inputs or [],
            "context_outputs": context_outputs or [],
            "risks": risks or [],
            "artifacts": artifacts or [],
        }
    )


def _build_agentic_flow_prompt(video_name: str, video_context: Dict[str, Any]) -> str:
    trace = video_context.get("agentic_trace", [])
    lines = [
        f"Video: {video_name}",
        "You are auditing an agentic video-demo pipeline.",
        "Analyze how context is accumulated step by step, how later steps depend on earlier outputs, and where wrong context can propagate.",
        "",
        "Per-step trace:",
    ]
    for item in trace:
        lines.extend(
            [
                f"- Step {item.get('step_id')} {item.get('title')}",
                f"  Description: {item.get('description', '')}",
                f"  Status: {item.get('status', 'unknown')}",
                f"  Context received: {', '.join(item.get('context_inputs', [])) or 'none'}",
                f"  Context produced: {', '.join(item.get('context_outputs', [])) or 'none'}",
                f"  Risks: {', '.join(item.get('risks', [])) or 'none'}",
                f"  Artifacts: {', '.join(item.get('artifacts', [])) or 'none'}",
            ]
        )

    lines.extend(
        [
            "",
            "Write markdown with these sections exactly:",
            "## Flow Summary",
            "Short explanation of how context evolves through the pipeline.",
            "## Step-by-Step Agentic Context",
            "Use one bullet per step. For each step explain what context entered, what new context was created, and what later steps rely on it.",
            "## Risk Register",
            "Use one bullet per step. Explicitly call out misidentification risk, wrong-context risk, and propagation risk.",
            "## Highest-Risk Context Failures",
            "List the most important compounded failure modes across the pipeline.",
            "## Mitigations",
            "Recommend concrete checks or gates to reduce context corruption.",
            "",
            "Be specific. Focus on agentic context flow, not generic ML commentary.",
        ]
    )
    return "\n".join(lines)


def _build_agentic_flow_prompt_compact(video_name: str, video_context: Dict[str, Any]) -> str:
    """Compact audit prompt tuned for slow reasoning models on Ollama."""
    trace = video_context.get("agentic_trace", [])
    lines = [
        f"Video: {video_name}",
        "Audit the agentic pipeline context flow.",
        "Return markdown with these exact sections:",
        "## Flow Summary",
        "## Step-by-Step Agentic Context",
        "## Risk Register",
        "## Highest-Risk Context Failures",
        "## Mitigations",
        "",
        "Per-step trace:",
    ]
    for item in trace:
        lines.append(
            f"- {item.get('step_id')} {item.get('title')} | "
            f"status={item.get('status', 'unknown')} | "
            f"in={'; '.join(item.get('context_inputs', [])[:3]) or 'none'} | "
            f"out={'; '.join(item.get('context_outputs', [])[:3]) or 'none'} | "
            f"risks={'; '.join(item.get('risks', [])[:3]) or 'none'}"
        )
    lines += [
        "",
        "Be concise and specific.",
        "Keep the whole answer under 900 words.",
        "Focus on context propagation, stale context, misidentification, and mitigation.",
    ]
    return "\n".join(lines)


def _reasoning_timeout_for_model(model: str) -> float:
    base = float(getattr(settings, "REASONING_TIMEOUT_SEC", 240))
    m = (model or "").lower()
    if any(tag in m for tag in ("32b", "30b", "27b", "26b")):
        return max(base, 600.0)
    if any(tag in m for tag in ("14b", "12b")):
        return max(base, 360.0)
    return base


def _fallback_agentic_flow_analysis(video_context: Dict[str, Any]) -> str:
    trace = video_context.get("agentic_trace", [])
    lines = [
        "## Flow Summary",
        "The demo pipeline accumulates context progressively: frame sampling establishes the timeline, multimodal steps add semantic and geometric evidence, and later reasoning steps consume that evidence to produce higher-level conclusions. The main agentic risk is not a single wrong model output but error carry-over from early observations into later narrative and structured reasoning.",
        "",
        "## Step-by-Step Agentic Context",
    ]
    for item in trace:
        received = ", ".join(item.get("context_inputs", [])) or "no prior context"
        produced = ", ".join(item.get("context_outputs", [])) or "no durable context"
        lines.append(
            f"- **{item.get('step_id')} {item.get('title')}** receives {received}; "
            f"produces {produced}; downstream consumers inherit both its evidence and its errors."
        )
    lines += ["", "## Risk Register"]
    for item in trace:
        risk_text = "; ".join(item.get("risks", [])) or "low direct risk"
        lines.append(f"- **{item.get('step_id')} {item.get('title')}**: {risk_text}.")
    lines += [
        "",
        "## Highest-Risk Context Failures",
        "- Qwen detailed captioning is the most exposed step because it consumes accumulated Florence, ASR, OCR, depth, detection, and prior-Qwen state. One bad upstream cue can shift the frame narrative.",
        "- Final synthesis can convert uncertain intermediate evidence into confident-looking ontology or narrative text if confidence and disagreement are not surfaced explicitly.",
        "- Distillation and fine-tuning can preserve or amplify weak teacher assumptions if retrieval gains are accepted without semantic validation.",
        "",
        "## Mitigations",
        "- Gate downstream prompts with confidence and disagreement summaries rather than only positive evidence.",
        "- Keep per-step provenance in artifacts so a reviewer can trace each claim to its source step.",
        "- Add contradiction checks between captions, OCR, ASR, detections, and final narratives before exporting final conclusions.",
    ]
    return "\n".join(lines)


def step_agentic_flow_artifact(
    video_name: str,
    video_dir: Path,
    video_context: Dict[str, Any],
    api_url: str,
    model: str,
) -> Dict[str, Any]:
    """Final step: generate an artifact tracing agentic context and risks."""
    result: Dict[str, Any] = {"skipped": True, "llm_used": False, "model": model or "deterministic"}
    output_path = video_dir / "agentic_flow.md"
    llm_analysis = ""
    t0 = time.time()
    _log_vram_snapshot("before reasoning sidecar use")

    if api_url:
        try:
            import httpx

            endpoint = f"{api_url.rstrip('/')}/chat/completions"
            timeout_sec = _reasoning_timeout_for_model(model)
            attempts = [
                {
                    "prompt": _build_agentic_flow_prompt_compact(video_name, video_context),
                    "max_tokens": 1100,
                },
                {
                    "prompt": _build_agentic_flow_prompt(video_name, video_context),
                    "max_tokens": 1500,
                },
            ]
            last_exc: Optional[Exception] = None
            for idx, attempt in enumerate(attempts, 1):
                try:
                    _log.info(
                        "  Agentic flow reasoning attempt %d/%d (model=%s timeout=%.0fs max_tokens=%d)",
                        idx, len(attempts), model, timeout_sec, attempt["max_tokens"],
                    )
                    resp = httpx.post(
                        endpoint,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": attempt["prompt"]}],
                            "max_tokens": attempt["max_tokens"],
                            "temperature": 0.2,
                        },
                        timeout=timeout_sec,
                    )
                    resp.raise_for_status()
                    llm_analysis = resp.json()["choices"][0]["message"]["content"].strip()
                    if llm_analysis:
                        result["llm_used"] = True
                        _log.info("  ✓ Agentic flow analysis generated with %s", model)
                        break
                except Exception as exc:
                    last_exc = exc
                    _log.warning("  Agentic flow reasoning attempt %d failed (%s)", idx, exc)
            if not llm_analysis and last_exc is not None:
                raise last_exc
        except Exception as exc:
            _log.warning("  Agentic flow reasoning failed (%s) — using deterministic fallback", exc)

    if not llm_analysis:
        llm_analysis = _fallback_agentic_flow_analysis(video_context)
        result["model"] = "deterministic-fallback"

    elapsed = time.time() - t0
    write_agentic_flow_md(
        output_path,
        video_name,
        video_context.get("agentic_trace", []),
        elapsed,
        result["model"],
        llm_analysis,
    )
    _log_vram_snapshot("after reasoning sidecar use")
    result.update({"skipped": False, "elapsed_sec": elapsed, "output_path": str(output_path)})
    return result


def step_video_synthesis(
    video_name: str,
    video_dir: Path,
    video_context: Dict[str, Any],
    api_url: str,
    model: str,
) -> Dict[str, Any]:
    """Step Z: synthesise video ontology + narrative via Ollama/vLLM API.

    Uses all accumulated context from steps A–H as input.  No local model is
    loaded — this is a pure API call, so CLIP+DINO can remain offloaded.
    Writes ``video_synthesis.md`` and ``video_ontology.json``.
    """
    result: Dict[str, Any] = {"skipped": True, "ontology": {}, "narrative": ""}
    if not api_url:
        _log.info("  Synthesis skipped (no QWEN_API_URL / --qwen-api-url set)")
        return result

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — skipping video synthesis")
        return result

    context_str = _build_context_prompt(video_name, video_context)
    endpoint    = f"{api_url.rstrip('/')}/chat/completions"
    t0          = time.time()
    _log_vram_snapshot("before synthesis sidecar use")
    ontology: Dict[str, Any] = {}
    narrative = ""

    # 1. Request structured ontology JSON
    ontology_prompt = (
        f"{context_str}\n\n"
        "Based on all the above observations, produce a structured video ontology "
        "as valid JSON with these fields:\n"
        '{\n'
        '  "domain": "string (e.g. outdoor_surveillance, urban_traffic, aerial_reconnaissance)",\n'
        '  "environment": "string (terrain/setting description)",\n'
        '  "primary_activities": ["list of main activities observed"],\n'
        '  "key_objects": ["list of key objects/entities"],\n'
        '  "temporal_structure": "string (how scene evolves over time)",\n'
        '  "scene_complexity": "low|medium|high",\n'
        '  "confidence": 0.0\n'
        '}\n\n'
        "Output only the JSON object, no other text."
    )
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": ontology_prompt}],
                "max_tokens": 512,
                "temperature": 0.1,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ontology = json.loads(raw.strip())
        _log.info("  ✓ Video ontology generated  (domain=%s)", ontology.get("domain", "?"))
    except Exception as exc:
        _log.warning("  Ontology generation failed (%s)", exc)

    # 2. Request fine-grained narrative
    narrative_prompt = (
        f"{context_str}\n\n"
        "Write a fine-grained narrative description of this video in markdown. Cover:\n"
        "1. **Opening scene** — what is visible in the first frames\n"
        "2. **Main activity** — primary events, motion, and content\n"
        "3. **Environmental context** — terrain, lighting, setting details\n"
        "4. **Notable details** — specific objects, text, audio cues if any\n"
        "5. **Temporal evolution** — how the scene changes over time\n"
        "6. **Summary** — one-sentence overall description\n\n"
        "Be specific and grounded in the observations above. Use technical language "
        "appropriate for outdoor robotics and surveillance contexts."
    )
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": narrative_prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        narrative = resp.json()["choices"][0]["message"]["content"].strip()
        _log.info("  ✓ Video narrative generated (%d chars)", len(narrative))
    except Exception as exc:
        _log.warning("  Narrative generation failed (%s)", exc)

    elapsed = time.time() - t0
    _log.info("  ✓ Video synthesis complete in %.1fs", elapsed)

    write_video_synthesis_md(
        video_dir / "video_synthesis.md",
        video_name, ontology, narrative, elapsed, model,
    )
    if ontology:
        (video_dir / "video_ontology.json").write_text(
            json.dumps(ontology, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log.info("  ✓ Ontology saved → video_ontology.json")

    result.update({"skipped": False, "ontology": ontology,
                   "narrative": narrative, "elapsed_sec": elapsed})
    _log_vram_snapshot("after synthesis sidecar use")
    return result


# ── Per-video orchestrator ────────────────────────────────────────────────────

_TOTAL_STEPS = 19
_VIDEO_EXTS  = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def find_videos(videos_dir: Path) -> List[Path]:
    return sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)


def run_video_pipeline(
    args: Any,
    video_path: Path,
    output_dir: Path,
    models: Dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
    _out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all pipeline steps for a single video. Returns per-video stats dict.

    *_out* is an optional external dict that is used as the stats container.
    When provided, callers can inspect it for partial results if an exception
    escapes — the timings and frame counts recorded up to the failure point
    are preserved.
    """
    video_name = video_path.stem
    video_id   = video_name.replace(" ", "_").lower()
    video_dir  = output_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    _banner(f"Processing video: {video_path.name}")
    _log.info("Output directory: %s", video_dir)

    # Use the shared container when provided so partial state is visible outside.
    if _out is None:
        _out = {}
    _out.update({"name": video_name, "video_path": str(video_path), "timings": {}})
    stats: Dict[str, Any] = _out
    T = stats["timings"]

    # Accumulated context passed through the pipeline; enriches synthesis at step Z.
    video_context: Dict[str, Any] = {"video_name": video_name}
    agentic_trace: List[Dict[str, Any]] = []
    video_context["agentic_trace"] = agentic_trace

    # Tracks whether CLIP+DINO backbones are on GPU (relevant only when device=="cuda").
    clip_dino_on_gpu = (device == "cuda" and _models_on_device(models, "cuda"))

    # A: Extract frames
    _step(1, _TOTAL_STEPS, "Frame extraction")
    with _Timer(T, "A_extract"):
        a = step_extract_frames(video_path, video_id, video_dir, fps=args.fps)
    frame_list: List[Tuple[str, float]] = a["frame_list"]
    stats["frames"]       = a["meta"]["frame_count"]
    stats["duration_sec"] = a["meta"]["duration_sec"]
    video_context["meta"] = {
        "frame_count": stats["frames"],
        "duration_sec": stats["duration_sec"],
    }
    _append_agentic_step(
        agentic_trace,
        step_id="A",
        title="Frame extraction",
        description="Decode the source video into a timestamped frame sequence that every later step reuses.",
        status="ok" if frame_list else "empty",
        context_inputs=["raw video bytes"],
        context_outputs=[
            f"{len(frame_list)} timestamped frames",
            f"duration {stats['duration_sec']:.1f}s",
            "frame timeline for all downstream alignment",
        ],
        risks=[
            "sampling can miss short-lived objects or events",
            "timestamp drift can misalign later ASR/OCR/detection context",
            "wrong extraction rate biases all downstream context",
        ],
        artifacts=["frames_metadata.json"],
    )
    if not frame_list:
        _log.error("No frames extracted — skipping video %s", video_path.name)
        return stats

    # Agentic knowledge accumulator — enriches downstream steps as each completes.
    knowledge = VideoKnowledge(
        video_name=video_name,
        duration_sec=stats["duration_sec"],
        frame_count=stats["frames"],
    )

    # B: Index — needs CLIP+DINO on GPU
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(2, _TOTAL_STEPS, "Vector store indexing")
    with _Timer(T, "B_index"):
        b = step_index_to_store(video_path, video_id, store, is_qdrant, models, frame_list)
    if device == "cuda":
        clip_dino_on_gpu = _models_on_device(models, device)
    stats["index_sec"] = b["elapsed_sec"]
    _append_agentic_step(
        agentic_trace,
        step_id="B",
        title="Vector store indexing",
        description="Embed frames for retrieval and establish the baseline semantic memory used by search steps.",
        status="ok",
        context_inputs=["timestamped frames", "base CLIP/DINO embeddings"],
        context_outputs=[
            "retrieval index populated",
            f"index latency {b['elapsed_sec']:.1f}s",
            "baseline visual neighborhoods",
        ],
        risks=[
            "embedding collisions can mix semantically different frames",
            "duplicate-heavy footage can distort nearest-neighbor context",
            "wrong baseline neighborhoods affect later search comparisons",
        ],
        artifacts=[],
    )

    # J: Gemma open-weight multimodal analysis
    _step(3, _TOTAL_STEPS, "Gemma multimodal analysis → gemma_analysis.md")
    with _Timer(T, "J_gemma"):
        j = step_gemma_analysis(
            video_path, video_id, video_name, video_dir, frame_list, models,
            gemma_api_url=getattr(args, "gemma_api_url", ""),
            gemma_api_model=getattr(args, "gemma_api_model", ""),
        )
    if not j.get("skipped"):
        video_context["gemma_analysis"] = {
            "n_frames":        j.get("n_frames", 0),
            "n_tasks":         len(j.get("task_results", {})),
            "mnn_rate_dino":   j.get("dino_comparison", {}).get("mnn_rate"),
            "mnn_rate_clip":   j.get("clip_comparison", {}).get("mnn_rate"),
        }
        knowledge.add_gemma(
            j.get("task_results", {}),
            mnn_dino=j.get("dino_comparison", {}).get("mnn_rate") or 0.0,
        )
    _append_agentic_step(
        agentic_trace,
        step_id="J",
        title="Gemma multimodal analysis",
        description="Run coarse video-level reasoning to infer dominant scene type, transitions, clusters, and teacher-signal compatibility.",
        status="skipped" if j.get("skipped") else "ok",
        context_inputs=["sampled video frames", "existing embeddings"],
        context_outputs=[
            f"scene type {knowledge.scene_type or 'unknown'}",
            f"{knowledge.n_transitions} transitions",
            f"{knowledge.n_clusters} semantic clusters",
            "domain hint for captioning and later reasoning",
        ] if not j.get("skipped") else ["no persistent Gemma context"],
        risks=[
            "scene classification can over-generalize from sparse samples",
            "wrong domain hint can bias Florence and Qwen toward the wrong narrative",
            "teacher-similarity judgments can be mistaken for semantic truth",
        ],
        artifacts=["gemma_analysis.md"] if not j.get("skipped") else [],
    )
    # Unload Gemma from Ollama immediately after analysis — frees ~12+ GiB for Florence.
    _gemma_api_url_j = settings.GEMMA_API_URL or getattr(args, "gemma_api_url", "")
    _gemma_api_model_j = settings.GEMMA_API_MODEL or getattr(args, "gemma_api_model", "")
    if _gemma_api_url_j and _gemma_api_model_j and device == "cuda":
        _unload_ollama_model(_gemma_api_url_j, _gemma_api_model_j)

    # L: Scene captioning — offloads CLIP+DINO internally, does NOT restore them
    caption_results: List[Dict[str, Any]] = []
    if not args.no_caption:
        _step(4, _TOTAL_STEPS, "Florence-2 scene captioning → scene_captions.md")
        with _Timer(T, "L_caption"):
            l_cap = step_scene_captioning(
                frame_list, video_name, video_dir, device,
                models=models,
                qwen_api_url=getattr(args, "qwen_api_url", ""),
                qwen_model=getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
                florence_api_url=getattr(args, "florence_api_url", ""),
                florence_model=getattr(args, "florence_model", ""),
                domain_hint=knowledge.domain_hint(),
            )
        caption_results = l_cap.get("captions", [])
        knowledge.add_captions(caption_results)
        if device == "cuda":
            clip_dino_on_gpu = False  # Florence offloaded them; we keep them off for M–Q
    else:
        T["L_caption"] = 0.0
        _step(4, _TOTAL_STEPS, "Scene captioning (skipped — --no-caption)")
    video_context["captions"] = caption_results
    _append_agentic_step(
        agentic_trace,
        step_id="L",
        title="Scene captioning",
        description="Generate per-frame scene captions and coarse temporal segments to seed later context-aware reasoning.",
        status="skipped" if args.no_caption else "ok",
        context_inputs=[
            "timestamped frames",
            knowledge.domain_hint() or "no domain hint",
        ],
        context_outputs=[
            f"{len(caption_results)} scene captions",
            f"{len(getattr(knowledge, '_segments', []))} caption segments",
            "frame-level prior scene descriptions",
        ] if caption_results else ["no caption context"],
        risks=[
            "caption hallucinations can create false scene priors",
            "repeated captions may hide real transitions",
            "wrong segment boundaries can contaminate later frame context",
        ],
        artifacts=["scene_captions.md"] if caption_results else [],
    )

    # M: ASR — no CLIP/DINO needed; Whisper manages its own VRAM
    asr_result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    if args.asr:
        _step(5, _TOTAL_STEPS, "ASR transcription → asr_subtitles.md")
        with _Timer(T, "M_asr"):
            asr_result = step_asr_transcription(video_path, frame_list, video_name, video_dir)
    else:
        T["M_asr"] = 0.0
    video_context["asr_segments"] = asr_result.get("segments", [])
    knowledge.add_asr(asr_result.get("subtitle_map", {}))
    _append_agentic_step(
        agentic_trace,
        step_id="M",
        title="ASR transcription",
        description="Transcribe audio and align subtitles to frames so later reasoning can use speech context.",
        status="skipped" if asr_result.get("skipped") else "ok",
        context_inputs=["video audio stream", "frame timestamps"],
        context_outputs=[
            f"{len(asr_result.get('segments', []))} ASR segments",
            f"{asr_result.get('covered_frames', 0)} subtitle-covered frames",
            "audio context aligned to timestamps",
        ] if not asr_result.get("skipped") else ["no audio context"],
        risks=[
            "transcription errors can inject false entities or actions",
            "language mismatch can produce wrong context with high confidence",
            "subtitle-frame misalignment can contaminate visual reasoning",
        ],
        artifacts=["asr_subtitles.md"] if not asr_result.get("skipped") else [],
    )

    # N: OCR
    ocr_result: Dict[str, Any] = {"skipped": True, "ocr_results": []}
    if args.ocr:
        _step(6, _TOTAL_STEPS, "OCR text extraction")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "N_ocr"):
            ocr_result = step_ocr_extraction(
                frame_list,
                video_name,
                video_dir,
                caption_results=caption_results,
            )
    else:
        T["N_ocr"] = 0.0
    video_context["ocr"] = ocr_result.get("ocr_results", [])
    knowledge.add_ocr(ocr_result.get("ocr_results", []))
    _append_agentic_step(
        agentic_trace,
        step_id="N",
        title="OCR extraction",
        description="Extract visible text from frames to enrich object and scene interpretation.",
        status="skipped" if ocr_result.get("skipped") else "ok",
        context_inputs=["frames", "caption-confidence prescreen when available"],
        context_outputs=[
            f"{ocr_result.get('non_empty', 0)} frames with OCR text",
            "visible-text evidence for Qwen and final synthesis",
        ] if not ocr_result.get("skipped") else ["no OCR context"],
        risks=[
            "small or low-contrast text can be missed",
            "false OCR tokens can create wrong named-entity context",
            "prescreen skips may discard frames with useful text",
        ],
        artifacts=[],
    )

    # O: Depth
    depth_result: Dict[str, Any] = {"skipped": True, "depth_results": []}
    if args.depth:
        _step(7, _TOTAL_STEPS, "Depth estimation")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "O_depth"):
            depth_result = step_depth_estimation(frame_list, video_name, video_dir)
        knowledge.add_depth(depth_result.get("depth_results", []))
    else:
        T["O_depth"] = 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="O",
        title="Depth estimation",
        description="Estimate relative scene geometry for near/far reasoning and scene-structure cues.",
        status="skipped" if depth_result.get("skipped") else "ok",
        context_inputs=["frames"],
        context_outputs=[
            f"{depth_result.get('ok_count', 0)} depth-estimated frames",
            "relative geometry cues for later prompts",
        ] if not depth_result.get("skipped") else ["no depth context"],
        risks=[
            "monocular depth can confuse scale and elevation",
            "depth failure in low-texture scenes can misstate geometry",
            "wrong depth priors can bias later scene explanations",
        ],
        artifacts=[],
    )

    # P: Detection — accumulate per-label object counts into context
    det_result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    if args.detection:
        _step(8, _TOTAL_STEPS, "Object detection")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "P_detection"):
            det_result = step_object_detection(frame_list, video_name, video_dir)
        knowledge.add_detections(det_result.get("detection_results", []))
    else:
        T["P_detection"] = 0.0
    if not det_result.get("skipped"):
        obj_counts: Dict[str, int] = {}
        for _r in det_result.get("detection_results", []):
            for _d in _r.get("detections", []):
                lbl = _d.get("label", "unknown")
                obj_counts[lbl] = obj_counts.get(lbl, 0) + 1
        video_context["detections"] = obj_counts
    _append_agentic_step(
        agentic_trace,
        step_id="P",
        title="Object detection",
        description="Detect frame-level entities so later reasoning can reference concrete objects instead of only global scene text.",
        status="skipped" if det_result.get("skipped") else "ok",
        context_inputs=["frames"],
        context_outputs=[
            f"{det_result.get('total_objects', 0)} detected objects",
            f"top entities: {', '.join(knowledge.known_entities[:5]) or 'none'}",
        ] if not det_result.get("skipped") else ["no detection context"],
        risks=[
            "class confusion can misidentify critical objects",
            "open-vocabulary labels can drift semantically across frames",
            "false positives can become persistent agentic context",
        ],
        artifacts=[],
    )

    # Q: World model
    world_result: Dict[str, Any] = {"skipped": True, "world_results": []}
    if args.world_model:
        _step(9, _TOTAL_STEPS, "World model video embeddings")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "Q_world"):
            world_result = step_world_model_pass(frame_list, video_name, video_dir)
    else:
        T["Q_world"] = 0.0
    if not world_result.get("skipped"):
        video_context["world_model_clips"] = world_result.get("ok_count", 0)
    _append_agentic_step(
        agentic_trace,
        step_id="Q",
        title="World model pass",
        description="Compress clips into temporal embeddings to capture motion-level context not visible in single frames.",
        status="skipped" if world_result.get("skipped") else "ok",
        context_inputs=["ordered frame clips"],
        context_outputs=[
            f"{world_result.get('ok_count', 0)} temporal clip embeddings",
            "coarse motion-context signal",
        ] if not world_result.get("skipped") else ["no temporal clip context"],
        risks=[
            "clip pooling can smooth away rare but important events",
            "temporal embeddings are hard to interpret and easy to overtrust",
            "wrong clip-level context can bias synthesis without clear provenance",
        ],
        artifacts=[],
    )

    # R: Qwen — uses ASR + OCR context from previous steps (already agentic)
    qwen_result: Dict[str, Any] = {"skipped": True, "results": []}
    if args.qwen:
        _step(10, _TOTAL_STEPS, "Qwen VLM detailed captioning → detailed_captions.md")
        with _Timer(T, "R_qwen"):
            qwen_result = step_qwen_captioning(
                frame_list, video_name, video_dir,
                subtitle_map=asr_result.get("subtitle_map", {}),
                ocr_results=ocr_result.get("ocr_results", []),
                # Pass a passthrough so QwenModel never creates a second CLIP
                # embedder (OpenCLIPTagger) that competes for VRAM.  In demo
                # mode we want full coverage; prescreening is not needed.
                clip_prescreen_fn=lambda _img: True,
                knowledge=knowledge,
            )
    else:
        T["R_qwen"] = 0.0
    if not qwen_result.get("skipped"):
        video_context["qwen_captions"] = qwen_result.get("results", [])
    _append_agentic_step(
        agentic_trace,
        step_id="R",
        title="Qwen detailed captioning",
        description="Fuse visual frames with accumulated Florence, ASR, OCR, depth, detections, and prior-Qwen state for structured per-frame reasoning.",
        status="skipped" if qwen_result.get("skipped") else "ok",
        context_inputs=[
            "frame image",
            "Florence scene priors",
            "ASR-aligned subtitle context",
            "OCR/depth/detection cues",
            "previous Qwen structured state",
        ],
        context_outputs=[
            f"{qwen_result.get('ok_count', 0)} detailed captions",
            "structured scene facts for downstream synthesis",
            "updated prior-state chain across frames",
        ] if not qwen_result.get("skipped") else ["no detailed reasoning context"],
        risks=[
            "upstream misidentification compounds inside one prompt",
            "previous-frame state can anchor the model to stale or wrong context",
            "rich prompt context can make uncertain claims look internally consistent",
        ],
        artifacts=["detailed_captions.md"] if not qwen_result.get("skipped") else [],
    )

    if any([args.asr, args.ocr, args.depth, args.detection, args.world_model, args.qwen]):
        _mm_md = video_dir / "multimodal_features.md"
        write_multimodal_md(_mm_md, video_name, asr_result, ocr_result,
                            depth_result, det_result, world_result, qwen_result)

    # C: Base model search — restore CLIP+DINO to GPU if needed
    if device == "cuda" and not clip_dino_on_gpu:
        # Evict Ollama (reloaded during step R) before restoring CLIP+DINO.
        # After step R Ollama holds ~13 GiB; without eviction DINO restore OOMs.
        if getattr(args, "qwen", False):
            _qwen_url   = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
            _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
            if _qwen_url and _qwen_model:
                _unload_ollama_model(_qwen_url, _qwen_model)
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(11, _TOTAL_STEPS, "Base model transformation test → base_search.md")
    with _Timer(T, "C_base_search"):
        c = step_base_model_search_test(frame_list, store, is_qdrant, models,
                                        video_id, video_name, video_dir, top_k=args.top_k)
    base_results = c["results"]; query_frame = c["query_frame"]; query_t_sec = c["query_t_sec"]
    stats["base_top_score"] = base_results[0]["score"] if base_results else 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="C",
        title="Base search test",
        description="Measure retrieval behavior of the base model as the control reference for adaptation steps.",
        status="ok",
        context_inputs=["retrieval index", "query frame"],
        context_outputs=[
            f"top-{len(base_results)} baseline matches",
            f"query at {query_t_sec:.1f}s",
        ],
        risks=[
            "search quality may favor visual similarity over semantic identity",
            "one query frame can underrepresent broader retrieval behavior",
            "baseline errors can distort later before/after comparisons",
        ],
        artifacts=["base_search.md"],
    )

    # D: SSL fine-tuning — DINOFineTuner loads its own separate DINO; offload ours first.
    # Skipped when using an API-based embedder — no local backbone to fine-tune.
    checkpoint_path = ""
    if models.get("uses_api_embedder"):
        T["D_finetune"] = 0.0
        T["E_distill"] = 0.0
        _step(12, _TOTAL_STEPS, "SSL DINOv3 fine-tuning (skipped — API embedder)")
        _step(13, _TOTAL_STEPS, "Knowledge distillation (skipped — API embedder)")
        student_backbone = None; student_dim = 768
    else:
        if device == "cuda" and clip_dino_on_gpu:
            _offload_models_to_cpu(models)
            clip_dino_on_gpu = False
        _step(12, _TOTAL_STEPS, "SSL DINOv3 fine-tuning → finetune_stats.md")
        with _Timer(T, "D_finetune"):
            d = step_ssl_finetune(video_id, video_name, video_dir, frame_list, device,
                                  epochs=args.epochs, batch_size=args.batch_size)
        stats["best_loss"] = d["best_loss"]; stats["ckpt_mb"] = d["ckpt_mb"]
        checkpoint_path    = d["checkpoint"]
        _append_agentic_step(
            agentic_trace,
            step_id="D",
            title="SSL fine-tuning",
            description="Adapt the local DINO backbone to mission-specific footage so retrieval neighborhoods reflect this video domain more closely.",
            status="ok",
            context_inputs=["frame sequence", "base DINO initialization"],
            context_outputs=[
                f"best loss {d['best_loss']:.4f}",
                "mission-adapted backbone checkpoint",
            ],
            risks=[
                "small-video adaptation can overfit to accidental patterns",
                "temporal positives can encode wrong sameness assumptions",
                "adapted features can improve scores while harming semantics",
            ],
            artifacts=["finetune_stats.md", "checkpoints/dino_ssl_best.pt"],
        )

        # E: Distillation — maximum-hydration chain (Gemma teacher + caption anchor when available)
        student_backbone = None; student_dim = 768
        if not args.no_distill:
            # Build caption anchor embeddings from Florence captions via CLIP text encoder
            _cap_anchor_embs: Optional[np.ndarray] = None
            _scene_captions = caption_results  # set by step_scene_captioning (step L)
            if _scene_captions and models.get("clip"):
                try:
                    _cap_texts = [r.get("caption") or "" for r in _scene_captions]
                    _cap_texts = [t for t in _cap_texts if t.strip()]
                    if _cap_texts:
                        _clip_model = models["clip"]
                        # Only OpenCLIPEmbedder has encode_texts suitable for anchoring
                        if hasattr(_clip_model, "encode_texts") and not isinstance(_clip_model, GemmaEmbedder if _HAS_GEMMA else type(None)):
                            _cap_anchor_embs = _clip_model.encode_texts(_cap_texts)
                            _log.info(
                                "  Distillation caption anchors: %d CLIP text embeddings from Florence captions",
                                len(_cap_anchor_embs),
                            )
                except Exception as _exc:
                    _log.debug("  Caption anchor prep failed (%s) — distilling without anchor", _exc)

            # Use Gemma embedder as teacher when loaded and MODEL_NAME=gemma
            _gemma_teacher = None
            if _HAS_GEMMA and isinstance(models.get("clip"), GemmaEmbedder):
                _gemma_teacher = models["clip"]
                _log.info("  Using GemmaVisionTeacher for distillation (max hydration)")

            _step(13, _TOTAL_STEPS, "Knowledge distillation (max hydration) → ViT-S/14 student")
            with _Timer(T, "E_distill"):
                e_distill = step_distill(
                    checkpoint_path, frame_list, video_name, video_dir, device,
                    distill_epochs=args.distill_epochs, batch_size=args.batch_size,
                    caption_embeddings=_cap_anchor_embs,
                    gemma_embedder=_gemma_teacher,
                )
            if not e_distill["skipped"]:
                student_backbone         = e_distill["student_backbone"]
                student_dim              = e_distill["student_dim"]
                stats["distill_loss"]    = e_distill["best_loss"]
                stats["student_ckpt_mb"] = e_distill["ckpt_mb"]
                stats["student_dim"]     = student_dim
                stats["teacher_dim"]     = e_distill["teacher_dim"]
            _append_agentic_step(
                agentic_trace,
                step_id="E",
                title="Knowledge distillation",
                description="Compress teacher geometry and optional language anchors into a smaller student suitable for deployment.",
                status="skipped" if e_distill.get("skipped") else "ok",
                context_inputs=[
                    "fine-tuned teacher checkpoint",
                    "optional Gemma teacher alignment",
                    "optional Florence caption anchors",
                ],
                context_outputs=[
                    f"student dim {student_dim}",
                    f"best distill loss {e_distill.get('best_loss', float('nan')):.4f}",
                    "student deployment checkpoint",
                ] if not e_distill.get("skipped") else ["no distilled student"],
                risks=[
                    "teacher mistakes transfer into the student representation",
                    "caption anchors can inject wrong semantics into retrieval space",
                    "compression can erase rare but important distinctions",
                ],
                artifacts=["distill_stats.md", "checkpoints/student_best.pt"] if not e_distill.get("skipped") else [],
            )
        else:
            T["E_distill"] = 0.0
            _step(13, _TOTAL_STEPS, "Knowledge distillation (skipped — --no-distill)")
            _append_agentic_step(
                agentic_trace,
                step_id="E",
                title="Knowledge distillation",
                description="Compress teacher knowledge into a smaller deployable student.",
                status="skipped",
                context_inputs=["fine-tuned teacher checkpoint"],
                context_outputs=["no distilled student"],
                risks=[
                    "without this step, deployment relies on the larger teacher or ONNX export only",
                    "no compression audit is produced",
                ],
                artifacts=[],
            )
    if models.get("uses_api_embedder"):
        _append_agentic_step(
            agentic_trace,
            step_id="D",
            title="SSL fine-tuning",
            description="Adapt the local DINO backbone to mission-specific footage.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no local fine-tuning checkpoint"],
            risks=[
                "no task-specific adaptation is learned in API-embedder mode",
            ],
            artifacts=[],
        )
        _append_agentic_step(
            agentic_trace,
            step_id="E",
            title="Knowledge distillation",
            description="Compress teacher knowledge into a smaller deployable student.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no distillation artifacts"],
            risks=[
                "no student compression path is available in API-embedder mode",
            ],
            artifacts=[],
        )

    # F: ONNX export + gallery — restore CLIP+DINO (export uses models["dino"])
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(14, _TOTAL_STEPS, "ONNX export + gallery build → edge_models/")
    with _Timer(T, "F_export"):
        e = step_export_model(checkpoint_path, frame_list, video_dir, device, models,
                              no_onnx=args.no_onnx,
                              student_backbone=student_backbone, student_dim=student_dim)
    stats["onnx_mb"] = e.get("onnx_mb", 0.0); stats["onnx_exported"] = e.get("exported", False)
    _append_agentic_step(
        agentic_trace,
        step_id="F",
        title="ONNX export",
        description="Package the best available backbone and gallery into deployment artifacts.",
        status="ok",
        context_inputs=["teacher or student backbone", "retrieval gallery frames"],
        context_outputs=[
            f"onnx exported={e.get('exported', False)}",
            "gallery.npz for edge classification",
        ],
        risks=[
            "export mismatches can change runtime behavior versus training",
            "gallery coverage can be too narrow for field use",
            "deployment artifacts can hide upstream semantic errors behind good latency",
        ],
        artifacts=["edge_models/dino_demo.onnx", "edge_models/gallery.npz"],
    )

    # G: Fine-tuned search — CLIP+DINO already on GPU
    _step(15, _TOTAL_STEPS, "Fine-tuned model transformation test → finetuned_search.md")
    with _Timer(T, "G_ft_search"):
        f = step_finetuned_model_search_test(frame_list, store, is_qdrant, models,
                                             query_frame, query_t_sec, video_id, video_name,
                                             video_dir, top_k=args.top_k)
    ft_results = f["results"]; stats["ft_top_score"] = ft_results[0]["score"] if ft_results else 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="G",
        title="Fine-tuned search test",
        description="Re-run retrieval after adaptation to quantify search-space changes.",
        status="ok",
        context_inputs=["fine-tuned or distilled backbone", "same query frame as baseline"],
        context_outputs=[
            f"top-{len(ft_results)} adapted matches",
            f"top score {stats['ft_top_score']:.4f}",
        ],
        risks=[
            "score improvements can hide semantic regressions",
            "query reuse can overstate adaptation gains",
            "retrieval differences may reflect memorization rather than better context",
        ],
        artifacts=["finetuned_search.md"],
    )

    # H: Comparison + description — CLIP+DINO on GPU; populates top_descriptions context
    _step(16, _TOTAL_STEPS, "Model comparison + video description → comparison.md, description.md")
    with _Timer(T, "H_compare"):
        g = step_compare_and_describe(frame_list, store, is_qdrant, base_results, ft_results,
                                      models, video_id, video_name, video_dir,
                                      stats["ckpt_mb"], stats["onnx_mb"])
    if g:
        stats["base_infer_ms"]   = g.get("base_infer_ms", 0.0)
        stats["ft_infer_ms"]     = g.get("ft_infer_ms", 0.0)
        stats["top_description"] = g.get("top_description", "")
        video_context["top_descriptions"] = g.get("text_descriptions", [])
    _append_agentic_step(
        agentic_trace,
        step_id="H",
        title="Comparison and description",
        description="Summarize retrieval changes and derive a CLIP-based coarse natural-language description of the video.",
        status="ok",
        context_inputs=["baseline and adapted retrieval outputs", "sampled frame embeddings"],
        context_outputs=[
            f"top description: {stats.get('top_description', 'unknown')}",
            "comparison summary across model variants",
        ],
        risks=[
            "top text prompt may sound plausible but be too coarse or wrong",
            "comparison metrics can privilege ranking stability over semantics",
            "narrative labels can bias the final synthesis context",
        ],
        artifacts=["comparison.md", "description.md"],
    )

    # I: 3D map + Gaussian Splat
    _step(17, _TOTAL_STEPS, "3D map + Gaussian Splat → 3d_map/")
    with _Timer(T, "I_3dmap"):
        h = step_create_3d_map(
            video_path, video_id, video_dir, frame_list, models,
            run_sfm_flag=not args.no_sfm,
            run_gsplat_flag=not getattr(args, "no_gsplat", False),
            device=device,
        )
    stats["sfm_poses"]     = h["sfm_poses"]
    stats["map_method"]    = h["method"]
    stats["map_points"]    = int(h["points"].shape[0]) if h.get("points") is not None else 0
    stats["gsplat_method"] = h.get("gsplat_method", "skipped")
    stats["splat_ply"]     = h.get("splat_ply")
    if h.get("splat_ply"):
        _log.info("  ✓ Gaussian Splat → %s", h["splat_ply"])
        _log.info("  ✓ Interactive viewer → %s", h.get("viewer_html", ""))
    video_context["map"] = {
        "method":        h["method"],
        "points":        stats["map_points"],
        "sfm_poses":     h["sfm_poses"],
        "gsplat_method": stats["gsplat_method"],
        "splat_ply":     stats["splat_ply"],
    }
    _append_agentic_step(
        agentic_trace,
        step_id="I",
        title="3D map creation",
        description="Recover scene geometry and export sparse-map or splat artifacts for spatial interpretation.",
        status="ok",
        context_inputs=["video frames", "camera-motion consistency"],
        context_outputs=[
            f"{stats['map_points']} map points",
            f"{stats['sfm_poses']} SfM poses",
            f"map method {stats['map_method']}",
        ],
        risks=[
            "geometry failure can create confident but wrong spatial context",
            "SfM fallback outputs may look valid while lacking metric truth",
            "map artifacts can be overinterpreted as semantic evidence",
        ],
        artifacts=["3d_map/sparse_map.npz", "3d_map/map_stats.json"],
    )

    # Z: Video synthesis — offload CLIP+DINO; Ollama API call only (no local model)
    if device == "cuda" and clip_dino_on_gpu:
        _offload_models_to_cpu(models)
        clip_dino_on_gpu = False  # noqa: F841
    _step(18, _TOTAL_STEPS, "Video synthesis (ontology + narrative) → video_synthesis.md")
    _qwen_url   = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
    _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
    with _Timer(T, "Z_synthesis"):
        step_video_synthesis(
            video_name, video_dir, video_context,
            api_url=_qwen_url, model=_qwen_model,
        )
    _append_agentic_step(
        agentic_trace,
        step_id="Z",
        title="Video synthesis",
        description="Use accumulated multimodal context to generate a structured ontology and narrative summary of the whole video.",
        status="ok" if _qwen_url else "skipped",
        context_inputs=[
            "Gemma summary",
            "captions, ASR, OCR, detections, Qwen frame reasoning",
            "retrieval description and map summary",
        ],
        context_outputs=[
            "video ontology",
            "global narrative summary",
        ] if _qwen_url else ["no synthesis output"],
        risks=[
            "final narrative can collapse uncertain evidence into a single confident story",
            "contradictions across modalities may be hidden in the synthesized summary",
            "wrong high-level framing can mask the original source of context errors",
        ],
        artifacts=["video_synthesis.md", "video_ontology.json"] if _qwen_url else [],
    )

    # AA: Agentic flow artifact — prefer Gemma reasoning, fall back to Qwen, then deterministic text.
    _step(19, _TOTAL_STEPS, "Agentic flow audit → agentic_flow.md")
    _agentic_url = (
        getattr(args, "reasoning_api_url", "")
        or getattr(settings, "REASONING_API_URL", "")
        or getattr(args, "gemma_api_url", "")
        or settings.GEMMA_API_URL
        or _qwen_url
    )
    _agentic_model = (
        getattr(args, "reasoning_model", "")
        or getattr(settings, "REASONING_MODEL", "")
        or getattr(args, "gemma_api_model", "")
        or settings.GEMMA_API_MODEL
        or _qwen_model
    )
    _append_agentic_step(
        agentic_trace,
        step_id="AA",
        title="Agentic flow audit",
        description="Audit the full context chain, explain step-to-step reasoning state, and register per-step risks of misidentification and wrong context.",
        status="ok",
        context_inputs=["complete pipeline trace", "all accumulated artifacts and summaries"],
        context_outputs=["agentic_flow.md audit report"],
        risks=[
            "reasoning model can restate upstream errors coherently",
            "audit quality depends on provenance captured from earlier steps",
            "fallback deterministic summary is less nuanced than the LLM audit",
        ],
        artifacts=["agentic_flow.md"],
    )
    with _Timer(T, "AA_agentic"):
        step_agentic_flow_artifact(
            video_name,
            video_dir,
            video_context,
            api_url=_agentic_url,
            model=_agentic_model,
        )
    if device == "cuda":
        _unload_known_sidecars(
            [
                (_agentic_url, _agentic_model),
                (_qwen_url, _qwen_model),
                (getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
                 getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL),
            ]
        )

    stats["pipeline_sec"] = sum(T.values())

    _banner(f"✓ Video complete: {video_path.name}")
    _log.info("  Output dir: %s", video_dir)
    return stats


def _run_video_pipeline_safe(
    args: Any,
    video_path: "Path",
    output_dir: "Path",
    models: Dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
) -> Dict[str, Any]:
    """Wrapper around :func:`run_video_pipeline` that always returns a stats dict.

    On exception, returns the partial stats dict with timings recorded up to
    the failure point so step times and frame counts are not lost.
    """
    _out: Dict[str, Any] = {}
    try:
        run_video_pipeline(args, video_path, output_dir, models, store, is_qdrant, device, _out=_out)
    except Exception as exc:
        _log.error("Pipeline failed for %s: %s", video_path.name, exc, exc_info=True)
        _out.setdefault("name", video_path.stem)
        _out["error"] = str(exc)
        _out.setdefault("timings", {})
        _out.setdefault("frames", 0)
        _out.setdefault("duration_sec", 0.0)
        timings = _out.get("timings", {})
        _out.setdefault("pipeline_sec", sum(timings.values()))
    return _out


# ── Main entry point ──────────────────────────────────────────────────────────

def run_demo(args: Any) -> None:
    """Run the end-to-end demo pipeline.

    Called by ``main.py --mode demo``.
    Env vars must be set by the caller (via :func:`apply_demo_env`) **before**
    this module is imported.
    """
    _configure_logging()
    _configure_warnings()

    output_dir = Path(args.output_dir).resolve()

    # --view-npz shortcut: just visualise existing NPZ files
    if getattr(args, "view_npz", None) is not None:
        if not _HAS_MPL:
            _log.error("matplotlib is required for the 3D viewer.  Install: pip install matplotlib")
            sys.exit(1)
        view_npz(args.view_npz if args.view_npz is not None else "", output_dir)
        return

    t_start = time.time()
    _banner("selfsuvis — End-to-End Demo Pipeline")
    _log.info("Videos directory : %s", args.videos_dir)
    _log.info("Output directory : %s", output_dir)
    _log.info("Device           : %s", args.device)
    _log.info("Epochs           : %d", args.epochs)
    _log.info("Qdrant           : %s", "disabled" if args.no_qdrant else "auto-detect")
    _log.info("SfM              : %s", "disabled" if args.no_sfm else "auto-detect (pycolmap)")
    multimodal_active = [args.asr, args.ocr, args.depth, args.detection, args.world_model, args.qwen]
    if any(multimodal_active):
        _log.info("Multimodal steps : %s",
                  " ".join(s for s, e in [("ASR", args.asr), ("OCR", args.ocr),
                                           ("Depth", args.depth), ("Detection", args.detection),
                                           ("WorldModel", args.world_model),
                                           ("Qwen", args.qwen)] if e))

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        _log.error("Videos directory does not exist: %s", videos_dir)
        _log.error("Create it with:  mkdir -p %s", videos_dir)
        sys.exit(1)

    videos = find_videos(videos_dir)
    if not videos:
        _log.error("No video files found in %s", videos_dir)
        _log.error("Supported formats: %s", " ".join(sorted(_VIDEO_EXTS)))
        sys.exit(1)

    _log.info("Found %d video(s): %s", len(videos), [v.name for v in videos])

    device   = _resolve_device(args.device)
    _log.info("Using device: %s", device)

    from pipeline.model_registry import detect_resources  # noqa: PLC0415

    if device == "cuda":
        _unload_known_sidecars(
            [
                (getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
                 getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL),
                (getattr(args, "qwen_api_url", "") or getattr(settings, "QWEN_API_URL", ""),
                 getattr(args, "qwen_model", "") or getattr(settings, "QWEN_MODEL", "")),
                (getattr(args, "reasoning_api_url", "") or getattr(settings, "REASONING_API_URL", ""),
                 getattr(args, "reasoning_model", "") or getattr(settings, "REASONING_MODEL", "")),
            ]
        )
    resources = detect_resources()
    _log.info(
        "Detected resources: VRAM total %.1f GiB | VRAM free %.1f GiB | RAM %.1f GiB",
        resources.get("vram_gb", 0.0),
        resources.get("free_vram_gb", 0.0),
        resources.get("ram_gb", 0.0),
    )
    if device == "cuda" and resources.get("vram_gb", 0.0) <= 0.0:
        _log.warning(
            "CUDA was requested but VRAM auto-detection returned 0.0 GiB. "
            "If the NVIDIA driver is temporarily inaccessible, set GPU_TOTAL_GB_HINT "
            "and optionally GPU_FREE_GB_HINT to preserve correct model planning."
        )

    explicit_gemma_model = getattr(args, "gemma_api_model", "") or os.getenv("GEMMA_API_MODEL", "")
    explicit_reasoning_model = getattr(args, "reasoning_model", "") or os.getenv("REASONING_MODEL", "")
    auto_analysis_model, auto_reasoning_model = _recommend_gemma_sidecar_models(resources)
    if not explicit_gemma_model:
        os.environ["GEMMA_API_MODEL"] = auto_analysis_model
        settings.GEMMA_API_MODEL = auto_analysis_model  # type: ignore[misc]
    if not explicit_reasoning_model:
        os.environ["REASONING_MODEL"] = auto_reasoning_model
        settings.REASONING_MODEL = auto_reasoning_model  # type: ignore[misc]
    if not os.getenv("REASONING_API_URL") and not getattr(args, "reasoning_api_url", ""):
        fallback_reasoning_url = (
            getattr(args, "gemma_api_url", "")
            or settings.GEMMA_API_URL
            or getattr(args, "qwen_api_url", "")
            or settings.QWEN_API_URL
        )
        if fallback_reasoning_url:
            os.environ["REASONING_API_URL"] = fallback_reasoning_url
            settings.REASONING_API_URL = fallback_reasoning_url  # type: ignore[misc]

    _log.info(
        "Demo LLM plan: analysis model=%s | reasoning model=%s",
        settings.GEMMA_API_MODEL or auto_analysis_model,
        settings.REASONING_MODEL or auto_reasoning_model,
    )

    # Pre-flight: if a Gemma API URL is configured, verify it responds before
    # loading any models.  Fail loudly rather than silently skipping later.
    _gemma_url = settings.GEMMA_API_URL or getattr(args, "gemma_api_url", "")
    if _gemma_url:
        _gemma_model_cfg = getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL
        # Auto-resolve: swap for a model that's actually available in Ollama
        _gemma_model = _resolve_ollama_gemma_model(_gemma_url, _gemma_model_cfg)
        if _gemma_model != _gemma_model_cfg:
            # Persist resolution so all downstream steps see the correct model
            os.environ["GEMMA_API_MODEL"] = _gemma_model
            settings.GEMMA_API_MODEL = _gemma_model  # type: ignore[misc]
        _log.info("Gemma API pre-flight check (url=%s  model=%s) …", _gemma_url, _gemma_model)
        try:
            import httpx as _httpx
            _r = _httpx.post(
                f"{_gemma_url.rstrip('/')}/chat/completions",
                json={"model": _gemma_model, "messages": [{"role": "user", "content": "ping"}],
                      "max_tokens": 1},
                timeout=20.0,
            )
            if _r.status_code == 404:
                _log.error(
                    "Gemma model '%s' not found in Ollama (HTTP 404). "
                    "Pull it with: ollama pull %s\n"
                    "Available models: %s",
                    _gemma_model, _gemma_model,
                    _list_ollama_models(_gemma_url),
                )
                sys.exit(1)
            if _r.status_code >= 500:
                _log.error(
                    "Gemma API pre-flight failed (HTTP %d). "
                    "Ensure Ollama is running: ollama pull %s",
                    _r.status_code, _gemma_model,
                )
                sys.exit(1)
            _log.info("  ✓ Gemma API reachable (HTTP %d  model=%s)", _r.status_code, _gemma_model)
        except Exception as _exc:
            _log.error(
                "Gemma API pre-flight error: %s. "
                "Check that Ollama is running at %s",
                _exc, _gemma_url,
            )
            sys.exit(1)

    _reasoning_url = (
        getattr(args, "reasoning_api_url", "")
        or getattr(settings, "REASONING_API_URL", "")
    )
    if _reasoning_url:
        _reasoning_model_cfg = getattr(args, "reasoning_model", "") or getattr(settings, "REASONING_MODEL", "")
        if (
            (getattr(args, "reasoning_backend", "") or getattr(settings, "REASONING_BACKEND", "")).lower() == "ollama"
            or ":11434" in _reasoning_url
        ):
            _reasoning_model = _resolve_ollama_reasoning_model(_reasoning_url, _reasoning_model_cfg)
        else:
            _reasoning_model = _reasoning_model_cfg
        if _reasoning_model != _reasoning_model_cfg:
            os.environ["REASONING_MODEL"] = _reasoning_model
            settings.REASONING_MODEL = _reasoning_model  # type: ignore[misc]
        _log.info(
            "Reasoning API pre-flight check (url=%s  model=%s) …",
            _reasoning_url, _reasoning_model,
        )
        try:
            import httpx as _httpx
            _r = _httpx.post(
                f"{_reasoning_url.rstrip('/')}/chat/completions",
                json={"model": _reasoning_model, "messages": [{"role": "user", "content": "ping"}],
                      "max_tokens": 1},
                timeout=20.0,
            )
            if _r.status_code == 404:
                _log.error(
                    "Reasoning model '%s' not found at %s (HTTP 404). "
                    "Pull or serve it before running the demo.",
                    _reasoning_model, _reasoning_url,
                )
                sys.exit(1)
            if _r.status_code >= 500:
                _log.error(
                    "Reasoning API pre-flight failed (HTTP %d) for model '%s'.",
                    _r.status_code, _reasoning_model,
                )
                sys.exit(1)
            _log.info("  ✓ Reasoning API reachable (HTTP %d  model=%s)", _r.status_code, _reasoning_model)
        except Exception as _exc:
            _log.error(
                "Reasoning API pre-flight error: %s. Check endpoint %s",
                _exc, _reasoning_url,
            )
            sys.exit(1)

    t_init   = time.time()
    models   = init_models(device)
    store, is_qdrant = init_store(models, use_qdrant=not args.no_qdrant)
    init_elapsed = time.time() - t_init

    per_video_stats: List[Dict[str, Any]] = []
    try:
        for i, video_path in enumerate(videos, 1):
            _banner(f"Video {i}/{len(videos)}: {video_path.name}")
            try:
                vstats = _run_video_pipeline_safe(args, video_path, output_dir,
                                                  models, store, is_qdrant, device)
            except KeyboardInterrupt:
                raise
            per_video_stats.append(vstats)

    except KeyboardInterrupt:
        _log.warning("")
        _log.warning("Interrupted by user (Ctrl-C) — shutting down gracefully …")
        _log.warning("  %d/%d video(s) completed.", len(per_video_stats), len(videos))
        if per_video_stats:
            total_elapsed = time.time() - t_start
            stats_path    = output_dir / "final_stats.md"
            write_final_stats_md(stats_path, per_video_stats, total_elapsed)
            print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)
            _log.warning("  Partial results written to: %s", stats_path)
        _log.warning("  Re-run to process remaining videos.")
        sys.exit(130)

    if not args.no_view:
        view_npz("", output_dir)

    total_elapsed = time.time() - t_start
    stats_path    = output_dir / "final_stats.md"
    write_final_stats_md(stats_path, per_video_stats, total_elapsed)
    print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)

    _log.info("  Final statistics: %s", stats_path)
    _log.info("")
    _log.info("  Next steps:")
    _log.info("    • Edge inference:  EdgeClassifier('edge_models/dino_demo.onnx', 'edge_models/gallery.npz')")
    _log.info("    • Full stack:      make up")
    _log.info("    • Fine-tune rerun: DINO_CHECKPOINT=<path> python main.py --mode demo")
    _log.info("")
    _banner("Done — thank you for using selfsuvis!")
