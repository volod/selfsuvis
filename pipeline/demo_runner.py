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


def write_scene_captions_md(
    output_path: Path,
    video_name: str,
    caption_results: List[Dict[str, Any]],
    elapsed_sec: float,
) -> None:
    lines = [
        f"# Scene Captions — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: Florence-2-large (MORE_DETAILED_CAPTION)",
        f"Frames captioned: {len(caption_results)}",
        f"Elapsed: {elapsed_sec:.1f}s",
        f"",
        f"## Per-Frame Captions",
        f"",
        f"| Frame | t (s) | Confidence | Caption |",
        f"|-------|-------|------------|---------|",
    ]
    for r in caption_results:
        fp   = r.get("frame_path", "")
        name = Path(fp).name if fp else "—"
        t    = r.get("t_sec", 0.0)
        conf = r.get("caption_confidence", 0.0) or 0.0
        cap  = (r.get("caption") or "").replace("|", "\\|")
        lines.append(f"| `{name}` | {t:.1f} | {conf:.3f} | {cap} |")
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · Florence-2-large · phase1 captioning*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_finetune_stats_md(
    output_path: Path,
    video_name: str,
    cfg: FinetuneConfig,
    best_loss: float,
    checkpoint_path: str,
    elapsed_sec: float,
    loss_history: List[float],
) -> None:
    ckpt_mb    = os.path.getsize(checkpoint_path) / 1e6 if os.path.exists(checkpoint_path) else 0
    best_epoch = int(np.argmin(loss_history)) + 1 if loss_history else 0
    lines = [
        f"# SSL Fine-Tuning Statistics — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
        f"| Frozen blocks | {cfg.freeze_blocks} |",
        f"| Device | `{cfg.device}` |",
        f"",
        f"## Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best loss | {best_loss:.4f} |",
        f"| Best epoch | {best_epoch}/{cfg.epochs} |",
        f"| Training time | {elapsed_sec:.1f}s |",
        f"| Checkpoint size | {ckpt_mb:.1f} MB |",
        f"| Checkpoint path | `{checkpoint_path}` |",
        f"",
        f"## Loss Curve",
        f"",
        f"| Epoch | Loss |",
        f"|-------|------|",
    ]
    for ep, loss in enumerate(loss_history, 1):
        lines.append(f"| {ep} | {loss:.4f} |")
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
    lines = [
        f"# Detailed Scene Captions — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Frames processed: {ok}/{len(results)}",
        f"Elapsed: {elapsed_sec:.1f}s",
        f"",
        f"## Per-Frame Analysis",
        f"",
        f"| Frame | t (s) | Caption / Scene Facts | Audio Context |",
        f"|-------|-------|----------------------|---------------|",
    ]
    for r in results:
        fp       = r.get("frame_path", "")
        name     = Path(fp).name if fp else "—"
        t        = r.get("t_sec", 0.0)
        subtitle = (r.get("subtitle_text") or "").replace("|", "\\|")[:60]
        if r.get("service_unavailable"):
            caption = "*sidecar unavailable*"
        elif r.get("skipped"):
            caption = "*skipped*"
        else:
            facts = r.get("caption") or r.get("scene_description") or ""
            if not facts and isinstance(r, dict):
                parts = []
                for k, v in r.items():
                    if k not in ("frame_path", "t_sec", "subtitle_text", "ocr_text") and v:
                        parts.append(f"{k}: {v}")
                facts = "; ".join(parts[:4])
            caption = str(facts).replace("|", "\\|")[:200]
        lines.append(f"| `{name}` | {t:.1f} | {caption} | {subtitle} |")
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
    models: Dict[str, Any] = {"device": device}

    _log.info("Loading OpenCLIP ViT-B-16 …")
    t0 = time.time()
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


# ── Memory helpers for GPU-constrained machines ───────────────────────────────

def _offload_models_to_cpu(models: Dict[str, Any]) -> None:
    """Move CLIP and DINO backbones to CPU and flush the CUDA allocator cache.

    Called before loading a large model (Florence-2, ASR) when VRAM is tight.
    The embedders keep their ``self.device`` attribute unchanged so they work
    correctly once the backbone is moved back by :func:`_restore_models_to_gpu`.
    """
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


def _restore_models_to_gpu(models: Dict[str, Any], device: str) -> None:
    """Move CLIP and DINO backbones back to *device* after a large model releases."""
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


def _caption_via_florence_api(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    api_url: str,
    model: str,
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
                        {"type": "text", "text": "<MORE_DETAILED_CAPTION>"},
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
                             "Describe this image in one or two sentences. "
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
        except Exception as exc:
            _log.debug("  Qwen caption error for %s: %s", Path(fp).name, exc)

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
        # Offload CLIP+DINO while API captions run (they aren't needed until step C)
        if models and device == "cuda":
            _offload_models_to_cpu(models)
        return _caption_via_florence_api(
            frame_list, video_name, video_dir,
            effective_florence_api_url, effective_florence_model,
        )

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

    # Step 2: if Ollama is running, unload its model to free ~11-12 GiB
    if qwen_api_url and qwen_model and device == "cuda":
        _unload_ollama_model(qwen_api_url, qwen_model)

    _log.info("Loading Florence-2-large on %s …", device)
    t0 = time.time()
    try:
        florence = FlorenceModel()
    except Exception as exc:
        if qwen_api_url and qwen_model:
            _log.warning("  Florence-2 load failed (%s) — using Qwen API fallback", exc)
            return _caption_via_qwen_api(frame_list, video_name, video_dir, qwen_api_url, qwen_model)
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
) -> Dict[str, Any]:
    """Step E: distil fine-tuned teacher → student."""
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
    cfg         = DistillConfig(student_model="dinov2_vits14", epochs=distill_epochs,
                                batch_size=batch_size, device=device)
    frame_paths = [fp for fp, _ in frame_list]
    _log.info("Starting distillation: teacher=ViT-B/14 → student=ViT-S/14  "
              "epochs=%d  frames=%d", cfg.epochs, len(frame_paths))
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
    return result


def step_qwen_captioning(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: Dict[float, str],
    ocr_results: List[Dict[str, Any]],
    clip_prescreen_fn=None,
) -> Dict[str, Any]:
    """Step R: Qwen VLM detailed scene captioning with ASR + OCR context."""
    out_md = video_dir / "detailed_captions.md"
    result: Dict[str, Any] = {"skipped": True, "results": []}
    try:
        from pipeline.qwen_model import QwenModel
    except ImportError as exc:
        _log.warning("  Qwen model unavailable (%s) — skipping", exc)
        return result
    qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn)
    if not qwen.is_enabled():
        _log.info("  Qwen disabled (QWEN_API_URL not set) — skipping detailed captioning")
        _log.info("  To enable: --qwen-api-url http://localhost:8010/v1  (or set QWEN_API_URL)")
        return result
    ocr_map: Dict[float, str] = {r["t_sec"]: r["ocr_text"]
                                  for r in ocr_results
                                  if r.get("t_sec") is not None and r.get("ocr_text")}
    _log.info("Running Qwen detailed captioning on %d frames (model=%s) …",
              len(frame_list), settings.QWEN_MODEL)
    t0 = time.time()
    batch_results = _run_batched_frame_inference(
        frame_list,
        batch_size=4,
        batch_fn=lambda batch, imgs: qwen.extract_batch(
            imgs,
            subtitle_texts=[subtitle_map.get(t_sec) or None for _fp, t_sec in batch],
            ocr_texts=[ocr_map.get(t_sec) or None for _fp, t_sec in batch],
        ),
        warning_label="Qwen",
        error_result={"service_unavailable": True},
    )
    caption_results: List[Dict[str, Any]] = []
    for r in batch_results:
        t_sec = r.get("t_sec", 0.0)
        caption_results.append({**r, "subtitle_text": subtitle_map.get(t_sec) or ""})
    elapsed = time.time() - t0
    ok             = sum(1 for r in caption_results
                         if not r.get("service_unavailable") and not r.get("skipped"))
    subtitle_used  = sum(1 for r in caption_results if r.get("subtitle_text"))
    _log.info("  ✓ Qwen: %d/%d frames captioned in %.1fs (%d with ASR context)",
              ok, len(frame_list), elapsed, subtitle_used)
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
    return result


# ── Per-video orchestrator ────────────────────────────────────────────────────

_TOTAL_STEPS = 17
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
    if not frame_list:
        _log.error("No frames extracted — skipping video %s", video_path.name)
        return stats

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

    # L: Scene captioning — offloads CLIP+DINO internally, does NOT restore them
    caption_results: List[Dict[str, Any]] = []
    if not args.no_caption:
        _step(3, _TOTAL_STEPS, "Florence-2 scene captioning → scene_captions.md")
        with _Timer(T, "L_caption"):
            l_cap = step_scene_captioning(
                frame_list, video_name, video_dir, device,
                models=models,
                qwen_api_url=getattr(args, "qwen_api_url", ""),
                qwen_model=getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
                florence_api_url=getattr(args, "florence_api_url", ""),
                florence_model=getattr(args, "florence_model", ""),
            )
        caption_results = l_cap.get("captions", [])
        if device == "cuda":
            clip_dino_on_gpu = False  # Florence offloaded them; we keep them off for M–Q
    else:
        T["L_caption"] = 0.0
        _step(3, _TOTAL_STEPS, "Scene captioning (skipped — --no-caption)")
    video_context["captions"] = caption_results

    # M: ASR — no CLIP/DINO needed; Whisper manages its own VRAM
    asr_result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    if args.asr:
        _step(4, _TOTAL_STEPS, "ASR transcription → asr_subtitles.md")
        with _Timer(T, "M_asr"):
            asr_result = step_asr_transcription(video_path, frame_list, video_name, video_dir)
    else:
        T["M_asr"] = 0.0
    video_context["asr_segments"] = asr_result.get("segments", [])

    # N: OCR
    ocr_result: Dict[str, Any] = {"skipped": True, "ocr_results": []}
    if args.ocr:
        _step(5, _TOTAL_STEPS, "OCR text extraction")
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

    # O: Depth
    depth_result: Dict[str, Any] = {"skipped": True, "depth_results": []}
    if args.depth:
        _step(6, _TOTAL_STEPS, "Depth estimation")
        with _Timer(T, "O_depth"):
            depth_result = step_depth_estimation(frame_list, video_name, video_dir)
    else:
        T["O_depth"] = 0.0

    # P: Detection — accumulate per-label object counts into context
    det_result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    if args.detection:
        _step(7, _TOTAL_STEPS, "Object detection")
        with _Timer(T, "P_detection"):
            det_result = step_object_detection(frame_list, video_name, video_dir)
    else:
        T["P_detection"] = 0.0
    if not det_result.get("skipped"):
        obj_counts: Dict[str, int] = {}
        for _r in det_result.get("detection_results", []):
            for _d in _r.get("detections", []):
                lbl = _d.get("label", "unknown")
                obj_counts[lbl] = obj_counts.get(lbl, 0) + 1
        video_context["detections"] = obj_counts

    # Q: World model
    world_result: Dict[str, Any] = {"skipped": True, "world_results": []}
    if args.world_model:
        _step(8, _TOTAL_STEPS, "World model video embeddings")
        with _Timer(T, "Q_world"):
            world_result = step_world_model_pass(frame_list, video_name, video_dir)
    else:
        T["Q_world"] = 0.0
    if not world_result.get("skipped"):
        video_context["world_model_clips"] = world_result.get("ok_count", 0)

    # R: Qwen — uses ASR + OCR context from previous steps (already agentic)
    qwen_result: Dict[str, Any] = {"skipped": True, "results": []}
    if args.qwen:
        _step(9, _TOTAL_STEPS, "Qwen VLM detailed captioning → detailed_captions.md")
        with _Timer(T, "R_qwen"):
            qwen_result = step_qwen_captioning(
                frame_list, video_name, video_dir,
                subtitle_map=asr_result.get("subtitle_map", {}),
                ocr_results=ocr_result.get("ocr_results", []),
                # Pass a passthrough so QwenModel never creates a second CLIP
                # embedder (OpenCLIPTagger) that competes for VRAM.  In demo
                # mode we want full coverage; prescreening is not needed.
                clip_prescreen_fn=lambda _img: True,
            )
    else:
        T["R_qwen"] = 0.0
    if not qwen_result.get("skipped"):
        video_context["qwen_captions"] = qwen_result.get("results", [])

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
    _step(10, _TOTAL_STEPS, "Base model transformation test → base_search.md")
    with _Timer(T, "C_base_search"):
        c = step_base_model_search_test(frame_list, store, is_qdrant, models,
                                        video_id, video_name, video_dir, top_k=args.top_k)
    base_results = c["results"]; query_frame = c["query_frame"]; query_t_sec = c["query_t_sec"]
    stats["base_top_score"] = base_results[0]["score"] if base_results else 0.0

    # D: SSL fine-tuning — DINOFineTuner loads its own separate DINO; offload ours first
    if device == "cuda" and clip_dino_on_gpu:
        _offload_models_to_cpu(models)
        clip_dino_on_gpu = False
    _step(11, _TOTAL_STEPS, "SSL DINOv3 fine-tuning → finetune_stats.md")
    with _Timer(T, "D_finetune"):
        d = step_ssl_finetune(video_id, video_name, video_dir, frame_list, device,
                              epochs=args.epochs, batch_size=args.batch_size)
    stats["best_loss"] = d["best_loss"]; stats["ckpt_mb"] = d["ckpt_mb"]
    checkpoint_path    = d["checkpoint"]

    # E: Distillation — also creates its own teacher + student; CLIP+DINO stay offloaded
    student_backbone = None; student_dim = 768
    if not args.no_distill:
        _step(12, _TOTAL_STEPS, "Knowledge distillation: ViT-B/14 teacher → ViT-S/14 student")
        with _Timer(T, "E_distill"):
            e_distill = step_distill(checkpoint_path, frame_list, video_name, video_dir, device,
                                     distill_epochs=args.distill_epochs, batch_size=args.batch_size)
        if not e_distill["skipped"]:
            student_backbone         = e_distill["student_backbone"]
            student_dim              = e_distill["student_dim"]
            stats["distill_loss"]    = e_distill["best_loss"]
            stats["student_ckpt_mb"] = e_distill["ckpt_mb"]
            stats["student_dim"]     = student_dim
            stats["teacher_dim"]     = e_distill["teacher_dim"]
    else:
        T["E_distill"] = 0.0
        _step(12, _TOTAL_STEPS, "Knowledge distillation (skipped — --no-distill)")

    # F: ONNX export + gallery — restore CLIP+DINO (export uses models["dino"])
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(13, _TOTAL_STEPS, "ONNX export + gallery build → edge_models/")
    with _Timer(T, "F_export"):
        e = step_export_model(checkpoint_path, frame_list, video_dir, device, models,
                              no_onnx=args.no_onnx,
                              student_backbone=student_backbone, student_dim=student_dim)
    stats["onnx_mb"] = e.get("onnx_mb", 0.0); stats["onnx_exported"] = e.get("exported", False)

    # G: Fine-tuned search — CLIP+DINO already on GPU
    _step(14, _TOTAL_STEPS, "Fine-tuned model transformation test → finetuned_search.md")
    with _Timer(T, "G_ft_search"):
        f = step_finetuned_model_search_test(frame_list, store, is_qdrant, models,
                                             query_frame, query_t_sec, video_id, video_name,
                                             video_dir, top_k=args.top_k)
    ft_results = f["results"]; stats["ft_top_score"] = ft_results[0]["score"] if ft_results else 0.0

    # H: Comparison + description — CLIP+DINO on GPU; populates top_descriptions context
    _step(15, _TOTAL_STEPS, "Model comparison + video description → comparison.md, description.md")
    with _Timer(T, "H_compare"):
        g = step_compare_and_describe(frame_list, store, is_qdrant, base_results, ft_results,
                                      models, video_id, video_name, video_dir,
                                      stats["ckpt_mb"], stats["onnx_mb"])
    if g:
        stats["base_infer_ms"]   = g.get("base_infer_ms", 0.0)
        stats["ft_infer_ms"]     = g.get("ft_infer_ms", 0.0)
        stats["top_description"] = g.get("top_description", "")
        video_context["top_descriptions"] = g.get("text_descriptions", [])

    # I: 3D map + Gaussian Splat
    _step(16, _TOTAL_STEPS, "3D map + Gaussian Splat → 3d_map/")
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

    # Z: Video synthesis — offload CLIP+DINO; Ollama API call only (no local model)
    if device == "cuda" and clip_dino_on_gpu:
        _offload_models_to_cpu(models)
        clip_dino_on_gpu = False  # noqa: F841
    _step(17, _TOTAL_STEPS, "Video synthesis (ontology + narrative) → video_synthesis.md")
    _qwen_url   = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
    _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
    with _Timer(T, "Z_synthesis"):
        step_video_synthesis(
            video_name, video_dir, video_context,
            api_url=_qwen_url, model=_qwen_model,
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
