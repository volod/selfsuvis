#!/usr/bin/env python3
"""selfsuvis end-to-end demo pipeline.

Runs the full perception stack on every video file in a source directory:

  Core steps (always run):
    A. Frame extraction + metadata      → {video_dir}/frames_metadata.json
    B. Index frames (CLIP + DINOv3)     → Qdrant or in-memory store
    C. Base-model search test           → {video_dir}/base_search.md
    D. SSL DINOv3 fine-tuning           → {video_dir}/finetune_stats.md
    E. Knowledge distillation           → {video_dir}/distill_stats.md
    F. ONNX export + gallery build      → {video_dir}/edge_models/
    G. Fine-tuned model search test     → {video_dir}/finetuned_search.md
    H. Comparison + video description   → {video_dir}/comparison.md, description.md
    I. 3D sparse map (SfM or PCA)       → {video_dir}/3d_map/
    J. Interactive 3D viewers (one window per video)
    K. Final statistics                 → output/final_stats.md

  Optional multimodal steps (off by default; enable with flags below):
    L. Florence-2 scene captioning      → {video_dir}/scene_captions.md
    M. ASR — Whisper speech-to-text     → {video_dir}/asr_subtitles.md
    N. OCR — text extraction per frame  → merged into multimodal_features.md
    O. Depth estimation per frame       → merged into multimodal_features.md
    P. Object detection per frame       → merged into multimodal_features.md
    Q. World model video embeddings     → merged into multimodal_features.md
    R. Qwen VLM detailed captioning     → {video_dir}/detailed_captions.md
       (uses ASR subtitles + OCR text as context; requires QWEN_API_URL / --qwen-api-url)

  Edge model outputs (step F):
    edge_models/dino_demo.onnx          Student or teacher backbone (ONNX)
    edge_models/gallery.npz             Embedding gallery for 1-NN classification

Usage:
    python demo.py                              # default: data_test/videos/
    python demo.py --videos-dir /path/to/videos
    python demo.py --device cuda --epochs 5
    python demo.py --no-qdrant --no-sfm        # offline / CPU-only demo

  Multimodal flags (each loads its model lazily on first frame):
    python demo.py --asr                        # Whisper ASR from audio track
    python demo.py --ocr                        # OCR text extraction
    python demo.py --depth                      # Depth estimation
    python demo.py --detection                  # Object detection
    python demo.py --world-model                # World model video embeddings
    python demo.py --asr --ocr --depth --detection  # all optional steps
    python demo.py --qwen --qwen-api-url http://localhost:8010/v1  # Qwen detailed captioning
    python demo.py --asr --qwen --qwen-api-url http://localhost:8010/v1  # Qwen + ASR context

  Model selection (auto = GPU-aware, see pipeline/model_registry.py):
    python demo.py --asr --asr-model openai/whisper-large-v3
    python demo.py --ocr --ocr-model ucaslcl/GOT-OCR2_0
    python demo.py --depth --depth-model depth-anything/Depth-Anything-V2-Large-hf
    python demo.py --detection --detection-model IDEA-Research/grounding-dino-base
    python demo.py --world-model --world-model-id facebook/vjepa2-vitg-fpc64-256
"""

# ── Early arg parse — must happen BEFORE pipeline imports so env vars are set ─
import argparse
import os
import sys
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="selfsuvis end-to-end demo pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--videos-dir", default="data_test/videos",
                   help="Directory containing input .mp4/.mov/.mkv files")
    p.add_argument("--output-dir", default="data_test/output",
                   help="Root directory for all demo output")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Torch device (auto selects CUDA when available)")
    p.add_argument("--epochs", type=int, default=3,
                   help="SSL fine-tuning epochs per video")
    p.add_argument("--batch-size", type=int, default=4,
                   help="SSL fine-tuning batch size")
    p.add_argument("--top-k", type=int, default=5,
                   help="Nearest neighbours to show in search tests")
    p.add_argument("--no-qdrant", action="store_true",
                   help="Skip Qdrant; use in-memory cosine search")
    p.add_argument("--no-sfm", action="store_true",
                   help="Skip pycolmap SfM; use PCA point-cloud fallback")
    p.add_argument("--no-caption", action="store_true",
                   help="Skip Florence-2 scene captioning (step L)")
    p.add_argument("--distill-epochs", type=int, default=5,
                   help="Knowledge distillation epochs (student ViT-S/14, default 5)")
    p.add_argument("--no-distill", action="store_true",
                   help="Skip knowledge distillation; export teacher to ONNX instead")
    p.add_argument("--no-onnx", action="store_true",
                   help="Skip ONNX export (requires torch.onnx)")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Frame-extraction rate (fps)")
    p.add_argument("--view-npz", metavar="PATH", nargs="?", const="",
                   help=(
                       "Visualize existing sparse_map.npz without running the pipeline. "
                       "Pass a .npz file path, a video output directory, or omit to scan "
                       "--output-dir for all sparse_map.npz files."
                   ))
    p.add_argument("--no-view", action="store_true",
                   help="Skip the interactive 3D map viewer at the end of the pipeline")
    # ── Optional multimodal step flags ────────────────────────────────────────
    p.add_argument("--asr", action="store_true",
                   help="Enable ASR step (Whisper speech-to-text from audio track)")
    p.add_argument("--asr-model", default="auto",
                   help="Whisper model ID or 'auto' (GPU-aware selection)")
    p.add_argument("--asr-language", default="",
                   help="Force ASR language code (e.g. 'en', 'uk'). Empty = auto-detect")
    p.add_argument("--ocr", action="store_true",
                   help="Enable OCR text extraction per frame")
    p.add_argument("--ocr-model", default="auto",
                   help="OCR model ID or 'auto'")
    p.add_argument("--depth", action="store_true",
                   help="Enable depth estimation per frame")
    p.add_argument("--depth-model", default="auto",
                   help="Depth model ID or 'auto'")
    p.add_argument("--detection", action="store_true",
                   help="Enable object detection per frame")
    p.add_argument("--detection-model", default="auto",
                   help="Detection model ID or 'auto'")
    p.add_argument("--detection-labels", default="",
                   help="Comma-separated labels for open-vocabulary detection")
    p.add_argument("--world-model", action="store_true",
                   help="Enable world model video embeddings")
    p.add_argument("--world-model-id", default="auto",
                   help="World model ID or 'auto'")
    p.add_argument("--qwen", action="store_true",
                   help="Enable Qwen VLM detailed scene captioning (step R); requires --qwen-api-url or QWEN_API_URL env var")
    p.add_argument("--qwen-api-url", default="",
                   help="Qwen vLLM/ollama sidecar URL (e.g. http://localhost:8010/v1). Sets QWEN_API_URL env var.")
    p.add_argument("--qwen-model", default="",
                   help="Qwen model ID to use; empty = use QWEN_MODEL env var default (Qwen/Qwen2.5-VL-7B-Instruct)")
    p.add_argument("--qwen-backend", default="",
                   choices=["", "vllm", "ollama"],
                   help="Qwen sidecar backend type. Empty = auto-detect (ollama inferred from port 11434).")
    return p


args = _build_arg_parser().parse_args()
_OUTPUT_DIR = Path(args.output_dir).resolve()
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set env vars before any pipeline.config import
os.environ.setdefault("DATA_DIR", str(_OUTPUT_DIR))
os.environ.setdefault("MODEL_NAME", "dinov3")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("QDRANT_COLLECTION", "demo_video_semantic")
os.environ.setdefault("DEVICE", args.device)
os.environ.setdefault("USE_FP16", "false")
os.environ.setdefault("SAMPLE_FPS_MAX", str(args.fps))
os.environ.setdefault("SFM_FPS", "1")
os.environ.setdefault("ALLOWED_INDEX_PATHS", "")
os.environ.setdefault("API_KEY", "")
# ── Multimodal model env vars (set before pipeline.config import) ─────────
os.environ.setdefault("ASR_ENABLED",  "true" if args.asr           else "false")
os.environ.setdefault("ASR_MODEL",    args.asr_model)
os.environ.setdefault("ASR_LANGUAGE", args.asr_language)
os.environ.setdefault("OCR_ENABLED",  "true" if args.ocr           else "false")
os.environ.setdefault("OCR_MODEL",    args.ocr_model)
os.environ.setdefault("DEPTH_ENABLED","true" if args.depth         else "false")
os.environ.setdefault("DEPTH_MODEL",  args.depth_model)
os.environ.setdefault("DETECTION_ENABLED","true" if args.detection else "false")
os.environ.setdefault("DETECTION_MODEL",  args.detection_model)
os.environ.setdefault("DETECTION_LABELS", args.detection_labels)
os.environ.setdefault("WORLD_MODEL_ENABLED","true" if args.world_model else "false")
os.environ.setdefault("WORLD_MODEL",  args.world_model_id)
if args.qwen_api_url:
    os.environ["QWEN_API_URL"] = args.qwen_api_url
os.environ.setdefault("QWEN_API_URL", "")
if args.qwen_model:
    os.environ["QWEN_MODEL"] = args.qwen_model
if args.qwen_backend:
    os.environ["QWEN_BACKEND"] = args.qwen_backend

# ── Standard imports ──────────────────────────────────────────────────────────
import json
import logging
import math
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── Pipeline imports (after env vars) ─────────────────────────────────────────
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



# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_FMT = "%(asctime)s  %(levelname)-7s  %(message)s"
_DATE_FMT = "%H:%M:%S"

logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_DATE_FMT)
log = logging.getLogger("demo")

# Reduce noise from heavy deps
for _noisy in ("urllib3", "PIL", "filelock", "torch", "timm"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _banner(msg: str) -> None:
    width = 72
    log.info("=" * width)
    log.info("  %s", msg)
    log.info("=" * width)


def _step(n: int, total: int, name: str) -> None:
    log.info("─── Step %d/%d: %s", n, total, name)


class _Timer:
    """Context manager that stores elapsed seconds into a dict under a key.

    Usage::
        timings: Dict[str, float] = {}
        with _Timer(timings, "extract"):
            result = expensive_function()
        # timings["extract"] == elapsed seconds
    """
    def __init__(self, store: Dict[str, float], key: str) -> None:
        self._store = store
        self._key   = key
        self._t0    = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.time()
        return self

    def __exit__(self, *_: Any) -> None:
        self._store[self._key] = time.time() - self._t0


# ── Model & store initialisation ──────────────────────────────────────────────

def _resolve_device() -> str:
    import torch
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def init_models(device: str) -> Dict[str, Any]:
    """Load CLIP and (optionally) DINO models. Returns a dict of model instances."""
    _banner("Initialising models")
    models: Dict[str, Any] = {"device": device}

    log.info("Loading OpenCLIP ViT-B-16 …")
    t0 = time.time()
    models["clip"] = OpenCLIPEmbedder()
    log.info("  ✓ CLIP ready in %.1fs  (dim=%d)", time.time() - t0, models["clip"].image_dim())

    if _HAS_DINO:
        log.info("Loading DINOv3 ViT-B/14 …  (first run downloads ~330 MB)")
        t0 = time.time()
        try:
            models["dino"] = DINOEmbedder("dinov3_vitb14")
            log.info("  ✓ DINO ready in %.1fs  (dim=%d)",
                     time.time() - t0, models["dino"].image_dim())
        except Exception as exc:
            log.warning("  ✗ DINOv3 load failed (%s) — using CLIP only", exc)
            models["dino"] = None
    else:
        log.warning("  ✗ models.dino_model unavailable — using CLIP only")
        models["dino"] = None

    return models


def init_store(models: Dict[str, Any], use_qdrant: bool) -> Tuple[Any, bool]:
    """Try to connect to Qdrant; fall back to InMemoryStore.

    Returns (store, is_qdrant).
    """
    if not use_qdrant:
        log.info("Qdrant disabled (--no-qdrant) — using in-memory cosine store")
        return InMemoryStore(), False

    try:
        from pipeline.qdrant_utils import QdrantStore
        clip_dim = models["clip"].image_dim()
        dino_dim = models["dino"].image_dim() if models.get("dino") else None
        store = QdrantStore(clip_dim=clip_dim, dino_dim=dino_dim)
        # Ping by listing collections
        store.client.get_collections()
        log.info("✓ Qdrant connected at %s:%s  collection=%s",
                 settings.QDRANT_HOST, settings.QDRANT_PORT, settings.QDRANT_COLLECTION)
        return store, True
    except Exception as exc:
        log.warning("Qdrant unavailable (%s) — falling back to in-memory store", exc)
        log.warning("  To enable: docker run -p 6333:6333 qdrant/qdrant")
        return InMemoryStore(), False


# ── Text descriptions for video-to-text captioning ────────────────────────────

_TEXT_PROMPTS = [
    # ── General scene types ────────────────────────────────────────────────────
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
    # ── Radar & sensor infrastructure ─────────────────────────────────────────
    "radar antenna or rotating radar dish on a rooftop or tower",
    "military radar installation in open terrain",
    "phased array radar or sensor array on a vehicle or structure",
    "surveillance radar dome or radome on a building",
    "weather radar tower in a field",
    "radar site with large parabolic antenna",
    "electronic warfare sensor mast on a ship or vehicle",
    # ── Vehicles — panoramic / top-down view ──────────────────────────────────
    "panoramic wide-angle view of vehicles on a road",
    "multiple cars and trucks visible in a wide scene",
    "convoy of military vehicles on a road viewed from above",
    "vehicles moving along a highway in a panoramic shot",
    "armoured vehicles or tanks in an open field",
    "trucks and heavy transport vehicles at an industrial site",
    "emergency vehicles with lights visible from aerial view",
    "vehicles parked in an open area viewed from a drone",
    # ── Radar + vehicle combined ───────────────────────────────────────────────
    "mobile radar unit mounted on a truck in a field",
    "radar vehicle or electronic warfare truck in a convoy",
    "surveillance vehicle with antenna array on a road",
    # ── Serpentine / slalom patterns with small road objects ──────────────────
    "small vehicles weaving in a serpentine pattern along a road",
    "tiny cars following a zigzag slalom course on a wide road",
    "overhead view of vehicles navigating obstacles in a serpentine layout",
    "small objects moving in curved paths on a straight road from above",
    "miniature vehicles visible as small dots arranged in a winding line",
    "drone view of traffic slowing and weaving around road obstacles",
    "serpentine convoy of small vehicles on an open road from altitude",
    # ── Simple and portable radars ─────────────────────────────────────────────
    "simple portable radar unit on a tripod in a field",
    "small ground surveillance radar deployed on the roadside",
    "handheld or man-portable radar device in open terrain",
    "compact radar sensor on a pole or mast near a road",
    "short-range radar unit with small dish antenna on the ground",
    "mobile radar system on a lightweight trailer or cart",
    "radar detector or traffic speed radar on a road",
]


# ── MD file writers ───────────────────────────────────────────────────────────

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
    """Write a nearest-neighbour search result as Markdown."""
    if output_path.exists():
        log.info("  Skipping %s (already exists)", output_path.name)
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
        t = payload.get("t_sec", 0.0)
        score = r.get("score", 0.0)
        rel = os.path.relpath(fp, output_path.parent) if fp else ""
        lines.append(f"| {i} | {score:.4f} | {t:.2f}s | {_md_image(rel, f'match {i}')} |")

    lines += [
        f"",
        f"---",
        f"*Artifact produced by `demo.py`. Re-run the demo to regenerate.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


def write_scene_captions_md(
    output_path: Path,
    video_name: str,
    caption_results: List[Dict[str, Any]],
    elapsed_sec: float,
) -> None:
    """Write per-frame Florence-2 captions as Markdown."""
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
        fp = r.get("frame_path", "")
        name = Path(fp).name if fp else "—"
        t = r.get("t_sec", 0.0)
        conf = r.get("caption_confidence", 0.0) or 0.0
        cap = (r.get("caption") or "").replace("|", "\\|")
        lines.append(f"| `{name}` | {t:.1f} | {conf:.3f} | {cap} |")
    lines += [
        f"",
        f"---",
        f"*Produced by `demo.py` · Florence-2-large · phase1 captioning*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


def write_finetune_stats_md(
    output_path: Path,
    video_name: str,
    cfg: FinetuneConfig,
    best_loss: float,
    checkpoint_path: str,
    elapsed_sec: float,
    loss_history: List[float],
) -> None:
    """Write SSL fine-tuning statistics as Markdown."""
    ckpt_mb = os.path.getsize(checkpoint_path) / 1e6 if os.path.exists(checkpoint_path) else 0
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
        f"# Load fine-tuned model for search:",
        f"export DINO_CHECKPOINT={checkpoint_path}",
        f"# Then start the API or run inference:",
        f"python demo.py --videos-dir data_test/videos",
        f"```",
        f"",
        f"---",
        f"*Artifact produced by `demo.py`. See `edge_models/` for ONNX export.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


def write_distill_stats_md(
    output_path: Path,
    video_name: str,
    stats: Dict[str, Any],
) -> None:
    """Write knowledge distillation stats as Markdown."""
    loss_history = stats.get("loss_history", [])
    lines = [
        f"# Knowledge Distillation — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Teacher | DINOv3 ViT-B/14 (fine-tuned SSL) — dim={stats.get('teacher_dim', 768)} |",
        f"| Student | {stats.get('student_model', 'dinov2_vits14')} — dim={stats.get('student_dim', 384)} |",
        f"| Loss | Cosine feature distillation: 1 − cos(proj(student), teacher) |",
        f"| Epochs | {len(loss_history)} |",
        f"| Elapsed | {stats.get('elapsed', 0):.1f}s |",
        f"",
        f"## Results",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best distillation loss | {stats.get('best_loss', float('nan')):.4f} |",
        f"| Student params | ~22M (vs teacher ~86M — {86/22:.1f}× compression) |",
        f"| Student dim | {stats.get('student_dim', 384)} (vs teacher {stats.get('teacher_dim', 768)}) |",
        f"| Best checkpoint | `{Path(stats.get('best_path', '')).name}` |",
        f"",
        f"## Loss Curve",
        f"",
        f"| Epoch | Loss |",
        f"|-------|------|",
    ]
    for i, loss in enumerate(loss_history, 1):
        lines.append(f"| {i} | {loss:.4f} |")
    lines += [
        f"",
        f"## Architecture",
        f"",
        f"```",
        f"Teacher (frozen):  DINOv3 ViT-B/14  →  768-dim embedding",
        f"                         ↓ cosine distillation loss",
        f"Proj head (temp):  Linear(384 → 768)  [discarded after training]",
        f"                         ↑",
        f"Student (trained): DINOv2 ViT-S/14  →  384-dim embedding",
        f"```",
        f"",
        f"The student is 4× smaller and ~2× faster at inference.",
        f"The projection head is used only during training to align embedding spaces.",
        f"The saved checkpoint contains **only the student backbone weights**.",
        f"",
        f"---",
        f"*Artifact produced by `demo.py`. Student exported to `edge_models/dino_demo.onnx`.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


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
    """Write base-vs-fine-tuned comparison as Markdown."""
    # Overlap: frame paths in common
    base_paths = {r.get("payload", r).get("frame_path", "") for r in base_results}
    ft_paths   = {r.get("payload", r).get("frame_path", "") for r in ft_results}
    overlap = len(base_paths & ft_paths)

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
        f"# On-device inference example:",
        f"from pipeline.edge_inference import EdgeClassifier",
        f"clf = EdgeClassifier('edge_models/dino_demo.onnx', 'edge_models/gallery.npz')",
        f"labels = clf.classify(frame_pil)   # [(label, score), ...]",
        f"```",
        f"",
        f"---",
        f"*Artifact produced by `demo.py`.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


def write_description_md(
    output_path: Path,
    video_name: str,
    frame_list: List[Tuple[str, float]],
    text_descriptions: List[Tuple[str, float]],
    all_scored: List[Tuple[str, float]],
) -> None:
    """Write a dedicated image-to-text description report."""
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

    lines += [
        f"",
        f"## Sample Frames",
        f"",
        f"Frames used for description (evenly spaced, up to 32):",
        f"",
    ]
    step = max(1, len(frame_list) // 8)   # show up to 8 frame paths in the MD
    for fp, t_sec in frame_list[::step][:8]:
        rel = Path(fp).name
        lines.append(f"- `{rel}` (t={t_sec:.1f}s)")

    lines += [
        f"",
        f"---",
        f"*Produced by `demo.py` · model: OpenCLIP ViT-B/16 (openai)*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


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
        distill_str = f"{distill_loss:.4f}" if not math.isnan(distill_loss) else "skipped"
        lines.append(
            f"| {v['name']} | {v.get('frames',0)} | "
            f"{v.get('index_sec',0):.1f} | "
            f"{v.get('best_loss', float('nan')):.4f} | "
            f"{distill_str} | "
            f"{v.get('sfm_poses',0)} | "
            f"{v.get('ckpt_mb',0):.1f} |"
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
        f"| `3d_map/sparse_map.npz` | 3D point cloud (from SfM or PCA fallback) |",
        f"| `3d_map/map_stats.json` | Point count, SfM pose count, scene count |",
        f"",
        f"---",
        f"*Run `python demo.py --help` for all options.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("✓ Final stats written to %s", output_path)


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
]


def _fmt_sec(sec: float) -> str:
    """Format seconds as '1h 23m 04s', '12m 34s', or '45.2s'."""
    if math.isnan(sec) or sec < 0:
        return "—"
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h}h {m:02d}m {s:02d}s"
    if sec >= 60:
        m = int(sec // 60)
        s = sec % 60
        return f"{m}m {s:04.1f}s"
    return f"{sec:.1f}s"


def print_run_stats(
    per_video: List[Dict[str, Any]],
    total_elapsed: float,
    init_elapsed: float,
    device: str,
) -> None:
    """Print a structured run-statistics table to the log."""
    W = 72
    SEP = "─" * W

    def _row(label: str, *cols: str) -> str:
        col_w = max(1, (W - 28) // max(len(cols), 1))
        parts = [f"  {label:<26}"]
        for c in cols:
            parts.append(f"{c:>{col_w}}")
        return "".join(parts)

    _banner("RUN STATISTICS")

    # ── Environment ────────────────────────────────────────────────────────────
    log.info("  Device       : %s", device.upper())
    log.info("  Videos       : %d", len(per_video))
    total_frames = sum(v.get("frames", 0) for v in per_video)
    total_duration = sum(v.get("duration_sec", 0.0) for v in per_video)
    log.info("  Total frames : %d  (%.1f min of video)", total_frames, total_duration / 60)
    log.info("  Total runtime: %s", _fmt_sec(total_elapsed))
    log.info("")

    # ── Time breakdown ─────────────────────────────────────────────────────────
    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]
    header_cols = names + ["TOTAL"]

    # Header
    log.info("  TIME BREAKDOWN")
    log.info("  " + SEP[:W-2])
    log.info(_row("Step", *header_cols))
    log.info("  " + SEP[:W-2])

    step_totals: Dict[str, float] = {}
    for key, label in _STEP_LABELS:
        vals = [v.get("timings", {}).get(key, 0.0) for v in per_video]
        total_step = sum(vals)
        step_totals[key] = total_step
        log.info(_row(label, *[_fmt_sec(s) for s in vals], _fmt_sec(total_step)))

    log.info("  " + SEP[:W-2])
    pipeline_per_video = [v.get("pipeline_sec", 0.0) for v in per_video]
    log.info(_row("Pipeline (steps sum)",
                  *[_fmt_sec(s) for s in pipeline_per_video],
                  _fmt_sec(sum(pipeline_per_video))))
    overhead = total_elapsed - sum(pipeline_per_video) - init_elapsed
    log.info(_row("Model init", _fmt_sec(init_elapsed), *([""] * (len(per_video) - 1)), ""))
    log.info(_row("Overhead (I/O, viewer, etc.)",
                  *([""] * len(per_video)), _fmt_sec(max(0.0, overhead))))
    log.info(_row("WALL CLOCK TOTAL",
                  *([""] * len(per_video)), _fmt_sec(total_elapsed)))
    log.info("")

    # ── Throughput ─────────────────────────────────────────────────────────────
    log.info("  THROUGHPUT")
    log.info("  " + SEP[:W-2])
    for v in per_video:
        t_extract = v.get("timings", {}).get("A_extract", 0.0) or 1e-9
        t_index   = v.get("timings", {}).get("B_index", 0.0) or 1e-9
        frames    = v.get("frames", 0)
        fps_ext   = frames / t_extract
        fps_idx   = frames / t_index
        log.info("  %-26s  extract: %5.1f fr/s   index: %5.1f fr/s",
                 v.get("name", "?"), fps_ext, fps_idx)
    log.info("")

    # ── Model metrics ──────────────────────────────────────────────────────────
    log.info("  MODEL METRICS")
    log.info("  " + SEP[:W-2])
    log.info(_row("Metric", *names))
    log.info("  " + SEP[:W-2])
    log.info(_row("SSL finetune loss",
                  *[f"{v.get('best_loss', float('nan')):.4f}" for v in per_video]))
    log.info(_row("Distill loss",
                  *[f"{v.get('distill_loss', float('nan')):.4f}"
                    if not math.isnan(v.get("distill_loss", float("nan")))
                    else "skipped"
                    for v in per_video]))
    log.info(_row("Teacher ckpt (MB)",
                  *[f"{v.get('ckpt_mb', 0.0):.1f}" for v in per_video]))
    log.info(_row("Student ckpt (MB)",
                  *[f"{v.get('student_ckpt_mb', 0.0):.1f}"
                    if v.get("student_ckpt_mb") else "—"
                    for v in per_video]))
    log.info(_row("ONNX size (MB)",
                  *[f"{v.get('onnx_mb', 0.0):.1f}"
                    if v.get("onnx_exported") else "—"
                    for v in per_video]))
    log.info(_row("Student embed dim",
                  *[str(v.get("student_dim", "—")) for v in per_video]))
    log.info(_row("Compression ratio",
                  *[f"{v['teacher_dim']/v['student_dim']:.1f}×"
                    if v.get("student_dim") and v.get("teacher_dim") else "—"
                    for v in per_video]))
    log.info(_row("Base infer (ms/fr)",
                  *[f"{v.get('base_infer_ms', 0.0):.1f}" for v in per_video]))
    log.info(_row("Fine-tuned infer (ms/fr)",
                  *[f"{v.get('ft_infer_ms', 0.0):.1f}" for v in per_video]))
    log.info("")

    # ── Search quality ─────────────────────────────────────────────────────────
    log.info("  SEARCH QUALITY  (top-1 cosine score, same query frame)")
    log.info("  " + SEP[:W-2])
    log.info(_row("Base model (pretrained)",
                  *[f"{v.get('base_top_score', 0.0):.4f}" for v in per_video]))
    log.info(_row("Fine-tuned model",
                  *[f"{v.get('ft_top_score', 0.0):.4f}" for v in per_video]))
    log.info("")

    # ── 3D map ─────────────────────────────────────────────────────────────────
    log.info("  3D MAP")
    log.info("  " + SEP[:W-2])
    log.info(_row("Method",     *[v.get("map_method", "—") for v in per_video]))
    log.info(_row("Points",     *[str(v.get("map_points", 0)) for v in per_video]))
    log.info(_row("SfM poses",  *[str(v.get("sfm_poses", 0)) for v in per_video]))
    log.info("")

    # ── Video descriptions ─────────────────────────────────────────────────────
    log.info("  TOP VIDEO DESCRIPTION  (CLIP text similarity)")
    log.info("  " + SEP[:W-2])
    for v in per_video:
        desc = v.get("top_description", "—") or "—"
        log.info("  %-20s  %s", v.get("name", "?"), desc)
    log.info("")
    log.info("  " + "═" * (W-2))


# ── Step implementations ──────────────────────────────────────────────────────

def step_extract_frames(
    video_path: Path,
    video_id: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step A: extract frames via ffmpeg, write metadata JSON."""
    log.info("Extracting frames from %s at %.1f fps …", video_path.name, args.fps)
    t0 = time.time()

    # extract_frames uses settings.FRAMES_DIR / video_id
    frame_list = extract_frames(str(video_path), video_id)
    elapsed = time.time() - t0

    # Persist metadata alongside video output dir
    meta = {
        "video": str(video_path),
        "video_id": video_id,
        "fps": args.fps,
        "frame_count": len(frame_list),
        "duration_sec": frame_list[-1][1] if frame_list else 0.0,
        "frames": [{"path": p, "t_sec": t} for p, t in frame_list],
        "extracted_at": datetime.now().isoformat(),
    }
    meta_path = video_dir / "frames_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("  ✓ %d frames extracted in %.1fs → %s",
             len(frame_list), elapsed, meta_path)
    log.info("  Artifacts: %s", meta_path)
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
    """Embed a batch and add to store (Qdrant or in-memory). Returns count added."""
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
    """Step B: embed frames (PIL-based, no cv2) and upsert into Qdrant or InMemoryStore."""
    t0 = time.time()
    dest = "Qdrant" if is_qdrant else "in-memory store"
    log.info("Embedding %d frames into %s …", len(frame_list), dest)

    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")
    batch_pil: List[Image.Image] = []
    batch_meta: List[Tuple[str, float]] = []
    indexed = 0

    for fp, t_sec in frame_list:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        batch_pil.append(img)
        batch_meta.append((fp, t_sec))

        if len(batch_pil) >= 32:
            indexed += _embed_and_flush(batch_pil, batch_meta, video_id,
                                        clip_model, dino_model, store, is_qdrant)
            batch_pil, batch_meta = [], []

    indexed += _embed_and_flush(batch_pil, batch_meta, video_id,
                                clip_model, dino_model, store, is_qdrant)

    elapsed = time.time() - t0
    log.info("  ✓ %d frames indexed into %s in %.1fs", indexed, dest, elapsed)
    return {"indexed": indexed, "elapsed_sec": elapsed}


def step_scene_captioning(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """Step L: Run Florence-2 detailed scene captioning on extracted frames.

    Loads Florence-2-large, captions all frames in FLORENCE_BATCH_SIZE batches,
    writes scene_captions.md.
    """
    out_md = video_dir / "scene_captions.md"

    try:
        from pipeline.florence_model import FlorenceModel
    except ImportError as exc:
        log.warning("  Florence-2 unavailable (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    log.info("Loading Florence-2-large on %s …", device)
    t0 = time.time()
    try:
        florence = FlorenceModel()
    except Exception as exc:
        log.warning("  Florence-2 load failed (%s) — skipping captioning", exc)
        return {"skipped": True, "reason": str(exc), "captions": []}

    load_sec = time.time() - t0
    log.info("  ✓ Florence-2-large loaded in %.1fs", load_sec)
    log.info("  Captioning %d frames …", len(frame_list))

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
            log.warning("  Florence batch %d failed: %s", batch_start, exc)
            captions_and_confs = [("", 0.5)] * len(batch)

        for (fp, t_sec), (cap, conf) in zip(batch, captions_and_confs):
            caption_results.append({
                "frame_path": fp,
                "t_sec": t_sec,
                "caption": cap,
                "caption_confidence": conf,
            })

    elapsed = time.time() - t0
    captioned = sum(1 for r in caption_results if r.get("caption"))
    log.info("  ✓ %d/%d frames captioned in %.1fs", captioned, len(frame_list), elapsed)

    write_scene_captions_md(out_md, video_name, caption_results, elapsed)
    log.info("  Artifact: %s", out_md)

    # Sample a few captions for the stats table
    samples = [(r["caption"], r["caption_confidence"]) for r in caption_results if r.get("caption")][:3]

    return {
        "skipped": False,
        "captions": caption_results,
        "captioned_count": captioned,
        "elapsed_sec": elapsed,
        "sample_captions": samples,
    }


def _pick_query_frame(frame_list: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Pick the middle frame as a representative query."""
    mid = len(frame_list) // 2
    return frame_list[mid]


def _embed_query(
    frame_path: str,
    models: Dict[str, Any],
    use_dino: bool = True,
) -> np.ndarray:
    """Embed a single frame with CLIP (primary) or DINO."""
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
    """Search the store for nearest neighbours of query_vec."""
    if is_qdrant:
        from qdrant_client.http import models as qmodels
        filt = qmodels.Filter(must=[
            qmodels.FieldCondition(
                key="video_id",
                match=qmodels.MatchValue(value=video_id),
            )
        ])
        raw = store.search(vector_name, query_vec, limit=top_k, payload_filter=filt)
        return [{"score": p.score, "payload": p.payload} for p in raw]
    else:
        return store.search(query_vec, limit=top_k)


def step_base_model_search_test(
    frame_list: List[Tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: Dict[str, Any],
    video_id: str,
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step C: embed query with base model, search, write base_search.md."""
    out_md = video_dir / "base_search.md"
    qfp, qt = _pick_query_frame(frame_list)
    log.info("Query frame: %s (t=%.2fs)", Path(qfp).name, qt)

    use_dino = models.get("dino") is not None
    t0 = time.time()
    query_vec = _embed_query(qfp, models, use_dino=use_dino)
    results = _search(query_vec, store, is_qdrant, args.top_k, video_id,
                      vector_name="dino" if use_dino else "clip")
    elapsed = time.time() - t0

    label = "Base DINOv3 (pretrained)" if use_dino else "Base CLIP (pretrained)"
    write_search_md(out_md, video_name, label, qfp, results, qt)
    log.info("  ✓ Search in %.2fs → top score %.4f", elapsed, results[0]["score"] if results else 0)
    log.info("  Artifact: %s", out_md)
    return {"results": results, "query_frame": qfp, "query_t_sec": qt}


def step_ssl_finetune(
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    device: str,
) -> Dict[str, Any]:
    """Step D: run SSL DINOv3 fine-tuning, write finetune_stats.md."""
    out_md = video_dir / "finetune_stats.md"
    ckpt_dir = video_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TemporalPairDataset needs parent-of-{video_id} as frames_dir
    frames_parent = settings.FRAMES_DIR      # .../output/frames/
    n_frames = len(frame_list)

    # Fall back to augment when not enough frames for temporal pairs
    approach = "temporal" if n_frames >= args.batch_size * 2 else "augment"
    if approach == "augment":
        log.info("  Only %d frames — using augment approach (no temporal pairs needed)", n_frames)

    cfg = FinetuneConfig(
        frames_dir=frames_parent,
        output_dir=str(ckpt_dir),
        model_name="dinov3_vitb14",
        approach=approach,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-5,
        weight_decay=0.04,
        temperature=0.07,
        freeze_blocks=10,
        embed_dim=768,
        proj_out_dim=128,
        num_workers=0,          # safe for demo (no multiprocessing issues)
        save_every=1,
        max_gap=3,
        device=device,
        seed=42,
    )

    log.info("Starting SSL fine-tuning: %d epochs, approach=%s, device=%s",
             args.epochs, approach, device)
    t0 = time.time()
    loss_history: List[float] = []

    # Monkey-patch DINOFineTuner to capture per-epoch losses
    import pipeline.ssl_finetune as _ssl_mod
    _orig_run = _ssl_mod.run_finetune

    def _run_capturing(c: FinetuneConfig) -> str:
        import torch, random
        random.seed(c.seed)
        torch.manual_seed(c.seed)
        ckpt_dir_inner = c.output_dir
        os.makedirs(ckpt_dir_inner, exist_ok=True)

        from pipeline.ssl_finetune import (
            build_augment_transform, TemporalPairDataset, AugmentPairDataset,
            DINOFineTuner, NTXentLoss,
        )
        from torch.utils.data import DataLoader

        transform = build_augment_transform()
        if c.approach == "temporal":
            dataset = TemporalPairDataset(c.frames_dir, transform=transform, max_gap=c.max_gap)
        else:
            dataset = AugmentPairDataset(c.frames_dir, transform=transform)

        loader = DataLoader(dataset, batch_size=c.batch_size, shuffle=True,
                            num_workers=c.num_workers, pin_memory=(c.device != "cpu"),
                            drop_last=True)

        tuner = DINOFineTuner(model_name=c.model_name, freeze_blocks=c.freeze_blocks,
                              device=c.device, embed_dim=c.embed_dim, proj_out_dim=c.proj_out_dim)
        optimizer = torch.optim.AdamW(tuner.trainable_params(), lr=c.lr, weight_decay=c.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
        loss_fn = NTXentLoss(temperature=c.temperature)

        best_loss = float("inf")
        best_path = os.path.join(ckpt_dir_inner, "dino_ssl_best.pt")

        for epoch in range(1, c.epochs + 1):
            tuner.train()
            epoch_losses = []
            for v1, v2 in loader:
                v1, v2 = v1.to(c.device), v2.to(c.device)
                loss = loss_fn(tuner.forward(v1), tuner.forward(v2))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_losses.append(loss.item())
            scheduler.step()
            avg = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            loss_history.append(avg)
            log.info("    Epoch %d/%d  loss=%.4f", epoch, c.epochs, avg)
            ckpt = os.path.join(ckpt_dir_inner, f"dino_ssl_{epoch:03d}.pt")
            tuner.save_checkpoint(ckpt)
            if avg < best_loss:
                best_loss = avg
                tuner.save_checkpoint(best_path)
        return best_path

    best_path = _run_capturing(cfg)
    elapsed = time.time() - t0
    best_loss = min(loss_history) if loss_history else float("nan")

    log.info("  ✓ Fine-tuning complete in %.1fs | best loss=%.4f | checkpoint: %s",
             elapsed, best_loss, best_path)
    log.info("  To use: export DINO_CHECKPOINT=%s", best_path)

    write_finetune_stats_md(out_md, video_name, cfg, best_loss, best_path, elapsed, loss_history)
    log.info("  Artifact: %s", out_md)

    ckpt_mb = os.path.getsize(best_path) / 1e6 if os.path.exists(best_path) else 0
    return {"checkpoint": best_path, "best_loss": best_loss,
            "elapsed_sec": elapsed, "ckpt_mb": ckpt_mb, "cfg": cfg}


def step_distill(
    teacher_checkpoint: str,
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """Step E: distil fine-tuned teacher (ViT-B/14) → student (ViT-S/14).

    Returns dict with keys: student_backbone, best_path, best_loss, student_dim,
    teacher_dim, student_model, ckpt_mb, skipped.
    """
    out_md = video_dir / "distill_stats.md"
    distill_dir = video_dir / "checkpoints"

    result: Dict[str, Any] = {
        "student_backbone": None, "best_path": "", "best_loss": float("nan"),
        "student_dim": 384, "teacher_dim": 768,
        "student_model": "dinov2_vits14", "ckpt_mb": 0.0, "skipped": False,
    }

    if not _HAS_DINO:
        log.warning("  DINO not available — skipping distillation")
        result["skipped"] = True
        return result

    # Load teacher backbone from checkpoint
    try:
        import torch
        from models.dino_model import hub_load_dino
        teacher_bb = hub_load_dino("dinov3_vitb14", pretrained=True).to(device)
        state = torch.load(teacher_checkpoint, map_location=device)
        teacher_bb.load_state_dict(state)
        teacher_bb.eval()
        log.info("  Teacher loaded from checkpoint: %s", teacher_checkpoint)
    except Exception as exc:
        log.warning("  Could not load teacher checkpoint (%s) — skipping distillation", exc)
        result["skipped"] = True
        return result

    cfg = DistillConfig(
        student_model="dinov2_vits14",
        epochs=args.distill_epochs,
        batch_size=args.batch_size,
        device=device,
    )
    frame_paths = [fp for fp, _ in frame_list]
    log.info("Starting distillation: teacher=ViT-B/14 → student=ViT-S/14  "
             "epochs=%d  frames=%d", cfg.epochs, len(frame_paths))

    try:
        stats = run_distillation(teacher_bb, frame_paths, distill_dir, cfg)
    except Exception as exc:
        log.warning("  Distillation failed (%s) — skipping", exc)
        result["skipped"] = True
        return result

    distiller = stats.pop("distiller")
    result.update(stats)
    result["student_backbone"] = distiller.student_backbone()
    result["ckpt_mb"] = os.path.getsize(stats["best_path"]) / 1e6

    log.info("  ✓ Distillation complete in %.1fs | best_loss=%.4f | student=%s (dim=%d)",
             stats["elapsed"], stats["best_loss"], stats["student_model"], stats["student_dim"])
    log.info("  Teacher: ViT-B/14 ~86M params  →  Student: ViT-S/14 ~22M params  "
             "(%.1f× compression)", 86 / 22)

    write_distill_stats_md(out_md, video_name, stats)
    log.info("  Artifact: %s", out_md)
    return result


def step_export_model(
    checkpoint_path: str,
    frame_list: List[Tuple[str, float]],
    video_dir: Path,
    device: str,
    models: Dict[str, Any],
    student_backbone: Optional[Any] = None,
    student_dim: int = 768,
) -> Dict[str, Any]:
    """Step F: export distilled student (or fine-tuned teacher) to ONNX + build gallery.npz.

    When student_backbone is provided the student is exported; otherwise the
    fine-tuned teacher weights are loaded from checkpoint_path and exported.

    Gallery fallback order:
      1. Use just-exported ONNX file (preferred — same weights, no PyTorch required)
      2. Use PyTorch backbone directly
      3. Fall back to CLIP (always available) when no DINO backbone exists
    """
    edge_dir = video_dir / "edge_models"
    edge_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(edge_dir / "dino_demo.onnx")
    gallery_path = str(edge_dir / "gallery.npz")

    result: Dict[str, Any] = {
        "onnx_path": onnx_path, "gallery_path": gallery_path,
        "onnx_mb": 0.0, "exported": False, "gallery_saved": False,
    }

    # ── Resolve backbone to export ─────────────────────────────────────────
    backbone_to_export = None
    if student_backbone is not None:
        backbone_to_export = student_backbone
        model_label = f"distilled student (ViT-S/14, dim={student_dim})"
    elif _HAS_DINO:
        dino = models.get("dino")
        if dino is None:
            log.warning("  DINO not available — will use CLIP for gallery only")
        else:
            try:
                log.info("Loading fine-tuned checkpoint: %s", checkpoint_path)
                dino.load_backbone_checkpoint(checkpoint_path)
                log.info("  ✓ Checkpoint loaded")
                backbone_to_export = dino.model.eval()
                model_label = "fine-tuned teacher (ViT-B/14)"
            except Exception as exc:
                log.warning("  Could not load checkpoint (%s) — will use base DINO for export", exc)
                backbone_to_export = dino.model.eval()
                model_label = "base DINOv3 teacher (ViT-B/14)"
    else:
        log.warning("  DINO not available — skipping ONNX export; will use CLIP for gallery")

    # ── ONNX export ────────────────────────────────────────────────────────
    if backbone_to_export is not None and not args.no_onnx:
        try:
            import torch
            # Move to CPU for ONNX export to avoid device/trace issues
            backbone_cpu = backbone_to_export.cpu().eval()
            dummy = torch.zeros(1, 3, 224, 224)
            log.info("Exporting ONNX (%s) to %s …", model_label, onnx_path)
            torch.onnx.export(
                backbone_cpu, dummy, onnx_path,
                opset_version=14,
                input_names=["pixel_values"],
                output_names=["embedding"],
                dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
                do_constant_folding=True,
            )
            if os.path.exists(onnx_path):
                onnx_mb = os.path.getsize(onnx_path) / 1e6
                result["onnx_mb"] = onnx_mb
                result["exported"] = True
                log.info("  ✓ ONNX export complete: %.1f MB → %s", onnx_mb, onnx_path)
                log.info("  Artifact: %s  (deploy on Jetson/Hailo-8/ARM)", onnx_path)
            else:
                log.warning("  ONNX export ran but file not found at %s", onnx_path)
            # Restore backbone to original device for gallery building
            backbone_to_export = backbone_to_export.to(device).eval()
        except Exception as exc:
            log.warning("  ONNX export failed (%s) — skipping", exc)
            if backbone_to_export is not None:
                try:
                    backbone_to_export = backbone_to_export.to(device).eval()
                except Exception:
                    pass
    elif backbone_to_export is not None:
        log.info("  ONNX export skipped (--no-onnx)")

    # ── Gallery build ──────────────────────────────────────────────────────
    log.info("Building embedding gallery from %d frames …", len(frame_list))
    try:
        step = max(1, len(frame_list) // 200)
        sampled = [fp for fp, _ in frame_list[::step]]
        # Filter to paths that actually exist
        sampled = [fp for fp in sampled if os.path.isfile(fp)]
        if not sampled:
            raise ValueError("No valid frame paths for gallery build")

        labels_map = {"scene": sampled}

        if result["exported"] and os.path.exists(onnx_path):
            # Preferred: use the just-written ONNX — consistent weights, no PyTorch needed
            build_gallery(labels_map=labels_map, output_path=gallery_path,
                          onnx_path=onnx_path)
            log.info("  Gallery built using ONNX model")
        elif backbone_to_export is not None:
            build_gallery(labels_map=labels_map, output_path=gallery_path,
                          backbone=backbone_to_export)
            log.info("  Gallery built using PyTorch backbone")
        else:
            # Fallback: use CLIP (always available)
            clip_model: OpenCLIPEmbedder = models["clip"]
            import torch
            import torch.nn.functional as F

            _device_clip = next(clip_model.model.parameters()).device
            transform_clip = __import__("torchvision").transforms.Compose([
                __import__("torchvision").transforms.Resize(
                    224, interpolation=__import__("torchvision").transforms.InterpolationMode.BICUBIC),
                __import__("torchvision").transforms.CenterCrop(224),
                __import__("torchvision").transforms.ToTensor(),
                __import__("torchvision").transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            all_embeds = []
            all_labels_g = []
            for fp in sampled:
                img = Image.open(fp).convert("RGB")
                emb = clip_model.encode_images([img])[0]
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                all_embeds.append(emb.astype(np.float32))
                all_labels_g.append("scene")
            emb_arr = np.stack(all_embeds, axis=0)
            np.savez(gallery_path,
                     embeddings=emb_arr,
                     labels=np.array(all_labels_g, dtype=object),
                     label_names=np.array(["scene"], dtype=object))
            log.info("  Gallery built using CLIP fallback (no DINO available)")

        if os.path.exists(gallery_path):
            gallery_mb = os.path.getsize(gallery_path) / 1e6
            result["gallery_saved"] = True
            log.info("  ✓ Gallery saved: %d embeddings → %s (%.1f MB)",
                     len(sampled), gallery_path, gallery_mb)
            log.info("  Artifact: %s  (use with EdgeClassifier for on-device 1-NN)", gallery_path)
        else:
            log.warning("  Gallery file not found after build: %s", gallery_path)

    except Exception as exc:
        log.warning("  Gallery build failed (%s)", exc, exc_info=True)

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
) -> Dict[str, Any]:
    """Step F: search with fine-tuned DINO, write finetuned_search.md."""
    out_md = video_dir / "finetuned_search.md"

    use_dino = models.get("dino") is not None
    t0 = time.time()
    query_vec = _embed_query(query_frame, models, use_dino=use_dino)
    results = _search(query_vec, store, is_qdrant, args.top_k, video_id,
                      vector_name="dino" if use_dino else "clip")
    ft_infer_ms = (time.time() - t0) * 1000 / max(len(frame_list), 1)

    write_search_md(out_md, video_name, "Fine-tuned DINOv3 (SSL adapted)",
                    query_frame, results, query_t_sec)
    log.info("  Artifact: %s", out_md)
    return {"results": results, "infer_ms": ft_infer_ms}


def step_asr_transcription(
    video_path: Path,
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step M: extract audio and run Whisper ASR to generate per-frame subtitles."""
    out_md = video_dir / "asr_subtitles.md"
    result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}

    try:
        from pipeline.audio_extractor import extract_audio, map_subtitles_to_frames
        from pipeline.asr_model import ASRModel
    except ImportError as exc:
        log.warning("  ASR unavailable (%s) — skipping", exc)
        return result

    asr = ASRModel()
    if not asr.is_enabled():
        log.info("  ASR disabled (ASR_ENABLED=false) — skipping")
        return result

    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    log.info("Extracting audio from %s …", video_path.name)
    wav_path = extract_audio(str(video_path), str(audio_dir))
    if not wav_path:
        log.warning("  No audio stream found in %s — ASR skipped", video_path.name)
        return result

    log.info("Transcribing audio with %s …", asr.model_id)
    t0 = time.time()
    segments = asr.transcribe(wav_path)
    elapsed = time.time() - t0

    if not segments:
        log.warning("  ASR returned no segments for %s", video_path.name)
        return result

    frame_timestamps = [t for _, t in frame_list]
    subtitle_map = map_subtitles_to_frames(segments, frame_timestamps,
                                           window_sec=settings.ASR_SUBTITLE_WINDOW_SEC)
    covered = sum(1 for t in frame_timestamps if t in subtitle_map)
    log.info("  ✓ ASR: %d segments → %d/%d frames have subtitles (%.1fs, model=%s)",
             len(segments), covered, len(frame_list), elapsed, asr.model_id)

    # Write subtitle markdown
    lines = [
        f"# ASR Subtitles — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: `{asr.model_id}`",
        f"Segments: {len(segments)}  |  Frames with subtitles: {covered}/{len(frame_list)}",
        f"Elapsed: {elapsed:.1f}s",
        f"",
        f"## Subtitle Segments",
        f"",
        f"| Start (s) | End (s) | Text |",
        f"|-----------|---------|------|",
    ]
    for seg in segments:
        ts = seg.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
        start, end = ts
        text = seg.get("text", "").strip().replace("|", "\\|")
        lines.append(f"| {start:.2f} | {end:.2f} | {text} |")
    lines += ["", "---", "*Produced by `demo.py` · ASR step M*"]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("  Artifact: %s", out_md)

    result.update({"skipped": False, "subtitle_map": subtitle_map,
                   "segments": segments, "elapsed_sec": elapsed,
                   "covered_frames": covered})
    return result


def step_ocr_extraction(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step N: extract visible text from each frame using OCR."""
    result: Dict[str, Any] = {"skipped": True, "ocr_results": []}

    try:
        from pipeline.ocr_model import OCRModel
    except ImportError as exc:
        log.warning("  OCR unavailable (%s) — skipping", exc)
        return result

    ocr = OCRModel()
    if not ocr.is_enabled():
        log.info("  OCR disabled (OCR_ENABLED=false) — skipping")
        return result

    log.info("Running OCR on %d frames (model=%s) …", len(frame_list), ocr.model_id)
    t0 = time.time()
    ocr_results: List[Dict[str, Any]] = []
    batch_size = settings.OCR_BATCH_SIZE

    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        imgs = []
        for fp, _t in batch:
            try:
                imgs.append(Image.open(fp).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
        try:
            batch_results = ocr.extract_text_batch(imgs)
        except Exception as exc:
            log.warning("  OCR batch %d failed: %s", batch_start, exc)
            batch_results = [{"ocr_text": "", "ocr_error": True}] * len(batch)
        for (fp, t_sec), r in zip(batch, batch_results):
            ocr_results.append({"frame_path": fp, "t_sec": t_sec, **r})

    elapsed = time.time() - t0
    non_empty = sum(1 for r in ocr_results if r.get("ocr_text"))
    log.info("  ✓ OCR: %d/%d frames have text in %.1fs", non_empty, len(frame_list), elapsed)
    result.update({"skipped": False, "ocr_results": ocr_results,
                   "non_empty": non_empty, "elapsed_sec": elapsed})
    return result


def step_depth_estimation(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step O: estimate depth for each frame (5-bucket percentile summary)."""
    result: Dict[str, Any] = {"skipped": True, "depth_results": []}

    try:
        from pipeline.depth_model import DepthModel
    except ImportError as exc:
        log.warning("  Depth model unavailable (%s) — skipping", exc)
        return result

    depth_model = DepthModel()
    if not depth_model.is_enabled():
        log.info("  Depth disabled (DEPTH_ENABLED=false) — skipping")
        return result

    log.info("Running depth estimation on %d frames (model=%s) …",
             len(frame_list), depth_model.model_id)
    t0 = time.time()
    depth_results: List[Dict[str, Any]] = []
    batch_size = 4

    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        imgs = []
        for fp, _t in batch:
            try:
                imgs.append(Image.open(fp).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
        try:
            batch_out = depth_model.estimate_batch(imgs)
        except Exception as exc:
            log.warning("  Depth batch %d failed: %s", batch_start, exc)
            batch_out = [{"depth_error": True}] * len(batch)
        for (fp, t_sec), r in zip(batch, batch_out):
            depth_results.append({"frame_path": fp, "t_sec": t_sec, **r})

    elapsed = time.time() - t0
    ok = sum(1 for r in depth_results if not r.get("depth_error"))
    log.info("  ✓ Depth: %d/%d frames estimated in %.1fs", ok, len(frame_list), elapsed)
    result.update({"skipped": False, "depth_results": depth_results,
                   "ok_count": ok, "elapsed_sec": elapsed})
    return result


def step_object_detection(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step P: run object detection on each frame."""
    result: Dict[str, Any] = {"skipped": True, "detection_results": []}

    try:
        from pipeline.detection_model import DetectionModel
    except ImportError as exc:
        log.warning("  Detection model unavailable (%s) — skipping", exc)
        return result

    det_model = DetectionModel()
    if not det_model.is_enabled():
        log.info("  Detection disabled (DETECTION_ENABLED=false) — skipping")
        return result

    log.info("Running object detection on %d frames (model=%s) …",
             len(frame_list), det_model.model_id)
    t0 = time.time()
    det_results: List[Dict[str, Any]] = []
    batch_size = 4

    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        imgs = []
        for fp, _t in batch:
            try:
                imgs.append(Image.open(fp).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
        try:
            batch_out = det_model.detect_batch(imgs)
        except Exception as exc:
            log.warning("  Detection batch %d failed: %s", batch_start, exc)
            batch_out = [{"detection_error": True}] * len(batch)
        for (fp, t_sec), r in zip(batch, batch_out):
            det_results.append({"frame_path": fp, "t_sec": t_sec, **r})

    elapsed = time.time() - t0
    total_objs = sum(len(r.get("detections", [])) for r in det_results)
    ok = sum(1 for r in det_results if not r.get("detection_error"))
    log.info("  ✓ Detection: %d objects across %d/%d frames in %.1fs",
             total_objs, ok, len(frame_list), elapsed)
    result.update({"skipped": False, "detection_results": det_results,
                   "total_objects": total_objs, "ok_count": ok, "elapsed_sec": elapsed})
    return result


def step_world_model_pass(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
) -> Dict[str, Any]:
    """Step Q: compute world model video embeddings for temporal clips."""
    result: Dict[str, Any] = {"skipped": True, "world_results": []}

    try:
        from pipeline.world_model import WorldModel
    except ImportError as exc:
        log.warning("  World model unavailable (%s) — skipping", exc)
        return result

    wm = WorldModel()
    if not wm.is_enabled():
        log.info("  World model disabled (WORLD_MODEL_ENABLED=false) — skipping")
        return result

    clip_frames = settings.WORLD_MODEL_CLIP_FRAMES
    log.info("Running world model on %d frames in clips of %d (model=%s) …",
             len(frame_list), clip_frames, wm.model_id)
    t0 = time.time()
    world_results: List[Dict[str, Any]] = []

    for clip_start in range(0, len(frame_list), clip_frames):
        clip = frame_list[clip_start : clip_start + clip_frames]
        imgs = []
        for fp, _t in clip:
            try:
                imgs.append(Image.open(fp).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
        try:
            clip_out = wm.process_clip(imgs)
        except Exception as exc:
            log.warning("  World model clip %d failed: %s", clip_start, exc)
            clip_out = {"world_model_error": True}
        # Assign result to middle frame of the clip
        mid = clip_start + len(clip) // 2
        fp, t_sec = frame_list[mid]
        world_results.append({"frame_path": fp, "t_sec": t_sec, **clip_out})

    elapsed = time.time() - t0
    ok = sum(1 for r in world_results if not r.get("world_model_error"))
    log.info("  ✓ World model: %d clips processed in %.1fs", ok, elapsed)
    result.update({"skipped": False, "world_results": world_results,
                   "ok_count": ok, "elapsed_sec": elapsed})
    return result


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
    """Write combined multimodal features report (OCR, depth, detection, world model)."""
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

    # OCR sample
    if not ocr_result.get("skipped"):
        lines += ["## OCR — Sample Text Extractions", ""]
        ocr_rows = [r for r in ocr_result.get("ocr_results", []) if r.get("ocr_text")][:10]
        if ocr_rows:
            lines += ["| t (s) | Extracted Text |", "|-------|----------------|"]
            for r in ocr_rows:
                txt = (r.get("ocr_text") or "").replace("|", "\\|")[:120]
                lines.append(f"| {r['t_sec']:.1f} | {txt} |")
        lines.append("")

    # Detection sample
    if not det_result.get("skipped"):
        lines += ["## Detection — Objects Found", ""]
        det_rows = [r for r in det_result.get("detection_results", [])
                    if r.get("detections")][:10]
        if det_rows:
            lines += ["| t (s) | Detections |", "|-------|------------|"]
            for r in det_rows:
                objs = ", ".join(
                    f"{d['label']} ({d['confidence']:.2f})"
                    for d in r["detections"][:5]
                )
                lines.append(f"| {r['t_sec']:.1f} | {objs} |")
        lines.append("")

    # Depth sample
    if not depth_result.get("skipped"):
        lines += ["## Depth — Percentile Summary (sample)", ""]
        depth_rows = [r for r in depth_result.get("depth_results", [])
                      if r.get("depth")][:5]
        if depth_rows:
            lines += ["| t (s) | p10 | p25 | p50 | p75 | p90 |",
                      "|-------|-----|-----|-----|-----|-----|"]
            for r in depth_rows:
                p = r["depth"].get("percentiles", [0]*5)
                lines.append(f"| {r['t_sec']:.1f} | "
                              f"{p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {p[3]:.3f} | {p[4]:.3f} |")
        lines.append("")

    lines += ["---", "*Produced by `demo.py` · multimodal steps M–R*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


def write_detailed_captions_md(
    output_path: Path,
    video_name: str,
    results: List[Dict[str, Any]],
    elapsed_sec: float,
    model_id: str,
) -> None:
    """Write per-frame Qwen VLM detailed captions as Markdown."""
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
        fp = r.get("frame_path", "")
        name = Path(fp).name if fp else "—"
        t = r.get("t_sec", 0.0)
        subtitle = (r.get("subtitle_text") or "").replace("|", "\\|")[:60]
        if r.get("service_unavailable"):
            caption = "*sidecar unavailable*"
        elif r.get("skipped"):
            caption = "*skipped*"
        else:
            # QwenModel returns a structured dict; join key facts as text
            facts = r.get("caption") or r.get("scene_description") or ""
            if not facts and isinstance(r, dict):
                # Flatten any nested fields returned by extract_frame_facts
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
        f"*Produced by `demo.py` · Qwen VLM step R · ASR subtitle context injected where available*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ Written %s", output_path)


def step_qwen_captioning(
    frame_list: List[Tuple[str, float]],
    video_name: str,
    video_dir: Path,
    subtitle_map: Dict[float, str],
    ocr_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Step R: Qwen VLM detailed scene captioning with ASR subtitle + OCR context.

    For each frame, looks up the nearest ASR subtitle from subtitle_map and any
    OCR text from the corresponding frame, then calls QwenModel.extract_batch
    with both as context.  Writes detailed_captions.md.

    Requires QWEN_API_URL to be set (either via --qwen-api-url flag or env var).
    """
    out_md = video_dir / "detailed_captions.md"
    result: Dict[str, Any] = {"skipped": True, "results": []}

    try:
        from pipeline.qwen_model import QwenModel
    except ImportError as exc:
        log.warning("  Qwen model unavailable (%s) — skipping", exc)
        return result

    qwen = QwenModel()
    if not qwen.is_enabled():
        log.info("  Qwen disabled (QWEN_API_URL not set) — skipping detailed captioning")
        log.info("  To enable: --qwen-api-url http://localhost:8010/v1  (or set QWEN_API_URL)")
        return result

    # Build OCR lookup: t_sec → ocr_text
    ocr_map: Dict[float, str] = {}
    for r in ocr_results:
        t = r.get("t_sec")
        txt = r.get("ocr_text") or ""
        if t is not None and txt:
            ocr_map[t] = txt

    log.info("Running Qwen detailed captioning on %d frames (model=%s) …",
             len(frame_list), settings.QWEN_MODEL)
    t0 = time.time()

    batch_size = 4  # Qwen sidecar is typically single-GPU; keep batches small
    caption_results: List[Dict[str, Any]] = []

    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        imgs: List[Image.Image] = []
        subtitles: List[Optional[str]] = []
        ocr_texts: List[Optional[str]] = []

        for fp, t_sec in batch:
            try:
                imgs.append(Image.open(fp).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
            subtitles.append(subtitle_map.get(t_sec) or None)
            ocr_texts.append(ocr_map.get(t_sec) or None)

        try:
            batch_out = qwen.extract_batch(imgs, subtitle_texts=subtitles, ocr_texts=ocr_texts)
        except Exception as exc:
            log.warning("  Qwen batch %d failed: %s", batch_start, exc)
            batch_out = [{"service_unavailable": True}] * len(batch)

        for (fp, t_sec), r in zip(batch, batch_out):
            caption_results.append({
                "frame_path": fp,
                "t_sec": t_sec,
                "subtitle_text": subtitle_map.get(t_sec) or "",
                **r,
            })

    elapsed = time.time() - t0
    ok = sum(1 for r in caption_results
             if not r.get("service_unavailable") and not r.get("skipped"))
    subtitle_used = sum(1 for r in caption_results if r.get("subtitle_text"))
    log.info("  ✓ Qwen: %d/%d frames captioned in %.1fs (%d with ASR context)",
             ok, len(frame_list), elapsed, subtitle_used)

    write_detailed_captions_md(out_md, video_name, caption_results, elapsed, settings.QWEN_MODEL)
    log.info("  Artifact: %s", out_md)

    result.update({"skipped": False, "results": caption_results,
                   "ok_count": ok, "subtitle_used": subtitle_used,
                   "elapsed_sec": elapsed})
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
    """Step G: compare results, caption video, write comparison.md."""
    out_md = video_dir / "comparison.md"

    # Measure inference times on a small sample
    sample_paths = [fp for fp, _ in frame_list[:10]]
    clip_model: OpenCLIPEmbedder = models["clip"]
    dino_model = models.get("dino")

    t0 = time.time()
    clip_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
    base_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)

    ft_infer_ms = base_infer_ms  # same model after hot-load; could differ on GPU
    if dino_model:
        t0 = time.time()
        dino_model.encode_images([Image.open(p).convert("RGB") for p in sample_paths])
        ft_infer_ms = (time.time() - t0) * 1000 / len(sample_paths)

    # Video-to-text: average CLIP frame embedding vs text prompts
    log.info("Computing video-to-text description …")
    try:
        # Embed up to 32 evenly-spaced frames
        step = max(1, len(frame_list) // 32)
        sampled_imgs = [Image.open(fp).convert("RGB") for fp, _ in frame_list[::step]]
        frame_embeds = clip_model.encode_images(sampled_imgs)  # (N, D)
        avg_embed = frame_embeds.mean(axis=0)                  # (D,)

        text_embeds = clip_model.encode_texts(_TEXT_PROMPTS)   # (T, D)
        scores = text_embeds @ avg_embed                        # (T,)
        ranked = sorted(zip(_TEXT_PROMPTS, scores.tolist()), key=lambda x: x[1], reverse=True)
        text_descriptions = ranked[:3]
        all_scored = ranked
        for desc, score in text_descriptions:
            log.info("  Video description: \"%s\" (sim=%.3f)", desc, score)
    except Exception as exc:
        log.warning("  Video-to-text failed (%s)", exc)
        text_descriptions = [("description unavailable", 0.0)]
        all_scored = text_descriptions

    write_comparison_md(
        out_md, video_name, base_results, ft_results,
        base_infer_ms, ft_infer_ms, ckpt_mb, onnx_mb, text_descriptions,
    )
    log.info("  Artifact: %s", out_md)

    desc_md = video_dir / "description.md"
    write_description_md(desc_md, video_name, frame_list, text_descriptions, all_scored)
    log.info("  Artifact: %s", desc_md)

    # Also echo summary to console
    log.info("── Comparison summary for %s ──────────────────────", video_name)
    log.info("  Base model avg score: %.4f", float(np.mean([r.get("score", 0) for r in base_results])) if base_results else 0)
    log.info("  Fine-tuned avg score: %.4f", float(np.mean([r.get("score", 0) for r in ft_results])) if ft_results else 0)
    log.info("  Inference time: base=%.1f ms/frame, fine-tuned=%.1f ms/frame",
             base_infer_ms, ft_infer_ms)
    log.info("  Checkpoint size: %.1f MB (PyTorch)  %.1f MB (ONNX)", ckpt_mb, onnx_mb)
    log.info("  Top video description: \"%s\"", text_descriptions[0][0] if text_descriptions else "—")

    return {
        "text_descriptions": text_descriptions,
        "base_infer_ms":     base_infer_ms,
        "ft_infer_ms":       ft_infer_ms,
        "top_description":   text_descriptions[0][0] if text_descriptions else "",
    }


def step_create_3d_map(
    video_path: Path,
    video_id: str,
    video_dir: Path,
    frame_list: List[Tuple[str, float]],
    models: Dict[str, Any],
    run_sfm_flag: bool,
) -> Dict[str, Any]:
    """Step H: build sparse 3D map → 3d_map/sparse_map.npz + map_stats.json."""
    return build_sparse_map(
        video_path=str(video_path),
        video_id=video_id,
        map_dir=video_dir / "3d_map",
        frame_list=frame_list,
        models=models,
        run_sfm_flag=run_sfm_flag,
    )


# ── Per-video orchestrator ─────────────────────────────────────────────────────

_TOTAL_STEPS = 16


def run_video_pipeline(
    video_path: Path,
    output_dir: Path,
    models: Dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
) -> Dict[str, Any]:
    """Run all pipeline steps for a single video. Returns per-video stats dict."""
    video_name = video_path.stem
    video_id = video_name.replace(" ", "_").lower()
    video_dir = output_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    _banner(f"Processing video: {video_path.name}")
    log.info("Output directory: %s", video_dir)

    stats: Dict[str, Any] = {
        "name": video_name,
        "video_path": str(video_path),
        "timings": {},
    }
    T = stats["timings"]   # shorthand

    # ── A: Extract frames ──────────────────────────────────────────────────────
    _step(1, _TOTAL_STEPS, "Frame extraction")
    with _Timer(T, "A_extract"):
        a = step_extract_frames(video_path, video_id, video_dir)
    frame_list: List[Tuple[str, float]] = a["frame_list"]
    stats["frames"]       = a["meta"]["frame_count"]
    stats["duration_sec"] = a["meta"]["duration_sec"]
    stats["video_fps"]    = a["meta"].get("fps", 0.0)

    if not frame_list:
        log.error("No frames extracted — skipping video %s", video_path.name)
        return stats

    # ── B: Index into store ────────────────────────────────────────────────────
    _step(2, _TOTAL_STEPS, "Vector store indexing")
    with _Timer(T, "B_index"):
        b = step_index_to_store(video_path, video_id, store, is_qdrant, models, frame_list)
    stats["index_sec"]       = b["elapsed_sec"]
    stats["indexed_frames"]  = b.get("indexed", 0)

    # ── L: Scene captioning ────────────────────────────────────────────────────
    caption_results: List[Dict[str, Any]] = []
    if not args.no_caption:
        _step(3, _TOTAL_STEPS, "Florence-2 scene captioning → scene_captions.md")
        with _Timer(T, "L_caption"):
            l_cap = step_scene_captioning(frame_list, video_name, video_dir, device)
        caption_results = l_cap.get("captions", [])
        if not l_cap.get("skipped"):
            stats["captioned_frames"] = l_cap.get("captioned_count", 0)
            stats["caption_elapsed_sec"] = l_cap.get("elapsed_sec", 0.0)
    else:
        T["L_caption"] = 0.0
        _step(3, _TOTAL_STEPS, "Scene captioning (skipped — --no-caption)")
        log.info("  Captioning skipped (--no-caption)")

    # ── M: ASR ────────────────────────────────────────────────────────────────
    asr_result: Dict[str, Any] = {"skipped": True, "subtitle_map": {}}
    if args.asr:
        _step(4, _TOTAL_STEPS, "ASR transcription → asr_subtitles.md")
        with _Timer(T, "M_asr"):
            asr_result = step_asr_transcription(video_path, frame_list, video_name, video_dir)
        if not asr_result.get("skipped"):
            stats["asr_segments"]      = len(asr_result.get("segments", []))
            stats["asr_covered_frames"] = asr_result.get("covered_frames", 0)
    else:
        T["M_asr"] = 0.0

    # ── N: OCR ────────────────────────────────────────────────────────────────
    ocr_result: Dict[str, Any] = {"skipped": True, "ocr_results": []}
    if args.ocr:
        _step(5, _TOTAL_STEPS, "OCR text extraction")
        with _Timer(T, "N_ocr"):
            ocr_result = step_ocr_extraction(frame_list, video_name, video_dir)
        if not ocr_result.get("skipped"):
            stats["ocr_frames_with_text"] = ocr_result.get("non_empty", 0)
    else:
        T["N_ocr"] = 0.0

    # ── O: Depth ──────────────────────────────────────────────────────────────
    depth_result: Dict[str, Any] = {"skipped": True, "depth_results": []}
    if args.depth:
        _step(6, _TOTAL_STEPS, "Depth estimation")
        with _Timer(T, "O_depth"):
            depth_result = step_depth_estimation(frame_list, video_name, video_dir)
        if not depth_result.get("skipped"):
            stats["depth_frames_ok"] = depth_result.get("ok_count", 0)
    else:
        T["O_depth"] = 0.0

    # ── P: Detection ──────────────────────────────────────────────────────────
    det_result: Dict[str, Any] = {"skipped": True, "detection_results": []}
    if args.detection:
        _step(7, _TOTAL_STEPS, "Object detection")
        with _Timer(T, "P_detection"):
            det_result = step_object_detection(frame_list, video_name, video_dir)
        if not det_result.get("skipped"):
            stats["det_total_objects"] = det_result.get("total_objects", 0)
    else:
        T["P_detection"] = 0.0

    # ── Q: World model ────────────────────────────────────────────────────────
    world_result: Dict[str, Any] = {"skipped": True, "world_results": []}
    if args.world_model:
        _step(8, _TOTAL_STEPS, "World model video embeddings")
        with _Timer(T, "Q_world"):
            world_result = step_world_model_pass(frame_list, video_name, video_dir)
        if not world_result.get("skipped"):
            stats["world_clips_ok"] = world_result.get("ok_count", 0)
    else:
        T["Q_world"] = 0.0

    # ── R: Qwen detailed captioning ───────────────────────────────────────────
    qwen_result: Dict[str, Any] = {"skipped": True, "results": []}
    if args.qwen:
        _step(9, _TOTAL_STEPS, "Qwen VLM detailed captioning → detailed_captions.md")
        with _Timer(T, "R_qwen"):
            qwen_result = step_qwen_captioning(
                frame_list, video_name, video_dir,
                subtitle_map=asr_result.get("subtitle_map", {}),
                ocr_results=ocr_result.get("ocr_results", []),
            )
        if not qwen_result.get("skipped"):
            stats["qwen_captioned"] = qwen_result.get("ok_count", 0)
            stats["qwen_subtitle_used"] = qwen_result.get("subtitle_used", 0)
    else:
        T["R_qwen"] = 0.0

    # Write combined multimodal report if any optional step ran
    _any_multimodal = (args.asr or args.ocr or args.depth or args.detection or args.world_model or args.qwen)
    if _any_multimodal:
        _mm_md = video_dir / "multimodal_features.md"
        write_multimodal_md(_mm_md, video_name, asr_result, ocr_result,
                            depth_result, det_result, world_result, qwen_result)
        log.info("  Artifact: %s", _mm_md)

    # ── C: Base model search test ──────────────────────────────────────────────
    _step(10, _TOTAL_STEPS, "Base model transformation test → base_search.md")
    with _Timer(T, "C_base_search"):
        c = step_base_model_search_test(
            frame_list, store, is_qdrant, models, video_id, video_name, video_dir,
        )
    base_results = c["results"]
    query_frame  = c["query_frame"]
    query_t_sec  = c["query_t_sec"]
    stats["base_top_score"] = base_results[0]["score"] if base_results else 0.0

    # ── D: SSL fine-tuning ─────────────────────────────────────────────────────
    _step(11, _TOTAL_STEPS, "SSL DINOv3 fine-tuning → finetune_stats.md")
    with _Timer(T, "D_finetune"):
        d = step_ssl_finetune(video_id, video_name, video_dir, frame_list, device)
    stats["best_loss"]    = d["best_loss"]
    stats["ckpt_mb"]      = d["ckpt_mb"]
    stats["ft_epochs"]    = d["cfg"].epochs
    stats["ft_loss_history"] = getattr(d.get("cfg"), "loss_history", [])
    checkpoint_path       = d["checkpoint"]

    # ── E: Knowledge distillation ─────────────────────────────────────────────
    student_backbone = None
    student_dim = 768
    if not args.no_distill:
        _step(12, _TOTAL_STEPS, "Knowledge distillation: ViT-B/14 teacher → ViT-S/14 student")
        with _Timer(T, "E_distill"):
            e_distill = step_distill(checkpoint_path, frame_list, video_name, video_dir, device)
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
        log.info("  Distillation skipped; exporting teacher to ONNX instead")

    # ── F: ONNX export + gallery ──────────────────────────────────────────────
    _step(13, _TOTAL_STEPS, "ONNX export + gallery build → edge_models/")
    with _Timer(T, "F_export"):
        e = step_export_model(checkpoint_path, frame_list, video_dir, device, models,
                              student_backbone=student_backbone, student_dim=student_dim)
    onnx_mb = e.get("onnx_mb", 0.0)
    stats["onnx_mb"]       = onnx_mb
    stats["onnx_exported"] = e.get("exported", False)

    # ── G: Fine-tuned model search test ───────────────────────────────────────
    _step(14, _TOTAL_STEPS, "Fine-tuned model transformation test → finetuned_search.md")
    with _Timer(T, "G_ft_search"):
        f = step_finetuned_model_search_test(
            frame_list, store, is_qdrant, models,
            query_frame, query_t_sec, video_id, video_name, video_dir,
        )
    ft_results = f["results"]
    stats["ft_top_score"] = ft_results[0]["score"] if ft_results else 0.0

    # ── H: Comparison + video description ────────────────────────────────────
    _step(15, _TOTAL_STEPS, "Model comparison + video description → comparison.md, description.md")
    with _Timer(T, "H_compare"):
        g = step_compare_and_describe(
            frame_list, store, is_qdrant, base_results, ft_results,
            models, video_id, video_name, video_dir,
            stats["ckpt_mb"], onnx_mb,
        )
    if g:
        stats["base_infer_ms"] = g.get("base_infer_ms", 0.0)
        stats["ft_infer_ms"]   = g.get("ft_infer_ms", 0.0)
        stats["top_description"] = g.get("top_description", "")

    # ── I: 3D map ─────────────────────────────────────────────────────────────
    _step(16, _TOTAL_STEPS, "3D map creation → 3d_map/sparse_map.npz + sparse_map.ply")
    with _Timer(T, "I_3dmap"):
        h = step_create_3d_map(
            video_path, video_id, video_dir, frame_list, models,
            run_sfm_flag=not args.no_sfm,
        )
    stats["sfm_poses"]   = h["sfm_poses"]
    stats["map_method"]  = h["method"]
    stats["map_points"]  = int(h["points"].shape[0]) if h.get("points") is not None else 0

    stats["pipeline_sec"] = sum(T.values())

    _banner(f"✓ Video complete: {video_path.name}")
    log.info("  Output dir: %s", video_dir)

    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def find_videos(videos_dir: Path) -> List[Path]:
    videos = sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
    return videos


def main() -> None:
    # --view-npz mode: just load and show existing NPZ files, then exit
    if args.view_npz is not None:
        if not _HAS_MPL:
            log.error("matplotlib is required for the 3D viewer.")
            log.error("  Install: pip install matplotlib")
            sys.exit(1)
        view_npz(args.view_npz if args.view_npz is not None else "", _OUTPUT_DIR)
        return

    t_start = time.time()

    _banner("selfsuvis — End-to-End Demo Pipeline")
    log.info("Videos directory : %s", args.videos_dir)
    log.info("Output directory : %s", _OUTPUT_DIR)
    log.info("Device           : %s", args.device)
    log.info("Epochs           : %d", args.epochs)
    log.info("Qdrant           : %s", "disabled" if args.no_qdrant else "auto-detect")
    log.info("SfM              : %s", "disabled" if args.no_sfm else "auto-detect (pycolmap)")
    if any([args.asr, args.ocr, args.depth, args.detection, args.world_model, args.qwen]):
        log.info("Multimodal steps   : %s",
                 " ".join(s for s, e in [("ASR", args.asr), ("OCR", args.ocr),
                                          ("Depth", args.depth),
                                          ("Detection", args.detection),
                                          ("WorldModel", args.world_model),
                                          ("Qwen", args.qwen)] if e))

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        log.error("Videos directory does not exist: %s", videos_dir)
        log.error("Create it with:  mkdir -p %s", videos_dir)
        log.error("Then place .mp4/.mov/.mkv files inside and re-run.")
        sys.exit(1)

    videos = find_videos(videos_dir)
    if not videos:
        log.error("No video files found in %s", videos_dir)
        log.error("Supported formats: %s", " ".join(sorted(_VIDEO_EXTS)))
        log.error("Download a sample: wget -P %s https://www.pexels.com/download/... (see README)", videos_dir)
        sys.exit(1)

    log.info("Found %d video(s): %s", len(videos), [v.name for v in videos])

    # Initialise shared resources
    device = _resolve_device()
    log.info("Using device: %s", device)

    t_init = time.time()
    models = init_models(device)
    store, is_qdrant = init_store(models, use_qdrant=not args.no_qdrant)
    init_elapsed = time.time() - t_init

    # Per-video pipeline
    per_video_stats: List[Dict[str, Any]] = []

    try:
        for i, video_path in enumerate(videos, 1):
            _banner(f"Video {i}/{len(videos)}: {video_path.name}")
            try:
                vstats = run_video_pipeline(
                    video_path, _OUTPUT_DIR, models, store, is_qdrant, device,
                )
            except KeyboardInterrupt:
                raise  # bubble up to the outer handler
            except Exception as exc:
                log.error("Pipeline failed for %s: %s", video_path.name, exc, exc_info=True)
                vstats = {"name": video_path.stem, "error": str(exc)}

            per_video_stats.append(vstats)

    except KeyboardInterrupt:
        log.warning("")
        log.warning("Interrupted by user (Ctrl-C) — shutting down gracefully …")
        log.warning("  %d/%d video(s) completed before interruption.", len(per_video_stats), len(videos))
        if per_video_stats:
            total_elapsed = time.time() - t_start
            stats_path = _OUTPUT_DIR / "final_stats.md"
            write_final_stats_md(stats_path, per_video_stats, total_elapsed)
            print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)
            log.warning("  Partial results written to: %s", stats_path)
        log.warning("  Re-run to process remaining videos.")
        sys.exit(130)  # standard exit code for Ctrl-C

    # Open 3D viewers from disk (default; skip with --no-view)
    if not args.no_view:
        view_npz("", _OUTPUT_DIR)

    # Final statistics
    total_elapsed = time.time() - t_start
    stats_path = _OUTPUT_DIR / "final_stats.md"
    write_final_stats_md(stats_path, per_video_stats, total_elapsed)

    # Rich stats table
    print_run_stats(per_video_stats, total_elapsed, init_elapsed, device)

    log.info("  Artifacts:")
    for v in per_video_stats:
        name = v.get("name", "?")
        log.info("    %s/  →  base_search.md  scene_captions.md  finetune_stats.md  distill_stats.md"
                 "  finetuned_search.md  comparison.md  3d_map/", name)
    log.info("")
    log.info("  Final statistics: %s", stats_path)
    log.info("")
    log.info("  Next steps:")
    log.info("    • Edge inference:  EdgeClassifier('edge_models/dino_demo.onnx', 'edge_models/gallery.npz')")
    log.info("    • Full stack:      make up")
    log.info("    • Fine-tune rerun: DINO_CHECKPOINT=<path> python demo.py")
    log.info("")
    _banner("Done — thank you for using selfsuvis!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("\nInterrupted — exiting.")
        sys.exit(130)
