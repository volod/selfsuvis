#!/usr/bin/env python3
"""selfsuvis end-to-end demo pipeline.

Runs the full perception stack on every video file in a source directory:
  A. Frame extraction + metadata
  B. Index frames into Qdrant vector store (or in-memory fallback)
  C. Base-model transformation test   → {video_dir}/base_search.md
  D. SSL DINOv3 fine-tuning           → {video_dir}/finetune_stats.md
  E. ONNX export + gallery build      → {video_dir}/edge_models/
  F. Fine-tuned model search test     → {video_dir}/finetuned_search.md
  G. Comparison + video description   → {video_dir}/comparison.md
  H. 3D sparse map (SfM or PCA)       → {video_dir}/3d_map/
  I. Interactive 3D viewers (one window per video)
  J. Final statistics                 → output/final_stats.md

Usage:
    python demo.py                            # default: data_test/videos/
    python demo.py --videos-dir /path/to/videos
    python demo.py --device cuda --epochs 5
    python demo.py --no-qdrant --no-sfm      # offline / CPU-only demo
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
        f"| Video | Frames | Index (s) | Finetune loss | SfM poses | Ckpt (MB) |",
        f"|-------|--------|-----------|---------------|-----------|-----------|",
    ]
    for v in per_video:
        lines.append(
            f"| {v['name']} | {v.get('frames',0)} | "
            f"{v.get('index_sec',0):.1f} | "
            f"{v.get('best_loss', float('nan')):.4f} | "
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
        f"| `finetune_stats.md` | SSL fine-tuning loss curve + config |",
        f"| `finetuned_search.md` | Nearest-neighbour results with fine-tuned DINOv3 |",
        f"| `comparison.md` | Base vs fine-tuned stats + video description |",
        f"| `checkpoints/dino_ssl_best.pt` | Fine-tuned backbone weights (PyTorch) |",
        f"| `edge_models/dino_demo.onnx` | ONNX export for on-device inference |",
        f"| `edge_models/gallery.npz` | Embedding gallery for 1-NN classification |",
        f"| `3d_map/sparse_map.npz` | 3D point cloud (from SfM or PCA fallback) |",
        f"| `3d_map/map_stats.json` | Point count, SfM pose count, scene count |",
        f"",
        f"---",
        f"*Run `python demo.py --help` for all options.*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("✓ Final stats written to %s", output_path)


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


def step_export_model(
    checkpoint_path: str,
    frame_list: List[Tuple[str, float]],
    video_dir: Path,
    device: str,
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """Step E: export fine-tuned DINOv3 to ONNX + build gallery.npz."""
    edge_dir = video_dir / "edge_models"
    edge_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(edge_dir / "dino_demo.onnx")
    gallery_path = str(edge_dir / "gallery.npz")

    result: Dict[str, Any] = {"onnx_path": onnx_path, "gallery_path": gallery_path,
                               "onnx_mb": 0.0, "exported": False}

    # Load fine-tuned weights into DINOEmbedder
    dino: Optional[DINOEmbedder] = models.get("dino")
    if dino is None:
        log.warning("  DINO not available — skipping ONNX export")
        return result

    try:
        log.info("Loading fine-tuned checkpoint: %s", checkpoint_path)
        dino.load_backbone_checkpoint(checkpoint_path)
        log.info("  ✓ Checkpoint loaded")
    except Exception as exc:
        log.warning("  Could not load checkpoint (%s) — skipping ONNX export", exc)
        return result

    # ONNX export
    if not args.no_onnx:
        try:
            import torch
            backbone = dino._model.eval()  # raw backbone
            log.info("Exporting ONNX to %s …", onnx_path)
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            torch.onnx.export(
                backbone, dummy, onnx_path,
                opset_version=14,
                input_names=["pixel_values"],
                output_names=["embedding"],
                dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
                do_constant_folding=True,
            )
            onnx_mb = os.path.getsize(onnx_path) / 1e6
            result["onnx_mb"] = onnx_mb
            result["exported"] = True
            log.info("  ✓ ONNX export complete: %.1f MB → %s", onnx_mb, onnx_path)
            log.info("  Artifact: %s  (deploy on Jetson/Hailo-8/ARM)", onnx_path)
        except Exception as exc:
            log.warning("  ONNX export failed (%s) — skipping", exc)
    else:
        log.info("  ONNX export skipped (--no-onnx)")

    # Gallery build: group all frames under one pseudo-label "scene"
    log.info("Building embedding gallery from %d frames …", len(frame_list))
    try:
        # Sample up to 200 frames evenly for gallery
        step = max(1, len(frame_list) // 200)
        sampled = [fp for fp, _ in frame_list[::step]]
        labels_map = {"scene": sampled}
        build_gallery(
            labels_map=labels_map,
            output_path=gallery_path,
            backbone=dino._model if hasattr(dino, "_model") else None,
        )
        gallery_mb = os.path.getsize(gallery_path) / 1e6
        log.info("  ✓ Gallery built: %d embeddings → %s (%.1f MB)",
                 len(sampled), gallery_path, gallery_mb)
        log.info("  Artifact: %s  (use with EdgeClassifier for on-device 1-NN)", gallery_path)
    except Exception as exc:
        log.warning("  Gallery build failed (%s)", exc)

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

    return {"text_descriptions": text_descriptions,
            "base_infer_ms": base_infer_ms, "ft_infer_ms": ft_infer_ms}


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

_TOTAL_STEPS = 8


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
    }

    # ── A: Extract frames ──────────────────────────────────────────────────────
    _step(1, _TOTAL_STEPS, "Frame extraction")
    a = step_extract_frames(video_path, video_id, video_dir)
    frame_list: List[Tuple[str, float]] = a["frame_list"]
    stats["frames"] = a["meta"]["frame_count"]
    stats["duration_sec"] = a["meta"]["duration_sec"]

    if not frame_list:
        log.error("No frames extracted — skipping video %s", video_path.name)
        return stats

    # ── B: Index into store ────────────────────────────────────────────────────
    _step(2, _TOTAL_STEPS, "Vector store indexing")
    b = step_index_to_store(video_path, video_id, store, is_qdrant, models, frame_list)
    stats["index_sec"] = b["elapsed_sec"]

    # ── C: Base model search test ──────────────────────────────────────────────
    _step(3, _TOTAL_STEPS, "Base model transformation test → base_search.md")
    c = step_base_model_search_test(
        frame_list, store, is_qdrant, models, video_id, video_name, video_dir,
    )
    base_results = c["results"]
    query_frame  = c["query_frame"]
    query_t_sec  = c["query_t_sec"]

    # ── D: SSL fine-tuning ─────────────────────────────────────────────────────
    _step(4, _TOTAL_STEPS, "SSL DINOv3 fine-tuning → finetune_stats.md")
    d = step_ssl_finetune(video_id, video_name, video_dir, frame_list, device)
    stats["best_loss"] = d["best_loss"]
    stats["ckpt_mb"]   = d["ckpt_mb"]
    checkpoint_path    = d["checkpoint"]

    # ── E: Export model ────────────────────────────────────────────────────────
    _step(5, _TOTAL_STEPS, "ONNX export + gallery build → edge_models/")
    e = step_export_model(checkpoint_path, frame_list, video_dir, device, models)
    onnx_mb = e.get("onnx_mb", 0.0)

    # ── F: Fine-tuned model search test ───────────────────────────────────────
    _step(6, _TOTAL_STEPS, "Fine-tuned model transformation test → finetuned_search.md")
    f = step_finetuned_model_search_test(
        frame_list, store, is_qdrant, models,
        query_frame, query_t_sec, video_id, video_name, video_dir,
    )
    ft_results = f["results"]

    # ── G: Comparison + video description ────────────────────────────────────
    _step(7, _TOTAL_STEPS, "Model comparison + video description → comparison.md, description.md")
    step_compare_and_describe(
        frame_list, store, is_qdrant, base_results, ft_results,
        models, video_id, video_name, video_dir,
        stats["ckpt_mb"], onnx_mb,
    )

    # ── H: 3D map ─────────────────────────────────────────────────────────────
    _step(8, _TOTAL_STEPS, "3D map creation → 3d_map/sparse_map.npz + sparse_map.ply")
    h = step_create_3d_map(
        video_path, video_id, video_dir, frame_list, models,
        run_sfm_flag=not args.no_sfm,
    )
    stats["sfm_poses"]  = h["sfm_poses"]
    stats["map_method"] = h["method"]

    _banner(f"✓ Video complete: {video_path.name}")
    log.info("  All artifacts written to: %s", video_dir)
    log.info("  Summary:")
    log.info("    Frames extracted    : %d", stats["frames"])
    log.info("    Indexed (store)     : %d", b.get("indexed", 0))
    log.info("    Best finetune loss  : %.4f", stats["best_loss"])
    log.info("    Checkpoint          : %s", checkpoint_path)
    log.info("    ONNX export         : %s", e.get("onnx_path", "skipped"))
    log.info("    3D map method       : %s  (%d pts)", stats["map_method"], h["points"].shape[0] if h["points"] is not None else 0)
    log.info("    3D map PLY          : %s", h.get("ply_path", "n/a"))
    log.info("    Output dir          : %s", video_dir)

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

    models = init_models(device)
    store, is_qdrant = init_store(models, use_qdrant=not args.no_qdrant)

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
            log.warning("  Partial results written to: %s", stats_path)
        log.warning("  Re-run to process remaining videos.")
        sys.exit(130)  # standard exit code for Ctrl-C

    # Open 3D viewers from disk (default; skip with --no-view)
    if not args.no_view:
        view_npz("", _OUTPUT_DIR)

    # Final statistics
    total_elapsed = time.time() - t_start
    _banner("Final Statistics")
    stats_path = _OUTPUT_DIR / "final_stats.md"
    write_final_stats_md(stats_path, per_video_stats, total_elapsed)

    log.info("")
    log.info("Pipeline complete in %.1fs", total_elapsed)
    log.info("")
    log.info("Artifacts summary:")
    for v in per_video_stats:
        name = v.get("name", "?")
        log.info("  %s/  →  base_search.md  finetune_stats.md  finetuned_search.md  comparison.md  description.md  3d_map/", name)
    log.info("")
    log.info("Final statistics: %s", stats_path)
    log.info("")
    log.info("Next steps:")
    log.info("  • Load ONNX model for edge inference:")
    log.info("      from pipeline.edge_inference import EdgeClassifier")
    log.info("      clf = EdgeClassifier('data_test/output/<name>/edge_models/dino_demo.onnx',")
    log.info("                           'data_test/output/<name>/edge_models/gallery.npz')")
    log.info("  • Start the full API stack:  make up")
    log.info("  • Re-run with fine-tuned model:  DINO_CHECKPOINT=... python demo.py")
    log.info("")
    _banner("Done — thank you for using selfsuvis!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("\nInterrupted — exiting.")
        sys.exit(130)
