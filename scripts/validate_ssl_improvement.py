#!/usr/bin/env python3
"""Multi-video SSL improvement validation harness.

Measures ΔR@1 (Recall@1) pre/post SSL fine-tuning across ≥3 diverse videos.

Diversity requirements (TODOS.md §validate_ssl_improvement):
  1. A daylight video (≥200 frames)
  2. A low-light or overcast video (≥200 frames)
  3. A video with ≥3 distinct GPS waypoints, real field footage (≥200 frames)

Methodology:
  - For each video: run 3 seeds; compute R@1 before and after SSL fine-tuning.
  - R@1 = fraction of queries whose top-1 nearest neighbour is the correct
    positive (temporal positive: next frame from same video).
  - ΔR@1 = R@1_post − R@1_pre. Median across seeds.
  - Gate: ΔR@1 > +0.02 on at least 2/3 videos.

Exit codes:
  0  Gate passed (ΔR@1 > +0.02 on ≥2/3 videos)
  1  Gate failed (insufficient improvement)
  2  Setup error (missing frames, CUDA unavailable when required, etc.)

Usage:
    python scripts/validate_ssl_improvement.py \\
        --video-dirs data/frames/daylight data/frames/lowlight data/frames/gps_multi \\
        --seeds 42 123 777 \\
        --epochs 5 \\
        --device cuda

    # Quick check with cpu (non-GPU environment):
    python scripts/validate_ssl_improvement.py \\
        --video-dirs data/frames/daylight data/frames/lowlight data/frames/gps_multi \\
        --device cpu --epochs 2
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_env_name = os.getenv("APP_ENV", "prod")
_env_file = Path(__file__).parent.parent / "env" / f"{_env_name}.env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_FRAMES = 200
_MIN_VIDEOS = 3
_SEEDS_DEFAULT = [42, 123, 777]
_GATE_DELTA = 0.02       # ΔR@1 must exceed this per-video
_GATE_VIDEOS = 2         # at least this many videos must pass


# ── Frame loading ─────────────────────────────────────────────────────────────

def _collect_frames(video_dir: Path) -> List[Path]:
    """Collect all JPEG/PNG frames under a directory, sorted."""
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        p for p in video_dir.rglob("*") if p.suffix.lower() in exts
    )
    return frames


def _load_embeddings(
    frame_paths: List[Path],
    backbone,
    transform,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed a list of frame paths using the given backbone.

    Args:
        frame_paths: Sorted list of image paths.
        backbone:    PyTorch nn.Module; outputs (B, dim) tensors.
        transform:   torchvision Transform to apply to each PIL image.
        device:      Torch device string.
        batch_size:  Inference batch size.

    Returns:
        float32 numpy array of shape (N, dim), L2-normalised.
    """
    import torch
    from PIL import Image

    backbone.eval()
    all_embs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(transform(img))
                except Exception:
                    logger.warning("Failed to load %s — skipping", p)
            if not tensors:
                continue
            import torch as _torch
            batch = _torch.stack(tensors).to(device)
            embs = backbone(batch)
            embs = embs.float().cpu().numpy()
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            all_embs.append(embs / norms)

    if not all_embs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(all_embs, axis=0).astype(np.float32)


# ── R@1 evaluation ────────────────────────────────────────────────────────────

def recall_at_1(embeddings: np.ndarray, window: int = 5) -> float:
    """Compute R@1 using temporal positives.

    For each frame i, its positive is any frame j in [i-window, i+window]
    (excluding i itself). R@1 is the fraction of frames whose top-1
    nearest neighbour (excluding self) is a temporal positive.

    Args:
        embeddings: (N, D) L2-normalised float32 array, frames in temporal order.
        window:     Temporal neighbourhood radius for positive definition.

    Returns:
        R@1 in [0, 1]. Returns 0.0 if N < 2.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    # Cosine similarity matrix
    sim = embeddings @ embeddings.T  # (N, N)
    np.fill_diagonal(sim, -2.0)  # exclude self

    hits = 0
    for i in range(n):
        nn_idx = int(np.argmax(sim[i]))
        if abs(nn_idx - i) <= window and nn_idx != i:
            hits += 1
    return hits / n


# ── Per-video SSL eval ────────────────────────────────────────────────────────

def evaluate_video(
    video_dir: Path,
    seed: int,
    device: str,
    epochs: int,
    model_name: str,
) -> Tuple[float, float]:
    """Run pre/post SSL evaluation for a single video and seed.

    Returns:
        (r1_pre, r1_post) tuple.
    """
    import torch
    from torchvision import transforms

    from pipeline.ssl_finetune import (
        DINOFineTuner,
        FinetuneConfig,
        build_eval_transform,
        run_finetune,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    frame_paths = _collect_frames(video_dir)
    if len(frame_paths) < 2:
        raise ValueError(f"{video_dir}: need at least 2 frames for R@1 eval")

    eval_transform = build_eval_transform()

    # ── Pre-SSL embeddings ────────────────────────────────────────────────────
    tuner_pre = DINOFineTuner(
        model_name=model_name,
        device=device,
        freeze_blocks=0,  # eval-only; no fine-tuning yet
    )
    tuner_pre.eval()
    embs_pre = _load_embeddings(frame_paths, tuner_pre.backbone, eval_transform, device)
    r1_pre = recall_at_1(embs_pre)

    # ── SSL fine-tuning ───────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as ckpt_dir:
        cfg = FinetuneConfig(
            frames_dir=str(video_dir),
            output_dir=ckpt_dir,
            model_name=model_name,
            approach="temporal",
            epochs=epochs,
            batch_size=16,
            lr=1e-5,
            seed=seed,
            device=device,
            num_workers=0,  # safer in subprocess context
        )
        best_ckpt = run_finetune(cfg)

        # ── Post-SSL embeddings ───────────────────────────────────────────────
        tuner_post = DINOFineTuner(
            model_name=model_name,
            device=device,
            freeze_blocks=0,
        )
        tuner_post.load_checkpoint(best_ckpt)
        tuner_post.eval()
        embs_post = _load_embeddings(frame_paths, tuner_post.backbone, eval_transform, device)
        r1_post = recall_at_1(embs_post)

    return r1_pre, r1_post


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--video-dirs",
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            f"Frame directories, one per video. At least {_MIN_VIDEOS} required. "
            f"Each must contain ≥{_MIN_FRAMES} frames."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=_SEEDS_DEFAULT,
        metavar="N",
        help="Random seeds (default: 42 123 777)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="SSL fine-tuning epochs per seed (default: 5)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto | cuda | cpu (default: auto)",
    )
    parser.add_argument(
        "--model-name",
        default="dinov3_vitb14",
        help="DINOv3 model name for ssl_finetune (default: dinov3_vitb14)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=_MIN_FRAMES,
        help=f"Minimum frames per video (default: {_MIN_FRAMES})",
    )
    args = parser.parse_args()

    # ── Resolve device ────────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch torchvision")
        return 2

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("Device: %s", device)

    # ── Validate video directories ────────────────────────────────────────────
    video_dirs: List[Path] = []
    for raw in args.video_dirs:
        p = Path(raw)
        if not p.is_dir():
            logger.error("Not a directory: %s", raw)
            return 2
        frames = _collect_frames(p)
        if len(frames) < args.min_frames:
            logger.error(
                "%s: only %d frames found (≥%d required)",
                raw, len(frames), args.min_frames,
            )
            return 2
        video_dirs.append(p)
        logger.info("Video %s: %d frames", p.name, len(frames))

    if len(video_dirs) < _MIN_VIDEOS:
        logger.error(
            "Need at least %d video directories; got %d",
            _MIN_VIDEOS, len(video_dirs),
        )
        return 2

    # ── Evaluate each video over all seeds ───────────────────────────────────
    video_results: Dict[str, Dict] = {}

    for video_dir in video_dirs:
        pre_scores: List[float] = []
        post_scores: List[float] = []

        for seed in args.seeds:
            logger.info(
                "Evaluating %s | seed=%d | device=%s | epochs=%d",
                video_dir.name, seed, device, args.epochs,
            )
            try:
                r1_pre, r1_post = evaluate_video(
                    video_dir, seed, device, args.epochs, args.model_name
                )
            except Exception as exc:
                logger.error("Evaluation failed for %s seed=%d: %s", video_dir.name, seed, exc)
                return 2

            delta = r1_post - r1_pre
            logger.info(
                "  %s seed=%d → R@1 pre=%.4f post=%.4f ΔR@1=%.4f",
                video_dir.name, seed, r1_pre, r1_post, delta,
            )
            pre_scores.append(r1_pre)
            post_scores.append(r1_post)

        median_pre  = float(np.median(pre_scores))
        median_post = float(np.median(post_scores))
        median_delta = median_post - median_pre
        gate_passed = median_delta > _GATE_DELTA

        video_results[video_dir.name] = {
            "r1_pre_seeds":   pre_scores,
            "r1_post_seeds":  post_scores,
            "r1_pre_median":  median_pre,
            "r1_post_median": median_post,
            "delta_median":   median_delta,
            "gate_passed":    gate_passed,
        }

        status = "PASS" if gate_passed else "FAIL"
        logger.info(
            "  %s | ΔR@1 median=%.4f | %s (gate: >%.2f)",
            video_dir.name, median_delta, status, _GATE_DELTA,
        )

    # ── Gate evaluation ───────────────────────────────────────────────────────
    n_passed = sum(1 for r in video_results.values() if r["gate_passed"])
    gate_ok  = n_passed >= _GATE_VIDEOS

    print("\n" + "=" * 60)
    print(f"SSL Improvement Gate: {n_passed}/{len(video_dirs)} videos pass ΔR@1 > +{_GATE_DELTA}")
    print("=" * 60)
    for name, r in video_results.items():
        status = "✓ PASS" if r["gate_passed"] else "✗ FAIL"
        print(
            f"  {status}  {name}: "
            f"pre={r['r1_pre_median']:.4f} "
            f"post={r['r1_post_median']:.4f} "
            f"Δ={r['delta_median']:+.4f}"
        )
    print()

    if gate_ok:
        print(
            f"GATE PASSED — SSL fine-tuning shows ΔR@1 > +{_GATE_DELTA} on "
            f"{n_passed}/{len(video_dirs)} videos.\n"
            "Proceed with GemmaSSLFinetuner (Phase 3) and GemmaVisionTeacher (Stage 0→1)."
        )
        return 0
    else:
        print(
            f"GATE FAILED — SSL fine-tuning improves only "
            f"{n_passed}/{len(video_dirs)} videos (need ≥{_GATE_VIDEOS}).\n"
            "Keep DINOv3→EfficientViT-S1 baseline (Phase 2) as the production edge model. "
            "Skip GemmaSSLFinetuner and GemmaVisionTeacher."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
