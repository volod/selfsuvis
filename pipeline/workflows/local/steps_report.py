"""All write_*_md functions, print_run_stats, and markdown helpers."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._common import (
    _log,
    _RUNNER_LABEL,
    _SCENE_CHANGE_THRESH,
    _analyze_caption_sequence,
    _jaccard,
)


# ── Markdown helpers ──────────────────────────────────────────────────────────

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


def write_finetune_stats_md(
    output_path: Path,
    video_name: str,
    cfg: Any,
    best_loss: float,
    checkpoint_path: str,
    elapsed_sec: float,
    loss_history: List[float],
) -> None:
    from .steps_ssl import _analyze_loss_curve, _loss_sparkline, _interpret_finetune_results

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
        f"python main.py --mode local --videos-dir data_test/videos",
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
        f"*Artifact produced by {_RUNNER_LABEL}. Student exported to `edge_models/dino_local.onnx`.*",
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
        f"- **`edge_models/dino_local.onnx`** — ONNX model for on-device inference (Jetson, Hailo-8)",
        f"- **`edge_models/gallery.npz`** — embedding gallery for 1-NN classification",
        f"- **`3d_map/`** — sparse 3D point cloud from Structure-from-Motion",
        f"",
        f"```python",
        f"from pipeline.training.edge_inference import EdgeClassifier",
        f"clf = EdgeClassifier('edge_models/dino_local.onnx', 'edge_models/gallery.npz')",
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
        f"# Local Full-Analysis Pipeline — Final Statistics",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total elapsed: {total_elapsed:.1f}s",
        f"Videos processed: {len(per_video)}",
        f"",
        f"## Step Timing",
        f"",
    ]
    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]
    header = "| Step | Type | " + " | ".join(names) + " | Total |"
    sep    = "|------|------|" + "|".join(["-------"] * len(names)) + "|-------|"
    lines += [header, sep]
    for key, label, comp_type in _STEP_LABELS:
        vals = [v.get("timings", {}).get(key, 0.0) for v in per_video]
        total_step = sum(vals)
        if total_step == 0 and key not in ("A_extract", "B_index"):
            continue
        dur_cells = " | ".join(_fmt_sec(s) for s in vals)
        lines.append(f"| {label} | {comp_type} | {dur_cells} | **{_fmt_sec(total_step)}** |")
    lines += [
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
        f"| `edge_models/dino_local.onnx` | ONNX export (student when distilled, teacher otherwise) |",
        f"| `edge_models/gallery.npz` | Embedding gallery for 1-NN classification |",
        f"| `asr_subtitles.md` | Whisper ASR segments + per-frame subtitle coverage (step M) |",
        f"| `multimodal_features.md` | OCR text, depth percentiles, detections, world model (steps N–Q) |",
        f"| `detailed_captions.md` | Qwen VLM detailed per-frame scene captions with ASR context (step R) |",
        f"| `unidrive_analysis.md` | UniDriveVLA understanding, perception, planning, and MoE consensus (step S) |",
        f"| `multi_model_comparison.md` | Gemma vs Qwen vs UniDriveVLA comparison and MoE agreement summary (step T) |",
        f"| `video_synthesis.md` | LLM video ontology + fine-grained narrative (step Z) |",
        f"| `agentic_flow.md` | Step-by-step agentic context trace, risk analysis, and context-propagation audit (step AA) |",
        f"| `video_ontology.json` | Structured ontology JSON (domain, environment, activities, objects) |",
        f"| `3d_map/sparse_map.npz` | 3D point cloud (from SfM or PCA fallback) |",
        f"| `3d_map/map_stats.json` | Point count, SfM pose count, scene count |",
        f"",
        f"---",
        f"*Run `python main.py --mode local --help` for all options.*",
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
    unidrive_result: Dict[str, Any],
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
        f"| UniDriveVLA expert analysis | {'✓' if not unidrive_result.get('skipped') else '—'} | "
        f"{unidrive_result.get('ok_count', 0)} frames analysed |",
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
    lines += ["---", f"*Produced by {_RUNNER_LABEL} · multimodal steps M–S*"]
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


def write_unidrive_analysis_md(
    output_path: Path,
    video_name: str,
    results: List[Dict[str, Any]],
    elapsed_sec: float,
    model_id: str,
) -> None:
    ok = sum(1 for r in results if not r.get("service_unavailable") and not r.get("parse_error"))
    lines = [
        f"# UniDriveVLA Expert Analysis — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_id}  |  Frames processed: {ok}/{len(results)}",
        f"Elapsed: {elapsed_sec:.1f}s",
        f"",
        f"| t (s) | Risk | Drivable | Expert Agreement | Understanding | Planning |",
        f"|-------|------|----------|------------------|---------------|----------|",
    ]
    for r in results:
        if r.get("service_unavailable"):
            lines.append(f"| {r.get('t_sec', 0.0):.1f} | — | — | — | *service unavailable* | — |")
            continue
        if r.get("parse_error"):
            lines.append(f"| {r.get('t_sec', 0.0):.1f} | — | — | — | *parse error* | — |")
            continue
        u = r.get("understanding", {}) or {}
        p = r.get("perception", {}) or {}
        plan = r.get("planning", {}) or {}
        moe = r.get("mixture_of_experts", {}) or {}
        understanding = (u.get("scene_summary", "") or "").replace("|", "\\|")[:70]
        planning = (plan.get("recommended_action", "") or "").replace("|", "\\|")[:70]
        lines.append(
            f"| {r.get('t_sec', 0.0):.1f} | {u.get('risk_level', 'unknown')} | "
            f"{p.get('drivable_area', 'unknown')} | {moe.get('expert_agreement', 'unknown')} | "
            f"{understanding} | {planning} |"
        )
    lines += ["", "## Mixture-of-Experts Consensus", ""]
    for r in results[:12]:
        moe = r.get("mixture_of_experts", {}) or {}
        consensus = (moe.get("consensus_summary", "") or "").strip()
        if not consensus:
            continue
        disagreements = moe.get("disagreement_points", []) or []
        dis_str = "; ".join(disagreements[:3]) if disagreements else "none"
        lines.append(
            f"- t={r.get('t_sec', 0.0):.1f}s: {consensus} "
            f"(agreement={moe.get('expert_agreement', 'unknown')}; disagreements: {dis_str})"
        )
    lines += ["", "---", f"*Produced by {_RUNNER_LABEL} · UniDriveVLA step S*"]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)


def write_multi_model_comparison_md(
    output_path: Path,
    video_name: str,
    gemma_result: Dict[str, Any],
    qwen_result: Dict[str, Any],
    unidrive_result: Dict[str, Any],
) -> Dict[str, Any]:
    qwen_rows = [r for r in qwen_result.get("results", []) if not r.get("service_unavailable")]
    uni_rows = [
        r for r in unidrive_result.get("results", [])
        if not r.get("service_unavailable") and not r.get("parse_error")
    ]

    def _nearest(rows: List[Dict[str, Any]], t_sec: float) -> Optional[Dict[str, Any]]:
        if not rows:
            return None
        return min(rows, key=lambda r: abs(float(r.get("t_sec", 0.0)) - t_sec))

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for u in uni_rows:
        q = _nearest(qwen_rows, float(u.get("t_sec", 0.0)))
        if q is None:
            continue
        if abs(float(u.get("t_sec", 0.0)) - float(q.get("t_sec", 0.0))) <= 2.0:
            pairs.append((q, u))

    agreement_scores: List[float] = []
    example_rows: List[Tuple[float, str, str, str, str]] = []
    for q, u in pairs[:10]:
        q_summary = str(q.get("scene_summary") or q.get("caption") or "")
        u_under = u.get("understanding", {}) or {}
        u_moe = u.get("mixture_of_experts", {}) or {}
        u_summary = str(u_under.get("scene_summary", "") or "")
        moe_summary = str(u_moe.get("consensus_summary", "") or "")
        agreement_scores.append(_jaccard(q_summary, u_summary or moe_summary))
        example_rows.append((
            float(u.get("t_sec", 0.0)),
            q_summary,
            u_summary,
            moe_summary,
            str(u_moe.get("expert_agreement", "unknown") or "unknown"),
        ))

    mean_agreement = float(np.mean(agreement_scores)) if agreement_scores else 0.0
    gemma_scene = ""
    task_results = gemma_result.get("task_results", {}) or {}
    clf = task_results.get("scene_classification", {}) or {}
    cat_dist = clf.get("category_distribution", {}) or {}
    if cat_dist:
        gemma_scene = next(iter(cat_dist))

    risk_levels = [((r.get("understanding") or {}).get("risk_level", "unknown")) for r in uni_rows]
    agreement_levels = [((r.get("mixture_of_experts") or {}).get("expert_agreement", "unknown")) for r in uni_rows]
    lines = [
        f"# Multi-Model Comparison — {video_name}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Coverage",
        f"",
        f"| Model family | Frames analysed | Primary output |",
        f"|-------------|-----------------|----------------|",
        f"| Gemma | {gemma_result.get('n_frames', 0)} | scene classification, clustering, cross-model probes |",
        f"| Qwen | {qwen_result.get('ok_count', 0)} | structured per-frame scene facts |",
        f"| UniDriveVLA | {len(uni_rows)} | understanding/perception/planning + MoE consensus |",
        f"",
        f"## Cross-Model Signals",
        f"",
        f"- Gemma dominant scene category: `{gemma_scene or 'unknown'}`",
        f"- Qwen ↔ UniDrive scene-summary token agreement: {mean_agreement:.3f} across {len(agreement_scores)} matched frames",
        f"- UniDrive risk profile: low={sum(1 for v in risk_levels if v == 'low')}, medium={sum(1 for v in risk_levels if v == 'medium')}, high={sum(1 for v in risk_levels if v == 'high')}",
        f"- UniDrive expert agreement: high={sum(1 for v in agreement_levels if v == 'high')}, medium={sum(1 for v in agreement_levels if v == 'medium')}, low={sum(1 for v in agreement_levels if v == 'low')}",
        f"",
        f"## Matched Examples",
        f"",
        f"| t (s) | Qwen summary | UniDrive understanding | UniDrive MoE consensus | Expert agreement |",
        f"|-------|--------------|------------------------|------------------------|------------------|",
    ]
    for t_sec, q_summary, u_summary, moe_summary, expert_agreement in example_rows:
        q_summary_md = q_summary.replace("|", "\\|")[:60]
        u_summary_md = u_summary.replace("|", "\\|")[:60]
        moe_summary_md = moe_summary.replace("|", "\\|")[:60]
        lines.append(
            f"| {t_sec:.1f} | {q_summary_md} | "
            f"{u_summary_md} | "
            f"{moe_summary_md} | {expert_agreement} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Qwen is the structured scene-facts baseline.",
        "- UniDrive adds explicit understanding, perception, and planning experts.",
        "- The UniDrive MoE consensus field is the best single input for downstream synthesis because it preserves both consensus and disagreement.",
        "",
        "---",
        f"*Produced by {_RUNNER_LABEL} · multi-model comparison step T*",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("  ✓ Written %s", output_path)
    return {
        "matched_frames": len(agreement_scores),
        "mean_qwen_unidrive_agreement": mean_agreement,
        "high_risk_frames": sum(1 for v in risk_levels if v == "high"),
    }


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

# (timing_key, step_label, computation_type)
# Ordered by typical execution sequence.
_STEP_LABELS: List[Tuple[str, str, str]] = [
    ("A_extract",         "A   Frame extraction",                  "I/O"           ),
    ("B_index",           "B   Vector store indexing",             "GPU embed"     ),
    ("J_gemma",           "J   Gemma multimodal analysis",         "LLM API"       ),
    ("L_caption",         "L   Scene captioning (Florence-2)",     "GPU vision"    ),
    ("M_asr",             "M   ASR (Whisper)",                     "GPU speech"    ),
    ("N_ocr",             "N   OCR (text extraction)",             "LLM API"       ),
    ("O_depth",           "O   Depth estimation",                  "GPU vision"    ),
    ("P_detection",       "P   Object detection",                  "GPU vision"    ),
    ("P2_yolo_sam",       "P2  YOLO11 + SAM2/3 detection",        "GPU vision"    ),
    ("P3_gemma_tracking", "P3  Gemma directed tracking",           "LLM API+GPU"  ),
    ("Q_world",           "Q   World model embeddings",            "GPU vision"    ),
    ("R_qwen",            "R   Qwen detailed captioning",          "LLM API"       ),
    ("S_unidrive",        "S   UniDriveVLA expert analysis",       "LLM API"       ),
    ("C_base_search",     "C   Base model search test",            "GPU embed"     ),
    ("I_3dmap",           "I   3D map (SfM + Gaussian Splat)",     "GPU 3D"        ),
    ("D_finetune",        "D   SSL DINOv3 fine-tuning",            "GPU train"     ),
    ("E_distill",         "E   Knowledge distillation",            "GPU train"     ),
    ("F_export",          "F   ONNX export + gallery",             "CPU"           ),
    ("G_ft_search",       "G   Fine-tuned search test",            "GPU embed"     ),
    ("H_compare",         "H   Model comparison + description",    "GPU embed"     ),
    ("T_multimodel",      "T   Multi-model comparison",            "GPU vision"    ),
    ("Z_synthesis",       "Z   Video synthesis (ontology+narr.)", "LLM API"       ),
    ("AA_agentic",        "AA  Agentic flow audit",                "LLM API"       ),
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
    from ._common import _banner

    # Column widths: step label, computation type, per-video durations, total
    LABEL_W = 34
    TYPE_W  = 14
    DUR_W   = 9
    n_vids  = len(per_video)
    W = LABEL_W + TYPE_W + DUR_W * (n_vids + 1) + 2
    SEP = "─" * W

    def _row(label: str, comp_type: str, *dur_cols: str) -> str:
        row = f"  {label:<{LABEL_W}}{comp_type:<{TYPE_W}}"
        for c in dur_cols:
            row += f"{c:>{DUR_W}}"
        return row

    _banner("RUN STATISTICS")
    _log.info("  Device       : %s", device.upper())
    _log.info("  Videos       : %d", len(per_video))
    total_frames   = sum(v.get("frames", 0) for v in per_video)
    total_duration = sum(v.get("duration_sec", 0.0) for v in per_video)
    _log.info("  Total frames : %d  (%.1f min of video)", total_frames, total_duration / 60)
    _log.info("  Total runtime: %s", _fmt_sec(total_elapsed))
    _log.info("")

    names = [v.get("name", f"video{i}") for i, v in enumerate(per_video)]
    _log.info("  STEP TIMING  (wall-clock per step)")
    _log.info("  " + SEP)
    _log.info(_row("Step", "Type", *(names + ["TOTAL"])))
    _log.info("  " + SEP)

    # Group by computation type for the subtotals
    by_type: Dict[str, float] = {}
    col_totals = [0.0] * n_vids
    grand_total = 0.0
    for key, label, comp_type in _STEP_LABELS:
        vals = [v.get("timings", {}).get(key, 0.0) for v in per_video]
        total_step = sum(vals)
        # Only show rows where at least one video ran this step
        if total_step > 0 or key in ("A_extract", "B_index"):
            _log.info(_row(label, comp_type, *[_fmt_sec(s) for s in vals], _fmt_sec(total_step)))
            for i, s in enumerate(vals):
                col_totals[i] += s
            grand_total += total_step
        by_type[comp_type] = by_type.get(comp_type, 0.0) + total_step
    _log.info("  " + SEP)
    _log.info(_row("TOTAL", "", *[_fmt_sec(s) for s in col_totals], _fmt_sec(grand_total)))

    _log.info("  " + SEP)
    pipeline_per_video = [v.get("pipeline_sec", 0.0) for v in per_video]
    _log.info(_row("Pipeline (steps sum)", "", *[_fmt_sec(s) for s in pipeline_per_video],
                   _fmt_sec(sum(pipeline_per_video))))
    overhead = total_elapsed - sum(pipeline_per_video) - init_elapsed
    _log.info(_row("Model initialisation", "", _fmt_sec(init_elapsed), *([""] * (n_vids - 1)), ""))
    _log.info(_row("Overhead (I/O, viewer, etc.)", "", *([""] * n_vids), _fmt_sec(max(0.0, overhead))))
    _log.info(_row("WALL CLOCK TOTAL", "", *([""] * n_vids), _fmt_sec(total_elapsed)))

    # ── Computation-type subtotals ─────────────────────────────────────────────
    _log.info("")
    _log.info("  COMPUTATION TYPE BREAKDOWN  (pipeline steps only)")
    _log.info("  " + "─" * (TYPE_W + DUR_W + LABEL_W + 2))
    TYPE_ORDER = ["I/O", "GPU embed", "GPU vision", "GPU speech", "GPU 3D",
                  "GPU train", "CPU", "LLM API", "LLM API+GPU"]
    for ct in TYPE_ORDER:
        t = by_type.get(ct, 0.0)
        if t > 0:
            pct = 100.0 * t / max(sum(by_type.values()), 1e-9)
            _log.info("  %-14s  %s  (%4.1f%%)", ct, _fmt_sec(t), pct)
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
