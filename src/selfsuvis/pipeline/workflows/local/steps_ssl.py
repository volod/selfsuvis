"""SSL fine-tuning steps and loss analysis helpers."""


import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.training import FinetuneConfig, run_finetune
from ._common import _log


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


def step_ssl_finetune(
    video_id: str,
    video_name: str,
    video_dir: Path,
    frame_list: List,
    device: str,
    epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Step 16: SSL DINOv3 fine-tuning, write finetune_stats.md."""
    from .steps_report import write_finetune_stats_md

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

    import selfsuvis.pipeline.training.ssl as _ssl_mod

    def _run_capturing(c: FinetuneConfig) -> str:
        import torch, random
        random.seed(c.seed); torch.manual_seed(c.seed)
        os.makedirs(c.output_dir, exist_ok=True)
        from selfsuvis.pipeline.training.ssl import (
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
        patience = 3
        no_improve = 0
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
            if avg < best_loss:
                best_loss = avg
                tuner.save_checkpoint(best_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    _log.info(
                        "    Early stop at epoch %d/%d (no improvement for %d epochs)",
                        epoch, c.epochs, patience,
                    )
                    break
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
