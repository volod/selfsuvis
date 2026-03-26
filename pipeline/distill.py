"""Knowledge distillation: fine-tuned DINOv3 ViT-B/14 teacher → ViT-S/14 student.

The teacher is the SSL fine-tuned backbone from step D.  The student is a smaller
DINOv2 ViT-S/14 (~22M params, 384-dim embeddings) vs the teacher's ViT-B/14
(~86M params, 768-dim).  After training the student is ~4× smaller and ~2× faster.

Training uses feature-level cosine distillation:
    L = mean(1 - cosine_similarity(proj(s_emb), t_emb.detach()))

A temporary projection head (Linear 384→768, no bias) aligns the student output
space with the teacher's during training.  It is discarded after training; only the
student backbone state dict is saved.  The student is initialised from pretrained
DINOv2 hub weights so it already has strong representations before distillation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pipeline.logging_utils import get_logger

logger = get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DistillConfig:
    """Hyperparameters for the distillation run."""
    student_model: str = "dinov2_vits14"   # ViT-S/14 — 22M params, 384-dim
    epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    image_size: int = 224
    device: str = "cuda"
    num_workers: int = 0


# ── Dataset ───────────────────────────────────────────────────────────────────

class _FrameDataset(Dataset):
    """Minimal dataset: loads frames from disk as normalised tensors."""

    def __init__(self, frame_paths: List[str], image_size: int = 224) -> None:
        self.paths = frame_paths
        self.transform = transforms.Compose([
            transforms.Resize(image_size,
                               interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ── Distiller ─────────────────────────────────────────────────────────────────

class KnowledgeDistiller:
    """Distils a large teacher backbone into a smaller student backbone.

    Args:
        teacher:  Fine-tuned PyTorch backbone in eval mode (weights frozen).
        config:   DistillConfig hyperparameters.
    """

    def __init__(self, teacher: torch.nn.Module, config: DistillConfig) -> None:
        self.config = config
        self.device = config.device

        # Teacher — frozen for the entire distillation run
        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Infer teacher output dimension
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self._t_dim: int = int(self.teacher(_dummy).shape[-1])
        logger.info("Teacher: %s  dim=%d (frozen)", type(teacher).__name__, self._t_dim)

        # Student backbone (pretrained, smaller)
        self.student = self._load_student()
        with torch.no_grad():
            self._s_dim: int = int(self.student(_dummy).shape[-1])
        logger.info("Student: %s  dim=%d (trainable)", config.student_model, self._s_dim)

        # Projection head (used only during training, then discarded)
        self._proj = nn.Linear(self._s_dim, self._t_dim, bias=False).to(self.device)
        nn.init.orthogonal_(self._proj.weight)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_student(self) -> torch.nn.Module:
        from models.dino_model import hub_load_dino
        logger.info("Loading student backbone: %s …", self.config.student_model)
        model = hub_load_dino(self.config.student_model, pretrained=True)
        return model.to(self.device).train()

    def _forward_teacher(self, batch: torch.Tensor) -> torch.Tensor:
        """Run teacher with AMP on CUDA; returns normalised float32 embeddings."""
        with torch.no_grad():
            if self.config.device == "cuda":
                with torch.cuda.amp.autocast():
                    t = self.teacher(batch)
            else:
                t = self.teacher(batch)
        return F.normalize(t.float(), dim=-1)   # (B, t_dim)

    def _forward_student(self, batch: torch.Tensor) -> torch.Tensor:
        """Run student + projection head with AMP; returns normalised embeddings."""
        if self.config.device == "cuda":
            with torch.cuda.amp.autocast():
                s = self.student(batch)
        else:
            s = self.student(batch)
        s_proj = self._proj(s.float())            # (B, t_dim)
        return F.normalize(s_proj, dim=-1)

    # ── Public API ────────────────────────────────────────────────────────────

    def distill(
        self,
        frame_paths: List[str],
        checkpoint_dir: Path,
    ) -> Dict[str, Any]:
        """Train the student and save checkpoints.

        Args:
            frame_paths:     Absolute paths to training frames.
            checkpoint_dir:  Directory to write per-epoch + best checkpoints.

        Returns:
            dict with keys: best_path, best_loss, loss_history, elapsed,
                            student_model, student_dim, teacher_dim.
        """
        cfg = self.config
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        dataset = _FrameDataset(frame_paths, cfg.image_size)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=len(dataset) > cfg.batch_size,
            pin_memory=(cfg.device == "cuda"),
        )

        trainable = list(self.student.parameters()) + list(self._proj.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1,
        )

        best_loss = float("inf")
        best_path = checkpoint_dir / "student_best.pt"
        loss_history: List[float] = []
        t0 = time.time()

        for epoch in range(1, cfg.epochs + 1):
            epoch_losses: List[float] = []
            self.student.train()
            self._proj.train()

            for batch in loader:
                batch = batch.to(self.device)

                t_emb = self._forward_teacher(batch)   # (B, t_dim) — no grad
                s_emb = self._forward_student(batch)   # (B, t_dim)

                # Cosine distillation loss: 0 = perfect alignment, 2 = opposite
                loss = (1.0 - (s_emb * t_emb).sum(dim=-1)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), cfg.grad_clip)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            epoch_loss = float(np.mean(epoch_losses))
            loss_history.append(epoch_loss)
            logger.info("Distill epoch %d/%d  loss=%.4f", epoch, cfg.epochs, epoch_loss)

            # Save student backbone only (no projection head)
            epoch_path = checkpoint_dir / f"student_{epoch:03d}.pt"
            torch.save(self.student.state_dict(), str(epoch_path))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.student.state_dict(), str(best_path))
                logger.info("  ↳ best checkpoint saved (loss=%.4f)", best_loss)

        elapsed = time.time() - t0
        logger.info(
            "Distillation complete: %.1fs | best_loss=%.4f | student=%s (dim=%d → %d params)",
            elapsed, best_loss, cfg.student_model, self._s_dim,
            sum(p.numel() for p in self.student.parameters()),
        )
        return {
            "best_path":     str(best_path),
            "best_loss":     best_loss,
            "loss_history":  loss_history,
            "elapsed":       elapsed,
            "student_model": cfg.student_model,
            "student_dim":   self._s_dim,
            "teacher_dim":   self._t_dim,
        }

    def student_backbone(self) -> torch.nn.Module:
        """Return the trained student backbone in eval mode (projection head discarded)."""
        self.student.eval()
        return self.student


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_distillation(
    teacher_backbone: torch.nn.Module,
    frame_paths: List[str],
    checkpoint_dir: Path,
    config: DistillConfig,
) -> Dict[str, Any]:
    """Distil teacher into a smaller student and save checkpoints.

    Args:
        teacher_backbone: Fine-tuned PyTorch backbone (will be frozen).
        frame_paths:      Absolute paths to frames used for training.
        checkpoint_dir:   Directory for student checkpoints.
        config:           DistillConfig instance.

    Returns:
        dict with keys: best_path, best_loss, loss_history, elapsed,
                        student_model, student_dim, teacher_dim, distiller.
    """
    distiller = KnowledgeDistiller(teacher_backbone, config)
    stats = distiller.distill(frame_paths, checkpoint_dir)
    return {**stats, "distiller": distiller}
