"""Knowledge distillation: fine-tuned DINOv3 ViT-B/14 teacher → ViT-S/14 student.

The teacher is the SSL fine-tuned backbone from step D.  The student is a smaller
DINOv2 ViT-S/14 (~22M params, 384-dim embeddings) vs the teacher's ViT-B/14
(~86M params, 768-dim).  After training the student is ~4× smaller and ~2× faster.

Training uses Relational Knowledge Distillation with Distance + Angle losses (RKD-DA)
plus a KoLeo spread regulariser and a cosine anchor loss:

    L = λ_D · L_RKD_dist  +  λ_A · L_RKD_angle  +  λ_kd · L_cosine  +  λ_koleo · L_KoLeo

RKD-DA preserves pairwise neighbourhood topology in the student embedding space,
which directly optimises retrieval Recall@K.  Defaults: λ_D=25, λ_A=50 (paper values).

A temporary projection head (Linear 384→768, orthogonal init) aligns the student output
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
from torch import Tensor
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
    # RKD-DA loss weights (Park et al. 2019 defaults)
    lambda_rkd_d: float = 25.0     # pairwise distance preservation
    lambda_rkd_a: float = 50.0     # triplet angle preservation
    lambda_kd: float = 1.0         # cosine anchor (student proj ≈ teacher)
    lambda_koleo: float = 0.1      # KoLeo spread regulariser


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


# ── RKD loss functions ────────────────────────────────────────────────────────

def _pairwise_dist(x: Tensor) -> Tensor:
    """Squared Euclidean pairwise distance matrix, clamped ≥ 0."""
    xx = (x * x).sum(dim=-1, keepdim=True)           # (B, 1)
    dist2 = xx + xx.T - 2.0 * (x @ x.T)             # (B, B)
    dist = dist2.clamp(min=1e-12).sqrt()              # (B, B)
    mask = torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
    return dist.masked_fill(mask, 0.0)


def _mean_off_diagonal(x: Tensor) -> Tensor:
    if x.shape[0] < 2:
        return x.new_tensor(1.0)
    mask = ~torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
    vals = x.masked_select(mask)
    if vals.numel() == 0:
        return x.new_tensor(1.0)
    return vals.mean().clamp(min=1e-8)


def rkd_distance_loss(t: Tensor, s: Tensor) -> Tensor:
    """RKD-D: preserve pairwise distance structure (Huber loss on normalised distances).

    Both t and s should be L2-normalised (B, D) tensors.
    """
    td = _pairwise_dist(t)
    sd = _pairwise_dist(s)
    # Normalise by mean so scale differences don't dominate
    td = td / _mean_off_diagonal(td)
    sd = sd / _mean_off_diagonal(sd)
    return F.huber_loss(sd, td, delta=1.0)


def rkd_angle_loss(t: Tensor, s: Tensor) -> Tensor:
    """RKD-A: preserve triplet angle structure (Huber loss on cosine angles).

    For each vertex j and all pairs (i, k), the angle ∠(i, j, k) is the cosine
    similarity between the unit vectors (j→i) and (j→k).

    Both t and s should be L2-normalised (B, D) tensors.
    Memory: O(B² · D) for diff tensors, O(B³) for angle tensor — fine for B≤64.
    """
    # Difference vectors: td[i,j] = t[i] - t[j]  →  (B, B, D)
    td = t.unsqueeze(1) - t.unsqueeze(0)   # (B, B, D)
    sd = s.unsqueeze(1) - s.unsqueeze(0)
    td_n = F.normalize(td, dim=-1, eps=1e-8)
    sd_n = F.normalize(sd, dim=-1, eps=1e-8)
    # angle[i, j, k] = cos(∠ i-j-k) = td_n[i,j] · td_n[k,j]
    # einsum: ijd, kjd -> ijk
    t_angle = torch.einsum('ijd,kjd->ijk', td_n, td_n)  # (B, B, B)
    s_angle = torch.einsum('ijd,kjd->ijk', sd_n, sd_n)
    return F.huber_loss(s_angle, t_angle, delta=1.0)


def koleo_loss(s: Tensor) -> Tensor:
    """KoLeo regulariser: maximise minimum pairwise distance → prevent collapse.

    L_KoLeo = -1/n · Σ log ||s_i − nn(s_i)||

    s should be L2-normalised (B, D).
    """
    if s.shape[0] < 2:
        return s.new_tensor(0.0)
    dist = _pairwise_dist(s)                              # (B, B)
    mask = torch.eye(s.shape[0], device=s.device, dtype=torch.bool)
    dist = dist.masked_fill(mask, float('inf'))
    nn_dist = dist.min(dim=-1).values.clamp(min=1e-8)    # (B,)
    return -torch.log(nn_dist).mean()


# ── Recall@1 metric ───────────────────────────────────────────────────────────

@torch.no_grad()
def recall_at_1(teacher_embs: Tensor, student_embs: Tensor) -> float:
    """Fraction of samples whose nearest neighbour in student space matches teacher space.

    Both inputs should be L2-normalised (N, D) tensors.
    Returns a float in [0, 1]; higher = student preserved teacher's local structure.
    """
    t_sim = teacher_embs @ teacher_embs.T
    t_sim.fill_diagonal_(float('-inf'))
    t_nn = t_sim.argmax(dim=-1)

    s_sim = student_embs @ student_embs.T
    s_sim.fill_diagonal_(float('-inf'))
    s_nn = s_sim.argmax(dim=-1)

    return (t_nn == s_nn).float().mean().item()


# ── Distiller ─────────────────────────────────────────────────────────────────

class KnowledgeDistiller:
    """Distils a large teacher backbone into a smaller student backbone using RKD-DA.

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

        # Param counts for compression ratio
        self._teacher_params = sum(p.numel() for p in self.teacher.parameters())
        self._student_params = sum(p.numel() for p in self.student.parameters())
        compression = self._teacher_params / max(self._student_params, 1)
        logger.info(
            "Compression: %.1f× (teacher=%dM, student=%dM)",
            compression,
            self._teacher_params // 1_000_000,
            self._student_params // 1_000_000,
        )

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
                with torch.amp.autocast('cuda'):
                    t = self.teacher(batch)
            else:
                t = self.teacher(batch)
        return F.normalize(t.float(), dim=-1)   # (B, t_dim)

    def _forward_student(self, batch: torch.Tensor) -> tuple[Tensor, Tensor]:
        """Run student + projection head with AMP; returns normalised embeddings."""
        if self.config.device == "cuda":
            with torch.amp.autocast('cuda'):
                s = self.student(batch)
        else:
            s = self.student(batch)
        s_proj = self._proj(s.float())            # (B, t_dim)
        return F.normalize(s_proj, dim=-1), F.normalize(s.float(), dim=-1)

    @torch.no_grad()
    def _collect_embeddings(self, loader: DataLoader) -> tuple[Tensor, Tensor]:
        """Collect all teacher and (raw, pre-proj) student embeddings for Recall@1."""
        self.student.eval()
        t_embs, s_embs = [], []
        for batch in loader:
            batch = batch.to(self.device)
            t_embs.append(self._forward_teacher(batch))
            if self.config.device == "cuda":
                with torch.amp.autocast('cuda'):
                    s = self.student(batch)
            else:
                s = self.student(batch)
            s_embs.append(F.normalize(s.float(), dim=-1))
        self.student.train()
        return torch.cat(t_embs, dim=0), torch.cat(s_embs, dim=0)

    # ── Public API ────────────────────────────────────────────────────────────

    def distill(
        self,
        frame_paths: List[str],
        checkpoint_dir: Path,
    ) -> Dict[str, Any]:
        """Train the student with RKD-DA + KoLeo + cosine anchor and save checkpoints.

        Args:
            frame_paths:     Absolute paths to training frames.
            checkpoint_dir:  Directory to write per-epoch + best checkpoints.

        Returns:
            dict with keys: best_path, best_loss, loss_history, loss_components,
                            recall_history, best_recall, compression_ratio,
                            elapsed, student_model, student_dim, teacher_dim.
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
        # Separate eval loader (no shuffle, no drop_last) for Recall@1
        eval_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device == "cuda"),
        )

        trainable = list(self.student.parameters()) + list(self._proj.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1,
        )

        best_loss = float("inf")
        best_recall = 0.0
        best_path = checkpoint_dir / "student_best.pt"
        loss_history: List[float] = []
        recall_history: List[float] = []
        loss_components: Dict[str, List[float]] = {
            "rkd_d": [], "rkd_a": [], "cosine": [], "koleo": [],
        }
        t0 = time.time()

        for epoch in range(1, cfg.epochs + 1):
            ep_losses: Dict[str, List[float]] = {k: [] for k in loss_components}
            ep_total: List[float] = []
            self.student.train()
            self._proj.train()

            for batch in loader:
                batch = batch.to(self.device)

                t_emb = self._forward_teacher(batch)              # (B, t_dim)
                s_proj, s_raw = self._forward_student(batch)      # (B, t_dim), (B, s_dim)

                # ── Loss components ───────────────────────────────────────────
                l_rkd_d  = rkd_distance_loss(t_emb, s_proj)
                l_rkd_a  = rkd_angle_loss(t_emb, s_proj)
                l_cosine = (1.0 - (s_proj * t_emb).sum(dim=-1)).mean()
                l_koleo  = koleo_loss(s_raw)

                loss = (cfg.lambda_rkd_d  * l_rkd_d
                      + cfg.lambda_rkd_a  * l_rkd_a
                      + cfg.lambda_kd     * l_cosine
                      + cfg.lambda_koleo  * l_koleo)

                if not torch.isfinite(loss):
                    logger.warning(
                        "Non-finite distillation loss at epoch %d; "
                        "rkd_d=%s rkd_a=%s cosine=%s koleo=%s. Skipping batch.",
                        epoch,
                        float(l_rkd_d.detach().float().cpu()),
                        float(l_rkd_a.detach().float().cpu()),
                        float(l_cosine.detach().float().cpu()),
                        float(l_koleo.detach().float().cpu()),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), cfg.grad_clip)
                optimizer.step()

                ep_losses["rkd_d"].append(l_rkd_d.item())
                ep_losses["rkd_a"].append(l_rkd_a.item())
                ep_losses["cosine"].append(l_cosine.item())
                ep_losses["koleo"].append(l_koleo.item())
                ep_total.append(loss.item())

            scheduler.step()

            if not ep_total:
                logger.warning("Distill %d/%d produced no valid batches; stopping early.",
                               epoch, cfg.epochs)
                break

            epoch_loss = float(np.mean(ep_total))
            loss_history.append(epoch_loss)
            for k in loss_components:
                loss_components[k].append(float(np.mean(ep_losses[k])))

            # Recall@1 on full dataset (student raw dim, no proj)
            t_all, s_all = self._collect_embeddings(eval_loader)
            r1 = recall_at_1(t_all, s_all)
            recall_history.append(r1)

            logger.info(
                "Distill %d/%d  total=%.4f  rkd_d=%.4f  rkd_a=%.4f  cos=%.4f  koleo=%.4f  R@1=%.3f",
                epoch, cfg.epochs, epoch_loss,
                loss_components["rkd_d"][-1], loss_components["rkd_a"][-1],
                loss_components["cosine"][-1], loss_components["koleo"][-1],
                r1,
            )

            # Save student backbone only (no projection head)
            epoch_path = checkpoint_dir / f"student_{epoch:03d}.pt"
            torch.save(self.student.state_dict(), str(epoch_path))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_recall = r1
                torch.save(self.student.state_dict(), str(best_path))
                logger.info("  ↳ best checkpoint saved (loss=%.4f  R@1=%.3f)", best_loss, r1)

        elapsed = time.time() - t0
        compression_ratio = self._teacher_params / max(self._student_params, 1)
        logger.info(
            "Distillation complete: %.1fs | best_loss=%.4f | best_R@1=%.3f | "
            "compression=%.1f× | student=%s (dim=%d, %dM params)",
            elapsed, best_loss, best_recall, compression_ratio,
            cfg.student_model, self._s_dim, self._student_params // 1_000_000,
        )
        return {
            "best_path":        str(best_path),
            "best_loss":        best_loss,
            "best_recall":      best_recall,
            "loss_history":     loss_history,
            "loss_components":  loss_components,
            "recall_history":   recall_history,
            "compression_ratio": compression_ratio,
            "teacher_params":   self._teacher_params,
            "student_params":   self._student_params,
            "elapsed":          elapsed,
            "student_model":    cfg.student_model,
            "student_dim":      self._s_dim,
            "teacher_dim":      self._t_dim,
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
        dict with keys: best_path, best_loss, best_recall, loss_history,
                        loss_components, recall_history, compression_ratio,
                        elapsed, student_model, student_dim, teacher_dim, distiller.
    """
    distiller = KnowledgeDistiller(teacher_backbone, config)
    stats = distiller.distill(frame_paths, checkpoint_dir)
    return {**stats, "distiller": distiller}
