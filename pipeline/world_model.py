"""World model wrapper for video scene understanding and future-state prediction.

Implements an embedding/feature interface for world models that understand
scene dynamics, physical plausibility, and temporal relationships in video.

Target: LeWorldModel (arxiv.org/abs/2603.19312v1, March 2026)
  "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
   from Pixels" — Maes, Le Lidec, Scieur, LeCun, Balestriero
  Architecture: JEPA. ~15M params, trains end-to-end from raw pixels using
  only two loss terms (next-embedding prediction + Gaussian regularizer).
  Plans 48× faster than foundation-model world models.  Latent space encodes
  physical quantities; detects physically implausible events.
  HuggingFace model ID: not yet released as of 2026-Q1.
  → Set WORLD_MODEL=<hf_id> once it appears on HuggingFace.

Current auto-selection hierarchy (``WORLD_MODEL=auto``):
  V-JEPA2-ViT-G → V-JEPA2-ViT-L → VideoMAEv2-Huge → VideoMAE-Large → ...
  (largest model fitting in available VRAM – 2 GB safety margin)

World model output stored in ``frame_facts_json["world_model"]``:
  {
    "embedding": null,           # full embedding (omitted if WORLD_MODEL_STORE_EMBED=false)
    "embedding_dim": 768,
    "model": "facebook/vjepa2-vitg-fpc64-256",
    "temporal_window_frames": 8  # how many frames were aggregated
  }

Disabled by default (``WORLD_MODEL_ENABLED=false``).

Top-10 video understanding models (small → large):

  1. google/vivit-b-16x2-kinetics400              86 M  ~0.2 GB  ViViT-B Kinetics-400
  2. facebook/timesformer-base-finetuned-k400    122 M  ~0.3 GB  divided space-time attn
  3. MCG-NJU/videomae-base                       122 M  ~0.3 GB  masked video autoencoding
  4. google/videoprism-base-f16r288              300 M  ~0.7 GB  Google dual-encoder
  5. MCG-NJU/videomae-large                      307 M  ~0.6 GB  stronger features
  6. facebook/vjepa2-vitl-fpc64-256              307 M  ~0.7 GB  V-JEPA2 ViT-L, 64 frames
  7. OpenGVLab/VideoMAEv2-Huge                   600 M  ~1.3 GB  VideoMAEv2 Huge
  8. facebook/vjepa2-vitg-fpc64-256              1.0 B  ~2.0 GB  V-JEPA2 ViT-G (strongest)
  9. OpenGVLab/InternVideo2-Stage2_1B-224p-f4    1.0 B  ~2.0 GB  video-language model
 10. nvidia/Cosmos-1.0-Autoregressive-4B          4.0 B  ~8.0 GB  physical world model

CLI override::

    WORLD_MODEL_ENABLED=true WORLD_MODEL=facebook/vjepa2-vitg-fpc64-256
    WORLD_MODEL_ENABLED=true WORLD_MODEL=MCG-NJU/videomae-large
    WORLD_MODEL_ENABLED=true WORLD_MODEL=nvidia/Cosmos-1.0-Autoregressive-4B
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from pipeline.config import settings
from pipeline.logging_utils import get_logger
from pipeline.model_registry import auto_select, detect_resources

logger = get_logger(__name__)

_VIDEOMAE_PREFIXES = ("MCG-NJU/videomae", "MCG-NJU/VideoMAE")


def _resolve_model_id() -> str:
    cfg = settings.WORLD_MODEL.strip()
    if cfg and cfg.lower() != "auto":
        return cfg
    resources = detect_resources()
    return auto_select("world_model", resources) or "MCG-NJU/videomae-base"


class WorldModel:
    """World model interface for scene understanding from video frames.

    Operates in *aggregated clip* mode: collects a buffer of consecutive kept
    frames (up to ``WORLD_MODEL_CLIP_FRAMES``) then computes a single world
    embedding for the clip.  The result is assigned to the representative
    (middle) frame's ``frame_facts_json``.

    For the arxiv 2603.19312 model, once its HuggingFace ID is known, set
    ``WORLD_MODEL=<hf_id>`` and re-run — the interface is model-agnostic.
    """

    def __init__(self) -> None:
        self._feature_extractor = None
        self._model = None
        self._model_id: Optional[str] = None
        self._frame_buffer: List[Image.Image] = []
        self._clip_frames = settings.WORLD_MODEL_CLIP_FRAMES
        self._load_failed: bool = False

    def is_enabled(self) -> bool:
        return settings.WORLD_MODEL_ENABLED

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = _resolve_model_id()
        return self._model_id

    def process_clip(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Extract world-model features from a list of consecutive frames.

        Returns a dict suitable for merging into ``frame_facts_json``.
        """
        if not self.is_enabled():
            return {"world_model_disabled": True}

        model, feat_extractor = self._load_model()
        if model is None:
            return {"world_model_unavailable": True}

        try:
            import torch
            import numpy as np

            # Sample up to clip_frames evenly spaced from the buffer
            n = len(images)
            if n == 0:
                return {"world_model_unavailable": True}
            indices = _sample_indices(n, self._clip_frames)
            sampled = [images[i] for i in indices]

            inputs = feat_extractor(sampled, return_tensors="pt")
            device = _get_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Extract mean-pooled last hidden state as embedding
            hidden = outputs.last_hidden_state  # (1, T, D) or (1, D)
            embedding_np = hidden.mean(dim=1).squeeze(0).cpu().float().numpy()

            result: Dict[str, Any] = {
                "world_model": {
                    "embedding_dim": int(embedding_np.shape[0]),
                    "model": self.model_id,
                    "temporal_window_frames": len(sampled),
                }
            }
            if settings.WORLD_MODEL_STORE_EMBED:
                result["world_model"]["embedding"] = embedding_np.tolist()
            return result

        except Exception:
            logger.warning("World model inference failed", exc_info=True)
            return {"world_model_error": True}

    def _load_model(self):
        if self._model is not None:
            return self._model, self._feature_extractor
        if self._load_failed:
            return None, None

        logger.info("Loading world model: %s", self.model_id)
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel

            device = _get_device()
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if settings.USE_FP16 and device != "cpu" else torch.float32,
            ).to(device).eval()
            logger.info("World model loaded: %s on %s", self.model_id, device)
        except Exception:
            logger.warning(
                "Failed to load world model %s — run: python scripts/prepare_models.py --world-model",
                self.model_id, exc_info=True,
            )
            self._model = None
            self._feature_extractor = None
            self._load_failed = True

        return self._model, self._feature_extractor


def _sample_indices(n: int, target: int) -> List[int]:
    """Return up to *target* evenly spaced indices from [0, n)."""
    if n <= target:
        return list(range(n))
    step = n / target
    return [int(i * step) for i in range(target)]


def _get_device() -> str:
    cfg = settings.DEVICE.lower()
    try:
        import torch
        if cfg == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if cfg == "cuda" and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
