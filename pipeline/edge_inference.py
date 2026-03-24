"""Edge model hydration — ONNX-based inference wrapper for on-device object identification.

Exports the fine-tuned DINOv3 backbone to ONNX and provides a lightweight cosine-similarity
nearest-neighbour classifier (EdgeClassifier) that runs on edge hardware (Jetson Orin, Hailo-8,
CPU-only ARM SBC) without requiring PyTorch or CUDA.

Key classes and functions:
  EdgeClassifier  — loads quantized ONNX model + gallery NPZ; classifies PIL images.
  build_gallery   — embeds representative frames and saves an NPZ gallery file.

No PyTorch or torchvision imports at module level — only inside from_torch and helper
functions that are only called with a PyTorch backbone. This keeps the edge deployment
free of PyTorch.

Usage (on robot):
    from pipeline.edge_inference import EdgeClassifier
    clf = EdgeClassifier("dino_edge_int8.onnx", "mission_objects.npz")
    labels = clf.classify(frame_pil)   # [(label, score), ...]
"""

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalisation constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_image(image_pil: Image.Image, image_size: int = 224) -> np.ndarray:
    """Preprocess a PIL image into a float32 NCHW numpy array.

    Steps: resize shortest side → centre crop → normalise (ImageNet stats).

    Args:
        image_pil:  Input PIL image (any mode, will be converted to RGB).
        image_size: Output spatial resolution (default 224).

    Returns:
        numpy array of shape (1, 3, image_size, image_size), dtype float32.
    """
    img = image_pil.convert("RGB")

    # Resize shortest side to image_size (bicubic)
    w, h = img.size
    if w < h:
        new_w = image_size
        new_h = int(h * image_size / w)
    else:
        new_h = image_size
        new_w = int(w * image_size / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # Centre crop to image_size × image_size
    w, h = img.size
    left = (w - image_size) // 2
    top = (h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))

    # Convert to float32 HWC in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    # ImageNet normalisation
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD  # (H, W, 3)

    # HWC → CHW → NCHW
    arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]  # (1, 3, H, W)
    return arr.astype(np.float32)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D float32 array along the last axis."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


# ── EdgeClassifier ────────────────────────────────────────────────────────────

class EdgeClassifier:
    """Cosine-similarity nearest-neighbour classifier backed by an ONNX backbone.

    Loads a (quantized) ONNX model produced by scripts/export_onnx.py and a gallery
    NPZ file produced by build_gallery or scripts/build_gallery.py.  No PyTorch
    dependency at inference time — only onnxruntime and numpy.

    Args:
        onnx_path:    Path to the ONNX model file.
        gallery_path: Path to the gallery NPZ (keys: embeddings, labels, label_names).
        top_k:        Maximum number of (label, score) pairs to return from classify().
        device:       Inference device string passed to onnxruntime providers.
                      "cpu" → CPUExecutionProvider; "cuda" → CUDAExecutionProvider.
    """

    def __init__(
        self,
        onnx_path: str,
        gallery_path: str,
        top_k: int = 3,
        device: str = "cpu",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for EdgeClassifier. "
                "Install it with: pip install onnxruntime  (CPU) or  pip install onnxruntime-gpu  (CUDA)."
            ) from exc

        self.top_k = top_k
        self._device = device

        # Select providers
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name: str = self._session.get_inputs()[0].name
        logger.info("EdgeClassifier: loaded ONNX model from %s", onnx_path)

        # Load gallery
        data = np.load(gallery_path, allow_pickle=True)
        self._gallery_embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        self._gallery_labels: np.ndarray = data["labels"]
        logger.info(
            "EdgeClassifier: gallery loaded from %s  (%d embeddings, dim=%d)",
            gallery_path,
            len(self._gallery_embeddings),
            self._gallery_embeddings.shape[1] if self._gallery_embeddings.ndim == 2 else -1,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, image_pil: Image.Image) -> np.ndarray:
        """Embed a PIL image using the ONNX backbone.

        Args:
            image_pil: Input PIL image.

        Returns:
            L2-normalised embedding vector of shape (D,), float32.
        """
        x = _preprocess_image(image_pil)  # (1, 3, 224, 224)
        outputs = self._session.run(None, {self._input_name: x})
        vec = outputs[0].astype(np.float32)  # (1, D)
        vec = _l2_normalise(vec)
        return vec[0]  # (D,)

    def classify(self, image_pil: Image.Image) -> List[Tuple[str, float]]:
        """Classify a PIL image against the gallery.

        Args:
            image_pil: Input PIL image.

        Returns:
            List of (label, cosine_score) pairs, sorted by score descending.
            At most top_k results. Scores are in [-1, 1].
        """
        query = self.embed(image_pil)  # (D,)
        # Cosine similarity: query (D,) · gallery (N, D)^T → (N,)
        scores: np.ndarray = self._gallery_embeddings @ query  # (N,)

        k = min(self.top_k, len(scores))
        # Get indices of top-k scores (partial sort)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(str(self._gallery_labels[i]), float(scores[i])) for i in top_idx]

    # ── Alternative constructor (for testing / dev without ONNX) ─────────────

    @classmethod
    def from_torch(
        cls,
        backbone,
        gallery_path: str,
        top_k: int = 3,
        device: str = "cpu",
    ) -> "EdgeClassifier":
        """Create an EdgeClassifier that wraps a PyTorch backbone instead of ONNX.

        Convenience constructor for testing and development when an ONNX file has not
        been exported yet. Requires PyTorch and torchvision; NOT suitable for edge deploy.

        Args:
            backbone:     Pretrained/fine-tuned PyTorch backbone (e.g. DINOv3 ViT).
            gallery_path: Path to gallery NPZ.
            top_k:        Maximum results to return.
            device:       PyTorch device string.

        Returns:
            EdgeClassifier-compatible object with the same embed / classify interface.
        """
        import torch
        import torch.nn.functional as F

        instance = cls.__new__(cls)
        instance.top_k = top_k
        instance._device = device
        instance._session = None  # no ONNX session

        # Load gallery
        data = np.load(gallery_path, allow_pickle=True)
        instance._gallery_embeddings = data["embeddings"].astype(np.float32)
        instance._gallery_labels = data["labels"]

        # Store PyTorch backbone
        instance._torch_backbone = backbone.to(device).eval()

        def _embed_torch(self_inner: "EdgeClassifier", image_pil: Image.Image) -> np.ndarray:
            """Torch-based embed used by from_torch instances."""
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = image_pil.convert("RGB")
            x = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
            with torch.no_grad():
                out = self_inner._torch_backbone(x)  # (1, D)
            out = F.normalize(out, dim=-1)
            return out[0].cpu().numpy().astype(np.float32)  # (D,)

        import types
        instance.embed = types.MethodType(_embed_torch, instance)

        logger.info(
            "EdgeClassifier.from_torch: gallery loaded from %s  (%d embeddings)",
            gallery_path, len(instance._gallery_embeddings),
        )
        return instance


# ── Gallery builder ───────────────────────────────────────────────────────────

def build_gallery(
    labels_map: Dict[str, List[str]],
    output_path: str,
    onnx_path: Optional[str] = None,
    backbone=None,
    image_size: int = 224,
) -> None:
    """Embed representative frames and save a gallery NPZ.

    Exactly one of onnx_path or backbone must be provided.

    Args:
        labels_map:   Mapping of label name → list of frame file paths.
                      e.g. {"vehicle": ["path1.jpg", "path2.jpg"], "barrier": [...]}
        output_path:  Destination path for the NPZ file.
        onnx_path:    Path to ONNX model (used if provided; no PyTorch required).
        backbone:     PyTorch backbone (used when onnx_path is None).
        image_size:   Spatial resolution for preprocessing (default 224).

    Raises:
        ValueError:        If labels_map is empty.
        FileNotFoundError: If any frame path in labels_map does not exist.
    """
    if not labels_map:
        raise ValueError("labels_map must not be empty.")

    # Validate all paths exist upfront
    for label, paths in labels_map.items():
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"Frame file not found for label '{label}': {p}"
                )

    # Build embedder
    if onnx_path is not None:
        clf = EdgeClassifier(onnx_path=onnx_path, gallery_path="", top_k=1)
        # Override gallery (not needed for embed)
        clf._gallery_embeddings = np.zeros((0, 1), dtype=np.float32)
        clf._gallery_labels = np.array([], dtype=object)

        def _embed_fn(img: Image.Image) -> np.ndarray:
            return clf.embed(img)
    elif backbone is not None:
        import torch
        import torch.nn.functional as F
        try:
            import torchvision.transforms as T
        except ImportError as exc:
            raise ImportError("torchvision is required when using a PyTorch backbone.") from exc

        _device = next(backbone.parameters()).device
        transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        backbone_eval = backbone.eval()

        def _embed_fn(img: Image.Image) -> np.ndarray:
            img_rgb = img.convert("RGB")
            x = transform(img_rgb).unsqueeze(0).to(_device)
            with torch.no_grad():
                out = backbone_eval(x)  # (1, D)
            out = F.normalize(out, dim=-1)
            return out[0].cpu().numpy().astype(np.float32)
    else:
        raise ValueError("Either onnx_path or backbone must be provided.")

    # Embed all frames
    all_embeddings: List[np.ndarray] = []
    all_labels: List[str] = []

    for label in sorted(labels_map.keys()):
        paths = labels_map[label]
        for p in paths:
            img = Image.open(p).convert("RGB")
            emb = _embed_fn(img)
            emb = _l2_normalise(emb.reshape(1, -1))[0]
            all_embeddings.append(emb)
            all_labels.append(label)
            logger.debug("build_gallery: embedded %s → label=%s", p, label)

    embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)  # (N, D)
    labels_arr = np.array(all_labels, dtype=object)
    label_names = np.array(sorted(labels_map.keys()), dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        labels=labels_arr,
        label_names=label_names,
    )
    logger.info(
        "build_gallery: saved %d embeddings (%d labels) → %s",
        len(embeddings), len(label_names), output_path,
    )
