"""Object detection model wrapper — RT-DETR, Grounding DINO, OmDet-Turbo.

Detects objects in frame images and stores the result in
``frame_facts_json["detections"]`` as a list of
``{"label": str, "confidence": float, "bbox_norm": [x1, y1, x2, y2]}`` dicts.
Bounding boxes are normalised to [0, 1] relative to image width/height.

Disabled by default (``DETECTION_ENABLED=false``).  Enable with:

    DETECTION_ENABLED=true DETECTION_MODEL=auto python worker/main.py

Top-10 detection models (small → large, override with ``DETECTION_MODEL``):

  1. facebook/detr-resnet-50              41 M  ~0.1 GB  classic COCO, fast
  2. PekingU/rtdetr_r50vd                 42 M  ~0.1 GB  RT-DETR, 108 FPS on T4
  3. PekingU/rtdetr_r101vd                76 M  ~0.2 GB  RT-DETR larger backbone
  4. omlab/omdet-turbo-swin-tiny-hf      108 M  ~0.3 GB  open-vocabulary
  5. omlab/omdet-turbo-swin-large-hf     218 M  ~0.5 GB  open-vocab, best speed
  6. IDEA-Research/grounding-dino-tiny   173 M  ~0.4 GB  text-guided, zero-shot
  7. IDEA-Research/grounding-dino-base   341 M  ~0.7 GB  text-guided, stronger
  8. microsoft/conditional-detr-resnet-101 62 M ~0.2 GB  faster convergence
  9. jozhang97/deta-swin-large           218 M  ~0.5 GB  63.5 COCO AP
 10. SenseTime/deformable-detr            40 M  ~0.1 GB  sparse attention DETR

CLI override::

    DETECTION_ENABLED=true DETECTION_MODEL=IDEA-Research/grounding-dino-tiny
    DETECTION_ENABLED=true DETECTION_LABELS="vehicle,person,weapon,infrastructure"
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from pipeline.config import settings
from pipeline.logging_utils import get_logger
from pipeline.model_registry import auto_select, detect_resources

logger = get_logger(__name__)


def _resolve_model_id() -> str:
    cfg = settings.DETECTION_MODEL.strip()
    if cfg and cfg.lower() != "auto":
        return cfg
    return auto_select("detection", detect_resources()) or "PekingU/rtdetr_r50vd"


class DetectionModel:
    """Object detector wrapping HuggingFace object-detection pipelines.

    For open-vocabulary models (Grounding DINO, OmDet), also accepts a
    ``candidate_labels`` list for zero-shot detection.
    """

    def __init__(self) -> None:
        self._pipe = None
        self._model_id: Optional[str] = None
        self._load_failed: bool = False

    def is_enabled(self) -> bool:
        return settings.DETECTION_ENABLED

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            self._model_id = _resolve_model_id()
        return self._model_id

    def detect_batch(
        self,
        images: List[Image.Image],
        candidate_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run detection on a batch of images. Returns one result dict per image."""
        if not self.is_enabled():
            return [{"detection_disabled": True}] * len(images)
        pipe = self._get_pipe()
        if pipe is None:
            return [{"detection_unavailable": True}] * len(images)
        return [self._detect_one(img, pipe, candidate_labels) for img in images]

    def detect(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.is_enabled():
            return {"detection_disabled": True}
        pipe = self._get_pipe()
        if pipe is None:
            return {"detection_unavailable": True}
        return self._detect_one(image, pipe, candidate_labels)

    def _detect_one(
        self,
        image: Image.Image,
        pipe,
        candidate_labels: Optional[List[str]],
    ) -> Dict[str, Any]:
        try:
            kwargs: Dict[str, Any] = {"threshold": settings.DETECTION_CONFIDENCE}
            if candidate_labels is not None:
                kwargs["candidate_labels"] = candidate_labels
            elif settings.DETECTION_LABELS:
                kwargs["candidate_labels"] = [
                    lbl.strip() for lbl in settings.DETECTION_LABELS.split(",") if lbl.strip()
                ]
            raw = pipe(image, **kwargs)
            if not isinstance(raw, list):
                raw = []
            w, h = image.size
            detections = []
            for det in raw:
                box = det.get("box", {})
                x1 = box.get("xmin", 0) / w
                y1 = box.get("ymin", 0) / h
                x2 = box.get("xmax", w) / w
                y2 = box.get("ymax", h) / h
                detections.append({
                    "label": det.get("label", ""),
                    "confidence": round(float(det.get("score", 0.0)), 4),
                    "bbox_norm": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                })
            return {"detections": detections, "detection_model": self.model_id}
        except Exception:
            logger.warning("Detection failed", exc_info=True)
            return {"detection_error": True}

    def _get_pipe(self):
        if self._pipe is not None:
            return self._pipe
        if self._load_failed:
            return None
        logger.info("Loading detection model: %s", self.model_id)
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            device = _get_device()
            torch_dtype = torch.float16 if settings.USE_FP16 and device != "cpu" else torch.float32
            self._pipe = hf_pipeline(
                "object-detection",
                model=self.model_id,
                device=device,
                torch_dtype=torch_dtype,
            )
            logger.info("Detection model loaded: %s on %s", self.model_id, device)
        except Exception:
            logger.warning(
                "Failed to load detection model %s — run: python scripts/prepare_models.py --detection",
                self.model_id, exc_info=True,
            )
            self._pipe = None
            self._load_failed = True
        return self._pipe


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
