from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pipeline.config import settings
from pipeline.logging_utils import get_logger


class DINOEmbedder:
    def __init__(self, model_name: str = "dinov2_vitb14"):
        self.logger = get_logger(__name__)
        self.device = self._resolve_device()
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Cache the embedding dimension so we can return a correct empty array
        # without a forward pass. Dimension varies by model variant (e.g. 768 for
        # vitb14, 1024 for vitl14).
        dummy = Image.new("RGB", (224, 224))
        self._embed_dim: int = self.encode_images([dummy], batch_size=1).shape[1]
        self.logger.info("DINO loaded: %s on %s (dim=%d)", model_name, self.device, self._embed_dim)

    def _resolve_device(self) -> str:
        if settings.DEVICE == "cpu":
            return "cpu"
        if settings.DEVICE == "cuda":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self, model_name: str):
        try:
            if "dinov3" in model_name:
                repo = "facebookresearch/dinov3"
            else:
                repo = "facebookresearch/dinov2"
            model = torch.hub.load(repo, model_name, pretrained=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINO model. Ensure weights are available offline or predownloaded."
            ) from exc
        return model.to(self.device)

    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            with torch.no_grad():
                if settings.USE_FP16 and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        feats = self.model(tensors)
                else:
                    feats = self.model(tensors)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.detach().cpu().numpy())
        if not embeddings:
            return np.zeros((0, self._embed_dim), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    def image_dim(self) -> int:
        return self._embed_dim
