import os
os.environ.setdefault("DEVICE", "cpu")

import logging
import open_clip

from models.dino_model import DINOEmbedder
from pipeline.config import settings


def download_openclip():
    model, _, _ = open_clip.create_model_and_transforms(
        settings.OPENCLIP_MODEL,
        pretrained=settings.OPENCLIP_PRETRAINED,
        device="cpu",
    )
    return model


def download_dino(model_name: str):
    _ = DINOEmbedder(model_name=model_name)


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    logger = logging.getLogger(__name__)
    logger.info("Downloading OpenCLIP weights...")
    download_openclip()
    model_name = os.getenv("DINO_MODEL", "dinov2_vitb14")
    if os.getenv("DOWNLOAD_DINO", "false").lower() == "true":
        logger.info("Downloading DINO weights: %s", model_name)
        download_dino(model_name)
    logger.info("Done")


if __name__ == "__main__":
    main()
