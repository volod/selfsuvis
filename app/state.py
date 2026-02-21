from PIL import Image

from models.dino_model import DINOEmbedder
from models.openclip_model import OpenCLIPEmbedder
from pipeline.config import settings, validate_settings
from pipeline.job_db import init_db
from pipeline.logging_utils import get_logger
from pipeline.processed_db import init_db as init_processed_db
from pipeline.qdrant_utils import QdrantStore

logger = get_logger(__name__)

init_db()
init_processed_db()
validate_settings()

if settings.MAX_IMAGE_PIXELS > 0:
    Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

clip_model = OpenCLIPEmbedder()
dino_model = None
if settings.MODEL_NAME in {"dinov2", "dinov3"}:
    try:
        dino_model = DINOEmbedder("dinov2_vitb14")
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("DINO model failed to load, disabling: %s", exc)
        dino_model = None

store = QdrantStore(
    clip_dim=clip_model.image_dim(),
    dino_dim=dino_model.image_dim() if dino_model else None,
)
