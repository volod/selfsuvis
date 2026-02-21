import os
from typing import List, Optional

from pipeline.logging_utils import get_logger

logger = get_logger(__name__)


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def _parse_allowed_paths(val: Optional[str]) -> List[str]:
    """Parse ALLOWED_INDEX_PATHS as comma-separated list. Empty means no restriction."""
    if val is None or not val.strip():
        return []
    return [p.strip() for p in val.split(",") if p.strip()]


class Settings:
    DATA_DIR = _env("DATA_DIR", "./data")
    FRAMES_DIR = _env("FRAMES_DIR", os.path.join(DATA_DIR, "frames"))
    TILES_DIR = _env("TILES_DIR", os.path.join(DATA_DIR, "tiles"))
    VIDEOS_DIR = _env("VIDEOS_DIR", os.path.join(DATA_DIR, "videos"))

    MODEL_NAME = _env("MODEL_NAME", "openclip")
    OPENCLIP_MODEL = _env("OPENCLIP_MODEL", "ViT-B-16")
    OPENCLIP_PRETRAINED = _env("OPENCLIP_PRETRAINED", "openai")

    SAM_MODEL_TYPE = _env("SAM_MODEL_TYPE", "vit_h")
    SAM_CHECKPOINT = _env("SAM_CHECKPOINT", "")
    LABELS_FILE = _env("LABELS_FILE", os.path.join(DATA_DIR, "labels", "openclip_rich.txt"))

    DEVICE = _env("DEVICE", "auto")
    USE_FP16 = _env("USE_FP16", "true").lower() == "true"

    SAMPLE_FPS_BASE = float(_env("SAMPLE_FPS_BASE", "2"))
    SAMPLE_FPS_MIN = float(_env("SAMPLE_FPS_MIN", "0.5"))
    SAMPLE_FPS_MAX = float(_env("SAMPLE_FPS_MAX", "5"))

    HIST_THRESH = float(_env("HIST_THRESH", "0.25"))
    EMBED_DRIFT_THRESH = float(_env("EMBED_DRIFT_THRESH", "0.15"))
    MAX_GAP_SEC = float(_env("MAX_GAP_SEC", "10"))

    MOTION_LOW = float(_env("MOTION_LOW", "0.02"))
    MOTION_HIGH = float(_env("MOTION_HIGH", "0.08"))

    STAB_ENABLE = _env("STAB_ENABLE", "true").lower() == "true"
    STAB_SIZE = int(_env("STAB_SIZE", "64"))
    PHASECORR_MIN_RESPONSE = float(_env("PHASECORR_MIN_RESPONSE", "0.15"))
    STAB_MAX_SHIFT = float(_env("STAB_MAX_SHIFT", "12"))

    TILE_SIZE = int(_env("TILE_SIZE", "384"))
    STRIDE = int(_env("STRIDE", "256"))

    BLUR_LAPL_VAR_MIN_FRAME = float(_env("BLUR_LAPL_VAR_MIN_FRAME", "80"))
    BLUR_LAPL_VAR_MIN_TILE = float(_env("BLUR_LAPL_VAR_MIN_TILE", "60"))
    MEAN_INTENSITY_MIN = float(_env("MEAN_INTENSITY_MIN", "20"))
    MEAN_INTENSITY_MAX = float(_env("MEAN_INTENSITY_MAX", "235"))

    SKY_BLUE_RATIO_MAX = float(_env("SKY_BLUE_RATIO_MAX", "0.35"))
    EDGE_DENSITY_MIN = float(_env("EDGE_DENSITY_MIN", "0.02"))

    TILE_STD_MIN = float(_env("TILE_STD_MIN", "12"))
    TILE_ENTROPY_MIN = float(_env("TILE_ENTROPY_MIN", "3.5"))

    CELL_SIZE = int(_env("CELL_SIZE", str(STRIDE)))
    CELL_WINDOW_SEC = float(_env("CELL_WINDOW_SEC", "5"))

    PHASH_LRU_SIZE = int(_env("PHASH_LRU_SIZE", "50000"))
    PHASH_HAMMING_MAX = int(_env("PHASH_HAMMING_MAX", "6"))

    DEDUP_RECENT_TILES = int(_env("DEDUP_RECENT_TILES", "200000"))
    DEDUP_TTL_SEC = float(_env("DEDUP_TTL_SEC", "120"))
    DEDUP_COS_SIM_THRESH = float(_env("DEDUP_COS_SIM_THRESH", "0.95"))

    TILE_INDEX_IF_EMBED_DRIFT_GT = float(_env("TILE_INDEX_IF_EMBED_DRIFT_GT", "0.10"))
    MAX_TILES_PER_SEGMENT = int(_env("MAX_TILES_PER_SEGMENT", "200"))

    K_RETRIEVE = int(_env("K_RETRIEVE", "100"))
    K_RETURN = int(_env("K_RETURN", "20"))

    QDRANT_HOST = _env("QDRANT_HOST", "qdrant")
    QDRANT_PORT = int(_env("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "video_semantic")

    JOB_DB_PATH = _env("JOB_DB_PATH", os.path.join(DATA_DIR, "jobs.db"))

    # Security and limits
    ALLOWED_INDEX_PATHS = _parse_allowed_paths(os.getenv("ALLOWED_INDEX_PATHS"))
    MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024)  # 2 GB default
    MAX_DOWNLOAD_BYTES = _env_int("MAX_DOWNLOAD_BYTES", 2 * 1024 * 1024 * 1024)  # 2 GB default
    PRECHECK_URL_TIMEOUT = _env_int("PRECHECK_URL_TIMEOUT", 20)
    SQLITE_TIMEOUT = _env_float("SQLITE_TIMEOUT", 30.0)
    MAX_REDIRECTS = _env_int("MAX_REDIRECTS", 5)
    ALLOW_PRIVATE_URLS = _env("ALLOW_PRIVATE_URLS", "false").lower() == "true"

    # Worker and ffmpeg
    WORKER_POLL_INTERVAL = _env_float("WORKER_POLL_INTERVAL", 2.0)
    FFMPEG_TIMEOUT_SEC = _env_int("FFMPEG_TIMEOUT_SEC", 3600)  # 1 hour for long videos

    # Video extensions (indexing)
    VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".avi"})

    # API auth and rate limiting
    API_KEY = _env("API_KEY", "")
    RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 120)
    RATE_LIMIT_BURST = _env_int("RATE_LIMIT_BURST", 60)
    TRUST_PROXY_HEADERS = _env("TRUST_PROXY_HEADERS", "false").lower() == "true"

    # Input limits
    MAX_IMAGE_PIXELS = _env_int("MAX_IMAGE_PIXELS", 80_000_000)
    MAX_DIR_FILES = _env_int("MAX_DIR_FILES", 5000)
    MAX_DIR_BYTES = _env_int("MAX_DIR_BYTES", 50 * 1024 * 1024 * 1024)  # 50 GB
    MAX_DIR_DEPTH = _env_int("MAX_DIR_DEPTH", 10)


def validate_settings() -> None:
    """Validate critical settings at startup. Raises ValueError on invalid config."""
    if settings.QDRANT_PORT < 1 or settings.QDRANT_PORT > 65535:
        raise ValueError("QDRANT_PORT must be 1-65535")
    if settings.MODEL_NAME not in {"openclip", "dinov2", "dinov3"}:
        raise ValueError("MODEL_NAME must be openclip, dinov2, or dinov3")
    if settings.MAX_UPLOAD_BYTES < 0 or settings.MAX_DOWNLOAD_BYTES < 0:
        raise ValueError("MAX_UPLOAD_BYTES and MAX_DOWNLOAD_BYTES must be non-negative")
    if settings.FFMPEG_TIMEOUT_SEC < 1:
        raise ValueError("FFMPEG_TIMEOUT_SEC must be >= 1")
    if settings.TILE_SIZE < 1 or settings.STRIDE < 1:
        raise ValueError("TILE_SIZE and STRIDE must be >= 1")
    if settings.MOTION_LOW < 0 or settings.MOTION_HIGH < 0 or settings.MOTION_LOW > settings.MOTION_HIGH:
        raise ValueError("MOTION_LOW and MOTION_HIGH must be non-negative and MOTION_LOW <= MOTION_HIGH")
    if settings.SAMPLE_FPS_MIN <= 0 or settings.SAMPLE_FPS_MAX < settings.SAMPLE_FPS_MIN:
        raise ValueError("SAMPLE_FPS_MIN must be > 0 and SAMPLE_FPS_MAX >= SAMPLE_FPS_MIN")
    if settings.MAX_REDIRECTS < 0:
        raise ValueError("MAX_REDIRECTS must be >= 0")
    if settings.RATE_LIMIT_PER_MIN < 0 or settings.RATE_LIMIT_BURST < 0:
        raise ValueError("RATE_LIMIT_PER_MIN and RATE_LIMIT_BURST must be >= 0")
    if settings.MAX_IMAGE_PIXELS < 0:
        raise ValueError("MAX_IMAGE_PIXELS must be >= 0")
    if settings.MAX_DIR_FILES < 0 or settings.MAX_DIR_BYTES < 0 or settings.MAX_DIR_DEPTH < 0:
        raise ValueError("MAX_DIR_FILES/MAX_DIR_BYTES/MAX_DIR_DEPTH must be >= 0")
    logger.info("Settings validated successfully")


settings = Settings()
