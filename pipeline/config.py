import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from pipeline.logging_utils import get_logger

logger = get_logger(__name__)

# Load env file before reading settings. APP_ENV selects env/dev.env, env/test.env, or env/prod.env.
_env_name = os.getenv("APP_ENV", "dev")
_env_dir = Path(__file__).resolve().parent.parent / "env"
_env_file = _env_dir / f"{_env_name}.env"
if _env_file.exists():
    load_dotenv(_env_file)
else:
    load_dotenv()  # Fallback: project root .env if present


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

    # PostgreSQL (replaces SQLite in production)
    DATABASE_URL = _env("DATABASE_URL", "")

    # Reports and maps output directories
    REPORTS_DIR = _env("REPORTS_DIR", os.path.join(DATA_DIR, "reports"))
    MAPS_DIR = _env("MAPS_DIR", os.path.join(DATA_DIR, "maps"))

    # Pipeline — SfM and 3DGS
    SFM_FPS = _env_float("SFM_FPS", 2.0)
    PYCOLMAP_CAMERA_MODEL = _env("PYCOLMAP_CAMERA_MODEL", "SIMPLE_RADIAL")

    # Pipeline — Florence-2 captioning
    FLORENCE_BATCH_SIZE = _env_int("FLORENCE_BATCH_SIZE", 16)

    # Pipeline — GPS extraction
    GPS_SIDECAR_PATH = _env("GPS_SIDECAR_PATH", "")
    GPS_FILTER_2D = _env("GPS_FILTER_2D", "false").lower() == "true"

    # Active learning
    AL_TAG_K = _env_int("AL_TAG_K", 50)
    # Switch from KMeans to MiniBatchKMeans when total embedded frame count exceeds this.
    # Default 25_000 ≈ 50 missions × 500 frames.
    KMEANS_BATCH_THRESHOLD = _env_int("KMEANS_BATCH_THRESHOLD", 25_000)

    # Change detection cosine-distance thresholds (per model family)
    CHANGE_DETECTION_THRESHOLD_CLIP = _env_float("CHANGE_DETECTION_THRESHOLD_CLIP", 0.35)
    CHANGE_DETECTION_THRESHOLD_DINO = _env_float("CHANGE_DETECTION_THRESHOLD_DINO", 0.25)

    # Services
    STATIC_SERVER_URL = _env("STATIC_SERVER_URL", "http://localhost:8080")
    SUPERSPLAT_SERVER_URL = _env("SUPERSPLAT_SERVER_URL", "http://localhost:8090")
    NERFSTUDIO_API_URL = _env("NERFSTUDIO_API_URL", "http://nerfstudio:8000")
    # ICP fusion mapper service (docker-compose.override.yml, port 8100 on host / 8000 in container)
    MAPPER_API_URL = _env("MAPPER_API_URL", "http://mapper:8000")

    # CVAT annotation service (http://localhost:8091 when running via make cvat-up)
    CVAT_URL = _env("CVAT_URL", "http://localhost:8091")
    # HMAC-SHA256 secret for verifying CVAT webhook payloads (X-Hook-Secret header).
    # If empty, signature verification is skipped (dev mode only).
    CVAT_WEBHOOK_SECRET = _env("CVAT_WEBHOOK_SECRET", "")
    # JSON dict mapping CVAT label names to canonical vocabulary.
    # Example: '{"automobile":"car","person":"pedestrian"}'
    # Labels absent from the dict are passed through unchanged.
    # Applied in AnnotatedFrameDataset.from_xml() and from_db() to normalize
    # label names across CVAT tasks before building the class vocabulary.
    CVAT_LABEL_MAPPINGS: dict = __import__("json").loads(
        _env("CVAT_LABEL_MAPPINGS", "{}")
    )

    # Security and limits
    ALLOWED_INDEX_PATHS = _parse_allowed_paths(os.getenv("ALLOWED_INDEX_PATHS"))
    MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024)  # 2 GB default
    MAX_DOWNLOAD_BYTES = _env_int("MAX_DOWNLOAD_BYTES", 2 * 1024 * 1024 * 1024)  # 2 GB default
    PRECHECK_URL_TIMEOUT = _env_int("PRECHECK_URL_TIMEOUT", 20)
    SQLITE_TIMEOUT = _env_float("SQLITE_TIMEOUT", 30.0)
    MAX_REDIRECTS = _env_int("MAX_REDIRECTS", 5)
    ALLOW_PRIVATE_URLS = _env("ALLOW_PRIVATE_URLS", "false").lower() == "true"

    # Robot identity (used to tag Qdrant payloads for multi-robot filtering)
    ROBOT_ID = _env("ROBOT_ID", "robot_0")

    # Worker identity (used in gpu_jobs table for resource isolation)
    WORKER_ID = _env("WORKER_ID", __import__("socket").gethostname())
    # Maximum seconds a gpu_jobs entry may be held before it is considered stale.
    GPU_JOB_TIMEOUT_SEC = _env_int("GPU_JOB_TIMEOUT_SEC", 3600)

    # Model version provenance: ID stored in Qdrant frame payloads at embed/re-embed time.
    # Set automatically by the worker after a successful supervised fine-tune.
    # Override via env var to annotate frames embedded with a known checkpoint.
    MODEL_VERSION_ID = _env("MODEL_VERSION_ID", "base")

    # Self-supervised DINOv3 domain adaptation (scripts/finetune_dino.py)
    SSL_CHECKPOINT_DIR = _env("SSL_CHECKPOINT_DIR", os.path.join(DATA_DIR, "checkpoints"))
    # Supervised CVAT fine-tuning (scripts/supervised_finetune_dino.py)
    SUP_CHECKPOINT_DIR = _env("SUP_CHECKPOINT_DIR", os.path.join(DATA_DIR, "checkpoints", "supervised"))
    SSL_FINETUNE_EPOCHS = _env_int("SSL_FINETUNE_EPOCHS", 10)
    SSL_FINETUNE_LR = _env_float("SSL_FINETUNE_LR", 1e-5)
    SSL_FINETUNE_BATCH_SIZE = _env_int("SSL_FINETUNE_BATCH_SIZE", 32)
    SSL_FINETUNE_FREEZE_BLOCKS = _env_int("SSL_FINETUNE_FREEZE_BLOCKS", 10)
    SSL_FINETUNE_TEMPERATURE = _env_float("SSL_FINETUNE_TEMPERATURE", 0.07)
    # "temporal": consecutive frames from same video dir; "augment": two augmented views
    SSL_FINETUNE_APPROACH = _env("SSL_FINETUNE_APPROACH", "temporal")
    # Path to fine-tuned DINOv3 backbone weights. When set and the file exists,
    # DINOEmbedder loads this checkpoint instead of the pretrained hub weights.
    DINO_CHECKPOINT = _env("DINO_CHECKPOINT", "")

    SUP_FINETUNE_EPOCHS = _env_int("SUP_FINETUNE_EPOCHS", 10)
    SUP_FINETUNE_LR = _env_float("SUP_FINETUNE_LR", 1e-5)
    SUP_FINETUNE_BATCH_SIZE = _env_int("SUP_FINETUNE_BATCH_SIZE", 16)
    SUP_FINETUNE_FREEZE_BLOCKS = _env_int("SUP_FINETUNE_FREEZE_BLOCKS", 8)
    SUP_FINETUNE_TEMPERATURE = _env_float("SUP_FINETUNE_TEMPERATURE", 0.07)

    # Active learning loop closure
    # Whether the CVAT webhook auto-triggers supervised fine-tuning.
    SUP_AUTO_TRIGGER = _env("SUP_AUTO_TRIGGER", "true").lower() == "true"
    # Minimum annotated frames before any finetune job is enqueued.
    MIN_ANNOTATED_FRAMES = _env_int("MIN_ANNOTATED_FRAMES", 50)
    # Minimum new annotated frames since last retrain before re-triggering.
    MIN_NEW_ANNOTATED_SINCE_RETRAIN = _env_int("MIN_NEW_ANNOTATED_SINCE_RETRAIN", 100)
    # Batch size for re-embedding sweeps (frames per Qdrant upsert call).
    REEMBED_BATCH_SIZE = _env_int("REEMBED_BATCH_SIZE", 256)
    # CVAT API bearer token for fetching annotation labels via REST API.
    CVAT_API_TOKEN = _env("CVAT_API_TOKEN", "")
    # Fraction of annotated frames held out for eval gate (0.0 disables gate).
    SUP_EVAL_FRACTION = _env_float("SUP_EVAL_FRACTION", 0.1)
    # Minimum frames per class in the eval split (single-class → reject).
    SUP_MIN_PER_CLASS_EVAL = _env_int("SUP_MIN_PER_CLASS_EVAL", 2)
    # Minimum total frames required to run the eval gate; below this → reject.
    SUP_MIN_EVAL_GATE_FRAMES = _env_int("SUP_MIN_EVAL_GATE_FRAMES", 20)
    # Minimum 1-NN eval accuracy required to promote a checkpoint.
    SUP_EVAL_GATE_THRESHOLD = _env_float("SUP_EVAL_GATE_THRESHOLD", 0.6)
    # Intra-vs-inter cosine gap above this value triggers an overfitting warning (not a gate).
    SUP_OVERFITTING_SHIFT_THRESHOLD = _env_float("SUP_OVERFITTING_SHIFT_THRESHOLD", 0.9)

    # Edge model hydration (scripts/export_onnx.py, scripts/build_gallery.py, pipeline/edge_inference.py)
    EDGE_MODELS_DIR = _env("EDGE_MODELS_DIR", os.path.join(DATA_DIR, "models"))
    EDGE_GALLERY_DIR = _env("EDGE_GALLERY_DIR", os.path.join(DATA_DIR, "gallery"))
    EDGE_ONNX_PATH = _env("EDGE_ONNX_PATH", "")       # path to quantized ONNX for EdgeClassifier
    EDGE_GALLERY_PATH = _env("EDGE_GALLERY_PATH", "")  # path to gallery NPZ for EdgeClassifier
    EDGE_TOP_K = _env_int("EDGE_TOP_K", 3)

    # Worker and ffmpeg
    WORKER_POLL_INTERVAL = _env_float("WORKER_POLL_INTERVAL", 2.0)
    FFMPEG_TIMEOUT_SEC = _env_int("FFMPEG_TIMEOUT_SEC", 3600)  # 1 hour for long videos
    # Maximum RTSP recording duration in seconds (caps duration_sec in POST /index/rtsp)
    RTSP_MAX_DURATION_SEC = _env_int("RTSP_MAX_DURATION_SEC", 3600)

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
    if not settings.API_KEY:
        logger.warning(
            "API_KEY is not set; the API is unauthenticated and open to any caller. "
            "Set the API_KEY environment variable for production use."
        )
    if not settings.ALLOWED_INDEX_PATHS:
        logger.warning(
            "ALLOWED_INDEX_PATHS is not set; path-based indexing endpoints "
            "(/index/video path=, /index/dir, /index/precheck path=, /index/precheck_dir) "
            "are disabled. Set ALLOWED_INDEX_PATHS to a comma-separated list of allowed base directories."
        )
    logger.info("Settings validated successfully")


settings = Settings()
