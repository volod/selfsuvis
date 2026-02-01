import os


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


class Settings:
    DATA_DIR = _env("DATA_DIR", "./data")
    FRAMES_DIR = _env("FRAMES_DIR", os.path.join(DATA_DIR, "frames"))
    TILES_DIR = _env("TILES_DIR", os.path.join(DATA_DIR, "tiles"))
    VIDEOS_DIR = _env("VIDEOS_DIR", os.path.join(DATA_DIR, "videos"))

    MODEL_NAME = _env("MODEL_NAME", "openclip")
    OPENCLIP_MODEL = _env("OPENCLIP_MODEL", "ViT-B-16")
    OPENCLIP_PRETRAINED = _env("OPENCLIP_PRETRAINED", "openai")

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


settings = Settings()
