import asyncio

from PIL import Image

from selfsuvis.models.dino_model import DINOEmbedder
from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.pipeline.core import get_dino_model_name, get_logger, settings, validate_settings
from selfsuvis.pipeline.storage.processed import init_db as init_processed_db
from selfsuvis.pipeline.storage import QdrantStore

logger = get_logger(__name__)

init_processed_db()
validate_settings()

if settings.MAX_IMAGE_PIXELS > 0:
    Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

if settings.MODEL_NAME == "gemma":
    from selfsuvis.models.gemma_model import GemmaEmbedder  # noqa: E402
    clip_model = GemmaEmbedder(  # type: ignore[assignment]
        model_id=settings.GEMMA_MODEL_ID,
        device=settings.DEVICE,
        use_bf16=settings.GEMMA_USE_BF16,
    )
else:
    clip_model = OpenCLIPEmbedder()

# ── Single authoritative DINOv3 checkpoint source ────────────────────────────
# Resolution order (most-to-least authoritative):
#   1. system_state.active_dino_checkpoint DB row   (set by worker after fine-tune)
#   2. DINO_CHECKPOINT env var                       (manual override / cold-start)
#   3. No checkpoint → load pretrained backbone only
#
# This ensures replicas and restarts always load the same checkpoint without
# relying on env var drift.  On DB unavailability, fall back to env var silently.

def _resolve_dino_checkpoint() -> str:
    """Return the authoritative DINOv3 checkpoint path from DB, falling back to env."""
    db_url = settings.DATABASE_URL
    if not db_url:
        return settings.DINO_CHECKPOINT

    try:
        import asyncpg

        async def _fetch():
            conn = await asyncpg.connect(db_url, timeout=5)
            try:
                row = await conn.fetchrow(
                    "SELECT value FROM system_state WHERE key = 'active_dino_checkpoint'"
                )
                return row["value"] if row else None
            finally:
                await conn.close()

        db_ckpt = asyncio.run(_fetch())
        if db_ckpt:
            logger.info("DINOv3 checkpoint from DB (overrides env): %s", db_ckpt)
            # Override settings so DINOEmbedder._load_model picks up the DB checkpoint.
            settings.DINO_CHECKPOINT = db_ckpt
    except Exception as exc:
        logger.warning("Could not read active_dino_checkpoint from DB (falling back to env): %s", exc)


dino_model = None
if settings.MODEL_NAME in {"dinov2", "dinov3"}:
    _resolve_dino_checkpoint()   # may update settings.DINO_CHECKPOINT from DB
    try:
        dino_name = get_dino_model_name(settings.MODEL_NAME)
        if dino_name is None:
            raise ValueError(f"Unsupported DINO model family: {settings.MODEL_NAME}")
        dino_model = DINOEmbedder(dino_name)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("DINO model failed to load, disabling: %s", exc)
        dino_model = None

store = QdrantStore(
    clip_dim=clip_model.image_dim(),
    dino_dim=dino_model.image_dim() if dino_model else None,
)
# Alias used by robot.py and admin.py
qdrant_store = store

# Lock serialising hot-reload of dino_model (POST /admin/reload-model).
# Reference assignment to dino_model is GIL-atomic; in-flight inference holds
# its captured reference and completes normally with old weights.
dino_model_lock: asyncio.Lock = asyncio.Lock()

# Lock preventing concurrent supervised fine-tune jobs from being enqueued.
# Held by _maybe_trigger_finetune while writing a new job to the DB.
_finetune_lock: asyncio.Lock = asyncio.Lock()
