from fastapi import APIRouter
from fastapi.responses import JSONResponse

from selfsuvis.app.api_utils import ERROR_RESPONSES
from selfsuvis.app.state import logger, store

router = APIRouter(tags=["health"])


@router.get("/health", responses={503: ERROR_RESPONSES[503]})
def health():
    """Health check for container orchestration. Verifies Qdrant connectivity."""
    try:
        store.client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return JSONResponse({"error": "service unavailable"}, status_code=503)
