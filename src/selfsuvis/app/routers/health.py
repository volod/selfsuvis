"""Enhanced /health endpoint — correlator heartbeat, SSE, DLQ, postgres, redis, adapters."""

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from selfsuvis.app.api_utils import ERROR_RESPONSES
from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", responses={503: ERROR_RESPONSES[503]})
async def health(request: Request):
    """Health check. Reports correlator heartbeat, SSE subscribers, DLQ depth, and service status."""
    status = "ok"
    details: dict = {}

    # PostgreSQL
    pool = getattr(request.app.state, "db_pool", None)
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            details["postgres"] = "ok"
        except Exception as exc:
            logger.warning("Health: postgres error: %s", exc)
            details["postgres"] = "error"
            status = "down"
    else:
        details["postgres"] = "unconfigured"

    # Qdrant (legacy check kept for backward compat)
    try:
        from selfsuvis.app.state import app_state

        app_state.store.client.get_collections()
        details["qdrant"] = "connected"
    except Exception:
        details["qdrant"] = "error"

    # Redis, correlator heartbeat, DLQ
    details["redis"] = "unconfigured"
    details["correlator_heartbeat_age_s"] = None
    details["dlq_depth"] = 0

    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.HEALTH_REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        details["redis"] = "ok"

        # Correlator heartbeat
        hb = await r.get("fusion:correlator:heartbeat")
        if hb:
            try:
                hb_ts = datetime.fromisoformat(hb.decode())
                age = (datetime.now(timezone.utc) - hb_ts).total_seconds()
                details["correlator_heartbeat_age_s"] = int(age)
                if age > 30:
                    status = max_status(status, "degraded")
            except Exception:
                details["correlator_heartbeat_age_s"] = None
        else:
            if settings.CORRELATOR_ENABLED:
                details["correlator_heartbeat_age_s"] = ">60"
                status = max_status(status, "degraded")

        # DLQ depth
        dlq = await r.llen("fusion:alert:dlq")
        details["dlq_depth"] = dlq
        if dlq > 0:
            status = max_status(status, "degraded")

        await r.aclose()
    except Exception as exc:
        logger.warning("Health: redis error: %s", exc)
        details["redis"] = "error"
        status = "down"

    # SSE subscribers
    details["sse_subscribers"] = len(getattr(request.app.state, "sse_subscribers", {}))

    # Adapter liveness
    details["adapters"] = _get_adapter_status()

    response_body = {"status": status, **details}
    http_status = 200 if status != "down" else 503
    return JSONResponse(response_body, status_code=http_status)


def max_status(current: str, candidate: str) -> str:
    order = {"ok": 0, "degraded": 1, "down": 2}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current


def _get_adapter_status() -> dict:
    try:
        from selfsuvis.pipeline.fusion.adapters.registry import registry

        out = {}
        for name, adapter in registry.all().items():
            out[name] = {
                "status": "ok" if adapter.enabled else "disabled",
                "last_event_ts": getattr(adapter, "last_event_ts", None),
            }
        return out
    except Exception:
        drone_audio_enabled = bool(
            settings.DRONE_AUDIO_MODEL_PATH and settings.DRONE_AUDIO_WATCH_DIR
        )
        return {
            "drone_audio": {
                "status": "ok" if drone_audio_enabled else "disabled",
                "last_event_ts": None,
            }
        }
