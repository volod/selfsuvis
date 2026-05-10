import hashlib
import hmac
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from selfsuvis.pipeline.core import settings


def _get_client_key(request: Request) -> str:
    """Derive client key for rate limiting. When TRUST_PROXY_HEADERS is True,
    uses X-Forwarded-For. WARNING: Only enable TRUST_PROXY_HEADERS behind a
    trusted reverse proxy that strips/overwrites this header; otherwise clients
    can spoof it and bypass or dilute rate limiting."""
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not settings.API_KEY:
        if settings.API_AUTH_REQUIRED:
            raise HTTPException(
                status_code=503,
                detail="Server authentication is not configured",
            )
        return
    if not hmac.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")


@dataclass
class _TokenBucket:
    capacity: float
    refill_rate: float
    tokens: float
    last_ts: float

    def allow(self) -> bool:
        now = time.time()
        elapsed = max(0.0, now - self.last_ts)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_ts = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class _RateLimiter:
    MAX_CLIENTS = 50_000

    def __init__(self, per_second: float = 0.0, burst: float = 0.0) -> None:
        self._limiters: dict[str, _TokenBucket] = {}
        # 0 means: read from settings at check() time (global limiter behaviour).
        self._per_second = per_second
        self._burst = burst

    def _evict_oldest(self) -> None:
        if len(self._limiters) >= self.MAX_CLIENTS:
            oldest = next(iter(self._limiters))
            del self._limiters[oldest]

    def _get_or_create(self, key: str) -> _TokenBucket:
        bucket = self._limiters.get(key)
        if bucket is None:
            self._evict_oldest()
            if self._per_second > 0:
                capacity = self._burst if self._burst > 0 else self._per_second
                refill = self._per_second
            else:
                capacity = max(1.0, float(settings.RATE_LIMIT_BURST))
                refill = float(settings.RATE_LIMIT_PER_MIN) / 60.0
            bucket = _TokenBucket(
                capacity=capacity,
                refill_rate=refill,
                tokens=capacity,
                last_ts=time.time(),
            )
            self._limiters[key] = bucket
        return bucket

    def check(self, key: str) -> bool:
        return self._get_or_create(key).allow()


_rate_limiter = _RateLimiter()
_sensor_rate_limiter = _RateLimiter(per_second=10.0, burst=10.0)


def rate_limit(request: Request) -> None:
    if settings.RATE_LIMIT_PER_MIN <= 0:
        return
    key = _get_client_key(request)
    if not _rate_limiter.check(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def sensor_rate_limit(sensor_id: str) -> None:
    """10 req/s per sensor_id. Called after EventEnvelope is parsed."""
    if not _sensor_rate_limiter.check(sensor_id):
        raise HTTPException(status_code=429, detail="Sensor rate limit exceeded")


async def require_sensor_key(
    request: Request,
    x_sensor_key: str = Header(default=""),
) -> None:
    """Authenticate ingest requests.

    If no sensor_keys rows exist: fall back to site-level require_api_key.
    If rows exist and key missing/invalid: 401.
    If key found but scope 'ingest' absent: 403.
    """
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        require_api_key(x_sensor_key)
        return

    async with pool.acquire() as conn:
        row_count = await conn.fetchval("SELECT COUNT(*) FROM sensor_keys")

    if row_count == 0:
        require_api_key(x_sensor_key)
        return

    if not x_sensor_key:
        raise HTTPException(status_code=401, detail="X-Sensor-Key header required")

    key_hash = hashlib.sha256(x_sensor_key.encode()).hexdigest()

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT scopes FROM sensor_keys WHERE key_hash = $1", key_hash)

    if row is None:
        raise HTTPException(status_code=401, detail="Invalid sensor key")

    if "ingest" not in row["scopes"]:
        raise HTTPException(status_code=403, detail="Sensor key lacks ingest scope")
