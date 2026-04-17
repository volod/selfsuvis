import hmac
import time
from dataclasses import dataclass
from typing import Dict

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
    # Maximum number of distinct client keys tracked simultaneously.
    # When the cap is reached, the oldest entry is evicted (insertion-order LRU).
    # Prevents unbounded memory growth from spoofed/unique source IPs.
    MAX_CLIENTS = 50_000

    def __init__(self) -> None:
        self._limiters: Dict[str, _TokenBucket] = {}

    def _evict_oldest(self) -> None:
        if len(self._limiters) >= self.MAX_CLIENTS:
            oldest = next(iter(self._limiters))
            del self._limiters[oldest]

    def _get_or_create(self, key: str) -> _TokenBucket:
        bucket = self._limiters.get(key)
        if bucket is None:
            self._evict_oldest()
            bucket = _TokenBucket(
                capacity=max(1.0, float(settings.RATE_LIMIT_BURST)),
                refill_rate=float(settings.RATE_LIMIT_PER_MIN) / 60.0,
                tokens=float(settings.RATE_LIMIT_BURST),
                last_ts=time.time(),
            )
            self._limiters[key] = bucket
        return bucket

    def check(self, key: str) -> bool:
        return self._get_or_create(key).allow()


_rate_limiter = _RateLimiter()

# Module-level aliases for backward compatibility (tests and external callers).
_limiters = _rate_limiter._limiters
_MAX_LIMITERS = _RateLimiter.MAX_CLIENTS


def _evict_oldest_limiter() -> None:
    _rate_limiter._evict_oldest()


def rate_limit(request: Request) -> None:
    if settings.RATE_LIMIT_PER_MIN <= 0:
        return
    key = _get_client_key(request)
    if not _rate_limiter.check(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
