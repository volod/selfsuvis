import hmac
import time
from dataclasses import dataclass
from typing import Dict

from fastapi import Header, HTTPException, Request

from pipeline.config import settings


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


_limiters: Dict[str, _TokenBucket] = {}
# Maximum number of distinct client keys tracked simultaneously.
# When the cap is reached, the oldest entry is evicted (insertion-order LRU).
# Prevents unbounded memory growth from spoofed/unique source IPs.
_MAX_LIMITERS = 50_000


def _evict_oldest_limiter() -> None:
    if len(_limiters) >= _MAX_LIMITERS:
        oldest = next(iter(_limiters))
        del _limiters[oldest]


def rate_limit(request: Request) -> None:
    if settings.RATE_LIMIT_PER_MIN <= 0:
        return
    key = _get_client_key(request)
    bucket = _limiters.get(key)
    if bucket is None:
        _evict_oldest_limiter()
        bucket = _TokenBucket(
            capacity=max(1.0, float(settings.RATE_LIMIT_BURST)),
            refill_rate=float(settings.RATE_LIMIT_PER_MIN) / 60.0,
            tokens=float(settings.RATE_LIMIT_BURST),
            last_ts=time.time(),
        )
        _limiters[key] = bucket
    if not bucket.allow():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
