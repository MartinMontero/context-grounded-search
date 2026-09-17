"""Redis-backed fixed-window rate limiting, keyed by the authenticated principal.

The window key is ``ratelimit:<subject>:<window-bucket>``; ``INCR`` + ``EXPIRE``
run inside a MULTI/EXEC pipeline so the counter and its TTL are set atomically.
Falls open (with an error log) when Redis is unreachable unless
``RATE_LIMIT_FAIL_OPEN=false``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int


class RateLimiter:
    def __init__(
        self,
        redis: Redis,
        *,
        limit: int,
        window_seconds: int,
        prefix: str = "ratelimit",
    ) -> None:
        self.redis = redis
        self.limit = limit
        self.window = window_seconds
        self.prefix = prefix

    async def hit(self, key: str, *, now: float | None = None) -> RateLimitResult:
        now = time.time() if now is None else now
        bucket = int(now // self.window)
        redis_key = f"{self.prefix}:{key}:{bucket}"
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.incr(redis_key)
            pipe.expire(redis_key, self.window + 1)
            count, _ = await pipe.execute()
        count = int(count)
        reset = self.window - int(now % self.window)
        return RateLimitResult(
            allowed=count <= self.limit,
            limit=self.limit,
            remaining=max(self.limit - count, 0),
            reset_seconds=reset,
        )


def _client_key(request: Request) -> str:
    principal = getattr(request.state, "principal", None)
    if principal is not None:
        return f"sub:{principal.subject}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


async def rate_limit(request: Request, response: Response) -> None:
    """FastAPI dependency; place it AFTER ``require_service_token`` so the key is the JWT sub."""
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    settings = request.app.state.settings
    try:
        result = await limiter.hit(_client_key(request))
    except (RedisError, OSError) as exc:
        if settings.rate_limit_fail_open:
            log.error("rate limiter unavailable, failing open", extra={"error": str(exc)})
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limiter unavailable"
        ) from exc
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset": str(result.reset_seconds),
    }
    response.headers.update(headers)
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={**headers, "Retry-After": str(result.reset_seconds)},
        )
