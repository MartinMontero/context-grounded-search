"""Application factory: every service is assembled the same way.

    app = create_app(settings=..., title=..., routers=[...], on_startup=[...])

Provides: JSON logging, OpenTelemetry, the fixed middleware order, ``/health``,
``/ready``, ``/internal/dlq``, the shared Redis client, rate limiter and DLQ on
``app.state``, and the uniform error envelope.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from rag_common import dlq_router, health
from rag_common.dlq import DeadLetterQueue
from rag_common.errors import register_exception_handlers
from rag_common.logging import configure_logging
from rag_common.middleware import install_middleware
from rag_common.ratelimit import RateLimiter
from rag_common.redis_client import create_redis, redis_ready
from rag_common.settings import BaseServiceSettings
from rag_common.telemetry import configure_telemetry, instrument_app

Hook = Callable[[FastAPI], Awaitable[None]]
log = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {"name": "health", "description": "Liveness and readiness probes (unauthenticated)."},
    {"name": "dlq", "description": "Dead-letter queue publishing for orchestration failures."},
]


def create_app(
    *,
    settings: BaseServiceSettings,
    title: str,
    description: str,
    routers: Sequence[APIRouter] = (),
    on_startup: Sequence[Hook] = (),
    on_shutdown: Sequence[Hook] = (),
    openapi_tags: Sequence[dict[str, str]] = (),
) -> FastAPI:
    configure_logging(settings.service_name, settings.log_level)
    configure_telemetry(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.redis = create_redis(settings)
        app.state.rate_limiter = RateLimiter(
            app.state.redis,
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        app.state.dlq = DeadLetterQueue(app.state.redis, stream=settings.dlq_stream)
        app.state.health = health.HealthRegistry()
        app.state.health.register("redis", lambda: redis_ready(app.state.redis))
        for hook in on_startup:
            await hook(app)
        log.info("service started", extra={"environment": settings.environment})
        try:
            yield
        finally:
            for hook in on_shutdown:
                try:
                    await hook(app)
                except Exception:  # shutdown must run every hook
                    log.exception("shutdown hook failed")
            await app.state.redis.aclose()
            log.info("service stopped")

    app = FastAPI(
        title=title,
        description=description,
        version=settings.service_version,
        lifespan=lifespan,
        openapi_tags=[*OPENAPI_TAGS, *openapi_tags],
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings  # available before lifespan for route-level access
    install_middleware(app, settings)
    instrument_app(app)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(dlq_router.router)
    for router in routers:
        app.include_router(router)
    return app
