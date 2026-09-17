"""Async Redis connection factory (rate limiter, DLQ, readiness)."""

from __future__ import annotations

from redis.asyncio import Redis

from rag_common.settings import BaseServiceSettings


def create_redis(settings: BaseServiceSettings) -> Redis:
    return Redis.from_url(
        settings.redis_url.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        health_check_interval=30,
        retry_on_timeout=True,
    )


async def redis_ready(redis: Redis) -> tuple[bool, str]:
    pong = await redis.ping()
    return bool(pong), "pong" if pong else "no pong"
