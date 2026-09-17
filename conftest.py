"""Repository-wide pytest fixtures: deterministic env + in-memory Redis.

No test in this repository needs Docker, network access or API keys.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import fakeredis
import fakeredis.aioredis
import pytest

TEST_ENV = {
    "SERVICE_NAME": "test-service",
    "SERVICE_JWT_SECRET": "unit-test-secret-that-is-at-least-32-chars-long",
    "SERVICE_JWT_ISSUER": "contextual-rag",
    "SERVICE_JWT_AUDIENCE": "rag-services",
    "ALLOWED_HOSTS": "testserver,localhost,127.0.0.1",
    "CORS_ALLOW_ORIGINS": "http://localhost:5678",
    "RATE_LIMIT_REQUESTS": "5",
    "RATE_LIMIT_WINDOW_SECONDS": "60",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "",
    "LOG_LEVEL": "WARNING",
    "REDIS_URL": "redis://localhost:6379/1",
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "COHERE_API_KEY": "cohere-test",
    "QDRANT_API_KEY": "qdrant-test",
    "QDRANT_URL": "http://localhost:6333",
    "DEFUDDLE_SIDECAR_URL": "http://sidecar.test:3000",
}


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    # Run from an empty directory so a developer's .env can never be picked up.
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def fake_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
async def fake_redis(
    fake_server: fakeredis.FakeServer,
) -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    redis = fakeredis.aioredis.FakeRedis(server=fake_server, decode_responses=True)
    try:
        yield redis
    finally:
        await redis.aclose()


@pytest.fixture
def fake_redis_sync(fake_server: fakeredis.FakeServer) -> Iterator[fakeredis.FakeRedis]:
    """Synchronous view of the same in-memory server, for assertions inside sync tests."""
    redis = fakeredis.FakeRedis(server=fake_server, decode_responses=True)
    try:
        yield redis
    finally:
        redis.close()


@pytest.fixture
def patch_redis(monkeypatch: pytest.MonkeyPatch, fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """Make ``create_app`` use the in-memory Redis."""
    import rag_common.app_factory as factory

    monkeypatch.setattr(factory, "create_redis", lambda _settings: fake_redis)
