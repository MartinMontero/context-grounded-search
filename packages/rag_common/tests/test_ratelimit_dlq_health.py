from __future__ import annotations

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from rag_common.app_factory import create_app
from rag_common.auth import mint_service_token, require_service_token
from rag_common.dlq import DLQ_FIELDS, DeadLetterQueue
from rag_common.ratelimit import RateLimiter, rate_limit
from rag_common.settings import BaseServiceSettings

SECRET = "unit-test-secret-that-is-at-least-32-chars-long"


def _token(subject: str = "n8n") -> dict[str, str]:
    token = mint_service_token(
        secret=SECRET, issuer="contextual-rag", audience="rag-services", subject=subject
    )
    return {"Authorization": f"Bearer {token}"}


async def test_rate_limiter_counts_per_window(fake_redis) -> None:
    limiter = RateLimiter(fake_redis, limit=2, window_seconds=60)
    r1 = await limiter.hit("sub:a", now=1000.0)
    r2 = await limiter.hit("sub:a", now=1001.0)
    r3 = await limiter.hit("sub:a", now=1002.0)
    assert (r1.allowed, r2.allowed, r3.allowed) == (True, True, False)
    assert r3.remaining == 0 and 0 < r3.reset_seconds <= 60
    # A different subject and a new window are independent.
    assert (await limiter.hit("sub:b", now=1002.0)).allowed
    assert (await limiter.hit("sub:a", now=1061.0)).allowed


@pytest.mark.usefixtures("patch_redis")
def test_rate_limit_dependency_returns_429_with_headers() -> None:
    router = APIRouter(dependencies=[Depends(require_service_token), Depends(rate_limit)])

    @router.get("/limited")
    async def limited() -> dict:
        return {"ok": True}

    app = create_app(settings=BaseServiceSettings(), title="t", description="t", routers=[router])
    with TestClient(app) as client:
        headers = _token()
        for i in range(5):  # RATE_LIMIT_REQUESTS=5 in conftest
            r = client.get("/limited", headers=headers)
            assert r.status_code == 200, i
            assert r.headers["x-ratelimit-remaining"] == str(4 - i)
        blocked = client.get("/limited", headers=headers)
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) > 0
        assert blocked.json()["error"]["message"] == "rate limit exceeded"
        # Another caller is unaffected.
        assert client.get("/limited", headers=_token("other")).status_code == 200


async def test_dlq_entry_has_exact_fields(fake_redis) -> None:
    dlq = DeadLetterQueue(fake_redis, stream="rag.dlq")
    entry_id = await dlq.publish(
        document_id="doc-1", stage="chunking", error=ValueError("boom " + "x" * 5000)
    )
    assert entry_id
    entries = await fake_redis.xrange("rag.dlq")
    assert len(entries) == 1
    _, fields = entries[0]
    assert tuple(fields.keys()) == DLQ_FIELDS
    assert fields["document_id"] == "doc-1"
    assert fields["stage"] == "chunking"
    assert fields["error_type"] == "ValueError"
    assert len(fields["error_message"]) == 4000
    assert fields["timestamp"].endswith("+00:00")


async def test_dlq_guard_publishes_and_reraises(fake_redis) -> None:
    dlq = DeadLetterQueue(fake_redis)
    with pytest.raises(RuntimeError):
        async with dlq.guard(document_id="doc-2", stage="indexing"):
            raise RuntimeError("qdrant down")
    entries = await fake_redis.xrange("rag.dlq")
    assert entries[0][1]["stage"] == "indexing"
    assert entries[0][1]["error_message"] == "qdrant down"


@pytest.mark.usefixtures("patch_redis")
def test_internal_dlq_endpoint_requires_auth_and_publishes(fake_redis) -> None:
    app = create_app(settings=BaseServiceSettings(), title="t", description="t")
    body = {"document_id": "doc-3", "stage": "orchestration", "error_type": "n8n.Timeout"}
    with TestClient(app) as client:
        assert client.post("/internal/dlq", json=body).status_code == 401
        r = client.post("/internal/dlq", json=body, headers=_token())
        assert r.status_code == 202
        assert r.json()["stream"] == "rag.dlq"
        assert r.json()["entry_id"]


@pytest.mark.usefixtures("patch_redis")
def test_ready_reports_failing_checks() -> None:
    async def failing(app: FastAPI) -> None:
        async def check() -> tuple[bool, str]:
            return False, "qdrant unreachable"

        app.state.health.register("qdrant", check)

    app = create_app(
        settings=BaseServiceSettings(), title="t", description="t", on_startup=[failing]
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["service"] == "test-service"
        ready = client.get("/ready")
        assert ready.status_code == 503
        checks = ready.json()["checks"]
        assert checks["redis"]["ok"] is True
        assert checks["qdrant"] == {"ok": False, "detail": "qdrant unreachable"}
