from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag_common.app_factory import create_app
from rag_common.errors import ForbiddenTargetError
from rag_common.middleware import RequestContextMiddleware
from rag_common.settings import BaseServiceSettings

pytestmark = pytest.mark.usefixtures("patch_redis")


def _app() -> tuple:
    router = APIRouter()

    @router.get("/big")
    async def big() -> dict:
        return {"payload": "x" * 5000}

    @router.get("/boom")
    async def boom() -> None:
        raise ForbiddenTargetError("private address blocked", details={"host": "10.0.0.1"})

    @router.get("/crash")
    async def crash() -> None:
        raise RuntimeError("unexpected")

    app = create_app(settings=BaseServiceSettings(), title="t", description="t", routers=[router])
    return app


def test_middleware_order_is_cors_trustedhost_gzip() -> None:
    app = _app()
    classes = [m.cls for m in app.user_middleware]
    assert classes[:3] == [CORSMiddleware, TrustedHostMiddleware, GZipMiddleware]
    assert classes[3] is RequestContextMiddleware


def test_request_id_and_security_headers() -> None:
    with TestClient(_app()) as client:
        r = client.get("/health", headers={"X-Request-ID": "abc-123"})
        assert r.status_code == 200
        assert r.headers["x-request-id"] == "abc-123"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["cache-control"] == "no-store"
        generated = client.get("/health").headers["x-request-id"]
        assert len(generated) == 32


def test_trusted_host_rejects_unknown_host() -> None:
    with TestClient(_app()) as client:
        r = client.get("/health", headers={"Host": "evil.example"})
        assert r.status_code == 400


def test_cors_preflight_for_allowed_origin() -> None:
    with TestClient(_app()) as client:
        r = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5678",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://localhost:5678"


def test_gzip_applied_to_large_responses() -> None:
    with TestClient(_app()) as client:
        r = client.get("/big", headers={"Accept-Encoding": "gzip"})
        assert r.status_code == 200
        assert r.headers.get("content-encoding") == "gzip"
        assert len(r.json()["payload"]) == 5000


def test_error_envelope_for_service_and_unhandled_errors() -> None:
    with TestClient(_app(), raise_server_exceptions=False) as client:
        r = client.get("/boom")
        assert r.status_code == 403
        body = r.json()["error"]
        assert body["type"] == "forbidden_target"
        assert body["details"] == {"host": "10.0.0.1"}
        assert body["request_id"]
        crash = client.get("/crash")
        assert crash.status_code == 500
        assert crash.json()["error"] == {
            "type": "internal_error",
            "message": "internal server error",
            "request_id": crash.headers["x-request-id"],
        }


def test_openapi_exposes_bearer_scheme_and_health_routes() -> None:
    with TestClient(_app()) as client:
        spec = client.get("/openapi.json").json()
        assert "/health" in spec["paths"] and "/ready" in spec["paths"]
        assert "/internal/dlq" in spec["paths"]
        assert "HTTPBearer" in spec["components"]["securitySchemes"]
