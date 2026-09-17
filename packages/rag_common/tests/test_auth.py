from __future__ import annotations

import time

import jwt
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from rag_common.app_factory import create_app
from rag_common.auth import (
    AuthError,
    ServicePrincipal,
    mint_service_token,
    require_service_token,
    verify_service_token,
)
from rag_common.settings import BaseServiceSettings

SECRET = "unit-test-secret-that-is-at-least-32-chars-long"
KW = {"secret": SECRET, "issuer": "contextual-rag", "audience": "rag-services"}


def test_mint_and_verify_roundtrip() -> None:
    token = mint_service_token(subject="n8n", ttl_seconds=60, **KW)
    principal = verify_service_token(token, **KW)
    assert principal.subject == "n8n"
    assert principal.claims["aud"] == "rag-services"
    assert len(principal.token_id) == 32


def test_expired_token_rejected() -> None:
    token = mint_service_token(subject="n8n", ttl_seconds=-120, **KW)
    with pytest.raises(AuthError, match="expired"):
        verify_service_token(token, **KW)


@pytest.mark.parametrize("bad", [{"audience": "other"}, {"issuer": "other"}, {"secret": "x" * 40}])
def test_wrong_audience_issuer_or_secret_rejected(bad: dict[str, str]) -> None:
    token = mint_service_token(subject="n8n", **KW)
    with pytest.raises(AuthError):
        verify_service_token(token, **{**KW, **bad})


def test_alg_none_and_missing_claims_rejected() -> None:
    now = int(time.time())
    unsigned = jwt.encode(
        {"iss": KW["issuer"], "aud": KW["audience"], "sub": "x", "exp": now + 60},
        key="",
        algorithm="none",
    )
    with pytest.raises(AuthError):
        verify_service_token(unsigned, **KW)
    no_jti = jwt.encode(
        {"iss": KW["issuer"], "aud": KW["audience"], "sub": "x", "iat": now, "exp": now + 60},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(AuthError):
        verify_service_token(no_jti, **KW)


@pytest.mark.usefixtures("patch_redis")
def test_dependency_enforces_bearer_token() -> None:
    router = APIRouter()

    @router.get("/secure")
    async def secure(principal: ServicePrincipal = Depends(require_service_token)) -> dict:
        return {"caller": principal.subject}

    app = create_app(settings=BaseServiceSettings(), title="t", description="t", routers=[router])
    with TestClient(app) as client:
        assert client.get("/secure").status_code == 401
        assert client.get("/secure", headers={"Authorization": "Bearer nope"}).status_code == 401
        token = mint_service_token(subject="n8n", **KW)
        ok = client.get("/secure", headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200
        assert ok.json() == {"caller": "n8n"}
        # Health endpoints stay unauthenticated.
        assert client.get("/health").status_code == 200
