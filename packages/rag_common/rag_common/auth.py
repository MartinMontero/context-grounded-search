"""Service-to-service authentication with HS256 JWTs.

Every internal caller (n8n workers, sibling services, operators using curl)
presents ``Authorization: Bearer <jwt>``. Tokens are minted with the shared
``SERVICE_JWT_SECRET`` and carry ``iss``/``aud``/``sub``/``exp``/``iat``/``jti``.
Only HS256 is accepted; the ``alg`` header cannot downgrade verification.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ALGORITHM = "HS256"
REQUIRED_CLAIMS = ["exp", "iat", "sub", "jti", "iss", "aud"]

bearer_scheme = HTTPBearer(auto_error=False, description="Service-to-service HS256 JWT")


class AuthError(Exception):
    """Raised when a token is missing, malformed, expired or not trusted."""


@dataclass(frozen=True)
class ServicePrincipal:
    subject: str
    token_id: str
    claims: dict[str, Any]


def mint_service_token(
    *,
    secret: str,
    issuer: str,
    audience: str,
    subject: str,
    ttl_seconds: int = 3600,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
        "jti": uuid.uuid4().hex,
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def verify_service_token(
    token: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
    leeway_seconds: int = 30,
) -> ServicePrincipal:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            audience=audience,
            issuer=issuer,
            leeway=leeway_seconds,
            options={"require": REQUIRED_CLAIMS},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token expired") from exc
    except jwt.InvalidTokenError as exc:  # covers signature, audience, issuer, claims
        raise AuthError(f"invalid token: {exc.__class__.__name__}") from exc
    return ServicePrincipal(subject=str(claims["sub"]), token_id=str(claims["jti"]), claims=claims)


async def require_service_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> ServicePrincipal:
    """FastAPI dependency: authenticate the caller and expose the principal on request.state."""
    settings = request.app.state.settings
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        principal = verify_service_token(
            credentials.credentials,
            secret=settings.service_jwt_secret.get_secret_value(),
            issuer=settings.service_jwt_issuer,
            audience=settings.service_jwt_audience,
            leeway_seconds=settings.service_jwt_leeway_seconds,
        )
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from exc
    request.state.principal = principal
    return principal
