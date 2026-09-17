"""Middleware stack, installed in a fixed order.

Starlette wraps middleware in *reverse* order of ``add_middleware`` calls (the
last one added is the outermost). ``install_middleware`` therefore adds the
innermost first so a request travels

    CORSMiddleware -> TrustedHostMiddleware -> GZipMiddleware -> RequestContext -> app

which is asserted by ``tests/test_middleware.py``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_common.errors import envelope
from rag_common.logging import request_id_var
from rag_common.settings import BaseServiceSettings

log = logging.getLogger("rag.access")
REQUEST_ID_HEADER = "x-request-id"
_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "cache-control": "no-store",
    "referrer-policy": "no-referrer",
}


class RequestContextMiddleware:
    """Pure-ASGI: request-id propagation, security headers and a structured access log."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1").strip()
        request_id = incoming[:128] if incoming else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                raw = list(message.get("headers") or [])
                raw.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                present = {k.lower() for k, _ in raw}
                for key, value in _SECURITY_HEADERS.items():
                    if key.encode() not in present:
                        raw.append((key.encode(), value.encode()))
                message["headers"] = raw
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            # Render the 500 envelope here (inside the stack) so the request id,
            # security headers and access log stay consistent for crashes too.
            logging.getLogger("rag_common.errors").exception(
                "unhandled exception", extra={"error_type": type(exc).__name__}
            )
            if response_started:
                raise
            body = json.dumps(envelope("internal_error", "internal server error")).encode()
            await send_wrapper(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send_wrapper({"type": "http.response.body", "body": body})
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            path = scope.get("path", "")
            if path not in ("/health", "/ready"):
                log.info(
                    "request completed",
                    extra={
                        "method": scope.get("method"),
                        "path": path,
                        "status_code": status_code,
                        "duration_ms": duration_ms,
                        "client_ip": (scope.get("client") or ("", 0))[0],
                    },
                )
            request_id_var.reset(token)


def install_middleware(app: FastAPI, settings: BaseServiceSettings) -> None:
    app.add_middleware(RequestContextMiddleware)  # innermost
    app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_minimum_size)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(  # outermost: preflights answered before host filtering
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
        max_age=600,
    )
