"""Uniform error envelope: ``{"error": {"type", "message", "request_id", "details"?}}``."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_common.logging import request_id_var

log = logging.getLogger(__name__)


class ServiceError(Exception):
    """Domain error with an HTTP status; message is safe to return to callers."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_type = "service_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UpstreamError(ServiceError):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_type = "upstream_error"


class PayloadTooLargeError(ServiceError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    error_type = "payload_too_large"


class ForbiddenTargetError(ServiceError):
    status_code = status.HTTP_403_FORBIDDEN
    error_type = "forbidden_target"


class InvalidDocumentError(ServiceError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_type = "invalid_document"


def envelope(error_type: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "error": {"type": error_type, "message": message, "request_id": request_id_var.get()}
    }
    if details:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope(exc.error_type, exc.message, exc.details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope("http_error", str(exc.detail)),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=envelope("validation_error", "request validation failed", exc.errors()),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled exception", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=envelope("internal_error", "internal server error"),
        )
