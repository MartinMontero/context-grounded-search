"""Liveness (``/health``) and readiness (``/ready``) endpoints.

``/health`` only proves the process serves HTTP. ``/ready`` runs every check the
service registered (Redis, Qdrant, sidecars, API keys present...) with a short
timeout and returns 503 until all of them pass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

ReadinessCheck = Callable[[], Awaitable[tuple[bool, str]]]


class HealthRegistry:
    def __init__(self, *, timeout_seconds: float = 3.0) -> None:
        self._checks: dict[str, ReadinessCheck] = {}
        self.timeout = timeout_seconds

    def register(self, name: str, check: ReadinessCheck) -> None:
        self._checks[name] = check

    async def run(self) -> dict[str, dict[str, str | bool]]:
        async def _run_one(name: str, check: ReadinessCheck) -> tuple[str, bool, str]:
            try:
                ok, detail = await asyncio.wait_for(check(), timeout=self.timeout)
            except TimeoutError:
                return name, False, f"timed out after {self.timeout}s"
            except Exception as exc:  # a failing check must never crash /ready
                return name, False, f"{type(exc).__name__}: {exc}"
            return name, ok, detail

        results = await asyncio.gather(*(_run_one(n, c) for n, c in self._checks.items()))
        return {name: {"ok": ok, "detail": detail} for name, ok, detail in results}


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str


class ReadyResponse(BaseModel):
    status: str
    service: str
    checks: dict[str, dict[str, str | bool]]


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    responses={503: {"model": ReadyResponse, "description": "One or more dependencies failing"}},
)
async def ready(request: Request, response: Response) -> ReadyResponse:
    settings = request.app.state.settings
    registry: HealthRegistry = request.app.state.health
    checks = await registry.run()
    all_ok = all(bool(c["ok"]) for c in checks.values())
    response.status_code = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        status="ready" if all_ok else "not_ready", service=settings.service_name, checks=checks
    )
