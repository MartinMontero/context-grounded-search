"""Extraction service entrypoint: ``uvicorn --factory extraction_service.main:build_app``."""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from extraction_service import router
from extraction_service.defuddle_client import DefuddleClient
from extraction_service.service import ExtractionService
from extraction_service.settings import ExtractionSettings, get_settings
from extraction_service.ssrf import SafeFetcher
from rag_common.app_factory import create_app
from rag_common.auth import mint_service_token

DESCRIPTION = """
Turns raw inputs into clean Markdown for the chunking stage.

* **HTML** — trafilatura (lxml + html_clean); Defuddle sidecar for Shadow-DOM / JS-heavy pages.
* **URL** — SSRF-hardened fetch (public IPs only, pinned DNS, redirect re-validation,
  size/time caps).
* **MIME** — `email.policy.default` parsing of multipart messages with attachment inventory.

All routes except `/health` and `/ready` require a service JWT and are rate limited per caller.
"""


def build_app(settings: ExtractionSettings | None = None) -> FastAPI:
    settings = settings or get_settings()

    async def startup(app: FastAPI) -> None:
        timeout = httpx.Timeout(settings.fetch_timeout_seconds, connect=5.0)
        app.state.http = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
            follow_redirects=False,
        )
        app.state.sidecar_http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

        def sidecar_token() -> str:
            return mint_service_token(
                secret=settings.service_jwt_secret.get_secret_value(),
                issuer=settings.service_jwt_issuer,
                audience=settings.service_jwt_audience,
                subject=settings.service_name,
                ttl_seconds=300,
            )

        defuddle = DefuddleClient(
            base_url=settings.defuddle_sidecar_url,
            client=app.state.sidecar_http,
            token_factory=sidecar_token,
        )
        fetcher = SafeFetcher(
            client=app.state.http,
            max_bytes=settings.max_payload_bytes,
            max_redirects=settings.max_redirects,
            allowed_ports=settings.allowed_ports,
            user_agent=settings.user_agent,
        )
        app.state.extraction = ExtractionService(
            settings=settings, fetcher=fetcher, defuddle=defuddle
        )
        app.state.health.register("defuddle_sidecar", defuddle.ready)

    async def shutdown(app: FastAPI) -> None:
        await app.state.http.aclose()
        await app.state.sidecar_http.aclose()

    return create_app(
        settings=settings,
        title="Extraction Service",
        description=DESCRIPTION,
        routers=[router.router],
        on_startup=[startup],
        on_shutdown=[shutdown],
        openapi_tags=[{"name": "extraction", "description": "HTML, URL and MIME extraction."}],
    )
