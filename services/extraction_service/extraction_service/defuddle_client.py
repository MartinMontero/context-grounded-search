"""HTTP client for the Node.js Defuddle sidecar (JSDOM parse or Playwright render)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from rag_common.errors import ForbiddenTargetError, PayloadTooLargeError, UpstreamError


@dataclass(frozen=True)
class DefuddleResult:
    content: str
    word_count: int
    title: str | None
    author: str | None
    description: str | None
    published: str | None
    site: str | None
    final_url: str | None


class DefuddleClient:
    def __init__(
        self, *, base_url: str, client: httpx.AsyncClient, token_factory: Callable[[], str]
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.token_factory = token_factory

    async def _post(self, path: str, payload: dict) -> DefuddleResult:
        try:
            response = await self.client.post(
                f"{self.base_url}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self.token_factory()}"},
            )
        except httpx.HTTPError as exc:
            raise UpstreamError(f"defuddle sidecar unreachable: {type(exc).__name__}") from exc
        if response.status_code == 403:
            raise ForbiddenTargetError(_message(response, "sidecar refused target"))
        if response.status_code == 413:
            raise PayloadTooLargeError(_message(response, "sidecar payload too large"))
        if response.status_code >= 400:
            raise UpstreamError(_message(response, f"defuddle sidecar HTTP {response.status_code}"))
        data = response.json()
        return DefuddleResult(
            content=(data.get("content") or "").strip(),
            word_count=int(data.get("wordCount") or 0),
            title=data.get("title") or None,
            author=data.get("author") or None,
            description=data.get("description") or None,
            published=data.get("published") or None,
            site=data.get("site") or None,
            final_url=data.get("finalUrl") or None,
        )

    async def parse_html(self, html: str, *, url: str | None) -> DefuddleResult:
        return await self._post("/parse", {"html": html, "url": url})

    async def render_url(self, url: str) -> DefuddleResult:
        return await self._post("/render", {"url": url})

    async def ready(self) -> tuple[bool, str]:
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=3.0)
        except httpx.HTTPError as exc:
            return False, f"sidecar unreachable: {type(exc).__name__}"
        return response.status_code == 200, f"HTTP {response.status_code}"


def _message(response: httpx.Response, fallback: str) -> str:
    try:
        return str(response.json().get("error") or fallback)
    except ValueError:
        return fallback
