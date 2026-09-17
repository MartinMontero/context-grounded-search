"""Cohere ``/v2/rerank`` with YAML-formatted documents and full-jitter retries.

Cohere's v2 rerank accepts plain strings; semi-structured chunks are rendered as
YAML (``sort_keys=False`` keeps the field order title -> context -> text, which
matters for how the model reads the document). The returned ``index`` values map
positions back to the candidates we sent; results are ordered by
``relevance_score``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cohere
import httpx
import yaml
from cohere.core.api_error import ApiError

from rag_common.errors import UpstreamError
from rag_common.retry import RetryExhaustedError, RetryPolicy, retry_async
from rag_common.telemetry import get_tracer
from retrieval_service.search import Candidate

log = logging.getLogger(__name__)
tracer = get_tracer(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: Candidate
    relevance_score: float
    original_index: int


def format_document(payload: dict[str, Any], *, max_chars: int = 12_000) -> str:
    """YAML rendering of the fields the reranker should see, in a fixed order."""
    doc: dict[str, Any] = {}
    for key, source in (
        ("title", "title"),
        ("context", "context"),
        ("text", "original_text"),
        ("source_url", "source_url"),
    ):
        value = payload.get(source)
        if value:
            doc[key] = str(value)[:max_chars]
    if "text" not in doc and payload.get("text"):
        doc["text"] = str(payload["text"])[:max_chars]
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    if isinstance(exc, ApiError):
        return exc.status_code in RETRYABLE_STATUS
    return False


def retry_after_hint(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None) or {}
    value = {k.lower(): v for k, v in headers.items()}.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class CohereReranker:
    def __init__(
        self,
        client: cohere.AsyncClientV2,
        *,
        model: str,
        policy: RetryPolicy | None = None,
        max_tokens_per_doc: int = 4096,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.policy = policy or RetryPolicy(max_retries=5, base_delay=1.0, max_delay=30.0)
        self.max_tokens_per_doc = max_tokens_per_doc
        self.sleep = sleep

    async def rerank(
        self, query: str, candidates: Sequence[Candidate], *, top_n: int
    ) -> tuple[list[RerankedCandidate], float]:
        if not candidates:
            return [], 0.0
        documents = [format_document(c.payload) for c in candidates]

        async def call() -> Any:
            return await self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                max_tokens_per_doc=self.max_tokens_per_doc,
            )

        t0 = time.perf_counter()
        with tracer.start_as_current_span("rerank.cohere") as span:
            span.set_attribute("cohere.model", self.model)
            span.set_attribute("rerank.documents", len(documents))
            try:
                response = await retry_async(
                    call,
                    policy=self.policy,
                    retry_on=(ApiError, httpx.HTTPError),
                    should_retry=is_retryable,
                    retry_after_hint=retry_after_hint,
                    sleep=self.sleep,
                    name="cohere.rerank",
                )
            except RetryExhaustedError as exc:
                raise UpstreamError(
                    "cohere rerank failed after retries",
                    details={"attempts": exc.attempts, "last_error": type(exc.last_error).__name__},
                ) from exc
            except ApiError as exc:
                raise UpstreamError(
                    f"cohere rerank rejected the request: HTTP {exc.status_code}",
                    details={"status_code": exc.status_code},
                ) from exc
            except httpx.HTTPError as exc:
                raise UpstreamError(f"cohere unreachable: {type(exc).__name__}") from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000

        reranked = [
            RerankedCandidate(
                candidate=candidates[item.index],
                relevance_score=float(item.relevance_score),
                original_index=int(item.index),
            )
            for item in response.results
            if 0 <= int(item.index) < len(candidates)
        ]
        reranked.sort(key=lambda r: r.relevance_score, reverse=True)
        return reranked, elapsed_ms
