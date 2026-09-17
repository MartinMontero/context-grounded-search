"""Anthropic Contextual Retrieval with prompt caching.

Request layout (identical for every chunk of a document)::

    system:   STATIC_SYSTEM_PROMPT
    messages: [ user: [ <document> ... </document>  <- cache_control ephemeral (breakpoint)
                        <chunk> ... </chunk> + instruction  <- varies per chunk ] ]

Rules enforced here
* The document block is built **once** per document and reused verbatim, so
  the cached prefix (system + document) is byte-identical across requests.
* The document block comes **before** the chunk text - anything after the
  breakpoint is uncached, anything before must not change.
* One breakpoint (of the 4 allowed) is enough: it covers the system prompt too.
* The first chunk is sent alone to *write* the cache; the rest fan out
  concurrently and *read* it. Sending all chunks at once would race the write.
* Cache eligibility is measured with ``count_tokens``: prefixes shorter than the
  model minimum silently do not cache, so the response reports it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic
from anthropic.types import MessageParam, TextBlock, TextBlockParam

from contextual_chunking_service.schemas import CacheStats
from rag_common.errors import InvalidDocumentError, UpstreamError
from rag_common.telemetry import get_tracer

log = logging.getLogger(__name__)
tracer = get_tracer(__name__)

SYSTEM_PROMPT = (
    "You situate chunks of a document for search retrieval. Given the full document and "
    "one chunk, reply with a short, succinct context (one to three sentences) that states what "
    "the chunk is about and how it relates to the overall document, including any entities, "
    "dates or identifiers needed to disambiguate it. Answer only with the context."
)
DOCUMENT_TEMPLATE = "<document>\n{document}\n</document>"
CHUNK_TEMPLATE = (
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)
CACHE_TTL = "5m"  # ephemeral breakpoints expire 5 minutes after their last read


@dataclass
class _UsageTotals:
    requests: int = 0
    cache_creation: int = 0
    cache_read: int = 0
    uncached: int = 0
    output: int = 0
    warnings: list[str] = field(default_factory=list)

    def add(self, usage: Any) -> None:
        self.requests += 1
        self.cache_creation += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.cache_read += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        self.uncached += int(getattr(usage, "input_tokens", 0) or 0)
        self.output += int(getattr(usage, "output_tokens", 0) or 0)


@dataclass(frozen=True)
class ContextualizationResult:
    contexts: list[str]
    cache: CacheStats
    warnings: list[str]


def build_document_block(document_text: str) -> TextBlockParam:
    """The static, cached block. Build once per document; never mutate."""
    return {
        "type": "text",
        "text": DOCUMENT_TEMPLATE.format(document=document_text),
        "cache_control": {"type": "ephemeral"},
    }


def build_messages(document_block: TextBlockParam, chunk_text: str) -> list[MessageParam]:
    """Document block first (cached prefix), chunk-specific text second."""
    return [
        {
            "role": "user",
            "content": [
                document_block,
                {"type": "text", "text": CHUNK_TEMPLATE.format(chunk=chunk_text)},
            ],
        }
    ]


class AnthropicContextualizer:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str,
        cache_min_tokens: int,
        max_concurrency: int = 4,
        max_tokens: int = 400,
        max_document_tokens: int = 150_000,
    ) -> None:
        self.client = client
        self.model = model
        self.cache_min_tokens = cache_min_tokens
        self.max_concurrency = max_concurrency
        self.max_tokens = max_tokens
        self.max_document_tokens = max_document_tokens

    async def measure_prefix(self, document_block: TextBlockParam) -> int:
        """Tokens in the cached prefix (system + document block)."""
        try:
            counted = await self.client.messages.count_tokens(
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": [document_block]}],
            )
        except anthropic.APIError as exc:
            raise _upstream(exc, "count_tokens") from exc
        return int(counted.input_tokens)

    async def contextualize(self, document_text: str, chunks: list[str]) -> ContextualizationResult:
        if not chunks:
            raise InvalidDocumentError("no chunks to contextualize")
        document_block = build_document_block(document_text)
        prefix_tokens = await self.measure_prefix(document_block)
        if prefix_tokens > self.max_document_tokens:
            raise InvalidDocumentError(
                f"document is {prefix_tokens} tokens; limit is {self.max_document_tokens}",
                details={"document_tokens": prefix_tokens},
            )
        eligible = prefix_tokens >= self.cache_min_tokens
        totals = _UsageTotals()
        if not eligible:
            totals.warnings.append(
                f"document prefix is {prefix_tokens} tokens, below the "
                f"{self.cache_min_tokens}-token cache minimum for {self.model}; "
                "requests will not be served from cache"
            )

        with tracer.start_as_current_span("contextualize.document") as span:
            span.set_attribute("anthropic.model", self.model)
            span.set_attribute("chunks", len(chunks))
            span.set_attribute("cache.eligible", eligible)
            contexts: list[str] = [""] * len(chunks)
            # 1) warm the cache with the first chunk (cache write)
            contexts[0] = await self._situate(document_block, chunks[0], totals)
            # 2) fan out the remainder (cache reads), bounded by the semaphore
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def run(index: int) -> None:
                async with semaphore:
                    contexts[index] = await self._situate(document_block, chunks[index], totals)

            await asyncio.gather(*(run(i) for i in range(1, len(chunks))))
            span.set_attribute("cache.read_tokens", totals.cache_read)

        if eligible and len(chunks) > 1 and totals.cache_read == 0:
            totals.warnings.append(
                "no cache reads observed across chunk requests; check for a silent invalidator"
            )
        stats = CacheStats(
            model=self.model,
            document_tokens=prefix_tokens,
            cache_min_tokens=self.cache_min_tokens,
            cache_eligible=eligible,
            requests=totals.requests,
            cache_creation_input_tokens=totals.cache_creation,
            cache_read_input_tokens=totals.cache_read,
            uncached_input_tokens=totals.uncached,
            output_tokens=totals.output,
        )
        return ContextualizationResult(contexts=contexts, cache=stats, warnings=totals.warnings)

    async def _situate(
        self, document_block: TextBlockParam, chunk_text: str, totals: _UsageTotals
    ) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=build_messages(document_block, chunk_text),
            )
        except anthropic.APIError as exc:
            raise _upstream(exc, "messages.create") from exc
        totals.add(response.usage)
        text = "".join(
            block.text for block in response.content if isinstance(block, TextBlock)
        ).strip()
        if response.stop_reason == "max_tokens":
            totals.warnings.append("a chunk context was truncated at max_tokens")
        if not text:
            raise UpstreamError("model returned an empty context")
        return text


def _upstream(exc: anthropic.APIError, operation: str) -> UpstreamError:
    status = getattr(exc, "status_code", None)
    log.error(
        "anthropic call failed",
        extra={"operation": operation, "error_type": type(exc).__name__, "status": status},
    )
    return UpstreamError(
        f"anthropic {operation} failed: {type(exc).__name__}",
        details={"status_code": status} if status else {},
    )
