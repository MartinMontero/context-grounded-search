"""chunk -> contextualize -> embed -> index, with per-stage dead-lettering."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence

from contextual_chunking_service.chunker import RawChunk, estimate_tokens, split_text
from contextual_chunking_service.contextualizer import AnthropicContextualizer
from contextual_chunking_service.indexer import QdrantIndexer, point_id
from contextual_chunking_service.schemas import (
    CacheStats,
    ChunkRequest,
    ChunkResponse,
    ContextualChunk,
    IndexResponse,
)
from contextual_chunking_service.settings import ChunkingSettings
from rag_common.dlq import DeadLetterQueue
from rag_common.embeddings import HybridEmbedder
from rag_common.errors import InvalidDocumentError

log = logging.getLogger(__name__)


def assemble_chunks(
    request: ChunkRequest, raw: Sequence[RawChunk], contexts: Sequence[str] | None
) -> list[ContextualChunk]:
    chunks: list[ContextualChunk] = []
    for i, r in enumerate(raw):
        context = contexts[i].strip() if contexts else ""
        contextualized = f"{context}\n\n{r.text}" if context else r.text
        chunks.append(
            ContextualChunk(
                chunk_id=point_id(request.document_id, r.index),
                document_id=request.document_id,
                chunk_index=r.index,
                text=r.text,
                context=context,
                contextualized_text=contextualized,
                start_char=r.start_char,
                end_char=r.end_char,
                token_estimate=estimate_tokens(contextualized),
                tenant_id=request.tenant_id,
                title=request.title,
                source=request.source,
                source_url=request.source_url,
                metadata=request.metadata,
            )
        )
    return chunks


class ChunkingPipeline:
    def __init__(
        self,
        *,
        settings: ChunkingSettings,
        contextualizer: AnthropicContextualizer,
        embedder: HybridEmbedder | None,
        indexer: QdrantIndexer | None,
        dlq: DeadLetterQueue,
    ) -> None:
        self.settings = settings
        self.contextualizer = contextualizer
        self.embedder = embedder
        self.indexer = indexer
        self.dlq = dlq

    async def chunk(self, request: ChunkRequest) -> ChunkResponse:
        async with self.dlq.guard(document_id=request.document_id, stage="chunking"):
            raw = split_text(
                request.text,
                chunk_words=self.settings.chunk_words,
                overlap_words=self.settings.chunk_overlap_words,
            )
            if not raw:
                raise InvalidDocumentError("document contains no text")
            if len(raw) > self.settings.max_chunks_per_document:
                raise InvalidDocumentError(
                    f"document splits into {len(raw)} chunks; limit is "
                    f"{self.settings.max_chunks_per_document}"
                )
            cache: CacheStats | None = None
            warnings: list[str] = []
            contexts: list[str] | None = None
            if request.contextualize:
                result = await self.contextualizer.contextualize(
                    request.text, [r.text for r in raw]
                )
                contexts, cache, warnings = result.contexts, result.cache, result.warnings
            chunks = assemble_chunks(request, raw, contexts)
        return ChunkResponse(
            document_id=request.document_id, chunks=chunks, cache=cache, warnings=warnings
        )

    async def index(self, request: ChunkRequest) -> IndexResponse:
        if self.embedder is None or self.indexer is None:
            raise RuntimeError("indexing is not configured")
        chunked = await self.chunk(request)
        async with self.dlq.guard(document_id=request.document_id, stage="indexing"):
            texts = [c.contextualized_text for c in chunked.chunks]
            vectors = await asyncio.to_thread(self.embedder.embed_documents, texts)
            count = await asyncio.to_thread(self.indexer.index, chunked.chunks, vectors)
        return IndexResponse(
            document_id=request.document_id,
            collection=self.indexer.collection,
            chunks_indexed=count,
            cache=chunked.cache,
            warnings=chunked.warnings,
        )


def new_document_id() -> str:
    return uuid.uuid4().hex
