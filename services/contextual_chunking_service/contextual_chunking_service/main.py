"""Chunking service entrypoint: ``uvicorn --factory contextual_chunking_service.main:build_app``."""

from __future__ import annotations

import asyncio

import anthropic
from fastapi import FastAPI

from contextual_chunking_service import router
from contextual_chunking_service.contextualizer import AnthropicContextualizer
from contextual_chunking_service.indexer import QdrantIndexer
from contextual_chunking_service.pipeline import ChunkingPipeline
from contextual_chunking_service.settings import ChunkingSettings, get_settings
from rag_common.app_factory import create_app
from rag_common.embeddings import HybridEmbedder
from rag_common.qdrant_schema import make_async_client, make_client, qdrant_ready

DESCRIPTION = """
Implements Anthropic **Contextual Retrieval**: each chunk is situated within its
document by Claude, using a prompt-cached document block, then embedded three ways
(dense 384-d, BM25 sparse with server-side IDF, ColBERT 128-d multivector) and
uploaded to Qdrant with `upload_points(batch_size=64, parallel=2, wait=True)`.

`POST /v1/chunk` returns the strict `ContextualChunk` objects without indexing;
`POST /v1/index` runs the whole pipeline. Failures dead-letter to `rag.dlq`.
"""


def build_app(settings: ChunkingSettings | None = None, *, with_indexing: bool = True) -> FastAPI:
    settings = settings or get_settings()

    async def startup(app: FastAPI) -> None:
        app.state.anthropic = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            max_retries=settings.anthropic_max_retries,
            timeout=settings.anthropic_timeout_seconds,
        )
        contextualizer = AnthropicContextualizer(
            app.state.anthropic,
            model=settings.anthropic_model,
            cache_min_tokens=settings.cache_min_tokens,
            max_concurrency=settings.anthropic_max_concurrency,
            max_tokens=settings.context_max_tokens,
            max_document_tokens=settings.max_document_tokens,
        )
        embedder = indexer = None
        if with_indexing:
            api_key = (
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            )
            app.state.qdrant = make_client(
                url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, api_key=api_key
            )
            app.state.qdrant_async = make_async_client(
                url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, api_key=api_key
            )
            indexer = QdrantIndexer(
                app.state.qdrant,
                collection=settings.qdrant_collection,
                batch_size=settings.upload_batch_size,
                parallel=settings.upload_parallel,
            )
            # Collection + payload indexes exist before any upload; model download is slow.
            await asyncio.to_thread(indexer.ensure_ready)
            embedder = await asyncio.to_thread(
                HybridEmbedder,
                dense_model=settings.dense_model,
                sparse_model=settings.sparse_model,
                colbert_model=settings.colbert_model,
                cache_dir=settings.fastembed_cache_path,
            )
            app.state.health.register(
                "qdrant",
                lambda: qdrant_ready(app.state.qdrant_async, settings.qdrant_collection),
            )
        app.state.pipeline = ChunkingPipeline(
            settings=settings,
            contextualizer=contextualizer,
            embedder=embedder,
            indexer=indexer,
            dlq=app.state.dlq,
        )

        async def anthropic_ready() -> tuple[bool, str]:
            return bool(settings.anthropic_api_key.get_secret_value()), settings.anthropic_model

        app.state.health.register("anthropic", anthropic_ready)

    async def shutdown(app: FastAPI) -> None:
        await app.state.anthropic.close()
        if getattr(app.state, "qdrant_async", None) is not None:
            await app.state.qdrant_async.close()
        if getattr(app.state, "qdrant", None) is not None:
            app.state.qdrant.close()

    return create_app(
        settings=settings,
        title="Contextual Chunking Service",
        description=DESCRIPTION,
        routers=[router.router],
        on_startup=[startup],
        on_shutdown=[shutdown],
        openapi_tags=[
            {"name": "chunking", "description": "Contextual chunking and hybrid indexing."}
        ],
    )
