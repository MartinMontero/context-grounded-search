"""Retrieval service entrypoint: ``uvicorn --factory retrieval_service.main:build_app``."""

from __future__ import annotations

import asyncio

import cohere
from fastapi import APIRouter, Depends, FastAPI, Request

from rag_common.app_factory import create_app
from rag_common.auth import require_service_token
from rag_common.embeddings import HybridEmbedder
from rag_common.qdrant_schema import make_async_client, qdrant_ready
from rag_common.ratelimit import rate_limit
from rag_common.retry import RetryPolicy
from retrieval_service.rerank import CohereReranker
from retrieval_service.schemas import SearchRequest, SearchResponse
from retrieval_service.search import HybridSearcher
from retrieval_service.service import RetrievalService
from retrieval_service.settings import RetrievalSettings, get_settings

DESCRIPTION = """
Hybrid retrieval over the contextual chunks:

1. **Qdrant Universal Query** — dense (bge-small) and BM25 sparse prefetches fused
   with reciprocal rank fusion inside a nested prefetch, then reranked by the
   ColBERT multivector (MaxSim) in the top-level query.
2. **Cohere rerank** (`rerank-v4.0-pro`) over the top candidates, documents rendered as
   YAML, with a full-jitter retry policy (5 retries, 1s base, 30s cap) for 429/5xx.

Set `colbert=false` / `rerank=false` on a request to compare stages (see `benchmarks/`).
"""

router = APIRouter(
    prefix="/v1",
    tags=["retrieval"],
    dependencies=[Depends(require_service_token), Depends(rate_limit)],
)


@router.post("/search", response_model=SearchResponse, summary="Hybrid search + rerank")
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    return await request.app.state.retrieval.search(body)


def build_app(settings: RetrievalSettings | None = None, *, load_models: bool = True) -> FastAPI:
    settings = settings or get_settings()

    async def startup(app: FastAPI) -> None:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        app.state.qdrant = make_async_client(
            url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, api_key=api_key
        )
        embedder = None
        if load_models:
            embedder = await asyncio.to_thread(
                HybridEmbedder,
                dense_model=settings.dense_model,
                sparse_model=settings.sparse_model,
                colbert_model=settings.colbert_model,
                cache_dir=settings.fastembed_cache_path,
            )
        app.state.cohere = cohere.AsyncClientV2(
            api_key=settings.cohere_api_key.get_secret_value(),
            timeout=settings.cohere_timeout_seconds,
            max_retries=0,  # our full-jitter policy is the single retry authority
        )
        reranker = CohereReranker(
            app.state.cohere,
            model=settings.cohere_rerank_model,
            policy=RetryPolicy(
                max_retries=settings.cohere_max_retries,
                base_delay=settings.cohere_base_delay,
                max_delay=settings.cohere_max_delay,
            ),
            max_tokens_per_doc=settings.cohere_max_tokens_per_doc,
        )
        searcher = HybridSearcher(
            client=app.state.qdrant,
            embedder=embedder,  # type: ignore[arg-type]
            collection=settings.qdrant_collection,
            prefetch_limit=settings.prefetch_limit,
            fusion_limit=settings.fusion_limit,
        )
        app.state.retrieval = RetrievalService(
            settings=settings, searcher=searcher, reranker=reranker, dlq=app.state.dlq
        )
        app.state.health.register(
            "qdrant", lambda: qdrant_ready(app.state.qdrant, settings.qdrant_collection)
        )

        async def cohere_ready() -> tuple[bool, str]:
            return bool(settings.cohere_api_key.get_secret_value()), settings.cohere_rerank_model

        app.state.health.register("cohere", cohere_ready)

    async def shutdown(app: FastAPI) -> None:
        await app.state.qdrant.close()

    return create_app(
        settings=settings,
        title="Retrieval Service",
        description=DESCRIPTION,
        routers=[router],
        on_startup=[startup],
        on_shutdown=[shutdown],
        openapi_tags=[{"name": "retrieval", "description": "Hybrid search and reranking."}],
    )
