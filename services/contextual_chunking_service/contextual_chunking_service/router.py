"""``/v1/chunk``, ``/v1/index`` and the chunk JSON schema."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from contextual_chunking_service.schemas import (
    ChunkRequest,
    ChunkResponse,
    ContextualChunk,
    IndexResponse,
)
from rag_common.auth import require_service_token
from rag_common.ratelimit import rate_limit

router = APIRouter(
    prefix="/v1",
    tags=["chunking"],
    dependencies=[Depends(require_service_token), Depends(rate_limit)],
)


@router.post(
    "/chunk",
    response_model=ChunkResponse,
    summary="Split a document and generate situating context for every chunk (no indexing)",
)
async def chunk(body: ChunkRequest, request: Request) -> ChunkResponse:
    return await request.app.state.pipeline.chunk(body)


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Chunk, contextualize, embed (dense + BM25 + ColBERT) and upload to Qdrant",
)
async def index(body: ChunkRequest, request: Request) -> IndexResponse:
    return await request.app.state.pipeline.index(body)


@router.get(
    "/schemas/contextual-chunk",
    summary="JSON Schema of the ContextualChunk contract",
    response_model=dict[str, Any],
)
async def chunk_schema() -> dict[str, Any]:
    return ContextualChunk.model_json_schema()
