"""OpenAPI schemas for the retrieval service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(
        default=None, ge=1, description="Final result count (default from settings)"
    )
    tenant_id: str | None = Field(default=None, max_length=128)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    metadata_filters: dict[str, str | int | bool] = Field(
        default_factory=dict, description="Exact-match filters on payload.metadata.<key>"
    )
    colbert: bool = Field(default=True, description="Rerank fused candidates with ColBERT MaxSim")
    rerank: bool = Field(default=True, description="Send candidates to Cohere rerank")
    candidates: int | None = Field(
        default=None, ge=1, le=200, description="Candidates passed to the reranker"
    )


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    chunk_index: int
    tenant_id: str | None = None
    title: str | None = None
    source_url: str | None = None
    context: str
    text: str = Field(description="Contextualized chunk text (what was embedded)")
    original_text: str
    score: float = Field(description="Final ordering score (Cohere relevance or Qdrant score)")
    qdrant_score: float
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Timings(BaseModel):
    embed_ms: float
    qdrant_ms: float
    rerank_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    candidates_considered: int
    stages: dict[str, Any]
    timings: Timings
