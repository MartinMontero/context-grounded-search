"""OpenAPI / JSON schemas for the chunking service.

``ContextualChunk`` is the strict contract consumed by the indexer, n8n and
downstream consumers. Its JSON Schema is exported to
``schemas/contextual_chunk.schema.json`` and a test keeps the two in sync.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, description="Extracted document text (Markdown or plain)")
    title: str | None = Field(default=None, max_length=512)
    source: str | None = Field(default=None, max_length=128, examples=["url", "mime", "html"])
    source_url: str | None = Field(default=None, max_length=4096)
    tenant_id: str = Field(default="default", min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)
    contextualize: bool = Field(
        default=True, description="Generate chunk context with Claude before embedding"
    )


class ContextualChunk(BaseModel):
    """One retrievable unit: the original chunk plus its LLM-generated situating context."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"title": "ContextualChunk"})

    chunk_id: str = Field(description="uuid5(document_id, chunk_index); stable across re-ingests")
    document_id: str
    chunk_index: int = Field(ge=0)
    text: str = Field(description="Original chunk text")
    context: str = Field(description="Succinct context situating the chunk in the document")
    contextualized_text: str = Field(
        description="`context + '\\n\\n' + text`; this is what gets embedded"
    )
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    token_estimate: int = Field(ge=1)
    tenant_id: str
    title: str | None = None
    source: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CacheStats(BaseModel):
    model: str
    document_tokens: int = Field(description="Tokens in system prompt + document block")
    cache_min_tokens: int
    cache_eligible: bool = Field(description="document_tokens >= model's cacheable minimum")
    requests: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    uncached_input_tokens: int
    output_tokens: int


class ChunkResponse(BaseModel):
    document_id: str
    chunks: list[ContextualChunk]
    cache: CacheStats | None = Field(default=None, description="Absent when contextualize=false")
    warnings: list[str] = Field(default_factory=list)


class IndexResponse(BaseModel):
    document_id: str
    collection: str
    chunks_indexed: int
    cache: CacheStats | None = None
    warnings: list[str] = Field(default_factory=list)
