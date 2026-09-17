from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr

from rag_common.embeddings import DEFAULT_COLBERT_MODEL, DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL
from rag_common.settings import BaseServiceSettings


class ChunkingSettings(BaseServiceSettings):
    service_name: str = Field(
        default="contextual-chunking-service", validation_alias="SERVICE_NAME"
    )

    # -- Anthropic contextual retrieval -------------------------------------
    anthropic_api_key: SecretStr = Field(validation_alias="ANTHROPIC_API_KEY")
    # claude-3-5-haiku-20241022 (2048-token cache minimum) was retired on
    # 2026-02-19; Haiku 4.5 is the current generation with a 4096-token minimum.
    anthropic_model: str = Field(default="claude-haiku-4-5", validation_alias="ANTHROPIC_MODEL")
    cache_min_tokens: int = Field(default=4096, ge=1, validation_alias="ANTHROPIC_CACHE_MIN_TOKENS")
    anthropic_max_concurrency: int = Field(
        default=4, ge=1, le=32, validation_alias="ANTHROPIC_MAX_CONCURRENCY"
    )
    anthropic_max_retries: int = Field(default=5, ge=0, validation_alias="ANTHROPIC_MAX_RETRIES")
    anthropic_timeout_seconds: float = Field(
        default=60.0, gt=0, validation_alias="ANTHROPIC_TIMEOUT_SECONDS"
    )
    context_max_tokens: int = Field(default=400, ge=32, validation_alias="CONTEXT_MAX_TOKENS")
    max_document_tokens: int = Field(
        default=150_000, ge=1000, validation_alias="MAX_DOCUMENT_TOKENS"
    )

    # -- chunking ------------------------------------------------------------
    chunk_words: int = Field(default=300, ge=20, validation_alias="CHUNK_WORDS")
    chunk_overlap_words: int = Field(default=40, ge=0, validation_alias="CHUNK_OVERLAP_WORDS")
    max_chunks_per_document: int = Field(
        default=2000, ge=1, validation_alias="MAX_CHUNKS_PER_DOCUMENT"
    )

    # -- Qdrant / embeddings ---------------------------------------------------
    qdrant_url: str = Field(default="http://qdrant:6333", validation_alias="QDRANT_URL")
    qdrant_grpc_port: int = Field(default=6334, validation_alias="QDRANT_GRPC_PORT")
    qdrant_api_key: SecretStr | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="documents", validation_alias="QDRANT_COLLECTION")
    upload_batch_size: int = Field(default=64, ge=1, validation_alias="QDRANT_UPLOAD_BATCH_SIZE")
    upload_parallel: int = Field(default=2, ge=1, validation_alias="QDRANT_UPLOAD_PARALLEL")
    dense_model: str = Field(default=DEFAULT_DENSE_MODEL, validation_alias="DENSE_MODEL")
    sparse_model: str = Field(default=DEFAULT_SPARSE_MODEL, validation_alias="SPARSE_MODEL")
    colbert_model: str = Field(default=DEFAULT_COLBERT_MODEL, validation_alias="COLBERT_MODEL")
    fastembed_cache_path: str | None = Field(default=None, validation_alias="FASTEMBED_CACHE_PATH")


@lru_cache
def get_settings() -> ChunkingSettings:
    return ChunkingSettings()
