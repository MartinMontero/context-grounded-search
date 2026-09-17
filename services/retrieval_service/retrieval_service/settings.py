from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr

from rag_common.embeddings import DEFAULT_COLBERT_MODEL, DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL
from rag_common.settings import BaseServiceSettings


class RetrievalSettings(BaseServiceSettings):
    service_name: str = Field(default="retrieval-service", validation_alias="SERVICE_NAME")

    # -- Cohere reranking ------------------------------------------------------
    cohere_api_key: SecretStr = Field(validation_alias="COHERE_API_KEY")
    cohere_rerank_model: str = Field(
        default="rerank-v4.0-pro", validation_alias="COHERE_RERANK_MODEL"
    )
    cohere_timeout_seconds: float = Field(
        default=20.0, gt=0, validation_alias="COHERE_TIMEOUT_SECONDS"
    )
    cohere_max_retries: int = Field(default=5, ge=0, validation_alias="COHERE_MAX_RETRIES")
    cohere_base_delay: float = Field(
        default=1.0, gt=0, validation_alias="COHERE_BASE_DELAY_SECONDS"
    )
    cohere_max_delay: float = Field(default=30.0, gt=0, validation_alias="COHERE_MAX_DELAY_SECONDS")
    cohere_max_tokens_per_doc: int = Field(
        default=4096, ge=256, validation_alias="COHERE_MAX_TOKENS_PER_DOC"
    )

    # -- search shape ------------------------------------------------------------
    prefetch_limit: int = Field(default=100, ge=1, validation_alias="SEARCH_PREFETCH_LIMIT")
    fusion_limit: int = Field(default=50, ge=1, validation_alias="SEARCH_FUSION_LIMIT")
    rerank_candidates: int = Field(default=20, ge=1, validation_alias="SEARCH_RERANK_CANDIDATES")
    default_top_k: int = Field(default=5, ge=1, validation_alias="SEARCH_DEFAULT_TOP_K")
    max_top_k: int = Field(default=50, ge=1, validation_alias="SEARCH_MAX_TOP_K")

    # -- Qdrant / embeddings ---------------------------------------------------
    qdrant_url: str = Field(default="http://qdrant:6333", validation_alias="QDRANT_URL")
    qdrant_grpc_port: int = Field(default=6334, validation_alias="QDRANT_GRPC_PORT")
    qdrant_api_key: SecretStr | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="documents", validation_alias="QDRANT_COLLECTION")
    dense_model: str = Field(default=DEFAULT_DENSE_MODEL, validation_alias="DENSE_MODEL")
    sparse_model: str = Field(default=DEFAULT_SPARSE_MODEL, validation_alias="SPARSE_MODEL")
    colbert_model: str = Field(default=DEFAULT_COLBERT_MODEL, validation_alias="COLBERT_MODEL")
    fastembed_cache_path: str | None = Field(default=None, validation_alias="FASTEMBED_CACHE_PATH")


@lru_cache
def get_settings() -> RetrievalSettings:
    return RetrievalSettings()
