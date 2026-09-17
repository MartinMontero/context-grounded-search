"""Qdrant indexing: deterministic point ids, payload indexes first, then ``upload_points``."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from qdrant_client import QdrantClient, models

from contextual_chunking_service.schemas import ContextualChunk
from rag_common.embeddings import HybridVectors
from rag_common.qdrant_schema import (
    COLBERT_VECTOR,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    assert_indexes_present,
    ensure_collection,
)
from rag_common.telemetry import get_tracer

log = logging.getLogger(__name__)
tracer = get_tracer(__name__)
POINT_NAMESPACE = uuid.UUID("2b6c3c6e-4f0f-4d5f-9d0a-6c0f2e6b1a11")


def point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


def build_points(
    chunks: Sequence[ContextualChunk], vectors: Sequence[HybridVectors]
) -> Iterable[models.PointStruct]:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors length mismatch")
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    for chunk, vec in zip(chunks, vectors, strict=True):
        yield models.PointStruct(
            id=point_id(chunk.document_id, chunk.chunk_index),
            vector={
                DENSE_VECTOR: vec.dense,
                SPARSE_VECTOR: models.SparseVector(
                    indices=vec.sparse.indices, values=vec.sparse.values
                ),
                COLBERT_VECTOR: vec.colbert,
            },
            payload={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "tenant_id": chunk.tenant_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.contextualized_text,
                "original_text": chunk.text,
                "context": chunk.context,
                "title": chunk.title,
                "source": chunk.source,
                "source_url": chunk.source_url,
                "created_at": created_at,
                "metadata": chunk.metadata,
            },
        )


class QdrantIndexer:
    def __init__(
        self,
        client: QdrantClient,
        *,
        collection: str,
        batch_size: int = 64,
        parallel: int = 2,
    ) -> None:
        self.client = client
        self.collection = collection
        self.batch_size = batch_size
        self.parallel = parallel

    def ensure_ready(self) -> None:
        ensure_collection(self.client, self.collection)

    def delete_document(self, document_id: str) -> None:
        """Remove stale chunks so a re-ingest with fewer chunks leaves no orphans."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def index(self, chunks: Sequence[ContextualChunk], vectors: Sequence[HybridVectors]) -> int:
        """Blocking; call via ``asyncio.to_thread`` from the request path."""
        if not chunks:
            return 0
        with tracer.start_as_current_span("qdrant.upload_points") as span:
            span.set_attribute("collection", self.collection)
            span.set_attribute("points", len(chunks))
            # Every payload index must exist BEFORE the first point lands.
            assert_indexes_present(self.client, self.collection)
            self.delete_document(chunks[0].document_id)
            self.client.upload_points(
                collection_name=self.collection,
                points=build_points(chunks, vectors),
                batch_size=self.batch_size,
                parallel=self.parallel,
                wait=True,
            )
        log.info(
            "chunks indexed",
            extra={
                "collection": self.collection,
                "document_id": chunks[0].document_id,
                "points": len(chunks),
            },
        )
        return len(chunks)
