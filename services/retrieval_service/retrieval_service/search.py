"""Hybrid search with Qdrant's Universal Query API.

Pipeline (one round-trip)::

    prefetch: [ Prefetch(                         <- nested prefetch object
                  prefetch=[dense top-N, sparse top-N],
                  query=RrfQuery(rrf=Rrf()),        <- fuse the two lists
                  limit=fusion_limit) ]
    query:    <colbert multivector>, using="colbert"  <- MaxSim rerank of the fused list

With ``use_colbert=False`` the RRF fusion becomes the top-level query (the
benchmark's "without ColBERT" arm).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from rag_common.embeddings import HybridEmbedder, HybridVectors
from rag_common.qdrant_schema import COLBERT_VECTOR, DENSE_VECTOR, SPARSE_VECTOR
from rag_common.telemetry import get_tracer

tracer = get_tracer(__name__)


@dataclass
class Candidate:
    point_id: str
    qdrant_score: float
    payload: dict[str, Any] = field(default_factory=dict)


def build_filter(
    *,
    tenant_id: str | None = None,
    document_ids: Sequence[str] | None = None,
    metadata_filters: dict[str, str | int | bool] | None = None,
) -> models.Filter | None:
    must: list[models.Condition] = []
    if tenant_id:
        must.append(
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
        )
    if document_ids:
        must.append(
            models.FieldCondition(key="document_id", match=models.MatchAny(any=list(document_ids)))
        )
    for key, value in (metadata_filters or {}).items():
        must.append(
            models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
        )
    return models.Filter(must=must) if must else None


def build_hybrid_query(
    vectors: HybridVectors,
    *,
    prefetch_limit: int,
    fusion_limit: int,
    limit: int,
    use_colbert: bool = True,
    query_filter: models.Filter | None = None,
) -> dict[str, Any]:
    """Keyword arguments for ``QdrantClient.query_points`` (sync or async)."""
    leaves = [
        models.Prefetch(
            query=vectors.dense, using=DENSE_VECTOR, limit=prefetch_limit, filter=query_filter
        ),
        models.Prefetch(
            query=models.SparseVector(indices=vectors.sparse.indices, values=vectors.sparse.values),
            using=SPARSE_VECTOR,
            limit=prefetch_limit,
            filter=query_filter,
        ),
    ]
    fusion = models.RrfQuery(rrf=models.Rrf())
    if not use_colbert:
        return {"prefetch": leaves, "query": fusion, "limit": limit, "with_payload": True}
    fused = models.Prefetch(prefetch=leaves, query=fusion, limit=fusion_limit)
    return {
        "prefetch": [fused],
        "query": vectors.colbert,  # list[list[float]] -> late-interaction MaxSim rerank
        "using": COLBERT_VECTOR,
        "limit": limit,
        "with_payload": True,
    }


class HybridSearcher:
    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        embedder: HybridEmbedder,
        collection: str,
        prefetch_limit: int,
        fusion_limit: int,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.prefetch_limit = prefetch_limit
        self.fusion_limit = fusion_limit

    async def search(
        self,
        query: str,
        *,
        limit: int,
        use_colbert: bool = True,
        query_filter: models.Filter | None = None,
    ) -> tuple[list[Candidate], dict[str, float]]:
        timings: dict[str, float] = {}
        with tracer.start_as_current_span("search.embed_query"):
            t0 = time.perf_counter()
            vectors = await asyncio.to_thread(self.embedder.embed_query, query)
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000
        kwargs = build_hybrid_query(
            vectors,
            prefetch_limit=self.prefetch_limit,
            fusion_limit=self.fusion_limit,
            limit=limit,
            use_colbert=use_colbert,
            query_filter=query_filter,
        )
        with tracer.start_as_current_span("search.qdrant_query") as span:
            span.set_attribute("qdrant.collection", self.collection)
            span.set_attribute("search.colbert", use_colbert)
            t0 = time.perf_counter()
            response = await self.client.query_points(collection_name=self.collection, **kwargs)
            timings["qdrant_ms"] = (time.perf_counter() - t0) * 1000
        candidates = [
            Candidate(
                point_id=str(p.id), qdrant_score=float(p.score), payload=dict(p.payload or {})
            )
            for p in response.points
        ]
        return candidates, timings
