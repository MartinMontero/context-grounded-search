"""Search orchestration: hybrid Qdrant retrieval -> optional Cohere rerank -> response."""

from __future__ import annotations

import time

from rag_common.dlq import DeadLetterQueue
from retrieval_service.rerank import CohereReranker
from retrieval_service.schemas import SearchHit, SearchRequest, SearchResponse, Timings
from retrieval_service.search import Candidate, HybridSearcher, build_filter
from retrieval_service.settings import RetrievalSettings


def _hit(rank: int, candidate: Candidate, *, score: float, rerank_score: float | None) -> SearchHit:
    p = candidate.payload
    return SearchHit(
        rank=rank,
        chunk_id=str(p.get("chunk_id") or candidate.point_id),
        document_id=str(p.get("document_id", "")),
        chunk_index=int(p.get("chunk_index", 0)),
        tenant_id=p.get("tenant_id"),
        title=p.get("title"),
        source_url=p.get("source_url"),
        context=str(p.get("context", "")),
        text=str(p.get("text", "")),
        original_text=str(p.get("original_text") or p.get("text", "")),
        score=score,
        qdrant_score=candidate.qdrant_score,
        rerank_score=rerank_score,
        metadata=dict(p.get("metadata") or {}),
    )


class RetrievalService:
    def __init__(
        self,
        *,
        settings: RetrievalSettings,
        searcher: HybridSearcher,
        reranker: CohereReranker | None,
        dlq: DeadLetterQueue,
    ) -> None:
        self.settings = settings
        self.searcher = searcher
        self.reranker = reranker
        self.dlq = dlq

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        candidates_n = request.candidates or self.settings.rerank_candidates
        query_filter = build_filter(
            tenant_id=request.tenant_id,
            document_ids=request.document_ids,
            metadata_filters=request.metadata_filters,
        )
        reranker = self.reranker if request.rerank else None
        use_rerank = reranker is not None
        limit = max(candidates_n, top_k) if use_rerank else top_k

        async with self.dlq.guard(document_id=f"query:{request.query[:64]}", stage="retrieval"):
            candidates, timings = await self.searcher.search(
                request.query, limit=limit, use_colbert=request.colbert, query_filter=query_filter
            )
        rerank_ms = 0.0
        if reranker is not None and candidates:
            async with self.dlq.guard(document_id=f"query:{request.query[:64]}", stage="rerank"):
                reranked, rerank_ms = await reranker.rerank(request.query, candidates, top_n=top_k)
            hits = [
                _hit(i + 1, r.candidate, score=r.relevance_score, rerank_score=r.relevance_score)
                for i, r in enumerate(reranked[:top_k])
            ]
        else:
            hits = [
                _hit(i + 1, c, score=c.qdrant_score, rerank_score=None)
                for i, c in enumerate(candidates[:top_k])
            ]
        total_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(
            query=request.query,
            hits=hits,
            candidates_considered=len(candidates),
            stages={
                "prefetch": {
                    "dense": self.searcher.prefetch_limit,
                    "sparse": self.searcher.prefetch_limit,
                },
                "fusion": "rrf",
                "colbert": request.colbert,
                "rerank": reranker.model if reranker is not None else None,
            },
            timings=Timings(
                embed_ms=round(timings["embed_ms"], 2),
                qdrant_ms=round(timings["qdrant_ms"], 2),
                rerank_ms=round(rerank_ms, 2),
                total_ms=round(total_ms, 2),
            ),
        )
