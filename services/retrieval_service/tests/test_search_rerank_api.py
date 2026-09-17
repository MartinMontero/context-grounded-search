from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from cohere.errors import BadRequestError, InternalServerError, TooManyRequestsError
from fastapi.testclient import TestClient
from qdrant_client import models

from rag_common.auth import mint_service_token
from rag_common.embeddings import HybridVectors, SparseVector
from rag_common.errors import UpstreamError
from rag_common.retry import RetryPolicy
from retrieval_service.main import build_app
from retrieval_service.rerank import CohereReranker, format_document, is_retryable, retry_after_hint
from retrieval_service.search import Candidate, build_filter, build_hybrid_query
from retrieval_service.settings import RetrievalSettings

SECRET = "unit-test-secret-that-is-at-least-32-chars-long"
VECTORS = HybridVectors(
    dense=[0.1] * 384,
    sparse=SparseVector(indices=[3, 9], values=[0.7, 0.2]),
    colbert=[[0.5] * 128, [0.4] * 128, [0.3] * 128],
)


def _auth() -> dict[str, str]:
    token = mint_service_token(
        secret=SECRET, issuer="contextual-rag", audience="rag-services", subject="n8n"
    )
    return {"Authorization": f"Bearer {token}"}


def _candidates(n: int) -> list[Candidate]:
    return [
        Candidate(
            point_id=f"p{i}",
            qdrant_score=1.0 - i * 0.1,
            payload={
                "chunk_id": f"c{i}",
                "document_id": "doc",
                "chunk_index": i,
                "tenant_id": "acme",
                "title": "Title",
                "context": f"ctx {i}",
                "text": f"ctx {i}\n\nbody {i}",
                "original_text": f"body {i}",
                "source_url": "https://x/y",
                "metadata": {"k": "v"},
            },
        )
        for i in range(n)
    ]


# --- query construction ------------------------------------------------------------


def test_nested_prefetch_rrf_then_colbert_rerank() -> None:
    q = build_hybrid_query(VECTORS, prefetch_limit=100, fusion_limit=50, limit=20)
    assert q["using"] == "colbert" and q["limit"] == 20 and q["with_payload"] is True
    assert q["query"] == VECTORS.colbert  # multivector reranks the fused candidates
    (fused,) = q["prefetch"]
    assert isinstance(fused, models.Prefetch)
    assert isinstance(fused.query, models.RrfQuery) and isinstance(fused.query.rrf, models.Rrf)
    assert fused.limit == 50
    dense, sparse = fused.prefetch
    assert (dense.using, dense.limit, dense.query) == ("dense", 100, VECTORS.dense)
    assert sparse.using == "sparse" and sparse.limit == 100
    assert isinstance(sparse.query, models.SparseVector)
    assert (sparse.query.indices, sparse.query.values) == ([3, 9], [0.7, 0.2])


def test_without_colbert_rrf_is_the_top_level_query_and_filters_propagate() -> None:
    flt = build_filter(tenant_id="acme", document_ids=["d1", "d2"], metadata_filters={"lang": "en"})
    q = build_hybrid_query(
        VECTORS, prefetch_limit=10, fusion_limit=5, limit=3, use_colbert=False, query_filter=flt
    )
    assert isinstance(q["query"], models.RrfQuery) and "using" not in q
    assert [p.using for p in q["prefetch"]] == ["dense", "sparse"]
    assert all(p.filter is flt for p in q["prefetch"])
    keys = [c.key for c in flt.must]
    assert keys == ["tenant_id", "document_id", "metadata.lang"]
    assert isinstance(flt.must[1].match, models.MatchAny)
    assert build_filter() is None


# --- reranker ----------------------------------------------------------------------


def test_format_document_is_ordered_yaml() -> None:
    text = format_document(_candidates(1)[0].payload)
    assert text.startswith("title: Title\ncontext: ctx 0\ntext: body 0\nsource_url:")
    assert yaml.safe_load(text) == {
        "title": "Title",
        "context": "ctx 0",
        "text": "body 0",
        "source_url": "https://x/y",
    }
    assert "text: only" in format_document({"text": "only"})


def test_retry_classification() -> None:
    assert is_retryable(TooManyRequestsError(body={}, headers={"Retry-After": "2"}))
    assert is_retryable(InternalServerError(body={}))
    assert not is_retryable(BadRequestError(body={}))
    assert retry_after_hint(TooManyRequestsError(body={}, headers={"Retry-After": "2"})) == 2.0
    assert retry_after_hint(InternalServerError(body={})) is None


class FakeCohere:
    def __init__(self, failures: list[Exception]) -> None:
        self.failures = failures
        self.calls: list[dict] = []

    async def rerank(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        n = len(kwargs["documents"])
        # Distinct, position-dependent scores so the index mapping is observable;
        # like the real API, return the top_n highest-scoring documents.
        results = [SimpleNamespace(index=i, relevance_score=(i + 1) / n) for i in range(n)]
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return SimpleNamespace(results=results[: kwargs["top_n"]])


async def test_rerank_retries_429_with_full_jitter_and_maps_indexes_back() -> None:
    sleeps: list[float] = []

    async def sleep(d: float) -> None:
        sleeps.append(d)

    client = FakeCohere(
        [TooManyRequestsError(body={}, headers={"retry-after": "2"}), InternalServerError(body={})]
    )
    reranker = CohereReranker(
        client, model="rerank-v4.0-pro", policy=RetryPolicy(5, 1.0, 30.0), sleep=sleep
    )
    candidates = _candidates(4)
    reranked, elapsed = await reranker.rerank("q", candidates, top_n=3)
    assert len(client.calls) == 3
    assert client.calls[0]["model"] == "rerank-v4.0-pro" and client.calls[0]["top_n"] == 3
    assert client.calls[0]["documents"][1].startswith("title: Title\ncontext: ctx 1")
    assert sleeps[0] == 2.0  # Retry-After honoured (>= the jittered delay)
    assert 0 <= sleeps[1] <= 2.0  # full jitter within [0, base * 2**1]
    assert [r.candidate.point_id for r in reranked] == ["p3", "p2", "p1"]  # sorted by score desc
    assert [r.original_index for r in reranked] == [3, 2, 1]
    assert reranked[0].relevance_score == 1.0 and elapsed >= 0


async def test_rerank_gives_up_after_policy_and_rejects_non_retryable() -> None:
    async def sleep(_: float) -> None:
        return None

    exhausted = CohereReranker(
        FakeCohere([TooManyRequestsError(body={}) for _ in range(6)]),
        model="m",
        policy=RetryPolicy(max_retries=5),
        sleep=sleep,
    )
    with pytest.raises(UpstreamError, match="after retries") as info:
        await exhausted.rerank("q", _candidates(2), top_n=2)
    assert info.value.details["attempts"] == 6
    bad = FakeCohere([BadRequestError(body={})])
    with pytest.raises(UpstreamError, match="HTTP 400"):
        await CohereReranker(bad, model="m", sleep=sleep).rerank("q", _candidates(2), top_n=2)
    assert len(bad.calls) == 1


# --- API ---------------------------------------------------------------------------


class FakeSearcher:
    prefetch_limit = 100

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    async def search(self, query, *, limit, use_colbert, query_filter):
        self.calls.append({"limit": limit, "colbert": use_colbert, "filter": query_filter})
        if self.fail:
            raise ConnectionError("qdrant down")
        return _candidates(min(limit, 6)), {"embed_ms": 1.0, "qdrant_ms": 2.0}


@pytest.fixture
def api(patch_redis, fake_redis_sync):
    app = build_app(RetrievalSettings(), load_models=False)
    with TestClient(app) as client:
        yield client, app, fake_redis_sync


def test_search_endpoint_reranks_and_orders_hits(api) -> None:
    client, app, _ = api
    app.state.retrieval.searcher = FakeSearcher()

    async def sleep(_: float) -> None:
        return None

    app.state.retrieval.reranker = CohereReranker(
        FakeCohere([]), model="rerank-v4.0-pro", sleep=sleep
    )
    body = {"query": "what is contextual retrieval", "top_k": 3, "tenant_id": "acme"}
    assert client.post("/v1/search", json=body).status_code == 401
    r = client.post("/v1/search", json=body, headers=_auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert [h["rank"] for h in data["hits"]] == [1, 2, 3]
    assert [h["chunk_id"] for h in data["hits"]] == ["c5", "c4", "c3"]  # reranker order wins
    assert data["hits"][0]["rerank_score"] == 1.0 and data["hits"][0]["score"] == 1.0
    assert data["hits"][0]["original_text"] == "body 5"
    assert data["stages"] == {
        "prefetch": {"dense": 100, "sparse": 100},
        "fusion": "rrf",
        "colbert": True,
        "rerank": "rerank-v4.0-pro",
    }
    assert data["candidates_considered"] == 6
    assert set(data["timings"]) == {"embed_ms", "qdrant_ms", "rerank_ms", "total_ms"}
    searcher_call = app.state.retrieval.searcher.calls[0]
    assert searcher_call["limit"] == 20 and searcher_call["filter"].must[0].key == "tenant_id"


def test_search_without_rerank_or_colbert_uses_qdrant_order(api) -> None:
    client, app, _ = api
    app.state.retrieval.searcher = FakeSearcher()
    r = client.post(
        "/v1/search",
        json={"query": "q", "top_k": 2, "rerank": False, "colbert": False},
        headers=_auth(),
    )
    assert r.status_code == 200
    assert [h["chunk_id"] for h in r.json()["hits"]] == ["c0", "c1"]
    assert r.json()["hits"][0]["rerank_score"] is None
    assert r.json()["stages"]["rerank"] is None and r.json()["stages"]["colbert"] is False
    assert app.state.retrieval.searcher.calls[0] == {"limit": 2, "colbert": False, "filter": None}


def test_search_failures_are_dead_lettered(api) -> None:
    client, app, redis = api
    app.state.retrieval.searcher = FakeSearcher(fail=True)
    r = client.post("/v1/search", json={"query": "boom"}, headers=_auth())
    assert r.status_code == 500
    entries = redis.xrange("rag.dlq")
    assert [(f["stage"], f["error_type"], f["document_id"]) for _, f in entries] == [
        ("retrieval", "ConnectionError", "query:boom")
    ]
