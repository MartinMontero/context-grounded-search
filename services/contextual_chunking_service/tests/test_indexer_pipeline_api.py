from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

from contextual_chunking_service.indexer import QdrantIndexer, build_points, point_id
from contextual_chunking_service.main import build_app
from contextual_chunking_service.pipeline import ChunkingPipeline, assemble_chunks
from contextual_chunking_service.schemas import ChunkRequest, ContextualChunk
from contextual_chunking_service.settings import ChunkingSettings
from rag_common.auth import mint_service_token
from rag_common.embeddings import HybridVectors, SparseVector
from rag_common.qdrant_schema import PAYLOAD_INDEXES

REPO = Path(__file__).resolve().parents[3]
SECRET = "unit-test-secret-that-is-at-least-32-chars-long"
DOC = "\n\n".join(" ".join(f"para{i}word{j}" for j in range(40)) for i in range(6))


def _auth() -> dict[str, str]:
    token = mint_service_token(
        secret=SECRET, issuer="contextual-rag", audience="rag-services", subject="n8n"
    )
    return {"Authorization": f"Bearer {token}"}


def _vectors(n: int) -> list[HybridVectors]:
    return [
        HybridVectors(
            dense=[0.1] * 384,
            sparse=SparseVector(indices=[1, 7], values=[0.5, 1.5]),
            colbert=[[0.2] * 128, [0.3] * 128],
        )
        for _ in range(n)
    ]


def _chunks(n: int = 3) -> list[ContextualChunk]:
    from contextual_chunking_service.chunker import split_text

    req = ChunkRequest(document_id="doc-1", text=DOC, title="T", source="html", tenant_id="acme")
    raw = split_text(DOC, chunk_words=80, overlap_words=0)[:n]
    return assemble_chunks(req, raw, [f"ctx {i}" for i in range(len(raw))])


# --- schema contract ---------------------------------------------------------------


def test_exported_json_schema_matches_model_and_is_strict() -> None:
    exported = json.loads((REPO / "schemas" / "contextual_chunk.schema.json").read_text("utf-8"))
    assert exported == ContextualChunk.model_json_schema()
    assert exported["additionalProperties"] is False
    assert set(exported["required"]) >= {
        "chunk_id",
        "document_id",
        "chunk_index",
        "text",
        "context",
        "contextualized_text",
    }
    with pytest.raises(ValueError):
        ContextualChunk.model_validate({**_chunks(1)[0].model_dump(), "unexpected": 1})


# --- indexer ----------------------------------------------------------------------


def test_points_have_stable_ids_named_vectors_and_payload() -> None:
    chunks = _chunks(2)
    points = list(build_points(chunks, _vectors(2)))
    assert [p.id for p in points] == [point_id("doc-1", 0), point_id("doc-1", 1)]
    assert points[0].id == chunks[0].chunk_id  # chunk_id and point id coincide
    assert set(points[0].vector) == {"dense", "sparse", "colbert"}
    assert isinstance(points[0].vector["sparse"], models.SparseVector)
    assert len(points[0].vector["dense"]) == 384 and len(points[0].vector["colbert"][0]) == 128
    payload = points[0].payload
    assert payload["text"] == chunks[0].contextualized_text and payload["context"] == "ctx 0"
    assert payload["tenant_id"] == "acme" and payload["chunk_index"] == 0
    assert payload["created_at"].endswith("+00:00")


def test_indexer_checks_payload_indexes_deletes_stale_then_uploads_with_spec_params() -> None:
    client = MagicMock()
    client.get_collection.return_value.payload_schema = dict.fromkeys(
        (f for f, _ in PAYLOAD_INDEXES), object()
    )
    indexer = QdrantIndexer(client, collection="documents", batch_size=64, parallel=2)
    chunks = _chunks(3)
    assert indexer.index(chunks, _vectors(3)) == 3
    names = [
        c[0] for c in client.mock_calls if c[0] in ("get_collection", "delete", "upload_points")
    ]
    assert names == ["get_collection", "delete", "upload_points"]
    upload = client.upload_points.call_args.kwargs
    assert upload["collection_name"] == "documents"
    assert (upload["batch_size"], upload["parallel"], upload["wait"]) == (64, 2, True)
    assert len(list(upload["points"])) == 3
    delete_filter = client.delete.call_args.kwargs["points_selector"].filter
    assert delete_filter.must[0].key == "document_id"
    assert delete_filter.must[0].match.value == "doc-1"


def test_indexer_refuses_to_upload_without_payload_indexes() -> None:
    client = MagicMock()
    client.get_collection.return_value.payload_schema = {"document_id": object()}
    indexer = QdrantIndexer(client, collection="documents")
    with pytest.raises(RuntimeError, match="payload indexes missing"):
        indexer.index(_chunks(1), _vectors(1))
    client.upload_points.assert_not_called()


# --- pipeline + API ---------------------------------------------------------------


class FakeContextualizer:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def contextualize(self, document_text: str, chunks: list[str]):
        if self.fail:
            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]
        from contextual_chunking_service.contextualizer import ContextualizationResult
        from contextual_chunking_service.schemas import CacheStats

        stats = CacheStats(
            model="claude-haiku-4-5",
            document_tokens=5000,
            cache_min_tokens=4096,
            cache_eligible=True,
            requests=len(chunks),
            cache_creation_input_tokens=5000,
            cache_read_input_tokens=5000 * (len(chunks) - 1),
            uncached_input_tokens=10,
            output_tokens=20,
        )
        return ContextualizationResult(
            contexts=[f"context {i}" for i in range(len(chunks))], cache=stats, warnings=[]
        )


class FakeEmbedder:
    def embed_documents(self, texts):
        return _vectors(len(texts))


async def test_pipeline_index_runs_all_stages_and_dead_letters_indexing_failures(
    fake_redis,
) -> None:
    from rag_common.dlq import DeadLetterQueue

    client = MagicMock()
    client.get_collection.return_value.payload_schema = dict.fromkeys(
        (f for f, _ in PAYLOAD_INDEXES), object()
    )
    settings = ChunkingSettings(chunk_words=80, chunk_overlap_words=0)
    pipeline = ChunkingPipeline(
        settings=settings,
        contextualizer=FakeContextualizer(),
        embedder=FakeEmbedder(),
        indexer=QdrantIndexer(client, collection="documents"),
        dlq=DeadLetterQueue(fake_redis),
    )
    response = await pipeline.index(ChunkRequest(document_id="doc-9", text=DOC))
    assert response.chunks_indexed == 3 and response.collection == "documents"
    assert response.cache.cache_read_input_tokens == 10000

    client.upload_points.side_effect = RuntimeError("grpc unavailable")
    with pytest.raises(RuntimeError):
        await pipeline.index(ChunkRequest(document_id="doc-10", text=DOC))
    entries = await fake_redis.xrange("rag.dlq")
    assert [(f["document_id"], f["stage"]) for _, f in entries] == [("doc-10", "indexing")]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, patch_redis, fake_redis_sync):
    import contextual_chunking_service.main as main_mod

    fake = SimpleNamespace(messages=None)

    async def close():
        return None

    fake.close = close
    monkeypatch.setattr(main_mod.anthropic, "AsyncAnthropic", lambda **_: fake)
    app = build_app(ChunkingSettings(chunk_words=80, chunk_overlap_words=0), with_indexing=False)
    with TestClient(app) as client:
        yield client, app, fake_redis_sync


def test_chunk_endpoint_returns_strict_chunks_and_cache_stats(api) -> None:
    client, app, _ = api
    app.state.pipeline.contextualizer = FakeContextualizer()
    body = {"document_id": "doc-api", "text": DOC, "title": "Doc", "tenant_id": "acme"}
    assert client.post("/v1/chunk", json=body).status_code == 401
    r = client.post("/v1/chunk", json=body, headers=_auth())
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["chunks"]) == 3
    first = data["chunks"][0]
    assert first["context"] == "context 0"
    assert first["contextualized_text"].startswith("context 0\n\npara0word0")
    assert first["chunk_id"] == point_id("doc-api", 0)
    assert data["cache"]["cache_eligible"] is True and data["cache"]["requests"] == 3
    ContextualChunk.model_validate(first)
    schema = client.get("/v1/schemas/contextual-chunk", headers=_auth()).json()
    assert schema["title"] == "ContextualChunk"
    rejected = client.post("/v1/chunk", json={**body, "bogus": 1}, headers=_auth())
    assert rejected.status_code == 422


def test_chunk_failures_are_dead_lettered_as_chunking_stage(api) -> None:
    client, app, redis = api
    app.state.pipeline.contextualizer = FakeContextualizer(fail=True)
    r = client.post("/v1/chunk", json={"document_id": "doc-fail", "text": DOC}, headers=_auth())
    assert r.status_code == 500
    entries = redis.xrange("rag.dlq")
    assert [(f["document_id"], f["stage"], f["error_type"]) for _, f in entries] == [
        ("doc-fail", "chunking", "APIConnectionError")
    ]
    empty = client.post(
        "/v1/chunk", json={"document_id": "doc-empty", "text": "  \n "}, headers=_auth()
    )
    assert empty.status_code == 422 and empty.json()["error"]["type"] == "invalid_document"
