"""Single source of truth for the Qdrant collection layout.

Named vectors
-------------
``dense``   384-d COSINE (FastEmbed ``BAAI/bge-small-en-v1.5``)
``sparse``  BM25 term weights with server-side IDF (``modifier=IDF``)
``colbert`` 128-d multivector, MAX_SIM comparator, HNSW disabled (``m=0``) -
            ColBERT is only ever used to *rerank* prefetched candidates, so an
            index over millions of token vectors would be pure overhead.

Every payload index is created before the first ``upload_points`` call.
"""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient, QdrantClient, models

log = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
COLBERT_VECTOR = "colbert"
DENSE_DIM = 384
COLBERT_DIM = 128

# (field, schema) — created before any upload; keyword fields drive tenant/document filters.
PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("document_id", models.PayloadSchemaType.KEYWORD),
    ("tenant_id", models.PayloadSchemaType.KEYWORD),
    ("source", models.PayloadSchemaType.KEYWORD),
    ("chunk_index", models.PayloadSchemaType.INTEGER),
    ("created_at", models.PayloadSchemaType.DATETIME),
)


def build_vectors_config() -> dict[str, models.VectorParams]:
    return {
        DENSE_VECTOR: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        COLBERT_VECTOR: models.VectorParams(
            size=COLBERT_DIM,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
        ),
    }


def build_sparse_config() -> dict[str, models.SparseVectorParams]:
    return {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}


def make_client(
    *, url: str, grpc_port: int, api_key: str | None, timeout: int = 30
) -> QdrantClient:
    return QdrantClient(
        url=url, grpc_port=grpc_port, api_key=api_key, prefer_grpc=True, timeout=timeout
    )


def make_async_client(
    *, url: str, grpc_port: int, api_key: str | None, timeout: int = 30
) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=url, grpc_port=grpc_port, api_key=api_key, prefer_grpc=True, timeout=timeout
    )


def ensure_collection(client: QdrantClient, collection: str) -> bool:
    """Create the collection and all payload indexes if missing. Returns True when created."""
    created = False
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=build_vectors_config(),
            sparse_vectors_config=build_sparse_config(),
            on_disk_payload=True,
        )
        created = True
        log.info("qdrant collection created", extra={"collection": collection})
    existing = set((client.get_collection(collection).payload_schema or {}).keys())
    for field, schema in PAYLOAD_INDEXES:
        if field in existing:
            continue
        client.create_payload_index(
            collection_name=collection, field_name=field, field_schema=schema, wait=True
        )
        log.info("qdrant payload index created", extra={"collection": collection, "field": field})
    return created


def assert_indexes_present(client: QdrantClient, collection: str) -> None:
    """Guard used right before ``upload_points``: every payload index must already exist."""
    schema = client.get_collection(collection).payload_schema or {}
    missing = [field for field, _ in PAYLOAD_INDEXES if field not in schema]
    if missing:
        raise RuntimeError(f"payload indexes missing on '{collection}': {missing}")


async def qdrant_ready(client: AsyncQdrantClient, collection: str) -> tuple[bool, str]:
    exists = await client.collection_exists(collection)
    return exists, f"collection '{collection}' {'present' if exists else 'missing'}"
