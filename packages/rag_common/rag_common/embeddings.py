"""Client-side hybrid embeddings with FastEmbed (dense + BM25 sparse + ColBERT).

The same class serves indexing (``embed_documents``) and querying
(``embed_query``); BM25 and ColBERT both have distinct document/query encoders,
so the two paths must not be mixed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding

log = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"  # pair with modifier=IDF server-side
DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"  # 128-d multivector


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class HybridVectors:
    dense: list[float]
    sparse: SparseVector
    colbert: list[list[float]]


class HybridEmbedder:
    def __init__(
        self,
        *,
        dense_model: str = DEFAULT_DENSE_MODEL,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        colbert_model: str = DEFAULT_COLBERT_MODEL,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        log.info(
            "loading fastembed models",
            extra={"dense": dense_model, "sparse": sparse_model, "colbert": colbert_model},
        )
        self.dense = TextEmbedding(model_name=dense_model, cache_dir=cache_dir, threads=threads)
        self.sparse = SparseTextEmbedding(
            model_name=sparse_model, cache_dir=cache_dir, threads=threads
        )
        self.colbert = LateInteractionTextEmbedding(
            model_name=colbert_model, cache_dir=cache_dir, threads=threads
        )
        self.dense_dim = self._dim(dense_model, TextEmbedding)
        self.colbert_dim = self._dim(colbert_model, LateInteractionTextEmbedding)

    @staticmethod
    def _dim(model_name: str, cls: type[TextEmbedding] | type[LateInteractionTextEmbedding]) -> int:
        for entry in cls.list_supported_models():
            if entry["model"] == model_name:
                return int(entry["dim"])
        raise ValueError(f"unknown FastEmbed model: {model_name}")

    def embed_documents(self, texts: Sequence[str], *, batch_size: int = 32) -> list[HybridVectors]:
        dense = [
            np.asarray(v, dtype=np.float32).tolist()
            for v in self.dense.embed(texts, batch_size=batch_size)
        ]
        sparse = [
            SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            for s in self.sparse.embed(texts, batch_size=batch_size)
        ]
        colbert = [
            np.asarray(m, dtype=np.float32).tolist()
            for m in self.colbert.embed(texts, batch_size=batch_size)
        ]
        if not (len(dense) == len(sparse) == len(colbert) == len(texts)):
            raise RuntimeError("embedding count mismatch")
        return [
            HybridVectors(dense=d, sparse=s, colbert=c)
            for d, s, c in zip(dense, sparse, colbert, strict=True)
        ]

    def embed_query(self, text: str) -> HybridVectors:
        dense = np.asarray(next(iter(self.dense.query_embed(text))), dtype=np.float32).tolist()
        s = next(iter(self.sparse.query_embed(text)))
        colbert = np.asarray(next(iter(self.colbert.query_embed(text))), dtype=np.float32).tolist()
        return HybridVectors(
            dense=dense,
            sparse=SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
            colbert=colbert,
        )
