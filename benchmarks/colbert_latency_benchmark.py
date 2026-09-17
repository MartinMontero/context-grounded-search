#!/usr/bin/env python
"""ColBERT latency benchmark: hybrid RRF retrieval with vs. without ColBERT reranking.

Seeds the fixture corpus (``evaluation/fixtures/corpus.json``) into a dedicated
Qdrant collection, embeds every fixture query once, then times the two query
shapes produced by ``retrieval_service.search.build_hybrid_query``:

* ``rrf``          prefetch dense + sparse, top-level RrfQuery
* ``rrf+colbert``  the same fusion nested in a prefetch, ColBERT MaxSim on top

Reports p50 / p95 / mean latency per arm plus recall@k against the corpus'
``relevant_document_ids`` so the latency cost can be weighed against quality.

    python benchmarks/colbert_latency_benchmark.py --seed --iterations 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "packages" / "rag_common"),
    str(ROOT / "services" / "contextual_chunking_service"),
    str(ROOT / "services" / "retrieval_service"),
]

from qdrant_client import models  # noqa: E402

from contextual_chunking_service.chunker import split_text  # noqa: E402
from rag_common.embeddings import HybridEmbedder  # noqa: E402
from rag_common.qdrant_schema import (  # noqa: E402
    COLBERT_VECTOR,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    assert_indexes_present,
    ensure_collection,
    make_client,
)
from retrieval_service.search import build_hybrid_query  # noqa: E402

ARMS = {"rrf": False, "rrf+colbert": True}


def seed(client, collection: str, corpus: dict, embedder: HybridEmbedder) -> int:
    ensure_collection(client, collection)
    assert_indexes_present(client, collection)
    texts, payloads = [], []
    for doc in corpus["documents"]:
        for chunk in split_text(doc["text"], chunk_words=120, overlap_words=20):
            # Poor-man's context (title) keeps the benchmark independent of the Anthropic API.
            texts.append(f"{doc['title']}\n\n{chunk.text}")
            payloads.append(
                {
                    "document_id": doc["document_id"],
                    "tenant_id": "benchmark",
                    "chunk_index": chunk.index,
                    "title": doc["title"],
                    "text": texts[-1],
                    "original_text": chunk.text,
                    "context": doc["title"],
                    "source": "benchmark",
                    "source_url": doc.get("source_url"),
                }
            )
    vectors = embedder.embed_documents(texts)
    points = [
        models.PointStruct(
            id=i,
            vector={
                DENSE_VECTOR: v.dense,
                SPARSE_VECTOR: models.SparseVector(
                    indices=v.sparse.indices, values=v.sparse.values
                ),
                COLBERT_VECTOR: v.colbert,
            },
            payload=p,
        )
        for i, (v, p) in enumerate(zip(vectors, payloads, strict=True))
    ]
    client.upload_points(collection, points=points, batch_size=64, parallel=2, wait=True)
    return len(points)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def run(args: argparse.Namespace) -> dict:
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    client = make_client(url=args.qdrant_url, grpc_port=args.grpc_port, api_key=args.api_key)
    embedder = HybridEmbedder(cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
    seeded = seed(client, args.collection, corpus, embedder) if args.seed else None

    queries = corpus["queries"]
    query_vectors = [embedder.embed_query(q["query"]) for q in queries]
    results: dict[str, dict] = {}
    for arm, use_colbert in ARMS.items():
        latencies: list[float] = []
        hits = 0
        for _ in range(args.warmup):
            client.query_points(
                collection_name=args.collection,
                **build_hybrid_query(
                    query_vectors[0],
                    prefetch_limit=args.prefetch,
                    fusion_limit=args.fusion,
                    limit=args.top_k,
                    use_colbert=use_colbert,
                ),
            )
        for _ in range(args.iterations):
            for q, vec in zip(queries, query_vectors, strict=True):
                kwargs = build_hybrid_query(
                    vec,
                    prefetch_limit=args.prefetch,
                    fusion_limit=args.fusion,
                    limit=args.top_k,
                    use_colbert=use_colbert,
                )
                t0 = time.perf_counter()
                response = client.query_points(collection_name=args.collection, **kwargs)
                latencies.append((time.perf_counter() - t0) * 1000)
                found = {p.payload.get("document_id") for p in response.points}
                hits += int(bool(found & set(q["relevant_document_ids"])))
        results[arm] = {
            "queries": len(queries) * args.iterations,
            "p50_ms": round(percentile(latencies, 50), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
            "mean_ms": round(statistics.fmean(latencies), 2),
            "max_ms": round(max(latencies), 2),
            f"recall@{args.top_k}": round(hits / (len(queries) * args.iterations), 3),
        }
    overhead = results["rrf+colbert"]["p50_ms"] - results["rrf"]["p50_ms"]
    return {
        "collection": args.collection,
        "seeded_points": seeded,
        "prefetch_limit": args.prefetch,
        "fusion_limit": args.fusion,
        "top_k": args.top_k,
        "arms": results,
        "colbert_p50_overhead_ms": round(overhead, 2),
    }


def render(report: dict) -> str:
    lines = [
        f"| arm | queries | p50 ms | p95 ms | mean ms | recall@{report['top_k']} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    recall_key = f"recall@{report['top_k']}"
    for arm, r in report["arms"].items():
        lines.append(
            f"| {arm} | {r['queries']} | {r['p50_ms']} | {r['p95_ms']} | {r['mean_ms']} "
            f"| {r[recall_key]} |"
        )
    lines.append(f"\nColBERT p50 overhead: {report['colbert_p50_overhead_ms']} ms")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--grpc-port", type=int, default=int(os.getenv("QDRANT_GRPC_PORT", "6334")))
    parser.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default="benchmark_documents")
    parser.add_argument("--corpus", default=str(ROOT / "evaluation" / "fixtures" / "corpus.json"))
    parser.add_argument("--seed", action="store_true", help="(re)index the fixture corpus first")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prefetch", type=int, default=100)
    parser.add_argument("--fusion", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output", default=str(ROOT / "benchmarks" / "results" / "colbert_latency.json")
    )
    args = parser.parse_args(argv)

    report = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render(report))
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
