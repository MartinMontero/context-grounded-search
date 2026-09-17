from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from qdrant_client import models

from rag_common.qdrant_schema import (
    COLBERT_VECTOR,
    DENSE_VECTOR,
    PAYLOAD_INDEXES,
    SPARSE_VECTOR,
    assert_indexes_present,
    build_sparse_config,
    build_vectors_config,
    ensure_collection,
)
from rag_common.retry import RetryExhaustedError, RetryPolicy, retry_async


def test_full_jitter_delay_is_bounded_by_cap() -> None:
    policy = RetryPolicy(max_retries=5, base_delay=1.0, max_delay=30.0)
    # rng() == 1.0 gives the cap: 1, 2, 4, 8, 16, then clamped to max_delay.
    caps = [policy.delay_for(a, rng=lambda: 1.0) for a in range(7)]
    assert caps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    # rng() == 0.0 gives 0 - "full" jitter spans the whole [0, cap] interval.
    assert policy.delay_for(4, rng=lambda: 0.0) == 0.0
    assert policy.delay_for(3, rng=lambda: 0.5) == 4.0


async def test_retry_async_retries_then_succeeds_and_honours_hint() -> None:
    attempts = 0
    sleeps: list[float] = []

    class ThrottledError(Exception):
        retry_after = 2.5

    async def op() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ThrottledError()
        return "ok"

    async def sleep(d: float) -> None:
        sleeps.append(d)

    result = await retry_async(
        op,
        policy=RetryPolicy(max_retries=5, base_delay=1.0, max_delay=30.0),
        retry_on=(ThrottledError,),
        retry_after_hint=lambda exc: getattr(exc, "retry_after", None),
        sleep=sleep,
        rng=lambda: 0.1,
    )
    assert result == "ok" and attempts == 3
    assert sleeps == [2.5, 2.5]  # hint wins over the smaller jittered delays 0.1 and 0.2


async def test_retry_async_gives_up_after_max_retries() -> None:
    calls = 0

    async def op() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError()

    async def sleep(_: float) -> None:
        return None

    with pytest.raises(RetryExhaustedError) as info:
        await retry_async(
            op, policy=RetryPolicy(max_retries=5), retry_on=(TimeoutError,), sleep=sleep
        )
    assert calls == 6 and info.value.attempts == 6


async def test_retry_async_respects_should_retry() -> None:
    async def op() -> None:
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        await retry_async(
            op, policy=RetryPolicy(), retry_on=(ValueError,), should_retry=lambda _: False
        )


def test_vector_config_matches_spec() -> None:
    vectors = build_vectors_config()
    dense = vectors[DENSE_VECTOR]
    assert (dense.size, dense.distance) == (384, models.Distance.COSINE)
    assert dense.multivector_config is None
    colbert = vectors[COLBERT_VECTOR]
    assert (colbert.size, colbert.distance) == (128, models.Distance.COSINE)
    assert colbert.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM
    assert colbert.hnsw_config.m == 0
    sparse = build_sparse_config()[SPARSE_VECTOR]
    assert sparse.modifier == models.Modifier.IDF


def test_ensure_collection_creates_collection_then_every_payload_index() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False
    client.get_collection.return_value.payload_schema = {}
    created = ensure_collection(client, "documents")
    assert created is True
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "documents"
    assert set(kwargs["vectors_config"]) == {DENSE_VECTOR, COLBERT_VECTOR}
    assert set(kwargs["sparse_vectors_config"]) == {SPARSE_VECTOR}
    expected = [
        call(collection_name="documents", field_name=field, field_schema=schema, wait=True)
        for field, schema in PAYLOAD_INDEXES
    ]
    assert client.create_payload_index.call_args_list == expected


def test_ensure_collection_is_idempotent_and_guard_detects_missing_index() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.payload_schema = dict.fromkeys(
        (f for f, _ in PAYLOAD_INDEXES), object()
    )
    assert ensure_collection(client, "documents") is False
    client.create_collection.assert_not_called()
    client.create_payload_index.assert_not_called()
    assert_indexes_present(client, "documents")
    client.get_collection.return_value.payload_schema = {"document_id": object()}
    with pytest.raises(RuntimeError, match="tenant_id"):
        assert_indexes_present(client, "documents")
