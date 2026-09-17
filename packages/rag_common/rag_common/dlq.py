"""Dead-letter queue on a Redis Stream (``rag.dlq``).

Every failed pipeline stage (extraction, chunking, indexing, ...) appends an
entry with exactly the fields ``document_id``, ``stage``, ``error_type``,
``error_message`` and ``timestamp``. Consumers (n8n, an operator, a replay job)
read it with ``XREAD``/``XREADGROUP``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

Stage = Literal["extraction", "chunking", "indexing", "retrieval", "rerank", "orchestration"]
DLQ_FIELDS = ("document_id", "stage", "error_type", "error_message", "timestamp")
_MAX_MESSAGE = 4000


class DeadLetterQueue:
    def __init__(self, redis: Redis, *, stream: str = "rag.dlq", maxlen: int = 100_000) -> None:
        self.redis = redis
        self.stream = stream
        self.maxlen = maxlen

    @staticmethod
    def build_entry(
        *,
        document_id: str,
        stage: Stage | str,
        error_type: str,
        error_message: str,
        timestamp: datetime | None = None,
    ) -> dict[str, str]:
        ts = timestamp or datetime.now(UTC)
        return {
            "document_id": document_id,
            "stage": stage,
            "error_type": error_type,
            "error_message": error_message[:_MAX_MESSAGE],
            "timestamp": ts.isoformat(timespec="milliseconds"),
        }

    async def publish(
        self,
        *,
        document_id: str,
        stage: Stage | str,
        error: BaseException | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> str | None:
        """Append an entry; returns the stream id, or ``None`` if Redis itself failed."""
        if error is not None:
            error_type = error_type or type(error).__name__
            error_message = error_message or (str(error) or repr(error))
        entry = self.build_entry(
            document_id=document_id,
            stage=stage,
            error_type=error_type or "UnknownError",
            error_message=error_message or "",
        )
        fields: dict[Any, Any] = dict(entry)
        try:
            entry_id = await self.redis.xadd(
                self.stream, fields, maxlen=self.maxlen, approximate=True
            )
        except (RedisError, OSError) as exc:  # never let the DLQ mask the original failure
            log.error("dlq publish failed", extra={"dlq_entry": entry, "error": str(exc)})
            return None
        log.warning("dlq entry published", extra={"dlq_entry": entry, "stream_id": entry_id})
        return str(entry_id)

    @asynccontextmanager
    async def guard(self, *, document_id: str, stage: Stage | str) -> AsyncIterator[None]:
        """Publish to the DLQ when the wrapped block raises, then re-raise."""
        try:
            yield
        except Exception as exc:
            await self.publish(document_id=document_id, stage=stage, error=exc)
            raise
