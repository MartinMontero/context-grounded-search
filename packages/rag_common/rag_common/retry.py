"""Exponential backoff with *full jitter* (AWS Architecture Blog, 2015).

    sleep = random_between(0, min(max_delay, base_delay * 2 ** attempt))

Used for third-party APIs that rate-limit (Cohere 429s, transient 5xx).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0

    def delay_for(self, attempt: int, *, rng: Callable[[], float] = random.random) -> float:
        """Full-jitter delay for a 0-based retry attempt."""
        cap = min(self.max_delay, self.base_delay * (2**attempt))
        return rng() * cap


class RetryExhaustedError(Exception):
    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(f"gave up after {attempts} attempts: {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    should_retry: Callable[[BaseException], bool] | None = None,
    retry_after_hint: Callable[[BaseException], float | None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
    name: str = "operation",
) -> T:
    """Run ``operation`` until it succeeds or ``policy.max_retries`` retries are spent.

    ``retry_after_hint`` lets the caller honour a server-provided ``Retry-After``
    (never exceeding ``max_delay``); otherwise the full-jitter delay is used.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except retry_on as exc:
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= policy.max_retries:
                raise RetryExhaustedError(attempt + 1, exc) from exc
            delay = policy.delay_for(attempt, rng=rng)
            hint = retry_after_hint(exc) if retry_after_hint else None
            if hint is not None:
                delay = min(max(delay, hint), policy.max_delay)
            log.warning(
                "retrying after error",
                extra={
                    "operation": name,
                    "attempt": attempt + 1,
                    "max_retries": policy.max_retries,
                    "delay_seconds": round(delay, 3),
                    "error_type": type(exc).__name__,
                },
            )
            await sleep(delay)
            attempt += 1
