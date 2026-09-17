from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock

from contextual_chunking_service.chunker import estimate_tokens, split_text
from contextual_chunking_service.contextualizer import (
    SYSTEM_PROMPT,
    AnthropicContextualizer,
    build_document_block,
    build_messages,
)
from rag_common.errors import InvalidDocumentError

PARAGRAPHS = [
    " ".join(f"p{i}w{j}" for j in range(60)) for i in range(10)
]  # 10 paragraphs x 60 words
DOC = "\n\n".join(PARAGRAPHS)


# --- chunker -------------------------------------------------------------------


def test_split_respects_paragraphs_budget_and_overlap() -> None:
    chunks = split_text(DOC, chunk_words=150, overlap_words=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # 150-word budget fits two 60-word paragraphs per chunk -> 5 chunks.
    assert len(chunks) == 5
    assert chunks[0].text.startswith("p0w0") and "p1w59" in chunks[0].text
    # Overlap: the second chunk starts with the last 10 words of the first body.
    assert chunks[1].text.split()[:10] == PARAGRAPHS[1].split()[-10:]
    assert chunks[1].text.split()[10] == "p2w0"
    assert all(c.word_count <= 150 + 10 for c in chunks)
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(DOC)
    assert [c.start_char for c in chunks] == sorted(c.start_char for c in chunks)


def test_split_long_paragraph_on_sentences_then_words() -> None:
    sentences = " ".join(f"Sentence number {i} has exactly six words." for i in range(40))
    chunks = split_text(sentences, chunk_words=50, overlap_words=0)
    assert len(chunks) == 6  # 40 sentences x 7 words = 280 words / 50 per chunk -> 6 chunks
    assert all(c.word_count <= 50 for c in chunks)
    assert all(c.text.endswith("words.") for c in chunks)  # sentence-aligned boundaries
    run_on = "word " * 120
    hard = split_text(run_on, chunk_words=50, overlap_words=0)
    assert [c.word_count for c in hard] == [50, 50, 20]


def test_split_edge_cases() -> None:
    assert split_text("   \n\n  ") == []
    assert split_text("one small paragraph", chunk_words=100)[0].text == "one small paragraph"
    with pytest.raises(ValueError):
        split_text("x", chunk_words=10, overlap_words=10)
    assert estimate_tokens("abcd" * 10) == 10 and estimate_tokens("") == 1


# --- contextualizer ------------------------------------------------------------


class FakeMessages:
    def __init__(self, prefix_tokens: int = 5000, fail_at: int | None = None) -> None:
        self.prefix_tokens = prefix_tokens
        self.calls: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.first_done = False
        self.started_before_first_done: list[int] = []
        self.fail_at = fail_at

    async def count_tokens(self, **kwargs):
        self.count_kwargs = kwargs
        return SimpleNamespace(input_tokens=self.prefix_tokens)

    async def create(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if index > 0 and not self.first_done:
            self.started_before_first_done.append(index)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        if index == 0:
            self.first_done = True
        if self.fail_at is not None and index == self.fail_at:
            import anthropic

            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]
        chunk_prompt = kwargs["messages"][0]["content"][1]["text"]
        chunk = chunk_prompt.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
        usage = SimpleNamespace(
            cache_creation_input_tokens=self.prefix_tokens if index == 0 else 0,
            cache_read_input_tokens=0 if index == 0 else self.prefix_tokens,
            input_tokens=80,
            output_tokens=25,
        )
        return SimpleNamespace(
            content=[TextBlock(type="text", text=f"This chunk covers {chunk[:12]}.")],
            usage=usage,
            stop_reason="end_turn",
        )


def _contextualizer(messages: FakeMessages, **overrides) -> AnthropicContextualizer:
    client = SimpleNamespace(messages=messages)
    kwargs = {"model": "claude-haiku-4-5", "cache_min_tokens": 4096, "max_concurrency": 3}
    kwargs.update(overrides)
    return AnthropicContextualizer(client, **kwargs)  # type: ignore[arg-type]


def test_document_block_is_first_static_and_marked_ephemeral() -> None:
    block = build_document_block("DOC TEXT")
    messages = build_messages(block, "chunk A")
    content = messages[0]["content"]
    assert content[0] is block
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[0]["text"] == "<document>\nDOC TEXT\n</document>"
    assert "cache_control" not in content[1]
    assert "<chunk>\nchunk A\n</chunk>" in content[1]["text"]
    assert messages[0]["role"] == "user" and len(messages) == 1


async def test_contextualize_warms_cache_then_fans_out_with_identical_prefix() -> None:
    messages = FakeMessages(prefix_tokens=5000)
    ctx = _contextualizer(messages)
    chunks = [f"chunk number {i} text" for i in range(8)]
    result = await ctx.contextualize("full document", chunks)

    assert len(messages.calls) == 8
    # count_tokens measured exactly the cached prefix: system + document block only.
    assert messages.count_kwargs["system"] == SYSTEM_PROMPT
    assert messages.count_kwargs["messages"][0]["content"] == [
        build_document_block("full document")
    ]
    # The document block is byte-identical in every request and always comes first.
    first_blocks = {
        json.dumps(c["messages"][0]["content"][0], sort_keys=True) for c in messages.calls
    }
    assert len(first_blocks) == 1
    assert all(c["system"] == SYSTEM_PROMPT for c in messages.calls)
    assert all(c["model"] == "claude-haiku-4-5" for c in messages.calls)
    # Chunk 0 completed alone (cache write) before any other request started.
    assert messages.started_before_first_done == []
    assert 1 < messages.max_in_flight <= 3
    # Results are in chunk order and usage is aggregated.
    assert result.contexts == [f"This chunk covers {c[:12]}." for c in chunks]
    assert result.cache.cache_eligible is True
    assert result.cache.requests == 8
    assert result.cache.cache_creation_input_tokens == 5000
    assert result.cache.cache_read_input_tokens == 5000 * 7
    assert result.cache.output_tokens == 25 * 8
    assert result.warnings == []


async def test_short_document_is_flagged_as_not_cacheable() -> None:
    messages = FakeMessages(prefix_tokens=1500)
    result = await _contextualizer(messages).contextualize("short", ["a", "b"])
    assert result.cache.cache_eligible is False
    assert result.cache.document_tokens == 1500 and result.cache.cache_min_tokens == 4096
    assert "below the 4096-token cache minimum" in result.warnings[0]
    # cache_control is still sent: harmless when ineligible, and correct if the doc grows.
    assert messages.calls[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


async def test_oversized_document_and_api_failures() -> None:
    with pytest.raises(InvalidDocumentError, match="limit is 1000"):
        await _contextualizer(
            FakeMessages(prefix_tokens=5000), max_document_tokens=1000
        ).contextualize("doc", ["a"])
    from rag_common.errors import UpstreamError

    with pytest.raises(UpstreamError, match=r"messages\.create"):
        await _contextualizer(FakeMessages(fail_at=2)).contextualize("doc", ["a", "b", "c", "d"])
