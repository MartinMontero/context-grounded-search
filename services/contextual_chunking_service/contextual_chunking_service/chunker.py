"""Deterministic, paragraph-aware text splitting with word overlap.

Paragraphs (blank-line separated) are packed into chunks of roughly
``chunk_words`` words. A paragraph longer than the budget is split on sentence
boundaries, then on words. Consecutive chunks share ``overlap_words`` words so
that a fact straddling a boundary is embedded at least once in full.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RawChunk:
    index: int
    text: str
    start_char: int
    end_char: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _words(text: str) -> int:
    return len(text.split())


def _split_long_paragraph(paragraph: str, chunk_words: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        n = _words(sentence)
        if n > chunk_words:  # a single run-on "sentence": hard split on words
            if current:
                pieces.append(" ".join(current))
                current, current_words = [], 0
            words = sentence.split()
            for i in range(0, len(words), chunk_words):
                pieces.append(" ".join(words[i : i + chunk_words]))
            continue
        if current_words + n > chunk_words and current:
            pieces.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def _units(text: str, chunk_words: int) -> list[str]:
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = _WHITESPACE.sub(" ", paragraph).strip()
        if not paragraph:
            continue
        if _words(paragraph) <= chunk_words:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, chunk_words))
    return units


def split_text(text: str, *, chunk_words: int = 300, overlap_words: int = 40) -> list[RawChunk]:
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")
    units = _units(text, chunk_words)
    if not units:
        return []

    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        n = _words(unit)
        if current and current_words + n > chunk_words:
            groups.append(current)
            current, current_words = [], 0
        current.append(unit)
        current_words += n
    if current:
        groups.append(current)

    chunks: list[RawChunk] = []
    cursor = 0
    previous_tail = ""
    for index, group in enumerate(groups):
        body = "\n\n".join(group)
        # Locate the body's first unit in the source to keep char offsets honest.
        anchor = text.find(group[0].split(" ", 1)[0], cursor)
        start = anchor if anchor >= 0 else cursor
        end = min(len(text), start + len(body))
        cursor = max(cursor, start + 1)
        chunk_text = f"{previous_tail}\n\n{body}".strip() if previous_tail else body
        chunks.append(RawChunk(index=index, text=chunk_text, start_char=start, end_char=end))
        if overlap_words:
            previous_tail = " ".join(body.split()[-overlap_words:])
    return chunks


def estimate_tokens(text: str) -> int:
    """Cheap upper-bound estimate (~4 chars/token for English prose)."""
    return max(1, -(-len(text) // 4))
