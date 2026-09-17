"""OpenAPI schemas for the extraction service."""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Engine(StrEnum):
    trafilatura = "trafilatura"
    defuddle = "defuddle"
    auto = "auto"  # trafilatura first, Defuddle when the result is thin


def new_document_id() -> str:
    return uuid.uuid4().hex


class Attachment(BaseModel):
    filename: str | None
    content_type: str
    size_bytes: int
    content_id: str | None = None
    text: str | None = Field(default=None, description="Decoded text for text/* attachments")


class ExtractedDocument(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "ExtractedDocument"})

    document_id: str
    source: Literal["html", "url", "mime"]
    source_url: str | None = None
    title: str | None = None
    author: str | None = None
    date: str | None = None
    description: str | None = None
    site_name: str | None = None
    language: str | None = None
    content: str = Field(description="Main content as Markdown")
    content_format: Literal["markdown", "text"] = "markdown"
    word_count: int
    engine: str
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractHtmlRequest(BaseModel):
    document_id: str | None = Field(default=None, max_length=256)
    html: str = Field(min_length=1)
    url: str | None = Field(
        default=None, description="Original URL, used for relative links/metadata"
    )
    engine: Engine = Engine.auto


class ExtractUrlRequest(BaseModel):
    document_id: str | None = Field(default=None, max_length=256)
    url: str = Field(min_length=8, max_length=4096, examples=["https://example.com/article"])
    engine: Engine = Engine.auto
    render: bool = Field(
        default=False,
        description="Render with headless Chromium in the Defuddle sidecar (Shadow DOM, JS apps)",
    )


class EmailHeaders(BaseModel):
    subject: str | None
    sender: str | None = Field(alias="from")
    to: list[str] = Field(default_factory=list)
    cc: list[str] = Field(default_factory=list)
    date: str | None
    message_id: str | None

    model_config = ConfigDict(populate_by_name=True)


class MimeExtractResponse(BaseModel):
    document_id: str
    headers: EmailHeaders
    body: ExtractedDocument
    attachments: list[Attachment]
