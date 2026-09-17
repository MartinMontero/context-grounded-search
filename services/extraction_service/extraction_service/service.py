"""Extraction orchestration: picks the engine, applies limits, shapes the response."""

from __future__ import annotations

import logging

from extraction_service.defuddle_client import DefuddleClient, DefuddleResult
from extraction_service.html_extractor import HtmlExtraction, extract_html
from extraction_service.mime_extractor import extract_mime
from extraction_service.schemas import (
    Engine,
    ExtractedDocument,
    MimeExtractResponse,
)
from extraction_service.settings import ExtractionSettings
from extraction_service.ssrf import SafeFetcher
from rag_common.errors import PayloadTooLargeError
from rag_common.telemetry import get_tracer

log = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ExtractionService:
    def __init__(
        self, *, settings: ExtractionSettings, fetcher: SafeFetcher, defuddle: DefuddleClient
    ) -> None:
        self.settings = settings
        self.fetcher = fetcher
        self.defuddle = defuddle

    # -- public API -----------------------------------------------------------

    async def from_html(
        self, *, document_id: str, html: str, url: str | None, engine: Engine
    ) -> ExtractedDocument:
        self._check_size(html.encode("utf-8", errors="ignore"))
        with tracer.start_as_current_span("extract.html") as span:
            span.set_attribute("extraction.engine", engine.value)
            doc = await self._extract(document_id, html, url=url, engine=engine, source="html")
        return doc

    async def from_url(
        self, *, document_id: str, url: str, engine: Engine, render: bool
    ) -> ExtractedDocument:
        with tracer.start_as_current_span("extract.url") as span:
            span.set_attribute("extraction.engine", engine.value)
            span.set_attribute("extraction.render", render)
            if render:
                # The sidecar drives Chromium; it repeats the SSRF checks on its side.
                result = await self.defuddle.render_url(url)
                return self._from_defuddle(document_id, result, url=url, source="url", warnings=[])
            fetched = await self.fetcher.fetch(url)
            span.set_attribute("http.redirect_count", fetched.redirects)
            html = fetched.body.decode("utf-8", errors="replace")
            doc = await self._extract(
                document_id, html, url=fetched.final_url, engine=engine, source="url"
            )
            doc.metadata.update(
                {
                    "final_url": fetched.final_url,
                    "content_type": fetched.content_type,
                    "bytes": len(fetched.body),
                }
            )
            return doc

    def from_mime(self, *, document_id: str, raw: bytes) -> MimeExtractResponse:
        self._check_size(raw)
        with tracer.start_as_current_span("extract.mime"):
            extraction = extract_mime(raw, max_attachments=self.settings.max_attachments)
        body = self._from_html_extraction(
            document_id,
            extraction.body,
            url=None,
            source="mime",
            engine="trafilatura" if extraction.body_content_type == "text/html" else "text",
            warnings=extraction.warnings,
        )
        body.metadata["body_content_type"] = extraction.body_content_type
        body.metadata["attachment_count"] = len(extraction.attachments)
        return MimeExtractResponse(
            document_id=document_id,
            headers=extraction.headers,
            body=body,
            attachments=extraction.attachments,
        )

    # -- internals ------------------------------------------------------------

    def _check_size(self, payload: bytes) -> None:
        if len(payload) > self.settings.max_payload_bytes:
            raise PayloadTooLargeError(
                f"payload of {len(payload)} bytes exceeds {self.settings.max_payload_bytes}"
            )

    async def _extract(
        self, document_id: str, html: str, *, url: str | None, engine: Engine, source: str
    ) -> ExtractedDocument:
        if engine is Engine.defuddle:
            result = await self.defuddle.parse_html(html, url=url)
            return self._from_defuddle(document_id, result, url=url, source=source, warnings=[])

        extraction = extract_html(html, url=url)
        warnings = list(extraction.warnings)
        if engine is Engine.auto and extraction.is_thin:
            try:
                result = await self.defuddle.parse_html(html, url=url)
            except Exception as exc:  # keep the trafilatura result if the sidecar is down
                warnings.append(f"defuddle fallback unavailable: {type(exc).__name__}")
                log.warning("defuddle fallback failed", extra={"error": str(exc)})
            else:
                if result.word_count > extraction.word_count:
                    warnings.append("trafilatura result was thin; used Defuddle")
                    return self._from_defuddle(
                        document_id, result, url=url, source=source, warnings=warnings
                    )
        return self._from_html_extraction(
            document_id, extraction, url=url, source=source, engine="trafilatura", warnings=warnings
        )

    @staticmethod
    def _from_html_extraction(
        document_id: str,
        extraction: HtmlExtraction,
        *,
        url: str | None,
        source: str,
        engine: str,
        warnings: list[str],
    ) -> ExtractedDocument:
        return ExtractedDocument(
            document_id=document_id,
            source=source,  # type: ignore[arg-type]
            source_url=url,
            title=extraction.title,
            author=extraction.author,
            date=extraction.date,
            description=extraction.description,
            site_name=extraction.site_name,
            language=extraction.language,
            content=extraction.content,
            content_format="markdown" if engine != "text" else "text",
            word_count=extraction.word_count,
            engine=engine,
            warnings=warnings,
        )

    @staticmethod
    def _from_defuddle(
        document_id: str,
        result: DefuddleResult,
        *,
        url: str | None,
        source: str,
        warnings: list[str],
    ) -> ExtractedDocument:
        return ExtractedDocument(
            document_id=document_id,
            source=source,  # type: ignore[arg-type]
            source_url=result.final_url or url,
            title=result.title,
            author=result.author,
            date=result.published,
            description=result.description,
            site_name=result.site,
            content=result.content,
            word_count=result.word_count or len(result.content.split()),
            engine="defuddle",
            warnings=warnings,
        )
