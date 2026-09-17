"""``/v1/extract/*`` endpoints (JWT + rate limited). Failures are dead-lettered."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from starlette.datastructures import UploadFile

from extraction_service.schemas import (
    ExtractedDocument,
    ExtractHtmlRequest,
    ExtractUrlRequest,
    MimeExtractResponse,
    new_document_id,
)
from rag_common.auth import require_service_token
from rag_common.errors import InvalidDocumentError, PayloadTooLargeError
from rag_common.ratelimit import rate_limit

router = APIRouter(
    prefix="/v1/extract",
    tags=["extraction"],
    dependencies=[Depends(require_service_token), Depends(rate_limit)],
)


async def _read_limited(request: Request, max_bytes: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise PayloadTooLargeError(f"request body exceeds {max_bytes} bytes")
    return bytes(body)


@router.post("/html", response_model=ExtractedDocument, summary="Extract content from raw HTML")
async def extract_html(body: ExtractHtmlRequest, request: Request) -> ExtractedDocument:
    document_id = body.document_id or new_document_id()
    async with request.app.state.dlq.guard(document_id=document_id, stage="extraction"):
        return await request.app.state.extraction.from_html(
            document_id=document_id, html=body.html, url=body.url, engine=body.engine
        )


@router.post(
    "/url",
    response_model=ExtractedDocument,
    summary="Fetch a URL (SSRF-protected) and extract its content",
    responses={
        403: {"description": "Target blocked by SSRF policy"},
        413: {"description": "Too large"},
    },
)
async def extract_url(body: ExtractUrlRequest, request: Request) -> ExtractedDocument:
    document_id = body.document_id or new_document_id()
    async with request.app.state.dlq.guard(document_id=document_id, stage="extraction"):
        return await request.app.state.extraction.from_url(
            document_id=document_id, url=body.url, engine=body.engine, render=body.render
        )


@router.post(
    "/mime",
    response_model=MimeExtractResponse,
    summary="Parse a multipart MIME message (raw message/rfc822 body or multipart 'file' upload)",
    openapi_extra={
        "requestBody": {
            "content": {
                "message/rfc822": {"schema": {"type": "string", "format": "binary"}},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                },
            },
            "required": True,
        }
    },
)
async def extract_mime(
    request: Request,
    document_id: str | None = Query(default=None, max_length=256),
) -> MimeExtractResponse:
    document_id = document_id or new_document_id()
    settings = request.app.state.settings
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    async with request.app.state.dlq.guard(document_id=document_id, stage="extraction"):
        if content_type == "multipart/form-data":
            form = await request.form(max_files=1, max_part_size=settings.max_payload_bytes)
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise InvalidDocumentError("multipart upload must contain a 'file' part")
            raw = await upload.read(settings.max_payload_bytes + 1)
            if len(raw) > settings.max_payload_bytes:
                raise PayloadTooLargeError(f"file exceeds {settings.max_payload_bytes} bytes")
        else:
            raw = await _read_limited(request, settings.max_payload_bytes)
        return request.app.state.extraction.from_mime(document_id=document_id, raw=raw)
