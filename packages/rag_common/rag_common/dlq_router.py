"""``POST /internal/dlq`` — lets n8n's Error Trigger workflow dead-letter a document.

The services dead-letter their own failures; this endpoint covers failures that
happen *between* services (HTTP timeouts, n8n node errors) where no service saw
the exception.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from rag_common.auth import require_service_token
from rag_common.dlq import DLQ_FIELDS
from rag_common.ratelimit import rate_limit

router = APIRouter(
    prefix="/internal",
    tags=["dlq"],
    dependencies=[Depends(require_service_token), Depends(rate_limit)],
)


class DLQPublishRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=256)
    stage: str = Field(min_length=1, max_length=64, examples=["orchestration"])
    error_type: str = Field(min_length=1, max_length=128)
    error_message: str = Field(default="", max_length=4000)


class DLQPublishResponse(BaseModel):
    stream: str
    entry_id: str | None
    fields: list[str] = list(DLQ_FIELDS)


@router.post(
    "/dlq",
    response_model=DLQPublishResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Publish a dead-letter entry to the rag.dlq stream",
)
async def publish_dlq(body: DLQPublishRequest, request: Request) -> DLQPublishResponse:
    dlq = request.app.state.dlq
    entry_id = await dlq.publish(
        document_id=body.document_id,
        stage=body.stage,
        error_type=body.error_type,
        error_message=body.error_message,
    )
    return DLQPublishResponse(stream=dlq.stream, entry_id=entry_id)
