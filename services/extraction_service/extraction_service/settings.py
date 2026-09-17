from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from rag_common.settings import BaseServiceSettings, _split_csv


class ExtractionSettings(BaseServiceSettings):
    service_name: str = Field(default="extraction-service", validation_alias="SERVICE_NAME")

    defuddle_sidecar_url: str = Field(
        default="http://defuddle-sidecar:3000", validation_alias="DEFUDDLE_SIDECAR_URL"
    )
    max_payload_bytes: int = Field(
        default=10 * 1024 * 1024, ge=1024, validation_alias="EXTRACTION_MAX_PAYLOAD_BYTES"
    )
    fetch_timeout_seconds: float = Field(
        default=15.0, gt=0, le=120, validation_alias="EXTRACTION_FETCH_TIMEOUT_SECONDS"
    )
    max_redirects: int = Field(default=3, ge=0, le=10, validation_alias="EXTRACTION_MAX_REDIRECTS")
    allowed_ports: Annotated[list[int], NoDecode] = Field(
        default=[80, 443], validation_alias="EXTRACTION_ALLOWED_PORTS"
    )
    user_agent: str = Field(
        default="contextual-rag-extractor/0.1 (+https://github.com/contextual-rag)",
        validation_alias="EXTRACTION_USER_AGENT",
    )
    max_attachments: int = Field(default=50, ge=0, validation_alias="EXTRACTION_MAX_ATTACHMENTS")

    @field_validator("allowed_ports", mode="before")
    @classmethod
    def _ports(cls, value: Any) -> Any:
        value = _split_csv(value)
        return [int(v) for v in value] if isinstance(value, list) else value


@lru_cache
def get_settings() -> ExtractionSettings:
    return ExtractionSettings()
