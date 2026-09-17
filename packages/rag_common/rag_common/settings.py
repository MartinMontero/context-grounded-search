"""Base settings shared by every service (pydantic-settings v2).

Each service subclasses :class:`BaseServiceSettings`, adds its own fields and
exposes an ``@lru_cache``-wrapped ``get_settings()`` so the environment is read
exactly once per process.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _split_csv(value: Any) -> Any:
    """Accept ``"a,b"`` from the environment as well as a real list."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_by_name=True,
        validate_by_alias=True,
    )

    # -- identity -----------------------------------------------------------
    service_name: str = Field(default="rag-service", validation_alias="SERVICE_NAME")
    service_version: str = Field(default="0.1.0", validation_alias="SERVICE_VERSION")
    environment: str = Field(default="development", validation_alias="APP_ENV")

    # -- observability -------------------------------------------------------
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_service_namespace: str = Field(
        default="contextual-rag", validation_alias="OTEL_SERVICE_NAMESPACE"
    )

    # -- infrastructure ------------------------------------------------------
    redis_url: SecretStr = Field(
        default=SecretStr("redis://localhost:6379/1"), validation_alias="REDIS_URL"
    )
    dlq_stream: str = Field(default="rag.dlq", validation_alias="DLQ_STREAM")

    # -- service-to-service auth --------------------------------------------
    service_jwt_secret: SecretStr = Field(validation_alias="SERVICE_JWT_SECRET")
    service_jwt_issuer: str = Field(default="contextual-rag", validation_alias="SERVICE_JWT_ISSUER")
    service_jwt_audience: str = Field(
        default="rag-services", validation_alias="SERVICE_JWT_AUDIENCE"
    )
    service_jwt_leeway_seconds: int = Field(default=30, ge=0, validation_alias="SERVICE_JWT_LEEWAY")

    # -- HTTP hardening ------------------------------------------------------
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default=["localhost", "127.0.0.1"], validation_alias="ALLOWED_HOSTS"
    )
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default=[], validation_alias="CORS_ALLOW_ORIGINS"
    )
    gzip_minimum_size: int = Field(default=1024, ge=0, validation_alias="GZIP_MINIMUM_SIZE")
    rate_limit_requests: int = Field(default=120, ge=1, validation_alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(
        default=60, ge=1, validation_alias="RATE_LIMIT_WINDOW_SECONDS"
    )
    rate_limit_fail_open: bool = Field(default=True, validation_alias="RATE_LIMIT_FAIL_OPEN")

    @field_validator("allowed_hosts", "cors_allow_origins", mode="before")
    @classmethod
    def _csv(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("otel_exporter_otlp_endpoint", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @field_validator("service_jwt_secret")
    @classmethod
    def _secret_strength(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("SERVICE_JWT_SECRET must be at least 32 characters")
        return value
