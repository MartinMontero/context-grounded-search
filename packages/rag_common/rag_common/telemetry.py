"""OpenTelemetry tracing: OTLP export when an endpoint is configured, no-op otherwise."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from rag_common.settings import BaseServiceSettings

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)
_configured = False


def configure_telemetry(settings: BaseServiceSettings) -> None:
    """Install a TracerProvider once per process and instrument httpx + redis."""
    global _configured
    if _configured:
        return
    resource = Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": settings.service_version,
            "service.namespace": settings.otel_service_namespace,
            "deployment.environment.name": settings.environment,
        }
    )
    provider = TracerProvider(resource=resource)
    if settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=settings.otel_exporter_otlp_endpoint.startswith("http://"),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        log.info(
            "otel exporter configured", extra={"endpoint": settings.otel_exporter_otlp_endpoint}
        )
    trace.set_tracer_provider(provider)
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    _configured = True


def instrument_app(app: FastAPI) -> None:
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,ready")


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
