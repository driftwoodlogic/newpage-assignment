from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings

_initialized = False


def init_tracer() -> None:
    global _initialized
    if _initialized:
        return

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "phoenix.project": settings.phoenix_project,
        }
    )
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=settings.phoenix_otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str = "rag"):
    return trace.get_tracer(name)
