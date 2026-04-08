"""OpenTelemetry distributed tracing setup.

Creates spans for:
  - Each audio chunk processed
  - Each STT transcription
  - Each TTS synthesis
  - Each transport message
  - Session lifecycle events
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_tracer = None


def setup_tracing(endpoint: str = "http://localhost:4317") -> None:
    """Initialize OpenTelemetry with OTLP gRPC exporter."""
    global _tracer

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": "sts-service",
            "service.version": "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer("sts-service")
    logger.info("tracing.initialized", endpoint=endpoint)


def get_tracer():
    """Get the STS service tracer, or a no-op tracer if tracing is disabled."""
    if _tracer is not None:
        return _tracer
    from opentelemetry import trace

    return trace.get_tracer("sts-service")
