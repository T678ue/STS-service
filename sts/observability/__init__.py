"""Observability setup: structured logging, Prometheus metrics, OpenTelemetry tracing, health."""

from sts.observability.logging import setup_logging
from sts.observability.metrics import setup_metrics
from sts.observability.tracing import setup_tracing

__all__ = ["setup_logging", "setup_metrics", "setup_tracing"]


def setup_observability(config) -> None:
    """Initialise all observability subsystems."""
    setup_logging(level=config.log_level, fmt=config.log_format)
    if config.enable_metrics:
        setup_metrics(port=config.metrics_port)
    if config.enable_health:
        from sts.observability.health import start_health_server
        start_health_server(port=config.health_port)
    if config.enable_tracing:
        setup_tracing(endpoint=config.otlp_endpoint)
