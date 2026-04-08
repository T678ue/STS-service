"""Prometheus metrics definitions and HTTP server.

Metrics are grouped by subsystem.  Each pipeline records its own metrics
via the module-level metric objects.
"""

from __future__ import annotations

import structlog
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Service-level
# ---------------------------------------------------------------------------
service_info = Info("sts_service", "STS service build information")
active_sessions = Gauge("sts_active_sessions", "Number of active client sessions")

# ---------------------------------------------------------------------------
# STT metrics
# ---------------------------------------------------------------------------


class _STTMetrics:
    transcription_duration = Histogram(
        "sts_stt_transcription_duration_seconds",
        "Time spent transcribing a speech segment",
        labelnames=["engine"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    segments_transcribed = Counter(
        "sts_stt_segments_total",
        "Total speech segments transcribed",
        labelnames=["engine"],
    )
    errors = Counter(
        "sts_stt_errors_total",
        "Total STT errors",
        labelnames=["engine"],
    )


stt_metrics = _STTMetrics()

# ---------------------------------------------------------------------------
# TTS metrics
# ---------------------------------------------------------------------------


class _TTSMetrics:
    synthesis_duration = Histogram(
        "sts_tts_synthesis_duration_seconds",
        "Time spent synthesizing audio",
        labelnames=["engine"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
    chars_synthesized = Counter(
        "sts_tts_chars_total",
        "Total characters synthesized",
        labelnames=["engine"],
    )
    errors = Counter(
        "sts_tts_errors_total",
        "Total TTS errors",
        labelnames=["engine"],
    )


tts_metrics = _TTSMetrics()

# ---------------------------------------------------------------------------
# Audio pipeline metrics
# ---------------------------------------------------------------------------
audio_bytes_received = Counter(
    "sts_audio_bytes_received_total",
    "Total audio bytes received from clients",
)
audio_bytes_sent = Counter(
    "sts_audio_bytes_sent_total",
    "Total audio bytes sent to clients",
)
vad_speech_segments = Counter(
    "sts_vad_speech_segments_total",
    "Total speech segments detected by VAD",
)

# ---------------------------------------------------------------------------
# Transport metrics
# ---------------------------------------------------------------------------
ws_connections = Gauge(
    "sts_ws_connections_active",
    "Active WebSocket connections",
)
grpc_streams = Gauge(
    "sts_grpc_streams_active",
    "Active gRPC streams",
)
transport_errors = Counter(
    "sts_transport_errors_total",
    "Transport-level errors",
    labelnames=["transport", "error_type"],
)


def setup_metrics(port: int = 9090) -> None:
    """Start the Prometheus metrics HTTP server."""
    from sts import __version__

    service_info.info({"version": __version__})
    start_http_server(port)
    logger.info("metrics.server_started", port=port)
