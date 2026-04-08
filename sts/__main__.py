"""Entry point: uv run python -m sts  or  sts-service CLI.

Usage:
    # Defaults (console logging, WebSocket on 8765, gRPC on 50051)
    uv run python -m sts

    # Development mode (verbose, console logs, no metrics/tracing)
    uv run python -m sts --dev

    # Custom ports
    uv run python -m sts --ws-port 9000 --grpc-port 9001

    # From a config file
    uv run python -m sts --config config.json

    # CPU-only, specific model
    uv run python -m sts --device cpu --stt-model small.en
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sts-service",
        description="STS Speech-to-Speech dialogue service",
    )

    p.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to JSON config file (overrides all other flags)",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: console logging, DEBUG level, metrics disabled",
    )

    # Transport
    g = p.add_argument_group("transport")
    g.add_argument("--ws-host", default=None, help="WebSocket bind host (default: 0.0.0.0)")
    g.add_argument("--ws-port", type=int, default=None, help="WebSocket port (default: 8765)")
    g.add_argument("--grpc-host", default=None, help="gRPC bind host (default: 0.0.0.0)")
    g.add_argument("--grpc-port", type=int, default=None, help="gRPC port (default: 50051)")
    g.add_argument("--no-ws", action="store_true", help="Disable WebSocket transport")
    g.add_argument("--no-grpc", action="store_true", help="Disable gRPC transport")

    # STT
    g = p.add_argument_group("stt")
    g.add_argument("--stt-engine", default=None, help="STT engine (default: faster-whisper)")
    g.add_argument("--stt-model", default=None, help="STT model (default: base.en)")
    g.add_argument("--device", default=None, help="Compute device: cpu | cuda | auto")
    g.add_argument("--compute-type", default=None, help="CTranslate2 compute type (default: int8)")

    # TTS
    g = p.add_argument_group("tts")
    g.add_argument("--tts-engine", default=None, help="TTS engine (default: piper)")
    g.add_argument("--tts-model", default=None, help="TTS model/voice (default: en_US-lessac-medium)")
    g.add_argument("--tts-data-dir", type=Path, default=None, help="Piper model directory")

    # Observability
    g = p.add_argument_group("observability")
    g.add_argument("--log-level", default=None, help="Log level (default: INFO)")
    g.add_argument("--log-format", default=None, choices=["json", "console"], help="Log output format")
    g.add_argument("--metrics-port", type=int, default=None, help="Prometheus metrics port")
    g.add_argument("--no-metrics", action="store_true", help="Disable Prometheus metrics")
    g.add_argument("--enable-tracing", action="store_true", help="Enable OpenTelemetry tracing")
    g.add_argument("--otlp-endpoint", default=None, help="OTLP collector endpoint")

    # Session
    g = p.add_argument_group("sessions")
    g.add_argument("--max-sessions", type=int, default=None, help="Max concurrent sessions")
    g.add_argument("--idle-timeout", type=float, default=None, help="Session idle timeout (seconds)")

    return p


def args_to_config(args: argparse.Namespace):
    """Layer CLI args on top of config file or defaults."""
    from sts.config import ServiceConfig

    # Start from config file if provided
    if args.config and args.config.is_file():
        import json
        data = json.loads(args.config.read_text())
        config = ServiceConfig.model_validate(data)
    else:
        config = ServiceConfig.from_env()

    # --dev preset: console logging, debug, no metrics
    if args.dev:
        config.observability.log_level = "DEBUG"
        config.observability.log_format = "console"
        config.observability.enable_metrics = False
        config.observability.enable_tracing = False
        config.tts.piper_data_dir = Path.home() / ".cache" / "sts" / "piper-models"

    # CLI overrides (only if explicitly provided)
    if args.ws_host is not None:
        config.transport.ws_host = args.ws_host
    if args.ws_port is not None:
        config.transport.ws_port = args.ws_port
    if args.grpc_host is not None:
        config.transport.grpc_host = args.grpc_host
    if args.grpc_port is not None:
        config.transport.grpc_port = args.grpc_port
    if args.no_ws:
        config.transport.enable_websocket = False
    if args.no_grpc:
        config.transport.enable_grpc = False

    if args.stt_engine is not None:
        config.stt.default_engine = args.stt_engine
    if args.stt_model is not None:
        config.stt.faster_whisper_model = args.stt_model
    if args.device is not None:
        config.stt.faster_whisper_device = args.device
    if args.compute_type is not None:
        config.stt.faster_whisper_compute_type = args.compute_type

    if args.tts_engine is not None:
        config.tts.default_engine = args.tts_engine
    if args.tts_model is not None:
        config.tts.piper_model = args.tts_model
    if args.tts_data_dir is not None:
        config.tts.piper_data_dir = args.tts_data_dir

    if args.log_level is not None:
        config.observability.log_level = args.log_level
    if args.log_format is not None:
        config.observability.log_format = args.log_format
    if args.metrics_port is not None:
        config.observability.metrics_port = args.metrics_port
    if args.no_metrics:
        config.observability.enable_metrics = False
    if args.enable_tracing:
        config.observability.enable_tracing = True
    if args.otlp_endpoint is not None:
        config.observability.otlp_endpoint = args.otlp_endpoint

    if args.max_sessions is not None:
        config.max_sessions = args.max_sessions
    if args.idle_timeout is not None:
        config.session_idle_timeout_s = args.idle_timeout

    return config


def main() -> None:
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args()

    config = args_to_config(args)

    from sts.server import STSServer
    server = STSServer(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_event = asyncio.Event()

    async def _shutdown():
        shutdown_event.set()
        await server.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_shutdown()))

    try:
        loop.run_until_complete(server.start())
        loop.run_until_complete(shutdown_event.wait())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()


if __name__ == "__main__":
    main()
