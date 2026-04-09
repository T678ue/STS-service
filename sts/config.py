"""Global service configuration.

Defines the top-level service config (ports, model paths, defaults).
Per-client session config lives in session/models.py.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TransportConfig(BaseModel):
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    enable_websocket: bool = True
    enable_grpc: bool = True


class AudioConfig(BaseModel):
    """Defaults applied when a session doesn't specify overrides."""

    internal_sample_rate: int = 16_000  # STT models expect 16 kHz
    internal_channels: int = 1
    vad_model_path: Path | None = None  # auto-download if None


class STTConfig(BaseModel):
    default_engine: str = "faster-whisper"
    faster_whisper_model: str = "base.en"
    faster_whisper_device: str = "auto"  # cpu | cuda | auto
    faster_whisper_compute_type: str = "int8"


class TTSConfig(BaseModel):
    default_engine: str = "piper"
    piper_model: str = "en_US-lessac-medium"
    piper_data_dir: Path = Path.home() / ".cache" / "sts" / "piper-models"


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    log_format: str = "json"  # json | console
    metrics_port: int = 9090
    enable_metrics: bool = True
    health_port: int = 8080
    enable_health: bool = True
    enable_tracing: bool = False
    otlp_endpoint: str = "http://localhost:4317"


class ServiceConfig(BaseModel):
    """Root configuration for the STS service."""

    transport: TransportConfig = Field(default_factory=TransportConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    max_sessions: int = 64
    session_idle_timeout_s: float = 300.0  # 5 minutes

    @classmethod
    def from_env(cls) -> ServiceConfig:
        """Load config, layering: defaults < config file < env vars."""
        import json
        import os

        path = os.environ.get("STS_CONFIG_PATH")
        if path and Path(path).is_file():
            data = json.loads(Path(path).read_text())
            return cls.model_validate(data)
        return cls()
