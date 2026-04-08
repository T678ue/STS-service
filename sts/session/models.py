"""Per-client session configuration and state.

Each connected client gets a Session with its own config.  Configs can be
updated mid-session (hot-reload) without tearing down the audio pipeline.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Audio format enum (what the CLIENT sends / wants to receive)
# ---------------------------------------------------------------------------
class AudioFormat(str, enum.Enum):
    PCM_S16LE = "pcm_s16le"      # raw signed 16-bit little-endian
    PCM_F32LE = "pcm_f32le"      # raw float32 little-endian
    WAV = "wav"
    OGG_OPUS = "ogg_opus"
    FLAC = "flac"
    MP3 = "mp3"
    RAW = "raw"                  # alias for PCM_S16LE


# ---------------------------------------------------------------------------
# VAD config
# ---------------------------------------------------------------------------
class VADConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.5       # speech probability threshold
    min_speech_ms: int = 250     # ignore speech shorter than this
    min_silence_ms: int = 600    # silence duration to mark end-of-speech
    padding_ms: int = 300        # pad speech boundaries to avoid clipping
    # max speech duration before forcing a segment break
    max_speech_s: float = 30.0


# ---------------------------------------------------------------------------
# STT session overrides
# ---------------------------------------------------------------------------
class STTSessionConfig(BaseModel):
    engine: str = "faster-whisper"
    model: str | None = None     # None = use service default
    language: str = "en"
    beam_size: int = 5
    enable_partials: bool = True  # stream partial transcriptions


# ---------------------------------------------------------------------------
# TTS session overrides
# ---------------------------------------------------------------------------
class TTSSessionConfig(BaseModel):
    engine: str = "piper"
    voice: str | None = None     # None = use service default
    speed: float = 1.0
    # Output format the CLIENT wants to receive
    output_format: AudioFormat = AudioFormat.PCM_S16LE
    output_sample_rate: int = 22050
    output_channels: int = 1


# ---------------------------------------------------------------------------
# Full session config (what the client sends on session create / update)
# ---------------------------------------------------------------------------
class SessionConfig(BaseModel):
    # Audio input description (what the client is sending)
    input_format: AudioFormat = AudioFormat.PCM_S16LE
    input_sample_rate: int = 16000
    input_channels: int = 1

    vad: VADConfig = Field(default_factory=VADConfig)
    stt: STTSessionConfig = Field(default_factory=STTSessionConfig)
    tts: TTSSessionConfig = Field(default_factory=TTSSessionConfig)

    # Client metadata (opaque, for observability)
    client_name: str = ""
    client_version: str = ""


# ---------------------------------------------------------------------------
# Session runtime state
# ---------------------------------------------------------------------------
class SessionMode(str, enum.Enum):
    STT = "stt"          # client sends audio, gets transcriptions
    TTS = "tts"          # client sends text, gets audio
    DUPLEX = "duplex"    # both directions active simultaneously


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    config: SessionConfig = Field(default_factory=SessionConfig)
    mode: SessionMode = SessionMode.STT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    # Runtime stats (populated by pipelines)
    audio_bytes_received: int = 0
    audio_bytes_sent: int = 0
    utterances_transcribed: int = 0
    synthesis_requests: int = 0

    # Version counter - incremented on each config update so pipelines
    # can detect changes without locks.
    config_version: int = 0
