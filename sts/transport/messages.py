"""Shared message types for all transports.

Both WebSocket and gRPC transports convert their wire format into these
internal message types.  This decouples transport details from the
session/pipeline logic.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class MessageType(str, enum.Enum):
    # Client → Service
    SESSION_CREATE = "session.create"
    SESSION_UPDATE = "session.update"
    SESSION_DESTROY = "session.destroy"
    SESSION_GET = "session.get"
    AUDIO_CHUNK = "audio.chunk"
    AUDIO_END = "audio.end"          # client signals end of audio stream
    TTS_SYNTHESIZE = "tts.synthesize"
    TTS_CANCEL = "tts.cancel"

    # Service → Client
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_DESTROYED = "session.destroyed"
    SESSION_INFO = "session.info"
    STT_TRANSCRIPTION = "stt.transcription"
    STT_VAD_STATE = "stt.vad_state"  # speech start/end events
    TTS_AUDIO = "tts.audio"
    TTS_DONE = "tts.done"
    ERROR = "error"
    PONG = "pong"


@dataclass(slots=True)
class InboundMessage:
    """Decoded message from a client."""

    type: MessageType
    session_id: str = ""
    payload: dict = field(default_factory=dict)
    audio_data: bytes = b""


@dataclass(slots=True)
class OutboundMessage:
    """Message to send to a client."""

    type: MessageType
    session_id: str = ""
    payload: dict = field(default_factory=dict)
    audio_data: bytes = b""
    is_binary: bool = False  # hint for transports: send as binary frame
