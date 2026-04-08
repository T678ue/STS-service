"""Abstract base class for STT engine adapters.

Every STT backend (faster-whisper, Vosk, Deepgram, etc.) implements this
interface.  The adapter is responsible ONLY for transcription — audio
preprocessing (format, resample, VAD) is handled upstream.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass(frozen=True, slots=True)
class Transcription:
    """A single transcription result."""

    text: str
    is_partial: bool = False     # True for interim/streaming results
    confidence: float = 0.0
    language: str = ""
    start_time: float = 0.0     # seconds, relative to segment start
    end_time: float = 0.0
    # Per-word timestamps (if the model supports it)
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float = 0.0


class STTAdapter(abc.ABC):
    """Interface that all STT engine adapters must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique engine identifier, e.g. 'faster-whisper'."""

    @abc.abstractmethod
    async def load(self, **kwargs) -> None:
        """Load model weights / initialise the engine.

        Called once at service startup.  Must be idempotent.
        """

    @abc.abstractmethod
    async def unload(self) -> None:
        """Release resources."""

    @abc.abstractmethod
    async def transcribe(
        self,
        audio: "np.ndarray",  # noqa: F821 — numpy imported by adapters
        *,
        language: str = "en",
        beam_size: int = 5,
    ) -> Transcription:
        """Transcribe a complete speech segment (post-VAD).

        Args:
            audio: float32, 16 kHz, mono.
            language: BCP-47 language code.
            beam_size: beam search width.

        Returns:
            Final transcription for this segment.
        """

    async def transcribe_stream(
        self,
        audio: "np.ndarray",
        *,
        language: str = "en",
        beam_size: int = 5,
    ) -> AsyncIterator[Transcription]:
        """Stream partial transcriptions as the segment is processed.

        Default implementation: yield a single final result.
        Override in adapters that support true streaming.
        """
        result = await self.transcribe(audio, language=language, beam_size=beam_size)
        yield result

    @property
    def supports_streaming(self) -> bool:
        """Whether this adapter produces meaningful partial results."""
        return False
