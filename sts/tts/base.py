"""Abstract base class for TTS engine adapters.

Every TTS backend (Piper, Coqui, ElevenLabs, etc.) implements this
interface.  The adapter produces raw float32 audio; format encoding
and resampling for the client is handled downstream.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """A chunk of synthesized audio."""

    audio: np.ndarray       # float32
    sample_rate: int
    is_final: bool = False  # last chunk for this synthesis request


class TTSAdapter(abc.ABC):
    """Interface that all TTS engine adapters must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique engine identifier, e.g. 'piper'."""

    @property
    @abc.abstractmethod
    def native_sample_rate(self) -> int:
        """The sample rate the engine produces natively."""

    @abc.abstractmethod
    async def load(self, **kwargs) -> None:
        """Load model / initialise engine. Must be idempotent."""

    @abc.abstractmethod
    async def unload(self) -> None:
        """Release resources."""

    @abc.abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """Synthesize text to audio in one shot.

        Returns a single SynthesisResult with the complete audio.
        """

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> AsyncIterator[SynthesisResult]:
        """Stream audio chunks as they're synthesized.

        Default: yield a single chunk.  Override for true streaming.

        For lower latency, implementations should split input text into
        sentences and yield audio per-sentence.
        """
        result = await self.synthesize(text, voice=voice, speed=speed)
        yield SynthesisResult(
            audio=result.audio,
            sample_rate=result.sample_rate,
            is_final=True,
        )

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def available_voices(self) -> list[str]:
        """List voices available for this engine."""
        return []
