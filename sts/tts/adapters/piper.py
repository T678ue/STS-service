"""Piper TTS adapter.

Piper is a fast, local neural TTS engine based on VITS.
Produces high-quality speech at 22050 Hz.

For streaming: we split input text at sentence boundaries and
synthesize each sentence independently, yielding audio chunks
as they complete.  This gives ~sentence-level latency.
"""

from __future__ import annotations

import asyncio
import re
from functools import partial
from typing import AsyncIterator

import numpy as np
import structlog

from sts.tts.base import SynthesisResult, TTSAdapter
from sts.tts.registry import TTSRegistry

logger = structlog.get_logger(__name__)

# Regex to split text into sentences (handles common cases)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@TTSRegistry.register
class PiperAdapter(TTSAdapter):
    NAME = "piper"

    def __init__(self) -> None:
        self._voice: str | None = None
        self._synthesize_fn = None  # piper's synthesize function
        self._piper = None
        self._sample_rate: int = 22050

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def native_sample_rate(self) -> int:
        return self._sample_rate

    async def load(self, **kwargs) -> None:
        from piper import PiperVoice

        model = kwargs.get("model", "en_US-lessac-medium")
        data_dir = kwargs.get("data_dir")

        logger.info("piper.loading", model=model, data_dir=data_dir)

        loop = asyncio.get_running_loop()

        # Piper downloads models on first use
        self._piper = await loop.run_in_executor(
            None,
            partial(PiperVoice.load, model, config_path=None, use_cuda=False),
        )
        self._voice = model

        if self._piper.config and hasattr(self._piper.config, "sample_rate"):
            self._sample_rate = self._piper.config.sample_rate

        logger.info("piper.loaded", model=model, sample_rate=self._sample_rate)

    async def unload(self) -> None:
        self._piper = None
        logger.info("piper.unloaded")

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        if self._piper is None:
            raise RuntimeError("Piper not loaded")

        loop = asyncio.get_running_loop()

        def _run():
            import io
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                self._piper.synthesize(
                    text,
                    wav,
                    length_scale=1.0 / speed if speed else 1.0,
                )
            buf.seek(0)
            with wave.open(buf, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio

        audio = await loop.run_in_executor(None, _run)

        return SynthesisResult(
            audio=audio,
            sample_rate=self._sample_rate,
            is_final=True,
        )

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> AsyncIterator[SynthesisResult]:
        """Stream synthesis by splitting at sentence boundaries.

        Yields audio for each sentence as soon as it's synthesized,
        reducing time-to-first-audio.
        """
        sentences = _SENTENCE_SPLIT.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return

        for i, sentence in enumerate(sentences):
            is_last = i == len(sentences) - 1
            result = await self.synthesize(sentence, voice=voice, speed=speed)
            yield SynthesisResult(
                audio=result.audio,
                sample_rate=result.sample_rate,
                is_final=is_last,
            )

    @property
    def supports_streaming(self) -> bool:
        return True
