"""TTS streaming pipeline.

Bridges text input from the transport layer with the TTS adapter.
Handles format conversion and resampling for the client's requested
output format.

    text → TTSAdapter → float32 audio → resample → encode → client
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog

from sts.audio.formats import encode_audio
from sts.audio.resample import resample
from sts.observability.metrics import tts_metrics
from sts.session.models import SessionState
from sts.tts.base import SynthesisResult, TTSAdapter

logger = structlog.get_logger(__name__)


class TTSPipeline:
    """Per-session TTS pipeline."""

    def __init__(
        self,
        session: SessionState,
        tts_adapter: TTSAdapter,
    ) -> None:
        self._session = session
        self._tts = tts_adapter

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text and return encoded audio bytes in the
        client's requested format."""
        cfg = self._session.config.tts

        with tts_metrics.synthesis_duration.labels(engine=cfg.engine).time():
            result = await self._tts.synthesize(
                text, voice=cfg.voice, speed=cfg.speed,
            )

        audio = self._postprocess(result)
        encoded = encode_audio(
            audio, cfg.output_format.value, cfg.output_sample_rate,
        )

        self._session.audio_bytes_sent += len(encoded)
        self._session.synthesis_requests += 1
        tts_metrics.chars_synthesized.labels(engine=cfg.engine).inc(len(text))

        return encoded

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks to the client."""
        cfg = self._session.config.tts

        with tts_metrics.synthesis_duration.labels(engine=cfg.engine).time():
            async for chunk in self._tts.synthesize_stream(
                text, voice=cfg.voice, speed=cfg.speed,
            ):
                audio = self._postprocess(chunk)
                encoded = encode_audio(
                    audio, cfg.output_format.value, cfg.output_sample_rate,
                )
                self._session.audio_bytes_sent += len(encoded)
                yield encoded

        self._session.synthesis_requests += 1
        tts_metrics.chars_synthesized.labels(engine=cfg.engine).inc(len(text))

    def _postprocess(self, result: SynthesisResult):
        """Resample TTS output to the client's requested rate."""
        cfg = self._session.config.tts
        audio = result.audio
        if result.sample_rate != cfg.output_sample_rate:
            audio = resample(audio, result.sample_rate, cfg.output_sample_rate)
        return audio
