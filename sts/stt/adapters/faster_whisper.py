"""faster-whisper STT adapter.

Uses CTranslate2-based Whisper inference for fast, accurate
speech-to-text with word-level timestamps.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import AsyncIterator

import numpy as np
import structlog

from sts.stt.base import STTAdapter, Transcription, WordTimestamp
from sts.stt.registry import STTRegistry

logger = structlog.get_logger(__name__)


@STTRegistry.register
class FasterWhisperAdapter(STTAdapter):
    NAME = "faster-whisper"

    def __init__(self) -> None:
        self._model = None
        self._model_size: str = "base.en"
        self._device: str = "auto"
        self._compute_type: str = "int8"

    @property
    def name(self) -> str:
        return self.NAME

    async def load(self, **kwargs) -> None:
        from faster_whisper import WhisperModel

        self._model_size = kwargs.get("model", self._model_size)
        self._device = kwargs.get("device", self._device)
        self._compute_type = kwargs.get("compute_type", self._compute_type)

        logger.info(
            "faster_whisper.loading",
            model=self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

        # Model loading is CPU-bound; run in thread to avoid blocking
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            partial(
                WhisperModel,
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            ),
        )
        logger.info("faster_whisper.loaded", model=self._model_size)

    async def unload(self) -> None:
        self._model = None
        logger.info("faster_whisper.unloaded")

    async def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str = "en",
        beam_size: int = 5,
    ) -> Transcription:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_running_loop()

        segments_iter, info = await loop.run_in_executor(
            None,
            partial(
                self._model.transcribe,
                audio,
                language=language if not self._model_size.endswith(".en") else None,
                beam_size=beam_size,
                word_timestamps=True,
                vad_filter=False,  # we handle VAD upstream
            ),
        )

        # Collect all segments (runs the generator)
        segments = await loop.run_in_executor(None, list, segments_iter)

        full_text = " ".join(seg.text.strip() for seg in segments)
        words = []
        for seg in segments:
            if seg.words:
                words.extend(
                    WordTimestamp(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                )

        return Transcription(
            text=full_text,
            is_partial=False,
            confidence=1.0 - info.language_probability if info.language_probability else 0.0,
            language=info.language or language,
            start_time=segments[0].start if segments else 0.0,
            end_time=segments[-1].end if segments else 0.0,
            words=words,
        )

    async def transcribe_stream(
        self,
        audio: np.ndarray,
        *,
        language: str = "en",
        beam_size: int = 5,
    ) -> AsyncIterator[Transcription]:
        """Stream segment-level results as they're decoded.

        faster-whisper processes segments sequentially, so we yield
        each segment as a partial result and then a final combined result.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_running_loop()

        segments_iter, info = await loop.run_in_executor(
            None,
            partial(
                self._model.transcribe,
                audio,
                language=language if not self._model_size.endswith(".en") else None,
                beam_size=beam_size,
                word_timestamps=True,
                vad_filter=False,
            ),
        )

        # Yield each segment as a partial result
        all_text_parts: list[str] = []
        all_words: list[WordTimestamp] = []
        first_start = 0.0
        last_end = 0.0

        def _next_segment(it):
            return next(it, None)

        while True:
            seg = await loop.run_in_executor(None, _next_segment, segments_iter)
            if seg is None:
                break

            text = seg.text.strip()
            all_text_parts.append(text)
            if not all_text_parts[1:]:
                first_start = seg.start
            last_end = seg.end

            words = []
            if seg.words:
                words = [
                    WordTimestamp(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]
                all_words.extend(words)

            yield Transcription(
                text=text,
                is_partial=True,
                language=info.language or language,
                start_time=seg.start,
                end_time=seg.end,
                words=words,
            )

        # Final combined result
        if all_text_parts:
            yield Transcription(
                text=" ".join(all_text_parts),
                is_partial=False,
                language=info.language or language,
                start_time=first_start,
                end_time=last_end,
                words=all_words,
            )

    @property
    def supports_streaming(self) -> bool:
        return True
