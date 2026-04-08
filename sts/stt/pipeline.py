"""STT streaming pipeline.

Bridges the audio pipeline (which produces speech segments) with the
STT adapter (which transcribes them).  Manages the async flow of:

    audio chunks → AudioPipeline → speech segments → STTAdapter → transcriptions
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np
import structlog

from sts.audio.pipeline import AudioPipeline
from sts.observability.metrics import stt_metrics
from sts.session.models import SessionConfig, SessionState
from sts.stt.base import STTAdapter, Transcription

logger = structlog.get_logger(__name__)


class STTPipeline:
    """Per-session STT pipeline.

    Owns an AudioPipeline instance and an STTAdapter reference.
    Produces transcription results from raw audio bytes.
    """

    def __init__(
        self,
        session: SessionState,
        audio_pipeline: AudioPipeline,
        stt_adapter: STTAdapter,
    ) -> None:
        self._session = session
        self._audio = audio_pipeline
        self._stt = stt_adapter
        self._result_queue: asyncio.Queue[Transcription | None] = asyncio.Queue()
        self._transcription_tasks: set[asyncio.Task] = set()

    async def feed_audio(self, raw_bytes: bytes) -> None:
        """Feed raw audio bytes from the transport layer.

        Speech segments are detected by the audio pipeline and dispatched
        to the STT adapter asynchronously.
        """
        self._session.audio_bytes_received += len(raw_bytes)
        segments = self._audio.process(raw_bytes)

        for segment in segments:
            task = asyncio.create_task(self._transcribe_segment(segment))
            self._transcription_tasks.add(task)
            task.add_done_callback(self._transcription_tasks.discard)

    async def _transcribe_segment(self, segment: np.ndarray) -> None:
        """Transcribe a single speech segment and enqueue results."""
        cfg = self._session.config.stt
        sid = self._session.session_id

        duration_s = len(segment) / 16_000
        logger.debug(
            "stt.transcribing_segment",
            session_id=sid,
            duration_s=round(duration_s, 2),
        )

        with stt_metrics.transcription_duration.labels(engine=cfg.engine).time():
            try:
                if cfg.enable_partials and self._stt.supports_streaming:
                    async for result in self._stt.transcribe_stream(
                        segment,
                        language=cfg.language,
                        beam_size=cfg.beam_size,
                    ):
                        await self._result_queue.put(result)
                else:
                    result = await self._stt.transcribe(
                        segment,
                        language=cfg.language,
                        beam_size=cfg.beam_size,
                    )
                    await self._result_queue.put(result)

                self._session.utterances_transcribed += 1
                stt_metrics.segments_transcribed.labels(engine=cfg.engine).inc()

            except Exception:
                logger.exception("stt.transcription_error", session_id=sid)
                stt_metrics.errors.labels(engine=cfg.engine).inc()

    async def flush(self) -> None:
        """Flush the audio pipeline and transcribe remaining audio."""
        segments = self._audio.flush()
        for segment in segments:
            await self._transcribe_segment(segment)
        await self._result_queue.put(None)  # sentinel: stream is done

    async def results(self) -> AsyncIterator[Transcription]:
        """Async iterator of transcription results.

        Yields Transcription objects as they become available.
        Terminates when flush() has been called and all segments processed.
        """
        while True:
            result = await self._result_queue.get()
            if result is None:
                break
            yield result

    def reset(self) -> None:
        self._audio.reset()

    @property
    def is_speaking(self) -> bool:
        return self._audio.is_speaking
