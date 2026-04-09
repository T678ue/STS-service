"""STT streaming pipeline.

Two modes of operation:

1. **Segment mode** (default): VAD detects complete speech segments,
   each is transcribed after silence is detected.  Lower CPU but
   higher latency — no text until the user stops talking.

2. **Realtime mode** (`enable_realtime_partials=True`): While the user
   is speaking, the growing speech buffer is periodically re-transcribed
   to produce interim partial results.  When silence is detected, the
   complete segment gets a final transcription.  This gives progressive
   text output while the user is still mid-utterance.

   Flow:
     audio chunk → AudioPipeline → VAD detects speech start
       → start periodic partial transcription loop (every N ms)
       → each loop: snapshot speech buffer → transcribe → emit partial
     silence detected → VAD emits complete segment
       → cancel partial loop → final transcription → emit final
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np
import structlog

from sts.audio.pipeline import AudioPipeline, AudioPipelineResult
from sts.audio.vad import VADEvent
from sts.observability.metrics import stt_metrics
from sts.session.models import SessionState
from sts.stt.base import STTAdapter, Transcription

logger = structlog.get_logger(__name__)

# Minimum speech buffer duration before attempting a partial transcription.
# Whisper needs ~0.5s minimum to produce anything useful.
_MIN_PARTIAL_DURATION_S = 0.5


class STTPipeline:
    """Per-session STT pipeline with realtime streaming support."""

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
        self._closed = False

        # Realtime partial transcription state
        self._partial_task: asyncio.Task | None = None
        self._last_partial_text: str = ""
        self._partial_running = False

    @property
    def _realtime_enabled(self) -> bool:
        return self._session.config.stt.enable_realtime_partials

    @property
    def _partial_interval_s(self) -> float:
        return self._session.config.stt.realtime_partial_interval_ms / 1000.0

    async def feed_audio(self, raw_bytes: bytes) -> None:
        """Feed raw audio bytes from the transport layer.

        Completed speech segments are dispatched for transcription.
        In realtime mode, speech-start events kick off periodic
        partial transcription of the in-progress buffer.
        """
        if self._closed:
            return

        self._session.audio_bytes_received += len(raw_bytes)
        result: AudioPipelineResult = self._audio.process(raw_bytes)

        # Handle VAD events
        for event in result.vad_events:
            if event == VADEvent.SPEECH_START:
                logger.debug(
                    "stt.vad.speech_start",
                    session_id=self._session.session_id,
                )
                # Emit VAD event to client
                await self._result_queue.put(
                    Transcription(text="", is_partial=True, _vad_event="speech_start")
                )
                # Start realtime partial loop
                if self._realtime_enabled:
                    self._start_partial_loop()

            elif event == VADEvent.SPEECH_END:
                logger.debug(
                    "stt.vad.speech_end",
                    session_id=self._session.session_id,
                )
                # Stop partial loop before final transcription
                self._stop_partial_loop()
                await self._result_queue.put(
                    Transcription(text="", is_partial=True, _vad_event="speech_end")
                )

        # Dispatch completed segments for final transcription
        for segment in result.completed_segments:
            task = asyncio.create_task(self._transcribe_segment(segment, is_final=True))
            self._transcription_tasks.add(task)
            task.add_done_callback(self._transcription_tasks.discard)

    # ------------------------------------------------------------------
    # Realtime partial transcription loop
    # ------------------------------------------------------------------

    def _start_partial_loop(self) -> None:
        """Start periodic partial transcription of in-progress speech."""
        if self._partial_task and not self._partial_task.done():
            return  # already running
        self._last_partial_text = ""
        self._partial_running = True
        self._partial_task = asyncio.create_task(self._partial_transcription_loop())

    def _stop_partial_loop(self) -> None:
        """Stop the partial transcription loop."""
        self._partial_running = False
        if self._partial_task and not self._partial_task.done():
            self._partial_task.cancel()
        self._partial_task = None

    async def _partial_transcription_loop(self) -> None:
        """Periodically transcribe the growing speech buffer.

        Runs every `realtime_partial_interval_ms` while the user is speaking.
        Each iteration snapshots the current speech buffer and transcribes it,
        emitting partial results only when the text actually changes.
        """
        interval = self._partial_interval_s
        sid = self._session.session_id
        cfg = self._session.config.stt

        logger.debug("stt.partial_loop.started", session_id=sid, interval_s=interval)

        try:
            while self._partial_running and not self._closed:
                await asyncio.sleep(interval)

                if not self._partial_running:
                    break

                # Snapshot the current speech buffer
                speech_audio = self._audio.get_speech_buffer()
                if speech_audio is None:
                    continue

                duration_s = len(speech_audio) / 16_000
                if duration_s < _MIN_PARTIAL_DURATION_S:
                    continue

                # Transcribe the snapshot (this is the expensive part)
                try:
                    result = await self._stt.transcribe(
                        speech_audio,
                        language=cfg.language,
                        beam_size=max(1, cfg.beam_size // 2),  # smaller beam for speed
                    )

                    # Only emit if text changed
                    if result.text.strip() and result.text.strip() != self._last_partial_text:
                        self._last_partial_text = result.text.strip()
                        partial = Transcription(
                            text=result.text.strip(),
                            is_partial=True,
                            confidence=result.confidence,
                            language=result.language,
                        )
                        await self._result_queue.put(partial)
                        logger.debug(
                            "stt.partial_result",
                            session_id=sid,
                            text=partial.text[:50],
                            buffer_duration_s=round(duration_s, 2),
                        )

                except Exception:
                    logger.exception("stt.partial_error", session_id=sid)

        except asyncio.CancelledError:
            pass

        logger.debug("stt.partial_loop.stopped", session_id=sid)

    # ------------------------------------------------------------------
    # Final segment transcription
    # ------------------------------------------------------------------

    async def _transcribe_segment(
        self, segment: np.ndarray, *, is_final: bool = True,
    ) -> None:
        """Transcribe a complete speech segment and enqueue results."""
        cfg = self._session.config.stt
        sid = self._session.session_id

        duration_s = len(segment) / 16_000
        logger.debug(
            "stt.transcribing_segment",
            session_id=sid,
            duration_s=round(duration_s, 2),
            is_final=is_final,
        )

        with stt_metrics.transcription_duration.labels(engine=cfg.engine).time():
            try:
                if cfg.enable_partials and self._stt.supports_streaming and not self._realtime_enabled:
                    # Use adapter-level streaming (segment-level partials)
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
                    # Mark as final (not partial)
                    final = Transcription(
                        text=result.text,
                        is_partial=False,
                        confidence=result.confidence,
                        language=result.language,
                        start_time=result.start_time,
                        end_time=result.end_time,
                        words=result.words,
                    )
                    await self._result_queue.put(final)

                self._session.utterances_transcribed += 1
                stt_metrics.segments_transcribed.labels(engine=cfg.engine).inc()

            except Exception:
                logger.exception("stt.transcription_error", session_id=sid)
                stt_metrics.errors.labels(engine=cfg.engine).inc()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Flush the audio pipeline and transcribe remaining audio."""
        self._stop_partial_loop()

        segments = self._audio.flush()
        for segment in segments:
            await self._transcribe_segment(segment)

        # Wait for any in-flight transcriptions to complete
        if self._transcription_tasks:
            await asyncio.gather(*self._transcription_tasks, return_exceptions=True)

        await self._result_queue.put(None)  # sentinel: stream is done

    def close(self) -> None:
        """Mark pipeline as closed.  Unblocks results() if it's waiting."""
        self._closed = True
        self._stop_partial_loop()
        try:
            self._result_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def results(self) -> AsyncIterator[Transcription]:
        """Async iterator of transcription results.

        Yields Transcription objects as they become available.
        In realtime mode, yields a mix of:
          - VAD events (text="", _vad_event set)
          - Partial transcriptions (is_partial=True)
          - Final transcriptions (is_partial=False)
        """
        while True:
            try:
                result = await self._result_queue.get()
            except asyncio.CancelledError:
                break
            if result is None:
                break
            yield result

    def reset(self) -> None:
        self._stop_partial_loop()
        self._audio.reset()

    @property
    def is_speaking(self) -> bool:
        return self._audio.is_speaking
