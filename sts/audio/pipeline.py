"""Audio preprocessing pipeline.

Orchestrates: decode → resample → mono → normalize → VAD
Consumes raw audio chunks from the transport layer and produces
clean speech segments ready for the STT engine.

For realtime streaming, also exposes the in-progress speech buffer
so the STT pipeline can periodically transcribe it for partial results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from sts.audio.formats import decode_chunk
from sts.audio.normalize import peak_normalize
from sts.audio.resample import resample, to_mono
from sts.audio.vad import SAMPLE_RATE as VAD_SAMPLE_RATE
from sts.audio.vad import SileroVAD, VADEvent, VADResult

if TYPE_CHECKING:
    from sts.session.models import SessionConfig

logger = structlog.get_logger(__name__)


class AudioPipelineResult:
    """Result from processing an audio chunk."""

    __slots__ = ("completed_segments", "vad_events", "speech_active", "speech_duration_s")

    def __init__(
        self,
        completed_segments: list[np.ndarray],
        vad_events: list[VADEvent],
        speech_active: bool,
        speech_duration_s: float,
    ) -> None:
        self.completed_segments = completed_segments
        self.vad_events = vad_events
        self.speech_active = speech_active
        self.speech_duration_s = speech_duration_s


class AudioPipeline:
    """Per-session audio preprocessing pipeline.

    Created when a session starts, configured by SessionConfig.
    Reconfigurable on-the-fly via `update_config()`.
    """

    def __init__(self, config: SessionConfig, vad_model_path=None) -> None:
        self._config = config
        self._vad: SileroVAD | None = None
        self._config_version = 0

        if config.vad.enabled:
            self._vad = SileroVAD(
                model_path=vad_model_path,
                threshold=config.vad.threshold,
                min_speech_ms=config.vad.min_speech_ms,
                min_silence_ms=config.vad.min_silence_ms,
                padding_ms=config.vad.padding_ms,
                max_speech_s=config.vad.max_speech_s,
            )

    def update_config(self, config: SessionConfig) -> None:
        """Hot-reload pipeline config.

        VAD parameters update without re-creating the ONNX session.
        """
        self._config = config
        if self._vad and config.vad.enabled:
            self._vad.threshold = config.vad.threshold
            self._vad.min_speech_samples = int(
                VAD_SAMPLE_RATE * config.vad.min_speech_ms / 1000
            )
            self._vad.min_silence_samples = int(
                VAD_SAMPLE_RATE * config.vad.min_silence_ms / 1000
            )
        self._config_version += 1

    def _preprocess(self, raw_bytes: bytes) -> np.ndarray:
        """Decode, resample, mono, normalize a raw audio chunk."""
        cfg = self._config

        audio, sr = decode_chunk(
            raw_bytes,
            input_format=cfg.input_format.value,
            sample_rate=cfg.input_sample_rate,
            channels=cfg.input_channels,
        )
        audio = to_mono(audio)
        audio = resample(audio, sr, VAD_SAMPLE_RATE)
        audio = peak_normalize(audio)
        return audio

    def process(self, raw_bytes: bytes) -> AudioPipelineResult:
        """Process a raw audio chunk through the full pipeline.

        Returns an AudioPipelineResult with completed speech segments,
        VAD events, and in-progress speech state.
        """
        audio = self._preprocess(raw_bytes)

        if self._vad:
            vad_result = self._vad.process_chunk(audio)
            return AudioPipelineResult(
                completed_segments=vad_result.completed_segments,
                vad_events=vad_result.events,
                speech_active=vad_result.speech_active,
                speech_duration_s=vad_result.speech_duration_s,
            )

        # No VAD — return the whole chunk as a single "segment"
        return AudioPipelineResult(
            completed_segments=[audio],
            vad_events=[],
            speech_active=False,
            speech_duration_s=0.0,
        )

    def get_speech_buffer(self) -> np.ndarray | None:
        """Get the in-progress speech audio for realtime partial transcription.

        Returns the accumulated speech buffer if VAD detects ongoing speech,
        or None if silent.  Does NOT consume the buffer.
        """
        if self._vad:
            return self._vad.get_speech_buffer()
        return None

    def flush(self) -> list[np.ndarray]:
        """Flush remaining audio on stream end."""
        if self._vad:
            return self._vad.flush()
        return []

    @property
    def is_speaking(self) -> bool:
        return self._vad.is_speaking if self._vad else False

    def reset(self) -> None:
        if self._vad:
            self._vad.reset()
