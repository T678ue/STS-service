"""Audio preprocessing pipeline.

Orchestrates: decode → resample → mono → normalize → VAD
Consumes raw audio chunks from the transport layer and produces
clean speech segments ready for the STT engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from sts.audio.formats import decode_chunk
from sts.audio.normalize import peak_normalize
from sts.audio.resample import resample, to_mono
from sts.audio.vad import SAMPLE_RATE as VAD_SAMPLE_RATE
from sts.audio.vad import SileroVAD

if TYPE_CHECKING:
    from sts.session.models import SessionConfig

logger = structlog.get_logger(__name__)


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

    def process(self, raw_bytes: bytes) -> list[np.ndarray]:
        """Process a raw audio chunk through the full pipeline.

        Returns a (possibly empty) list of speech segments as float32
        arrays at 16 kHz mono — ready for STT.
        """
        cfg = self._config

        # 1. Decode to float32
        audio, sr = decode_chunk(
            raw_bytes,
            input_format=cfg.input_format.value,
            sample_rate=cfg.input_sample_rate,
            channels=cfg.input_channels,
        )

        # 2. To mono
        audio = to_mono(audio)

        # 3. Resample to internal rate (16 kHz for STT/VAD)
        audio = resample(audio, sr, VAD_SAMPLE_RATE)

        # 4. Normalize
        audio = peak_normalize(audio)

        # 5. VAD segmentation
        if self._vad:
            return self._vad.process_chunk(audio)

        # No VAD — return the whole chunk as a single "segment"
        return [audio]

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
