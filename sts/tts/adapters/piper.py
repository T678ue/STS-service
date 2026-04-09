"""Piper TTS adapter.

Piper is a fast, local neural TTS engine based on VITS.
Produces high-quality speech at 22050 Hz.

For streaming: we split input text at sentence boundaries and
synthesize each sentence independently, yielding audio chunks
as they complete.  This gives ~sentence-level latency.

piper-tts API:
  PiperVoice.load(model_path: str | Path, config_path: str | Path | None = None)
  voice.synthesize(text, wav_file, length_scale=1.0, ...)
"""

from __future__ import annotations

import asyncio
import re
from functools import partial
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import structlog

from sts.tts.base import SynthesisResult, TTSAdapter
from sts.tts.registry import TTSRegistry

logger = structlog.get_logger(__name__)

# Split text at sentence boundaries for streaming
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _resolve_model_path(model: str, data_dir: str | None) -> Path:
    """Resolve model name to a .onnx file path.

    If `model` is already an absolute path or existing file, use it directly.
    Otherwise look in `data_dir` for <model>.onnx.
    """
    p = Path(model)
    if p.is_file():
        return p

    if data_dir:
        candidate = Path(data_dir) / f"{model}.onnx"
        if candidate.is_file():
            return candidate
        # Also try without the .onnx extension matching
        for f in Path(data_dir).glob(f"{model}*"):
            if f.suffix == ".onnx":
                return f

    # If nothing found, return the original — PiperVoice.load will error clearly
    return p


@TTSRegistry.register
class PiperAdapter(TTSAdapter):
    NAME = "piper"

    def __init__(self) -> None:
        self._voice_name: str | None = None
        self._piper = None  # PiperVoice instance
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
        model_path = _resolve_model_path(model, data_dir)

        # Config file is conventionally <model>.onnx.json
        config_path = Path(f"{model_path}.json")
        if not config_path.is_file():
            config_path = None

        logger.info(
            "piper.loading",
            model=model,
            model_path=str(model_path),
            config_path=str(config_path) if config_path else None,
        )

        loop = asyncio.get_running_loop()

        self._piper = await loop.run_in_executor(
            None,
            partial(
                PiperVoice.load,
                str(model_path),
                config_path=str(config_path) if config_path else None,
            ),
        )
        self._voice_name = model

        # Read native sample rate from the model config
        if hasattr(self._piper, "config") and self._piper.config:
            sr = getattr(self._piper.config, "sample_rate", None)
            if sr:
                self._sample_rate = int(sr)

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

        def _run() -> np.ndarray:
            import io
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self._sample_rate)
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
