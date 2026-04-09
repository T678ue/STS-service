"""Voice Activity Detection using Silero VAD (ONNX).

Runs the Silero VAD model via onnxruntime — no PyTorch dependency.
Provides a streaming-friendly interface that ingests fixed-size audio
frames and emits speech segment boundaries.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_SILERO_ONNX_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_DEFAULT_CACHE = Path.home() / ".cache" / "sts" / "silero_vad.onnx"

# Silero VAD expects 16 kHz, and processes fixed window sizes.
SAMPLE_RATE = 16_000
# Valid window sizes for Silero: 512 (32ms), 1024 (64ms), 1536 (96ms)
WINDOW_SAMPLES = 512  # 32 ms at 16 kHz


def _ensure_model(path: Path | None) -> Path:
    """Download the Silero VAD ONNX model if not cached."""
    target = path or _DEFAULT_CACHE
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("vad.downloading_model", url=_SILERO_ONNX_URL, dest=str(target))
    urllib.request.urlretrieve(_SILERO_ONNX_URL, target)
    return target


class SileroVAD:
    """Streaming VAD using the Silero ONNX model.

    Feed audio frames via `process_chunk()`.  It accumulates samples into
    the fixed window size the model expects, runs inference, and tracks
    speech/silence transitions.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 600,
        padding_ms: int = 300,
        max_speech_s: float = 30.0,
    ) -> None:
        import onnxruntime as ort

        path = _ensure_model(model_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), sess_options=opts)

        self.threshold = threshold
        self.min_speech_samples = int(SAMPLE_RATE * min_speech_ms / 1000)
        self.min_silence_samples = int(SAMPLE_RATE * min_silence_ms / 1000)
        self.padding_samples = int(SAMPLE_RATE * padding_ms / 1000)
        self.max_speech_samples = int(SAMPLE_RATE * max_speech_s)

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset ONNX model hidden state and segment tracking."""
        # Silero ONNX model state: h and c are (2, 1, 64) tensors
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

        self._buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._speech_buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self._reset_state()

    def _infer(self, window: np.ndarray) -> float:
        """Run one VAD window through the ONNX model."""
        inp = window.reshape(1, -1)
        sr = np.array([SAMPLE_RATE], dtype=np.int64)

        ort_inputs = {
            "input": inp,
            "sr": sr,
            "h": self._h,
            "c": self._c,
        }
        output, self._h, self._c = self._session.run(None, ort_inputs)
        return float(output[0][0])

    def process_chunk(self, audio: np.ndarray) -> list[np.ndarray]:
        """Feed a chunk of float32 16 kHz mono audio.

        Returns a (possibly empty) list of completed speech segments,
        each a float32 ndarray.
        """
        self._buffer = np.concatenate([self._buffer, audio])
        completed_segments: list[np.ndarray] = []

        while len(self._buffer) >= WINDOW_SAMPLES:
            window = self._buffer[:WINDOW_SAMPLES]
            self._buffer = self._buffer[WINDOW_SAMPLES:]

            prob = self._infer(window)

            if prob >= self.threshold:
                # Speech detected
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_samples = 0
                    self._silence_samples = 0
                    self._speech_buffer = []

                self._speech_buffer.append(window)
                self._speech_samples += WINDOW_SAMPLES
                self._silence_samples = 0

                # Force segment break if too long
                if self._speech_samples >= self.max_speech_samples:
                    segment = np.concatenate(self._speech_buffer)
                    completed_segments.append(segment)
                    self._in_speech = False
                    self._speech_buffer = []
                    self._speech_samples = 0
            else:
                # Silence detected
                if self._in_speech:
                    self._speech_buffer.append(window)  # keep padding
                    self._silence_samples += WINDOW_SAMPLES

                    if self._silence_samples >= self.min_silence_samples:
                        # End of speech
                        if self._speech_samples >= self.min_speech_samples:
                            segment = np.concatenate(self._speech_buffer)
                            completed_segments.append(segment)
                        # else: too short, discard
                        self._in_speech = False
                        self._speech_buffer = []
                        self._speech_samples = 0
                        self._silence_samples = 0

        return completed_segments

    def flush(self) -> list[np.ndarray]:
        """Flush any remaining buffered speech (e.g. on stream end)."""
        segments: list[np.ndarray] = []
        if self._in_speech and self._speech_buffer:
            if self._speech_samples >= self.min_speech_samples:
                segments.append(np.concatenate(self._speech_buffer))
        self._reset_state()
        return segments

    @property
    def is_speaking(self) -> bool:
        return self._in_speech
