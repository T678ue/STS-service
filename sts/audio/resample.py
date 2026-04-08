"""High-quality sample rate conversion using libsoxr."""

from __future__ import annotations

import numpy as np
import soxr


def resample(
    audio: np.ndarray,
    from_sr: int,
    to_sr: int,
) -> np.ndarray:
    """Resample audio from one sample rate to another.

    Args:
        audio: float32 array, shape (samples,) or (samples, channels).
        from_sr: source sample rate.
        to_sr: target sample rate.

    Returns:
        Resampled float32 array.
    """
    if from_sr == to_sr:
        return audio
    return soxr.resample(audio, from_sr, to_sr, quality="HQ")


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)
