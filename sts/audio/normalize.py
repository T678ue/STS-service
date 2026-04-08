"""Gain normalization for consistent input levels.

Applies peak normalization so the loudest sample sits at `target_peak`.
This helps VAD and STT models get consistent signal levels regardless
of the client's mic gain.
"""

from __future__ import annotations

import numpy as np


def peak_normalize(
    audio: np.ndarray,
    target_peak: float = 0.95,
    floor_db: float = -60.0,
) -> np.ndarray:
    """Normalize audio so peak amplitude equals `target_peak`.

    If the signal is below `floor_db` (i.e. essentially silence), return
    as-is to avoid amplifying noise.
    """
    peak = np.abs(audio).max()
    if peak < 1e-10:
        return audio

    peak_db = 20 * np.log10(peak)
    if peak_db < floor_db:
        return audio  # too quiet — likely silence / noise

    return audio * (target_peak / peak)


def rms_normalize(
    audio: np.ndarray,
    target_rms: float = 0.1,
    floor_db: float = -60.0,
) -> np.ndarray:
    """Normalize audio to a target RMS level."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-10:
        return audio

    rms_db = 20 * np.log10(rms)
    if rms_db < floor_db:
        return audio

    return audio * (target_rms / rms)
