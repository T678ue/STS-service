"""Audio format detection, decoding, and encoding.

Accepts whatever the client sends (raw PCM, WAV, Opus, FLAC, MP3 ...)
and normalises it to float32 numpy arrays for internal processing.
Also encodes float32 back to the client's requested output format.
"""

from __future__ import annotations

import io
import struct
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from sts.session.models import AudioFormat


# -- magic bytes for format sniffing ----------------------------------------
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"RIFF", "wav"),
    (b"OggS", "ogg_opus"),
    (b"fLaC", "flac"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
    (b"ID3", "mp3"),
]


def sniff_format(header: bytes) -> str | None:
    """Best-effort format detection from the first bytes of a stream."""
    for sig, fmt in _SIGNATURES:
        if header[: len(sig)] == sig:
            return fmt
    return None


# -- decoding ---------------------------------------------------------------

def decode_pcm(
    data: bytes,
    sample_rate: int,
    channels: int,
    fmt: str = "pcm_s16le",
) -> np.ndarray:
    """Decode raw PCM bytes to float32 ndarray, shape (samples, channels)."""
    if fmt in ("pcm_s16le", "raw"):
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif fmt == "pcm_f32le":
        arr = np.frombuffer(data, dtype=np.float32).copy()
    else:
        raise ValueError(f"Unknown PCM format: {fmt}")

    if channels > 1:
        arr = arr.reshape(-1, channels)
    return arr


def decode_file_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an encoded audio buffer (WAV, FLAC, OGG, MP3) → (float32, sr)."""
    buf = io.BytesIO(data)
    audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    return audio, int(sr)


def decode_chunk(
    data: bytes,
    input_format: str,
    sample_rate: int,
    channels: int,
) -> tuple[np.ndarray, int]:
    """Unified decoder: handles both raw PCM and encoded formats.

    Returns (float32_array, sample_rate).
    """
    if input_format in ("pcm_s16le", "pcm_f32le", "raw"):
        return decode_pcm(data, sample_rate, channels, input_format), sample_rate

    # For container formats, try soundfile
    return decode_file_bytes(data)


# -- encoding ---------------------------------------------------------------

def encode_pcm_s16le(audio: np.ndarray) -> bytes:
    """float32 → signed 16-bit little-endian bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def encode_pcm_f32le(audio: np.ndarray) -> bytes:
    return audio.astype(np.float32).tobytes()


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode_ogg_opus(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="OGG", subtype="OPUS")
    return buf.getvalue()


def encode_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC")
    return buf.getvalue()


_ENCODERS: dict[str, callable] = {  # type: ignore[valid-type]
    "pcm_s16le": lambda a, sr: encode_pcm_s16le(a),
    "raw": lambda a, sr: encode_pcm_s16le(a),
    "pcm_f32le": lambda a, sr: encode_pcm_f32le(a),
    "wav": encode_wav,
    "ogg_opus": encode_ogg_opus,
    "flac": encode_flac,
}


def encode_audio(audio: np.ndarray, fmt: str, sample_rate: int) -> bytes:
    """Encode float32 audio to the requested output format."""
    encoder = _ENCODERS.get(fmt)
    if encoder is None:
        raise ValueError(f"Unsupported output format: {fmt}")
    return encoder(audio, sample_rate)
