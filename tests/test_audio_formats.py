"""Tests for audio format detection, decoding, and encoding."""

from __future__ import annotations

import numpy as np
import pytest

from sts.audio.formats import (
    decode_chunk,
    decode_pcm,
    encode_audio,
    encode_pcm_s16le,
    sniff_format,
)


class TestSniffFormat:
    def test_wav(self):
        assert sniff_format(b"RIFF\x00\x00\x00\x00WAVE") == "wav"

    def test_ogg(self):
        assert sniff_format(b"OggS\x00\x00") == "ogg_opus"

    def test_flac(self):
        assert sniff_format(b"fLaC\x00\x00") == "flac"

    def test_mp3_sync(self):
        assert sniff_format(b"\xff\xfb\x90\x00") == "mp3"

    def test_unknown(self):
        assert sniff_format(b"\x00\x00\x00\x00") is None


class TestDecodePCM:
    def test_s16le_roundtrip(self):
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        encoded = (original * 32767).astype(np.int16).tobytes()
        decoded = decode_pcm(encoded, 16000, 1, "pcm_s16le")
        np.testing.assert_allclose(decoded, original, atol=1e-4)

    def test_f32le_passthrough(self):
        original = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        decoded = decode_pcm(original.tobytes(), 16000, 1, "pcm_f32le")
        np.testing.assert_array_equal(decoded, original)


class TestEncode:
    def test_pcm_s16le_encode(self):
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        encoded = encode_pcm_s16le(audio)
        assert len(encoded) == 6  # 3 samples * 2 bytes

    def test_roundtrip_pcm(self):
        audio = np.random.uniform(-1, 1, 1000).astype(np.float32)
        encoded = encode_audio(audio, "pcm_s16le", 16000)
        decoded = decode_pcm(encoded, 16000, 1, "pcm_s16le")
        np.testing.assert_allclose(decoded, audio, atol=1e-4)


class TestDecodeChunk:
    def test_raw_pcm(self, pcm_s16le_bytes):
        audio, sr = decode_chunk(pcm_s16le_bytes, "pcm_s16le", 16000, 1)
        assert sr == 16000
        assert audio.dtype == np.float32
        assert len(audio) == len(pcm_s16le_bytes) // 2
