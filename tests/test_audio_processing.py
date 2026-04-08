"""Tests for audio resampling and normalization."""

from __future__ import annotations

import numpy as np

from sts.audio.normalize import peak_normalize, rms_normalize
from sts.audio.resample import resample, to_mono


class TestResample:
    def test_identity(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_upsample(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = resample(audio, 16000, 48000)
        # 1s of 16kHz → 1s of 48kHz
        assert len(result) == pytest.approx(48000, abs=10)

    def test_downsample(self):
        audio = np.random.randn(48000).astype(np.float32)
        result = resample(audio, 48000, 16000)
        assert len(result) == pytest.approx(16000, abs=10)


class TestToMono:
    def test_already_mono(self):
        audio = np.random.randn(1000).astype(np.float32)
        result = to_mono(audio)
        np.testing.assert_array_equal(result, audio)

    def test_stereo_to_mono(self):
        stereo = np.random.randn(1000, 2).astype(np.float32)
        mono = to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == 1000
        np.testing.assert_allclose(mono, stereo.mean(axis=1))


class TestNormalize:
    def test_peak_normalize(self):
        audio = np.array([0.25, -0.25, 0.1], dtype=np.float32)
        result = peak_normalize(audio, target_peak=0.95)
        assert np.abs(result).max() == pytest.approx(0.95, abs=1e-6)

    def test_peak_normalize_silence(self):
        audio = np.zeros(100, dtype=np.float32)
        result = peak_normalize(audio)
        np.testing.assert_array_equal(result, audio)

    def test_rms_normalize(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        result = rms_normalize(audio, target_rms=0.1)
        rms = np.sqrt(np.mean(result**2))
        assert rms == pytest.approx(0.1, abs=1e-3)


# Need pytest import for approx
import pytest
