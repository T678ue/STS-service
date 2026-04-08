"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from sts.config import ServiceConfig
from sts.session.models import SessionConfig, SessionState


@pytest.fixture
def service_config() -> ServiceConfig:
    return ServiceConfig()


@pytest.fixture
def session_config() -> SessionConfig:
    return SessionConfig()


@pytest.fixture
def session_state(session_config: SessionConfig) -> SessionState:
    return SessionState(config=session_config)


@pytest.fixture
def silence_audio() -> np.ndarray:
    """1 second of silence at 16 kHz."""
    return np.zeros(16_000, dtype=np.float32)


@pytest.fixture
def sine_audio() -> np.ndarray:
    """1 second of 440 Hz sine wave at 16 kHz (simulates speech-level signal)."""
    t = np.linspace(0, 1, 16_000, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def pcm_s16le_bytes(sine_audio: np.ndarray) -> bytes:
    """Sine wave encoded as PCM signed 16-bit little-endian."""
    return (sine_audio * 32767).astype(np.int16).tobytes()
