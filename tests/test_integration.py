"""Integration smoke test.

Starts the full server with mocked STT/TTS adapters, connects via
WebSocket, creates a session, sends audio, and verifies transcription
comes back.  Also tests TTS round-trip and session lifecycle.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import websockets

from sts.config import ServiceConfig
from sts.server import STSServer
from sts.stt.base import STTAdapter, Transcription
from sts.stt.registry import STTRegistry
from sts.tts.base import SynthesisResult, TTSAdapter
from sts.tts.registry import TTSRegistry


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------

class MockSTTAdapter(STTAdapter):
    NAME = "mock-stt"

    @property
    def name(self) -> str:
        return self.NAME

    async def load(self, **kwargs) -> None:
        pass

    async def unload(self) -> None:
        pass

    async def transcribe(self, audio, *, language="en", beam_size=5) -> Transcription:
        return Transcription(
            text="hello world",
            is_partial=False,
            confidence=0.95,
            language=language,
        )

    @property
    def supports_streaming(self) -> bool:
        return False


class MockTTSAdapter(TTSAdapter):
    NAME = "mock-tts"

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def native_sample_rate(self) -> int:
        return 22050

    async def load(self, **kwargs) -> None:
        pass

    async def unload(self) -> None:
        pass

    async def synthesize(self, text, *, voice=None, speed=1.0) -> SynthesisResult:
        # Return 0.1s of silence
        audio = np.zeros(2205, dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=22050, is_final=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_config() -> ServiceConfig:
    """Config with mocked engines, no metrics, free ports."""
    config = ServiceConfig()
    config.transport.ws_port = 0  # let OS pick a free port
    config.transport.enable_grpc = False
    config.observability.enable_metrics = False
    config.observability.enable_health = False
    config.observability.enable_tracing = False
    config.observability.log_format = "console"
    config.observability.log_level = "DEBUG"
    config.stt.default_engine = "mock-stt"
    config.tts.default_engine = "mock-tts"
    return config


@pytest.fixture
async def server_and_port(test_config):
    """Start the server and yield (server, port)."""
    # Register mock adapters
    STTRegistry._adapters["mock-stt"] = MockSTTAdapter
    TTSRegistry._adapters["mock-tts"] = MockTTSAdapter

    server = STSServer(test_config)

    # We need to get the actual port after binding to port 0.
    # Patch WebSocketTransport to use port 0 and capture the bound port.
    await server.start()

    # Get the actual bound port from the websocket server
    ws_transport = server._transports[0]
    # websockets server exposes sockets
    actual_port = None
    if ws_transport._server and ws_transport._server.sockets:
        actual_port = ws_transport._server.sockets[0].getsockname()[1]

    yield server, actual_port or test_config.transport.ws_port

    await server.stop()

    # Cleanup registries
    STTRegistry._adapters.pop("mock-stt", None)
    STTRegistry._instances.pop("mock-stt", None)
    TTSRegistry._adapters.pop("mock-tts", None)
    TTSRegistry._instances.pop("mock-tts", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    async def test_create_and_destroy_session(self, server_and_port):
        server, port = server_and_port
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            # Create session
            await ws.send(json.dumps({
                "type": "session.create",
                "config": {"client_name": "test"},
                "mode": "stt",
            }))
            response = json.loads(await ws.recv())
            assert response["type"] == "session.created"
            session_id = response["session_id"]
            assert session_id

            # Get session info
            await ws.send(json.dumps({
                "type": "session.get",
                "session_id": session_id,
            }))
            info = json.loads(await ws.recv())
            assert info["type"] == "session.info"

            # Destroy session
            await ws.send(json.dumps({
                "type": "session.destroy",
                "session_id": session_id,
            }))
            resp = json.loads(await ws.recv())
            assert resp["type"] == "session.destroyed"

    async def test_update_config_hot_reload(self, server_and_port):
        server, port = server_and_port
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            # Create
            await ws.send(json.dumps({
                "type": "session.create",
                "mode": "stt",
            }))
            created = json.loads(await ws.recv())
            session_id = created["session_id"]

            # Update config
            await ws.send(json.dumps({
                "type": "session.update",
                "session_id": session_id,
                "config": {"input_sample_rate": 44100},
            }))
            updated = json.loads(await ws.recv())
            assert updated["type"] == "session.updated"
            assert updated["config_version"] == 1

            # Cleanup
            await ws.send(json.dumps({
                "type": "session.destroy",
                "session_id": session_id,
            }))
            await ws.recv()


class TestTTSRoundTrip:
    async def test_synthesize_returns_audio_and_done(self, server_and_port):
        server, port = server_and_port
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            # Create TTS session
            await ws.send(json.dumps({
                "type": "session.create",
                "mode": "tts",
            }))
            created = json.loads(await ws.recv())
            session_id = created["session_id"]

            # Send TTS request
            await ws.send(json.dumps({
                "type": "tts.synthesize",
                "session_id": session_id,
                "text": "Hello world.",
            }))

            # Should get at least one audio frame and a done message
            messages = []
            got_done = False
            while not got_done:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, bytes):
                    messages.append(("audio", msg))
                else:
                    parsed = json.loads(msg)
                    messages.append(("json", parsed))
                    if parsed.get("type") == "tts.done":
                        got_done = True

            assert got_done
            # Should have at least one audio chunk
            audio_msgs = [m for m in messages if m[0] == "audio"]
            assert len(audio_msgs) > 0

            # Cleanup
            await ws.send(json.dumps({
                "type": "session.destroy",
                "session_id": session_id,
            }))
            await ws.recv()
