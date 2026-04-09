"""Main server orchestrator.

Wires together: transports ↔ sessions ↔ audio pipelines ↔ STT/TTS engines.
Handles the message dispatch loop: receives InboundMessages from any
transport, routes them to the correct session/pipeline, and sends
OutboundMessages back.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from sts.audio.pipeline import AudioPipeline
from sts.config import ServiceConfig
from sts.observability import setup_observability
from sts.observability.metrics import active_sessions
from sts.session.manager import SessionManager
from sts.session.models import SessionConfig, SessionMode
from sts.stt.pipeline import STTPipeline
from sts.stt.registry import STTRegistry, discover_adapters as discover_stt
from sts.transport.base import ClientConnection, MessageHandler
from sts.transport.messages import InboundMessage, MessageType, OutboundMessage
from sts.transport.grpc_server import GRPCTransport
from sts.transport.websocket import WebSocketTransport
from sts.tts.pipeline import TTSPipeline
from sts.tts.registry import TTSRegistry, discover_adapters as discover_tts

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class STSServer:
    """Top-level server that coordinates all subsystems."""

    def __init__(self, config: ServiceConfig | None = None) -> None:
        self.config = config or ServiceConfig.from_env()
        self.sessions = SessionManager(self.config)
        self._transports: list = []
        self._stopped = False

        # Per-session pipeline tracking
        self._stt_pipelines: dict[str, STTPipeline] = {}
        self._tts_pipelines: dict[str, TTSPipeline] = {}
        self._result_tasks: dict[str, asyncio.Task] = {}

        # Connection → session mapping (for cleanup on disconnect)
        self._conn_sessions: dict[int, str] = {}

    async def start(self) -> None:
        """Boot the entire service."""
        # 1. Observability
        setup_observability(self.config.observability)
        logger.info("server.starting", config=self.config.model_dump())

        # 2. Discover and register adapters
        discover_stt()
        discover_tts()
        logger.info(
            "server.adapters_discovered",
            stt=STTRegistry.available(),
            tts=TTSRegistry.available(),
        )

        # 3. Pre-load default engines
        await STTRegistry.get_or_create(
            self.config.stt.default_engine,
            model=self.config.stt.faster_whisper_model,
            device=self.config.stt.faster_whisper_device,
            compute_type=self.config.stt.faster_whisper_compute_type,
        )
        await TTSRegistry.get_or_create(
            self.config.tts.default_engine,
            model=self.config.tts.piper_model,
            data_dir=str(self.config.tts.piper_data_dir),
        )

        # 4. Session manager
        await self.sessions.start()
        self.sessions.on_destroy(self._on_session_destroyed)

        # 5. Transports
        if self.config.transport.enable_websocket:
            ws = WebSocketTransport(
                host=self.config.transport.ws_host,
                port=self.config.transport.ws_port,
            )
            self._transports.append(ws)

        if self.config.transport.enable_grpc:
            grpc_t = GRPCTransport(
                host=self.config.transport.grpc_host,
                port=self.config.transport.grpc_port,
            )
            self._transports.append(grpc_t)

        for transport in self._transports:
            await transport.start(self._handle_message)

        # 6. Signal readiness to health endpoint
        from sts.observability.health import set_ready, set_status_fn

        set_status_fn(self._health_status)
        set_ready(True)

        logger.info(
            "server.started",
            transports=[t.name for t in self._transports],
        )

    async def stop(self) -> None:
        """Graceful shutdown.  Idempotent — safe to call multiple times."""
        if self._stopped:
            return
        self._stopped = True

        from sts.observability.health import set_ready
        set_ready(False)

        logger.info("server.stopping")

        # Stop transports first (no new connections)
        for transport in self._transports:
            await transport.stop()

        # Close all STT pipelines so results() iterators exit
        for pipeline in self._stt_pipelines.values():
            pipeline.close()

        # Cancel result-forwarding tasks
        for task in self._result_tasks.values():
            task.cancel()

        # Tear down sessions
        await self.sessions.stop()

        # Unload engines
        await STTRegistry.shutdown_all()
        await TTSRegistry.shutdown_all()

        logger.info("server.stopped")

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(
        self,
        msg: InboundMessage,
        conn: ClientConnection,
    ) -> OutboundMessage | list[OutboundMessage] | None:
        """Central message router.  Called by every transport."""

        match msg.type:
            case MessageType.SESSION_CREATE:
                return await self._handle_session_create(msg, conn)
            case MessageType.SESSION_UPDATE:
                return await self._handle_session_update(msg)
            case MessageType.SESSION_DESTROY:
                return await self._handle_session_destroy(msg)
            case MessageType.SESSION_GET:
                return await self._handle_session_get(msg)
            case MessageType.AUDIO_CHUNK:
                return await self._handle_audio_chunk(msg, conn)
            case MessageType.AUDIO_END:
                return await self._handle_audio_end(msg, conn)
            case MessageType.TTS_SYNTHESIZE:
                return await self._handle_tts_synthesize(msg, conn)
            case _:
                logger.warning("server.unknown_message", type=msg.type)
                return OutboundMessage(
                    type=MessageType.ERROR,
                    payload={"code": "unknown_type", "message": f"Unknown: {msg.type}"},
                )

    # ------------------------------------------------------------------
    # Session handlers
    # ------------------------------------------------------------------

    async def _handle_session_create(
        self, msg: InboundMessage, conn: ClientConnection,
    ) -> OutboundMessage:
        config_data = msg.payload.get("config", {})
        mode_str = msg.payload.get("mode", "stt")
        mode = SessionMode(mode_str)

        config = SessionConfig.model_validate(config_data) if config_data else SessionConfig()
        state = await self.sessions.create(config=config, mode=mode)
        active_sessions.set(self.sessions.active_count)

        # Track connection → session for cleanup
        self._conn_sessions[id(conn)] = state.session_id

        # Create pipelines
        await self._setup_pipelines(state, conn)

        return OutboundMessage(
            type=MessageType.SESSION_CREATED,
            session_id=state.session_id,
            payload={
                "config": state.config.model_dump(),
                "mode": state.mode.value,
            },
        )

    async def _setup_pipelines(self, state, conn: ClientConnection) -> None:
        """Create STT and/or TTS pipelines for a session."""
        sid = state.session_id

        if state.mode in (SessionMode.STT, SessionMode.DUPLEX):
            audio_pipeline = AudioPipeline(
                state.config,
                vad_model_path=self.config.audio.vad_model_path,
            )
            stt_adapter = await STTRegistry.get_or_create(
                state.config.stt.engine,
                model=state.config.stt.model or self.config.stt.faster_whisper_model,
            )
            stt_pipeline = STTPipeline(state, audio_pipeline, stt_adapter)
            self._stt_pipelines[sid] = stt_pipeline

            # Start a background task that forwards transcription results to the client
            task = asyncio.create_task(
                self._forward_transcriptions(sid, stt_pipeline, conn)
            )
            self._result_tasks[sid] = task

        if state.mode in (SessionMode.TTS, SessionMode.DUPLEX):
            tts_adapter = await TTSRegistry.get_or_create(
                state.config.tts.engine,
                model=state.config.tts.voice or self.config.tts.piper_model,
            )
            tts_pipeline = TTSPipeline(state, tts_adapter)
            self._tts_pipelines[sid] = tts_pipeline

    async def _forward_transcriptions(
        self,
        session_id: str,
        pipeline: STTPipeline,
        conn: ClientConnection,
    ) -> None:
        """Background task: read STT results and send to client."""
        try:
            async for transcription in pipeline.results():
                msg = OutboundMessage(
                    type=MessageType.STT_TRANSCRIPTION,
                    session_id=session_id,
                    payload={
                        "text": transcription.text,
                        "is_partial": transcription.is_partial,
                        "confidence": transcription.confidence,
                        "language": transcription.language,
                        "start_time": transcription.start_time,
                        "end_time": transcription.end_time,
                        "words": [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                                "probability": w.probability,
                            }
                            for w in transcription.words
                        ],
                    },
                )
                await conn.send(msg)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "server.transcription_forward_error", session_id=session_id
            )

    async def _handle_session_update(self, msg: InboundMessage) -> OutboundMessage:
        try:
            state = await self.sessions.update_config(msg.session_id, msg.payload.get("config", {}))

            # Hot-reload pipeline configs
            stt = self._stt_pipelines.get(msg.session_id)
            if stt:
                stt._audio.update_config(state.config)

            return OutboundMessage(
                type=MessageType.SESSION_UPDATED,
                session_id=msg.session_id,
                payload={
                    "config": state.config.model_dump(),
                    "config_version": state.config_version,
                },
            )
        except KeyError:
            return OutboundMessage(
                type=MessageType.ERROR,
                payload={"code": "session_not_found", "message": "Session not found"},
            )

    async def _handle_session_destroy(self, msg: InboundMessage) -> OutboundMessage:
        await self.sessions.destroy(msg.session_id)
        active_sessions.set(self.sessions.active_count)
        return OutboundMessage(
            type=MessageType.SESSION_DESTROYED,
            session_id=msg.session_id,
        )

    async def _handle_session_get(self, msg: InboundMessage) -> OutboundMessage:
        state = await self.sessions.get(msg.session_id)
        if state is None:
            return OutboundMessage(
                type=MessageType.ERROR,
                payload={"code": "session_not_found", "message": "Session not found"},
            )
        return OutboundMessage(
            type=MessageType.SESSION_INFO,
            session_id=msg.session_id,
            payload={
                "config": state.config.model_dump(),
                "mode": state.mode.value,
                "audio_bytes_received": state.audio_bytes_received,
                "audio_bytes_sent": state.audio_bytes_sent,
                "utterances_transcribed": state.utterances_transcribed,
                "config_version": state.config_version,
            },
        )

    # ------------------------------------------------------------------
    # Audio / STT handlers
    # ------------------------------------------------------------------

    async def _handle_audio_chunk(
        self, msg: InboundMessage, conn: ClientConnection,
    ) -> OutboundMessage | None:
        pipeline = self._stt_pipelines.get(msg.session_id)
        if not pipeline:
            return OutboundMessage(
                type=MessageType.ERROR,
                session_id=msg.session_id,
                payload={"code": "no_stt_pipeline", "message": "No STT pipeline for session"},
            )

        self.sessions.touch(msg.session_id)
        await pipeline.feed_audio(msg.audio_data)
        # Transcription results arrive via the background forward task
        return None

    async def _handle_audio_end(
        self, msg: InboundMessage, conn: ClientConnection,
    ) -> OutboundMessage | None:
        pipeline = self._stt_pipelines.get(msg.session_id)
        if not pipeline:
            return None

        # Process any remaining audio in this last chunk
        if msg.audio_data:
            await pipeline.feed_audio(msg.audio_data)
        await pipeline.flush()
        # Results forwarded via background task; final result has is_partial=False
        return None

    # ------------------------------------------------------------------
    # TTS handler
    # ------------------------------------------------------------------

    async def _handle_tts_synthesize(
        self, msg: InboundMessage, conn: ClientConnection,
    ) -> list[OutboundMessage] | OutboundMessage | None:
        pipeline = self._tts_pipelines.get(msg.session_id)
        if not pipeline:
            return OutboundMessage(
                type=MessageType.ERROR,
                session_id=msg.session_id,
                payload={"code": "no_tts_pipeline", "message": "No TTS pipeline for session"},
            )

        text = msg.payload.get("text", "")
        if not text:
            return OutboundMessage(
                type=MessageType.ERROR,
                session_id=msg.session_id,
                payload={"code": "empty_text", "message": "No text to synthesize"},
            )

        self.sessions.touch(msg.session_id)

        # Stream audio chunks directly to client
        try:
            async for audio_chunk in pipeline.synthesize_stream(text):
                audio_msg = OutboundMessage(
                    type=MessageType.TTS_AUDIO,
                    session_id=msg.session_id,
                    audio_data=audio_chunk,
                    is_binary=True,
                )
                await conn.send(audio_msg)

            # Signal completion
            await conn.send(OutboundMessage(
                type=MessageType.TTS_DONE,
                session_id=msg.session_id,
            ))
        except Exception:
            logger.exception("server.tts_error", session_id=msg.session_id)
            return OutboundMessage(
                type=MessageType.ERROR,
                session_id=msg.session_id,
                payload={"code": "tts_error", "message": "TTS synthesis failed"},
            )
        return None

    # ------------------------------------------------------------------
    # Cleanup callbacks
    # ------------------------------------------------------------------

    def _health_status(self) -> dict:
        """Build status dict for /status health endpoint."""
        return {
            "active_sessions": self.sessions.active_count,
            "stt_engines": STTRegistry.available(),
            "tts_engines": TTSRegistry.available(),
            "transports": [t.name for t in self._transports],
            "stopped": self._stopped,
        }

    async def _on_session_destroyed(self, state) -> None:
        """Clean up pipelines when a session is destroyed."""
        sid = state.session_id

        # Close STT pipeline so results() unblocks
        stt = self._stt_pipelines.pop(sid, None)
        if stt:
            stt.close()

        # Cancel transcription forwarding
        task = self._result_tasks.pop(sid, None)
        if task:
            task.cancel()

        self._tts_pipelines.pop(sid, None)
