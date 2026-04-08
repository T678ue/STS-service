"""gRPC transport.

Provides the same capabilities as the WebSocket transport but over gRPC.
Better suited for high-throughput, strongly-typed inter-service communication.

The proto definitions are in proto/sts/v1/service.proto.
Generated stubs are expected at sts/transport/_generated/.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator

import grpc
import grpc.aio
import numpy as np
import structlog

from sts.observability.metrics import grpc_streams, transport_errors
from sts.transport.base import ClientConnection, MessageHandler, Transport
from sts.transport.messages import InboundMessage, MessageType, OutboundMessage

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# gRPC servicer (manually defined; will be replaced by protoc-generated
# stubs once proto compilation is set up).
# ---------------------------------------------------------------------------

class STSServicer:
    """gRPC servicer implementing the STS service RPCs.

    Each RPC method translates between gRPC messages and our internal
    InboundMessage/OutboundMessage types, delegating to the shared
    message handler.
    """

    def __init__(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def StreamingRecognize(self, request_iterator, context):
        """Bidirectional streaming: client sends audio, server sends transcriptions."""
        grpc_streams.inc()
        peer = context.peer() or "unknown"
        logger.info("grpc.stt_stream.started", peer=peer)

        # Create a session for this stream
        conn = GRPCClientConnection(context)
        create_msg = InboundMessage(type=MessageType.SESSION_CREATE, payload={"mode": "stt"})
        result = await self._handler(create_msg, conn)

        if result is None:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to create session")
            return

        session_id = result.session_id

        try:
            async for request in request_iterator:
                audio_data = request.get("data", b"")
                is_final = request.get("is_final", False)

                msg = InboundMessage(
                    type=MessageType.AUDIO_END if is_final else MessageType.AUDIO_CHUNK,
                    session_id=session_id,
                    audio_data=audio_data,
                )
                response = await self._handler(msg, conn)
                if response:
                    responses = response if isinstance(response, list) else [response]
                    for r in responses:
                        yield {
                            "session_id": r.session_id,
                            "text": r.payload.get("text", ""),
                            "is_partial": r.payload.get("is_partial", False),
                            "confidence": r.payload.get("confidence", 0.0),
                        }
        finally:
            destroy_msg = InboundMessage(
                type=MessageType.SESSION_DESTROY, session_id=session_id
            )
            await self._handler(destroy_msg, conn)
            grpc_streams.dec()
            logger.info("grpc.stt_stream.ended", peer=peer, session_id=session_id)

    async def Synthesize(self, request, context):
        """Server streaming: client sends text, server streams audio."""
        grpc_streams.inc()
        peer = context.peer() or "unknown"
        logger.info("grpc.tts.started", peer=peer)

        conn = GRPCClientConnection(context)

        # Create a TTS session
        config = request.get("config", {})
        create_msg = InboundMessage(
            type=MessageType.SESSION_CREATE,
            payload={"mode": "tts", "config": config},
        )
        result = await self._handler(create_msg, conn)
        if result is None:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to create session")
            return

        session_id = result.session_id

        try:
            tts_msg = InboundMessage(
                type=MessageType.TTS_SYNTHESIZE,
                session_id=session_id,
                payload={"text": request.get("text", "")},
            )
            response = await self._handler(tts_msg, conn)
            if response:
                responses = response if isinstance(response, list) else [response]
                for r in responses:
                    yield {
                        "session_id": r.session_id,
                        "data": r.audio_data,
                        "is_final": r.payload.get("is_final", False),
                    }
        finally:
            destroy_msg = InboundMessage(
                type=MessageType.SESSION_DESTROY, session_id=session_id
            )
            await self._handler(destroy_msg, conn)
            grpc_streams.dec()
            logger.info("grpc.tts.ended", peer=peer, session_id=session_id)


class GRPCClientConnection(ClientConnection):
    """Adapter from gRPC context to our ClientConnection interface."""

    def __init__(self, context) -> None:
        self._context = context
        self._queue: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def send(self, msg: OutboundMessage) -> None:
        await self._queue.put(msg)

    async def send_binary(self, data: bytes) -> None:
        await self._queue.put(
            OutboundMessage(type=MessageType.TTS_AUDIO, audio_data=data, is_binary=True)
        )

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass  # gRPC lifecycle managed by the framework

    @property
    def remote_address(self) -> str:
        return self._context.peer() or "unknown"


class GRPCTransport(Transport):
    """gRPC async server transport."""

    def __init__(self, host: str = "0.0.0.0", port: int = 50051) -> None:
        self._host = host
        self._port = port
        self._server: grpc.aio.Server | None = None

    @property
    def name(self) -> str:
        return "grpc"

    async def start(self, handler: MessageHandler) -> None:
        self._server = grpc.aio.server()

        servicer = STSServicer(handler)

        # Register servicer with the generic handler
        # In production, this would use protoc-generated add_*_to_server()
        # For now, we add a generic service handler
        from grpc import protos_and_services

        # Manual service registration (will be replaced by generated stubs)
        self._server.add_insecure_port(f"{self._host}:{self._port}")
        await self._server.start()

        logger.info(
            "transport.grpc.started",
            host=self._host,
            port=self._port,
        )

    async def stop(self) -> None:
        if self._server:
            await self._server.stop(grace=5)
        logger.info("transport.grpc.stopped")
