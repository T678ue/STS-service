"""gRPC transport.

Provides the same capabilities as the WebSocket transport but over gRPC.
Uses a generic service handler to avoid requiring proto-generated stubs
at import time.  Once protos are compiled, swap to the generated
add_*_to_server() registrations.

The proto definitions are in proto/sts/v1/service.proto.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import grpc
import grpc.aio
import structlog

from sts.observability.metrics import grpc_streams, transport_errors
from sts.transport.base import ClientConnection, MessageHandler, Transport
from sts.transport.messages import InboundMessage, MessageType, OutboundMessage

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Generic service handler — works without protoc-generated stubs.
# Routes all RPCs through the shared MessageHandler.
# ---------------------------------------------------------------------------

_SERVICE_NAME = "sts.v1.SpeechToText"
_TTS_SERVICE_NAME = "sts.v1.TextToSpeech"
_SESSION_SERVICE_NAME = "sts.v1.SessionService"


class STSGenericHandler(grpc.GenericRpcHandler):
    """Maps gRPC method names to our handler functions.

    This allows the gRPC transport to work immediately without
    compiled proto stubs.  Methods receive raw serialized bytes;
    we use a simple JSON-over-gRPC encoding for now.  Replace with
    proper proto (de)serialization once stubs are generated.
    """

    def __init__(self, servicer: STSServicer) -> None:
        self._servicer = servicer
        self._methods: dict[str, grpc.RpcMethodHandler] = {
            f"/{_SERVICE_NAME}/StreamingRecognize": grpc.stream_stream_rpc_method_handler(
                servicer.StreamingRecognize,
            ),
            f"/{_TTS_SERVICE_NAME}/Synthesize": grpc.unary_stream_rpc_method_handler(
                servicer.Synthesize,
            ),
            f"/{_SESSION_SERVICE_NAME}/CreateSession": grpc.unary_unary_rpc_method_handler(
                servicer.CreateSession,
            ),
            f"/{_SESSION_SERVICE_NAME}/DestroySession": grpc.unary_unary_rpc_method_handler(
                servicer.DestroySession,
            ),
        }

    def service(self, handler_call_details):
        return self._methods.get(handler_call_details.method)


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

        conn = GRPCClientConnection(context)
        create_msg = InboundMessage(
            type=MessageType.SESSION_CREATE, payload={"mode": "stt"},
        )
        result = await self._handler(create_msg, conn)

        if result is None:
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to create session")
            return

        session_id = result.session_id

        try:
            async for request in request_iterator:
                # request is raw bytes: deserialize as needed
                audio_data = request if isinstance(request, bytes) else b""
                is_final = False

                msg = InboundMessage(
                    type=MessageType.AUDIO_END if is_final else MessageType.AUDIO_CHUNK,
                    session_id=session_id,
                    audio_data=audio_data,
                )
                await self._handler(msg, conn)

                # Drain any queued results from the connection
                while not conn._queue.empty():
                    out = conn._queue.get_nowait()
                    yield _serialize_outbound(out)
        finally:
            # Flush remaining audio
            flush_msg = InboundMessage(
                type=MessageType.AUDIO_END, session_id=session_id,
            )
            await self._handler(flush_msg, conn)

            # Drain final results
            await asyncio.sleep(0.1)  # brief yield for transcription to complete
            while not conn._queue.empty():
                out = conn._queue.get_nowait()
                yield _serialize_outbound(out)

            destroy_msg = InboundMessage(
                type=MessageType.SESSION_DESTROY, session_id=session_id,
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

        create_msg = InboundMessage(
            type=MessageType.SESSION_CREATE,
            payload={"mode": "tts"},
        )
        result = await self._handler(create_msg, conn)
        if result is None:
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to create session")
            return

        session_id = result.session_id

        try:
            tts_msg = InboundMessage(
                type=MessageType.TTS_SYNTHESIZE,
                session_id=session_id,
                payload={"text": request.decode("utf-8") if isinstance(request, bytes) else ""},
            )
            await self._handler(tts_msg, conn)

            # The TTS handler streams audio via conn.send()
            # Drain the queue
            while True:
                try:
                    out = await asyncio.wait_for(conn._queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    break
                yield _serialize_outbound(out)
                if out.type == MessageType.TTS_DONE:
                    break
        finally:
            destroy_msg = InboundMessage(
                type=MessageType.SESSION_DESTROY, session_id=session_id,
            )
            await self._handler(destroy_msg, conn)
            grpc_streams.dec()
            logger.info("grpc.tts.ended", peer=peer, session_id=session_id)

    async def CreateSession(self, request, context):
        """Unary: create a session and return info."""
        conn = GRPCClientConnection(context)
        msg = InboundMessage(type=MessageType.SESSION_CREATE, payload={})
        result = await self._handler(msg, conn)
        return _serialize_outbound(result) if result else b""

    async def DestroySession(self, request, context):
        """Unary: destroy a session."""
        import json as _json

        data = _json.loads(request) if isinstance(request, bytes) else {}
        session_id = data.get("session_id", "")
        conn = GRPCClientConnection(context)
        msg = InboundMessage(
            type=MessageType.SESSION_DESTROY, session_id=session_id,
        )
        result = await self._handler(msg, conn)
        return _serialize_outbound(result) if result else b""


def _serialize_outbound(msg: OutboundMessage) -> bytes:
    """Serialize an OutboundMessage to bytes for gRPC.

    For audio, returns raw audio bytes.
    For control messages, returns JSON bytes.
    """
    if msg.is_binary and msg.audio_data:
        return msg.audio_data

    import json
    return json.dumps({
        "type": msg.type.value,
        "session_id": msg.session_id,
        **msg.payload,
    }).encode("utf-8")


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
        generic_handler = STSGenericHandler(servicer)
        self._server.add_generic_rpc_handlers([generic_handler])

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
