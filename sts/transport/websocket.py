"""WebSocket transport.

Protocol:
  - Text frames: JSON control messages (session CRUD, TTS requests,
    transcription results, errors).
  - Binary frames: audio data prefixed with a header:
      [1 byte: direction/type][32 bytes: session ID hex (zero-padded)]

Audio direction bytes:
  0x01 = audio input  (client -> service, for STT)
  0x02 = audio output (service -> client, from TTS)
  0x03 = audio input final (last chunk from client)
  0x04 = audio output final (last chunk from service)
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog
import websockets
import websockets.asyncio.server

from sts.observability.metrics import transport_errors, ws_connections
from sts.transport.base import ClientConnection, MessageHandler, Transport
from sts.transport.messages import InboundMessage, MessageType, OutboundMessage

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Binary frame header: 1 byte type + 32 bytes session id (hex, zero-padded)
AUDIO_INPUT = 0x01
AUDIO_OUTPUT = 0x02
AUDIO_INPUT_FINAL = 0x03
AUDIO_OUTPUT_FINAL = 0x04
SESSION_ID_LEN = 32
HEADER_SIZE = 1 + SESSION_ID_LEN  # 33 bytes


class WebSocketClientConnection(ClientConnection):
    """Wraps a websockets connection."""

    def __init__(self, ws: websockets.asyncio.server.ServerConnection) -> None:
        self._ws = ws
        # Sessions owned by this connection (for cleanup on disconnect)
        self.session_ids: set[str] = set()

    async def send(self, msg: OutboundMessage) -> None:
        if msg.is_binary and msg.audio_data:
            sid_bytes = msg.session_id.encode("ascii")[:SESSION_ID_LEN].ljust(
                SESSION_ID_LEN, b"\0"
            )
            header = bytes([AUDIO_OUTPUT]) + sid_bytes
            await self._ws.send(header + msg.audio_data)
        else:
            payload = {
                "type": msg.type.value,
                "session_id": msg.session_id,
                **msg.payload,
            }
            await self._ws.send(json.dumps(payload))

    async def send_binary(self, data: bytes) -> None:
        await self._ws.send(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code, reason)

    @property
    def remote_address(self) -> str:
        try:
            addr = self._ws.remote_address
            if addr:
                return f"{addr[0]}:{addr[1]}"
        except Exception:
            pass
        return "unknown"


class WebSocketTransport(Transport):
    """WebSocket server transport."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._server = None
        self._handler: MessageHandler | None = None

    @property
    def name(self) -> str:
        return "websocket"

    async def start(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._server = await websockets.asyncio.server.serve(
            self._on_connection,
            self._host,
            self._port,
            max_size=10 * 1024 * 1024,  # 10 MB max frame
            ping_interval=20,
            ping_timeout=20,
        )
        logger.info(
            "transport.websocket.started",
            host=self._host,
            port=self._port,
        )

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("transport.websocket.stopped")

    async def _on_connection(
        self, ws: websockets.asyncio.server.ServerConnection,
    ) -> None:
        """Handle a single WebSocket connection lifecycle."""
        conn = WebSocketClientConnection(ws)
        ws_connections.inc()
        logger.info("ws.connected", remote=conn.remote_address)

        try:
            async for raw_message in ws:
                try:
                    msg = self._decode(raw_message)
                    if msg and self._handler:
                        result = await self._handler(msg, conn)
                        # Track sessions created on this connection
                        if (
                            result
                            and not isinstance(result, list)
                            and result.type == MessageType.SESSION_CREATED
                        ):
                            conn.session_ids.add(result.session_id)

                        if result:
                            if isinstance(result, list):
                                for r in result:
                                    await conn.send(r)
                            else:
                                await conn.send(result)
                except Exception:
                    logger.exception("ws.message_error", remote=conn.remote_address)
                    transport_errors.labels(
                        transport="websocket", error_type="message_handler"
                    ).inc()
                    error_msg = OutboundMessage(
                        type=MessageType.ERROR,
                        payload={"code": "internal_error", "message": "Internal error"},
                    )
                    await conn.send(error_msg)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.exception("ws.connection_error", remote=conn.remote_address)
            transport_errors.labels(
                transport="websocket", error_type="connection"
            ).inc()
        finally:
            # Destroy all sessions owned by this connection
            if self._handler:
                for sid in conn.session_ids:
                    try:
                        destroy_msg = InboundMessage(
                            type=MessageType.SESSION_DESTROY, session_id=sid,
                        )
                        await self._handler(destroy_msg, conn)
                    except Exception:
                        logger.exception("ws.disconnect_cleanup_error", session_id=sid)
            ws_connections.dec()
            logger.info("ws.disconnected", remote=conn.remote_address)

    def _decode(self, raw) -> InboundMessage | None:
        """Decode a WebSocket frame into an InboundMessage."""
        if isinstance(raw, bytes):
            # Binary frame: audio data
            if len(raw) < HEADER_SIZE:
                logger.warning("ws.binary_frame_too_short", size=len(raw))
                return None

            direction = raw[0]
            session_id = (
                raw[1:HEADER_SIZE].rstrip(b"\0").decode("ascii", errors="replace")
            )
            audio = raw[HEADER_SIZE:]

            if direction == AUDIO_INPUT_FINAL:
                return InboundMessage(
                    type=MessageType.AUDIO_END,
                    session_id=session_id,
                    audio_data=audio,
                )
            return InboundMessage(
                type=MessageType.AUDIO_CHUNK,
                session_id=session_id,
                audio_data=audio,
            )

        # Text frame: JSON control message
        data = json.loads(raw)
        msg_type = data.pop("type", None)
        session_id = data.pop("session_id", "")

        try:
            mt = MessageType(msg_type)
        except ValueError:
            logger.warning("ws.unknown_message_type", type=msg_type)
            return None

        return InboundMessage(
            type=mt,
            session_id=session_id,
            payload=data,
        )
