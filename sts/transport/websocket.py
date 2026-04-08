"""WebSocket transport.

Protocol:
  - Text frames: JSON control messages (session CRUD, TTS requests,
    transcription results, errors).
  - Binary frames: audio data prefixed with a 17-byte header:
      [1 byte: direction/type][16 bytes: session UUID hex]

Audio direction bytes:
  0x01 = audio input  (client → service, for STT)
  0x02 = audio output (service → client, from TTS)
  0x03 = audio input final (last chunk from client)
  0x04 = audio output final (last chunk from service)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING

import structlog
import websockets
from websockets.asyncio.server import ServerConnection

from sts.observability.metrics import transport_errors, ws_connections
from sts.transport.base import ClientConnection, MessageHandler, Transport
from sts.transport.messages import InboundMessage, MessageType, OutboundMessage

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Binary frame header constants
AUDIO_INPUT = 0x01
AUDIO_OUTPUT = 0x02
AUDIO_INPUT_FINAL = 0x03
AUDIO_OUTPUT_FINAL = 0x04
HEADER_SIZE = 17  # 1 byte type + 16 bytes session id (hex)


class WebSocketClientConnection(ClientConnection):
    """Wraps a websockets connection."""

    def __init__(self, ws: ServerConnection) -> None:
        self._ws = ws

    async def send(self, msg: OutboundMessage) -> None:
        if msg.is_binary and msg.audio_data:
            header = bytes([AUDIO_OUTPUT]) + msg.session_id[:16].encode().ljust(16, b"\0")
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
        self._server = await websockets.serve(
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

    async def _on_connection(self, ws: ServerConnection) -> None:
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
            session_id = raw[1:HEADER_SIZE].rstrip(b"\0").decode("ascii", errors="replace")
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
