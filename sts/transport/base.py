"""Abstract transport interface.

Transports handle the wire protocol (WebSocket, gRPC, etc.) and convert
to/from the internal message types.  The server orchestrator calls
`start()` and `stop()` on each enabled transport.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, Coroutine

if TYPE_CHECKING:
    from sts.transport.messages import InboundMessage, OutboundMessage

# Type alias for the message handler the server registers with each transport
MessageHandler = Callable[
    ["InboundMessage", "ClientConnection"],
    Coroutine[None, None, "OutboundMessage | list[OutboundMessage] | None"],
]


class ClientConnection(abc.ABC):
    """Represents a single client connection, regardless of transport."""

    @abc.abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to this client."""

    @abc.abstractmethod
    async def send_binary(self, data: bytes) -> None:
        """Send raw binary data (audio) to this client."""

    @abc.abstractmethod
    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection."""

    @property
    @abc.abstractmethod
    def remote_address(self) -> str:
        """Client's address for logging."""


class Transport(abc.ABC):
    """Interface for transport layer implementations."""

    @abc.abstractmethod
    async def start(self, handler: MessageHandler) -> None:
        """Start listening for connections. `handler` processes each message."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Graceful shutdown."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Transport identifier for metrics/logs."""
