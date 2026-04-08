"""Entry point: python -m sts  or  sts-service CLI."""

from __future__ import annotations

import asyncio
import signal
import sys


def main() -> None:
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass  # uvloop optional; falls back to default event loop

    from sts.config import ServiceConfig
    from sts.server import STSServer

    config = ServiceConfig.from_env()
    server = STSServer(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGTERM / SIGINT
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(server.stop()))

    try:
        loop.run_until_complete(server.start())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()


if __name__ == "__main__":
    main()
