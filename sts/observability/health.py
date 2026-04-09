"""Lightweight health check HTTP endpoint.

Serves /health (liveness) and /ready (readiness) on a configurable port.
Runs alongside the Prometheus metrics server, or standalone if metrics
are disabled.

/health → 200 if the process is alive
/ready  → 200 if STT and TTS engines are loaded and accepting sessions
/status → JSON blob with session counts, engine info, uptime
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_start_time = time.monotonic()
_ready = False
_status_fn: callable = lambda: {}  # type: ignore[assignment]


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/ready":
            if _ready:
                self._respond(200, {"status": "ready"})
            else:
                self._respond(503, {"status": "not_ready"})
        elif self.path == "/status":
            status = _status_fn()
            status["uptime_s"] = round(time.monotonic() - _start_time, 1)
            self._respond(200, status)
        else:
            self._respond(404, {"error": "not_found"})

    def _respond(self, code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args) -> None:
        pass  # suppress default stderr logging


def start_health_server(port: int = 8080) -> HTTPServer:
    """Start the health check HTTP server in a daemon thread."""
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("health.server_started", port=port)
    return server


def set_ready(ready: bool = True) -> None:
    global _ready
    _ready = ready


def set_status_fn(fn) -> None:
    global _status_fn
    _status_fn = fn
