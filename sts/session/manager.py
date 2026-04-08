"""Session lifecycle management.

Handles creation, lookup, update (hot-reload), expiry, and teardown of
per-client sessions.  Thread-safe via asyncio locks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sts.config import ServiceConfig

from sts.session.models import SessionConfig, SessionMode, SessionState

logger = structlog.get_logger(__name__)


class SessionManager:
    def __init__(self, service_config: ServiceConfig) -> None:
        self._config = service_config
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

        # Callbacks that transports / pipelines can register to react to
        # session events.
        self._on_create: list[asyncio.Future[None] | None] = []
        self._on_update_callbacks: list[
            callable  # type: ignore[valid-type]
        ] = []
        self._on_destroy_callbacks: list[
            callable  # type: ignore[valid-type]
        ] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._reap_idle_sessions())
        logger.info("session_manager.started", max_sessions=self._config.max_sessions)

    async def stop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
        # Destroy all remaining sessions
        session_ids = list(self._sessions.keys())
        for sid in session_ids:
            await self.destroy(sid)
        logger.info("session_manager.stopped", sessions_cleaned=len(session_ids))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create(
        self,
        config: SessionConfig | None = None,
        mode: SessionMode = SessionMode.STT,
    ) -> SessionState:
        async with self._lock:
            if len(self._sessions) >= self._config.max_sessions:
                raise RuntimeError(
                    f"Max sessions ({self._config.max_sessions}) reached"
                )

            state = SessionState(
                config=config or SessionConfig(),
                mode=mode,
            )
            self._sessions[state.session_id] = state

        logger.info(
            "session.created",
            session_id=state.session_id,
            mode=mode.value,
            client_name=state.config.client_name,
            input_format=state.config.input_format.value,
            input_sample_rate=state.config.input_sample_rate,
        )
        await self._fire_update(state)
        return state

    async def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    async def update_config(
        self, session_id: str, patch: dict
    ) -> SessionState:
        """Hot-reload session config.  Merges `patch` into existing config."""
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"Session {session_id} not found")

            updated = state.config.model_copy(update=patch)
            state.config = updated
            state.config_version += 1
            state.last_activity = datetime.now(timezone.utc)

        logger.info(
            "session.config_updated",
            session_id=session_id,
            config_version=state.config_version,
            changes=list(patch.keys()),
        )
        await self._fire_update(state)
        return state

    async def destroy(self, session_id: str) -> None:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
        if state:
            state.is_active = False
            logger.info(
                "session.destroyed",
                session_id=session_id,
                duration_s=(
                    datetime.now(timezone.utc) - state.created_at
                ).total_seconds(),
                audio_in=state.audio_bytes_received,
                audio_out=state.audio_bytes_sent,
                utterances=state.utterances_transcribed,
            )
            for cb in self._on_destroy_callbacks:
                try:
                    await cb(state)
                except Exception:
                    logger.exception("session.destroy_callback_error")

    def touch(self, session_id: str) -> None:
        """Bump last_activity timestamp (called on every audio chunk)."""
        state = self._sessions.get(session_id)
        if state:
            state.last_activity = datetime.now(timezone.utc)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def list_sessions(self) -> list[SessionState]:
        return list(self._sessions.values())

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_update(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._on_update_callbacks.append(callback)

    def on_destroy(self, callback) -> None:  # type: ignore[no-untyped-def]
        self._on_destroy_callbacks.append(callback)

    async def _fire_update(self, state: SessionState) -> None:
        for cb in self._on_update_callbacks:
            try:
                await cb(state)
            except Exception:
                logger.exception("session.update_callback_error")

    # ------------------------------------------------------------------
    # Idle reaper
    # ------------------------------------------------------------------

    async def _reap_idle_sessions(self) -> None:
        timeout = self._config.session_idle_timeout_s
        while True:
            await asyncio.sleep(30)  # check every 30 s
            now = datetime.now(timezone.utc)
            expired = [
                sid
                for sid, s in self._sessions.items()
                if (now - s.last_activity).total_seconds() > timeout
            ]
            for sid in expired:
                logger.warning("session.idle_expired", session_id=sid)
                await self.destroy(sid)
