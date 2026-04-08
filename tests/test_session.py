"""Tests for session management."""

from __future__ import annotations

import asyncio

import pytest

from sts.config import ServiceConfig
from sts.session.manager import SessionManager
from sts.session.models import SessionConfig, SessionMode


@pytest.fixture
def manager():
    config = ServiceConfig(max_sessions=3, session_idle_timeout_s=1.0)
    return SessionManager(config)


class TestSessionManager:
    async def test_create_session(self, manager):
        await manager.start()
        try:
            state = await manager.create()
            assert state.session_id
            assert state.is_active
            assert state.mode == SessionMode.STT
            assert manager.active_count == 1
        finally:
            await manager.stop()

    async def test_create_with_config(self, manager):
        await manager.start()
        try:
            config = SessionConfig(input_sample_rate=48000, client_name="test-app")
            state = await manager.create(config=config, mode=SessionMode.DUPLEX)
            assert state.config.input_sample_rate == 48000
            assert state.config.client_name == "test-app"
            assert state.mode == SessionMode.DUPLEX
        finally:
            await manager.stop()

    async def test_max_sessions(self, manager):
        await manager.start()
        try:
            await manager.create()
            await manager.create()
            await manager.create()
            with pytest.raises(RuntimeError, match="Max sessions"):
                await manager.create()
        finally:
            await manager.stop()

    async def test_update_config(self, manager):
        await manager.start()
        try:
            state = await manager.create()
            sid = state.session_id

            updated = await manager.update_config(sid, {"input_sample_rate": 44100})
            assert updated.config.input_sample_rate == 44100
            assert updated.config_version == 1
        finally:
            await manager.stop()

    async def test_destroy(self, manager):
        await manager.start()
        try:
            state = await manager.create()
            await manager.destroy(state.session_id)
            assert manager.active_count == 0
            assert await manager.get(state.session_id) is None
        finally:
            await manager.stop()

    async def test_destroy_callback(self, manager):
        destroyed_ids = []

        async def on_destroy(state):
            destroyed_ids.append(state.session_id)

        await manager.start()
        try:
            manager.on_destroy(on_destroy)
            state = await manager.create()
            await manager.destroy(state.session_id)
            assert state.session_id in destroyed_ids
        finally:
            await manager.stop()
