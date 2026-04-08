"""TTS adapter registry.

Mirrors the STT registry pattern.  Adapters register via decorator.
"""

from __future__ import annotations

import structlog

from sts.tts.base import TTSAdapter

logger = structlog.get_logger(__name__)


class TTSRegistry:
    _adapters: dict[str, type[TTSAdapter]] = {}
    _instances: dict[str, TTSAdapter] = {}

    @classmethod
    def register(cls, adapter_cls: type[TTSAdapter]) -> type[TTSAdapter]:
        name = getattr(adapter_cls, "NAME", None) or adapter_cls.__name__
        cls._adapters[name] = adapter_cls
        logger.debug("tts.adapter_registered", name=name)
        return adapter_cls

    @classmethod
    def get_class(cls, name: str) -> type[TTSAdapter]:
        if name not in cls._adapters:
            available = list(cls._adapters.keys())
            raise KeyError(f"TTS adapter '{name}' not found. Available: {available}")
        return cls._adapters[name]

    @classmethod
    async def get_or_create(cls, name: str, **kwargs) -> TTSAdapter:
        if name not in cls._instances:
            adapter_cls = cls.get_class(name)
            instance = adapter_cls()
            await instance.load(**kwargs)
            cls._instances[name] = instance
            logger.info("tts.adapter_loaded", name=name)
        return cls._instances[name]

    @classmethod
    async def shutdown_all(cls) -> None:
        for name, instance in cls._instances.items():
            logger.info("tts.adapter_unloading", name=name)
            await instance.unload()
        cls._instances.clear()

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._adapters.keys())


def discover_adapters() -> None:
    """Import all built-in adapter modules to trigger registration."""
    import sts.tts.adapters.piper  # noqa: F401
