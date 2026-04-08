"""STT adapter registry.

Discovers and manages STT engine adapters.  Adapters register themselves
at import time via `@STTRegistry.register`.
"""

from __future__ import annotations

import structlog

from sts.stt.base import STTAdapter

logger = structlog.get_logger(__name__)


class STTRegistry:
    _adapters: dict[str, type[STTAdapter]] = {}
    _instances: dict[str, STTAdapter] = {}

    @classmethod
    def register(cls, adapter_cls: type[STTAdapter]) -> type[STTAdapter]:
        """Class decorator to register an STT adapter."""
        # Instantiate temporarily to read the name property
        # (adapters must have a no-arg __init__ or we read from class attr)
        name = getattr(adapter_cls, "NAME", None) or adapter_cls.__name__
        cls._adapters[name] = adapter_cls
        logger.debug("stt.adapter_registered", name=name)
        return adapter_cls

    @classmethod
    def get_class(cls, name: str) -> type[STTAdapter]:
        if name not in cls._adapters:
            available = list(cls._adapters.keys())
            raise KeyError(f"STT adapter '{name}' not found. Available: {available}")
        return cls._adapters[name]

    @classmethod
    async def get_or_create(cls, name: str, **kwargs) -> STTAdapter:
        """Get a running instance, creating and loading if needed."""
        if name not in cls._instances:
            adapter_cls = cls.get_class(name)
            instance = adapter_cls()
            await instance.load(**kwargs)
            cls._instances[name] = instance
            logger.info("stt.adapter_loaded", name=name)
        return cls._instances[name]

    @classmethod
    async def shutdown_all(cls) -> None:
        for name, instance in cls._instances.items():
            logger.info("stt.adapter_unloading", name=name)
            await instance.unload()
        cls._instances.clear()

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._adapters.keys())


def discover_adapters() -> None:
    """Import all built-in adapter modules to trigger registration."""
    import sts.stt.adapters.faster_whisper  # noqa: F401
