"""Unified in-memory cache for provider results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Generic, Optional, TypeVar

from src.providers.base import ProviderResult, ProviderStatus

T = TypeVar("T")


@dataclass(frozen=True)
class CacheEntry(Generic[T]):
    """A cache entry with expiry metadata."""

    key: str
    value: T
    created_at: float
    expires_at: float

    @property
    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at


class CacheManager:
    """Small thread-safe TTL cache shared by providers."""

    def __init__(self, now_func=time.time) -> None:
        self._now = now_func
        self._entries: dict[str, CacheEntry[object]] = {}
        self._lock = RLock()

    def make_key(self, namespace: str, *parts: object) -> str:
        normalized_parts = [str(part).strip() for part in parts if part is not None]
        return ":".join([namespace.strip(), *normalized_parts])

    def get(self, key: str) -> Optional[object]:
        now = self._now()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired(now):
                self._entries.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: object, ttl_seconds: float) -> None:
        now = self._now()
        with self._lock:
            self._entries[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + max(0.0, ttl_seconds),
            )

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self, namespace: str = "") -> None:
        with self._lock:
            if not namespace:
                self._entries.clear()
                return
            prefix = f"{namespace.strip()}:"
            for key in list(self._entries):
                if key == namespace or key.startswith(prefix):
                    self._entries.pop(key, None)


def cached_provider_result(
    result: ProviderResult[T],
    *,
    cache_key: str,
) -> ProviderResult[T]:
    """Return a ProviderResult copy marked as served from cache."""
    return ProviderResult(
        status=ProviderStatus.CACHE,
        source=result.source,
        data=result.data,
        message=result.message,
        error_type=result.error_type,
        warnings=list(result.warnings),
        metadata={
            **result.metadata,
            "cache_hit": True,
            "cache_key": cache_key,
            "source_status": result.status.value,
        },
    )


_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Return the process-wide cache manager."""
    global _cache_manager  # noqa: PLW0603
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
