"""Cached provider wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.market.akshare_source import HistoryBar
from src.providers.base import KlineProvider, ProviderResult, SectorProvider
from src.providers.cache import CacheManager, cached_provider_result, get_cache_manager

if TYPE_CHECKING:
    from src.rating.sector_provider import SectorContext

DEFAULT_KLINE_CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_SECTOR_CACHE_TTL_SECONDS = 24 * 60 * 60


class CachedKlineProvider:
    """Cache successful historical K-line provider results."""

    def __init__(
        self,
        provider: KlineProvider,
        cache_manager: CacheManager | None = None,
        ttl_seconds: float = DEFAULT_KLINE_CACHE_TTL_SECONDS,
        namespace: str = "kline",
    ) -> None:
        self.provider = provider
        self.cache = cache_manager or get_cache_manager()
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    def get_history(
        self,
        symbol: str,
        period: int = 60,
    ) -> ProviderResult[list[HistoryBar]]:
        key = self.cache.make_key(self.namespace, symbol, period)
        cached = self.cache.get(key)
        if isinstance(cached, ProviderResult):
            return cached_provider_result(cached, cache_key=key)

        result = self.provider.get_history(symbol, period=period)
        if result.ok and result.data:
            self.cache.set(key, result, self.ttl_seconds)
        return result


class CachedSectorProvider:
    """Cache successful sector context provider results."""

    def __init__(
        self,
        provider: SectorProvider,
        cache_manager: CacheManager | None = None,
        ttl_seconds: float = DEFAULT_SECTOR_CACHE_TTL_SECONDS,
        namespace: str = "sector",
    ) -> None:
        self.provider = provider
        self.cache = cache_manager or get_cache_manager()
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    def get_sector_context(self, symbol: str) -> ProviderResult["SectorContext"]:
        key = self.cache.make_key(self.namespace, symbol)
        cached = self.cache.get(key)
        if isinstance(cached, ProviderResult):
            return cached_provider_result(cached, cache_key=key)

        result = self.provider.get_sector_context(symbol)
        if result.ok and result.data is not None:
            self.cache.set(key, result, self.ttl_seconds)
        return result
