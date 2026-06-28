"""Unified data provider interfaces."""

from src.providers.base import (
    KlineProvider,
    ProviderResult,
    ProviderStatus,
    RealtimeQuoteProvider,
    SectorProvider,
)
from src.providers.cache import CacheManager, get_cache_manager
from src.providers.cached_providers import CachedKlineProvider, CachedSectorProvider
from src.providers.health import (
    AkShareHistoryHealthCheck,
    CacheHealthCheck,
    EastMoneyQuoteHealthCheck,
    HealthCheckResult,
    HealthMonitor,
    HealthStatus,
    get_health_monitor,
    overall_status,
)

__all__ = [
    "AkShareHistoryHealthCheck",
    "CacheManager",
    "CacheHealthCheck",
    "CachedKlineProvider",
    "CachedSectorProvider",
    "EastMoneyQuoteHealthCheck",
    "HealthCheckResult",
    "HealthMonitor",
    "HealthStatus",
    "KlineProvider",
    "ProviderResult",
    "ProviderStatus",
    "RealtimeQuoteProvider",
    "SectorProvider",
    "get_cache_manager",
    "get_health_monitor",
    "overall_status",
]
