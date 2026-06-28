"""Unified provider cache tests."""

from types import SimpleNamespace

from src.market.akshare_source import HistoryBar
from src.providers import (
    CacheManager,
    CachedKlineProvider,
    CachedSectorProvider,
    ProviderResult,
    ProviderStatus,
)
from src.rating.rating_engine import InvestmentRatingEngine
from src.rating.sector_provider import SectorContext


class FakeClock:
    def __init__(self, value=1000.0):
        self.value = value

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def make_bar(index=1):
    return HistoryBar(
        date=f"2026-06-{index:02d}",
        open=10,
        high=11,
        low=9,
        close=10.5,
        volume=1000,
        amount=100000,
    )


def test_cache_manager_get_set_expiry_and_namespace_clear():
    clock = FakeClock()
    cache = CacheManager(now_func=clock.now)
    kline_key = cache.make_key("kline", "300001", 60)
    sector_key = cache.make_key("sector", "300001")

    cache.set(kline_key, "history", ttl_seconds=10)
    cache.set(sector_key, "sector", ttl_seconds=10)

    assert cache.get(kline_key) == "history"
    clock.advance(11)
    assert cache.get(kline_key) is None
    assert cache.get(sector_key) is None

    cache.set(kline_key, "history", ttl_seconds=10)
    cache.set(sector_key, "sector", ttl_seconds=10)
    cache.clear("kline")

    assert cache.get(kline_key) is None
    assert cache.get(sector_key) == "sector"


def test_cached_kline_provider_returns_cache_hit_after_success():
    clock = FakeClock()
    cache = CacheManager(now_func=clock.now)
    calls = {"count": 0}

    def get_history(symbol, period=60):
        calls["count"] += 1
        return ProviderResult.success([make_bar(calls["count"])], "UnitKline")

    provider = CachedKlineProvider(
        SimpleNamespace(get_history=get_history),
        cache_manager=cache,
        ttl_seconds=60,
    )

    first = provider.get_history("300001")
    second = provider.get_history("300001")

    assert first.status == ProviderStatus.SUCCESS
    assert second.status == ProviderStatus.CACHE
    assert second.metadata["cache_hit"] is True
    assert calls["count"] == 1


def test_cached_kline_provider_does_not_cache_failures():
    cache = CacheManager()
    calls = {"count": 0}

    def get_history(symbol, period=60):
        calls["count"] += 1
        return ProviderResult.failed("UnitKline", "failed")

    provider = CachedKlineProvider(
        SimpleNamespace(get_history=get_history),
        cache_manager=cache,
    )

    provider.get_history("300001")
    provider.get_history("300001")

    assert calls["count"] == 2


def test_cached_sector_provider_caches_sector_context():
    cache = CacheManager()
    calls = {"count": 0}

    def get_sector_context(symbol):
        calls["count"] += 1
        return ProviderResult.success(
            SectorContext(industry="半导体", concepts=["AI硬件"]),
            "UnitSector",
        )

    provider = CachedSectorProvider(
        SimpleNamespace(get_sector_context=get_sector_context),
        cache_manager=cache,
    )

    first = provider.get_sector_context("300001")
    second = provider.get_sector_context("300001")

    assert first.status == ProviderStatus.SUCCESS
    assert second.status == ProviderStatus.CACHE
    assert second.data.industry == "半导体"
    assert calls["count"] == 1


def test_rating_engine_uses_cache_manager_when_explicitly_enabled():
    cache = CacheManager()
    calls = {"history": 0}

    def get_history(symbol, period=60):
        calls["history"] += 1
        return [make_bar(calls["history"])]

    market_data = SimpleNamespace(
        get_quote=lambda symbol, market="CN": None,
        get_history=get_history,
        get_recent_bars=lambda symbol, market="CN", limit=60: [],
        get_stock_info=lambda symbol: None,
        get_index_quotes=lambda market="CN": [],
    )
    sector_provider = SimpleNamespace(
        get_sector_context=lambda symbol: ProviderResult.success(
            SectorContext(),
            "UnitSector",
        )
    )

    engine = InvestmentRatingEngine(
        market_data=market_data,
        rating_sector_provider=sector_provider,
        cache_manager=cache,
        persist_history=False,
    )

    engine.evaluate("300001")
    engine.evaluate("300001")

    assert calls["history"] == 1
