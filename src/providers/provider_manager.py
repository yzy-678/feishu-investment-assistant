"""ProviderManager: unified scheduling, cache and fallback layer."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.providers.akshare_provider import AkShareProvider
from src.providers.base import BaseProvider, ProviderResult, ProviderStatus
from src.providers.cache import CacheManager, cached_provider_result, get_cache_manager
from src.providers.eastmoney_provider import EastMoneyProvider
from src.rating.sector_provider import SectorContext

logger = logging.getLogger(__name__)


class ProviderManager:
    """Route all data requests through providers by priority and capability."""

    def __init__(
        self,
        providers: Optional[list[BaseProvider]] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        self.providers = sorted(
            providers or [EastMoneyProvider(), AkShareProvider()],
            key=lambda provider: provider.fallback_priority,
        )
        self.cache = cache_manager or get_cache_manager()

    def get_quote(self, symbol: str) -> ProviderResult[Any]:
        return self._first_success(
            "quote",
            symbol,
            lambda provider: provider.get_quote(symbol),
            providers=self._providers_by_source("EastMoney"),
        )

    def get_kline(self, symbol: str, period: int = 60) -> ProviderResult[Any]:
        return self._first_success(
            "kline",
            symbol,
            lambda provider: provider.get_kline(symbol, period=period),
            cache_parts=(period,),
            providers=self._providers_by_source("AkShare"),
        )

    def get_sector(self, symbol: str) -> ProviderResult[SectorContext]:
        cache_key = self.cache.make_key("provider-manager", "sector", symbol)
        cached = self.cache.get(cache_key)
        if isinstance(cached, ProviderResult):
            return cached_provider_result(cached, cache_key=cache_key)

        attempts: list[ProviderResult[Any]] = []
        name = ""
        industry = ""
        region_sector = ""
        concepts: list[str] = []
        sources: list[str] = []

        for provider in self.providers:
            sector_result = provider.get_sector(symbol)
            attempts.append(sector_result)
            context = sector_result.data
            if isinstance(context, SectorContext):
                if context.name and not name:
                    name = context.name
                if context.industry and not industry:
                    industry = context.industry
                if context.region_sector and not region_sector:
                    region_sector = context.region_sector
                self._merge_concepts(concepts, context.concepts)
                self._merge_sources(sources, context.data_source or sector_result.source)

            concepts_result = provider.get_concepts(symbol)
            attempts.append(concepts_result)
            self._merge_concepts(concepts, concepts_result.data or [])
            if concepts_result.ok and concepts_result.data:
                self._merge_sources(sources, concepts_result.source)

        context = SectorContext(
            name=name,
            industry=industry,
            region_sector=region_sector,
            concepts=concepts,
            data_source=", ".join(sources),
            industry_score=10.0 if industry else None,
            concept_score=10.0 if concepts else None,
            warning=self._sector_warning(bool(industry), bool(concepts)),
        )
        metadata = {
            "symbol": symbol,
            "sector_status": context.sector_status,
            "attempts": self._attempts_metadata(attempts),
        }
        if context.available:
            result = ProviderResult.success(
                context,
                context.data_source or "ProviderManager",
                metadata=metadata,
            )
        else:
            result = ProviderResult.partial(
                context,
                "ProviderManager",
                context.warning,
                warnings=[context.warning],
                metadata=metadata,
            )
        self.cache.set(cache_key, result, ttl_seconds=300)
        return result

    def get_concepts(self, symbol: str) -> ProviderResult[list[str]]:
        sector = self.get_sector(symbol)
        concepts = sector.data.concepts if sector.data is not None else []
        if concepts:
            return ProviderResult.success(concepts, sector.source, metadata=sector.metadata)
        return ProviderResult.partial(
            [],
            sector.source or "ProviderManager",
            "概念数据暂不可用",
            metadata=sector.metadata,
        )

    def get_fund_flow(self, symbol: str) -> ProviderResult[dict[str, Any]]:
        result = self._first_success(
            "fund_flow",
            symbol,
            lambda provider: provider.get_fund_flow(symbol),
        )
        return result if result.data is not None else ProviderResult.partial({}, result.source, result.message)

    def get_news(self, symbol: str) -> ProviderResult[list[Any]]:
        result = self._first_success("news", symbol, lambda provider: provider.get_news(symbol))
        return result if result.data is not None else ProviderResult.partial([], result.source, result.message)

    def get_index_quotes(self) -> ProviderResult[Any]:
        for provider in self.providers:
            getter = getattr(provider, "get_index_quotes", None)
            if not callable(getter):
                continue
            result = getter()
            if result.ok and result.data:
                return result
        return ProviderResult.partial([], "ProviderManager", "指数行情暂不可用")

    def health_check(self) -> dict[str, ProviderResult[dict[str, Any]]]:
        return {provider.data_source: provider.health_check() for provider in self.providers}

    def _first_success(
        self,
        namespace: str,
        symbol: str,
        fetcher,
        cache_parts: tuple[Any, ...] = (),
        providers: Optional[list[BaseProvider]] = None,
    ) -> ProviderResult[Any]:
        cache_key = self.cache.make_key("provider-manager", namespace, symbol, *cache_parts)
        cached = self.cache.get(cache_key)
        if isinstance(cached, ProviderResult):
            return cached_provider_result(cached, cache_key=cache_key)

        attempts: list[ProviderResult[Any]] = []
        for provider in providers or self.providers:
            result = fetcher(provider)
            attempts.append(result)
            if result.ok and result.data not in (None, [], {}):
                metadata = {
                    **result.metadata,
                    "attempts": self._attempts_metadata(attempts),
                    "fallback": len(attempts) > 1,
                    "fallback_from": [
                        item.source for item in attempts[:-1] if not item.ok
                    ],
                }
                final = ProviderResult(
                    status=result.status,
                    source=result.source,
                    data=result.data,
                    message=result.message,
                    error_type=result.error_type,
                    warnings=list(result.warnings),
                    metadata=metadata,
                )
                ttl = getattr(provider.config, "cache_ttl_seconds", 60.0)
                if provider.cache_enabled:
                    self.cache.set(cache_key, final, ttl_seconds=ttl)
                return final

        last = attempts[-1] if attempts else None
        return ProviderResult(
            status=ProviderStatus.FAILED,
            source=last.source if last else "ProviderManager",
            message=(last.message if last else "") or f"{namespace}数据不可用",
            error_type=last.error_type if last else "",
            metadata={"attempts": self._attempts_metadata(attempts)},
        )

    def _providers_by_source(self, source: str) -> list[BaseProvider]:
        matches = [
            provider
            for provider in self.providers
            if provider.data_source.strip().lower() == source.strip().lower()
        ]
        return matches or list(self.providers)

    @staticmethod
    def _merge_concepts(target: list[str], values: list[str]) -> None:
        for value in values:
            concept = str(value or "").strip()
            if concept and concept not in target:
                target.append(concept)

    @staticmethod
    def _merge_sources(target: list[str], value: str) -> None:
        for item in str(value or "").split(","):
            source = item.strip()
            if source and source not in target:
                target.append(source)

    @staticmethod
    def _attempts_metadata(attempts: list[ProviderResult[Any]]) -> list[dict[str, Any]]:
        return [
            {
                "source": item.source,
                "status": item.status.value,
                "message": item.message,
                "error_type": item.error_type,
            }
            for item in attempts
        ]

    @staticmethod
    def _sector_warning(industry_available: bool, concepts_available: bool) -> str:
        if industry_available and concepts_available:
            return ""
        if industry_available:
            return "概念数据暂不可用，板块评分部分纳入。"
        if concepts_available:
            return "行业数据暂不可用，板块评分部分纳入。"
        return "行业/概念数据暂不可用，板块评分暂未纳入。"


_provider_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Return the process-wide ProviderManager."""
    global _provider_manager  # noqa: PLW0603
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager
