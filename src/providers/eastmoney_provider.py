"""EastMoney provider implementation for realtime and sector data."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.market.service import MarketDataService, QuoteSnapshot
from src.providers.base import ProviderConfig, ProviderResult
from src.rating.sector_provider import EastMoneyRawSectorSource, SectorContext

logger = logging.getLogger(__name__)


class EastMoneyProvider:
    """Provider facade for EastMoney data sources."""

    def __init__(
        self,
        service: Optional[MarketDataService] = None,
        sector_source: Optional[EastMoneyRawSectorSource] = None,
        config: Optional[ProviderConfig] = None,
    ) -> None:
        self.config = config or ProviderConfig(
            data_source="EastMoney",
            timeout=8.0,
            cache_enabled=True,
            fallback_priority=10,
            cache_ttl_seconds=30.0,
        )
        self.service = service or MarketDataService(timeout=self.config.timeout)
        self.sector_source = sector_source or EastMoneyRawSectorSource(
            timeout=self.config.timeout
        )

    @property
    def data_source(self) -> str:
        return self.config.data_source

    @property
    def timeout(self) -> float:
        return self.config.timeout

    @property
    def cache_enabled(self) -> bool:
        return self.config.cache_enabled

    @property
    def fallback_priority(self) -> int:
        return self.config.fallback_priority

    def get_quote(self, symbol: str) -> ProviderResult[QuoteSnapshot]:
        try:
            quote = self.service.get_quote(symbol, market="CN")
        except TimeoutError as exc:
            logger.warning("EastMoney quote timeout: symbol=%s error=%s", symbol, exc)
            return ProviderResult.timeout(self.data_source, "实时行情超时")
        except Exception as exc:
            logger.warning("EastMoney quote failed: symbol=%s error=%s", symbol, exc)
            return ProviderResult.failed(
                self.data_source,
                "实时行情不可用",
                error_type=type(exc).__name__,
            )
        return ProviderResult.success(quote, quote.source or self.data_source)

    def get_kline(self, symbol: str, period: int = 60) -> ProviderResult[Any]:
        return ProviderResult.failed(
            self.data_source,
            "EastMoney K线未作为主数据源启用",
            metadata={"symbol": symbol, "period": period},
        )

    def get_sector(self, symbol: str) -> ProviderResult[SectorContext]:
        try:
            context = self.sector_source.get_sector_context(symbol)
        except TimeoutError as exc:
            logger.warning("EastMoney sector timeout: symbol=%s error=%s", symbol, exc)
            return ProviderResult.timeout(self.data_source, "行业/概念数据超时")
        except Exception as exc:
            logger.warning("EastMoney sector failed: symbol=%s error=%s", symbol, exc)
            return ProviderResult.failed(
                self.data_source,
                "行业/概念数据不可用",
                error_type=type(exc).__name__,
            )

        if context.available:
            return ProviderResult.success(
                context,
                context.data_source or self.data_source,
                metadata={"sector_status": context.sector_status},
            )
        return ProviderResult.partial(
            context,
            self.data_source,
            context.warning,
            warnings=[context.warning],
            metadata={"sector_status": context.sector_status},
        )

    def get_concepts(self, symbol: str) -> ProviderResult[list[str]]:
        sector = self.get_sector(symbol)
        concepts = sector.data.concepts if sector.data is not None else []
        if concepts:
            return ProviderResult.success(concepts, sector.source)
        return ProviderResult.partial(
            [],
            sector.source or self.data_source,
            "EastMoney概念数据暂不可用",
        )

    def get_fund_flow(self, symbol: str) -> ProviderResult[dict[str, Any]]:
        return ProviderResult.partial(
            {},
            self.data_source,
            "EastMoney资金流接口预留，暂未接入",
            metadata={"symbol": symbol},
        )

    def get_news(self, symbol: str) -> ProviderResult[list[Any]]:
        return ProviderResult.partial(
            [],
            self.data_source,
            "EastMoney新闻接口预留，暂未接入",
            metadata={"symbol": symbol},
        )

    def get_index_quotes(self) -> ProviderResult[list[QuoteSnapshot]]:
        try:
            quotes = self.service.get_index_quotes(market="CN")
        except Exception as exc:
            logger.warning("EastMoney index quote failed: error=%s", exc)
            return ProviderResult.failed(
                self.data_source,
                "指数行情不可用",
                error_type=type(exc).__name__,
            )
        if not quotes:
            return ProviderResult.partial([], self.data_source, "指数行情为空")
        return ProviderResult.success(quotes, self.data_source)

    def health_check(self) -> ProviderResult[dict[str, Any]]:
        quote = self.get_quote("000001")
        if quote.ok and quote.data is not None:
            return ProviderResult.success(
                {"quote": "ok", "symbol": quote.data.symbol},
                self.data_source,
            )
        return ProviderResult.failed(
            self.data_source,
            quote.message or "EastMoney健康检查失败",
            error_type=quote.error_type,
        )
