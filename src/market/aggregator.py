"""MarketDataAggregator: single market-data entry point for business logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.market.akshare_source import HistoryBar, MACDSnapshot, MASnapshot, StockInfo
from src.market.service import QuoteSnapshot
from src.providers.base import ProviderResult, ProviderStatus
from src.providers.provider_manager import ProviderManager, get_provider_manager
from src.rating.sector_provider import SectorContext
from src.time_utils import shanghai_now


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Unified, non-null market data snapshot for one symbol."""

    symbol: str
    quote: QuoteSnapshot
    kline: list[HistoryBar] = field(default_factory=list)
    sector: SectorContext = field(default_factory=SectorContext)
    concepts: list[str] = field(default_factory=list)
    stock_info: StockInfo = field(default_factory=lambda: StockInfo(symbol=""))
    fund_flow: dict[str, Any] = field(default_factory=dict)
    news: list[Any] = field(default_factory=list)
    index_change_pct: float = 0.0
    source_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def quote_available(self) -> bool:
        return self.source_map.get("quote", {}).get("status") in {
            ProviderStatus.SUCCESS.value,
            ProviderStatus.PARTIAL.value,
            ProviderStatus.CACHE.value,
        }

    @property
    def kline_available(self) -> bool:
        return bool(self.kline)

    def as_dict(self) -> dict[str, Any]:
        return {
            "quote": self.quote,
            "kline": self.kline,
            "sector": self.sector,
            "concepts": self.concepts,
            "fund_flow": self.fund_flow,
            "news": self.news,
            "source_map": self.source_map,
        }


class MarketDataAggregator:
    """Aggregate quote, kline, sector, fund flow and news through providers."""

    def __init__(self, provider_manager: Optional[ProviderManager] = None) -> None:
        self.provider_manager = provider_manager or get_provider_manager()

    def get_snapshot(self, symbol: str, period: int = 60) -> MarketDataSnapshot:
        normalized_symbol = str(symbol or "").strip().upper()
        quote_result = self.provider_manager.get_quote(normalized_symbol)
        kline_result = self.provider_manager.get_kline(normalized_symbol, period=period)
        sector_result = self.provider_manager.get_sector(normalized_symbol)
        concepts_result = self.provider_manager.get_concepts(normalized_symbol)
        fund_flow_result = self.provider_manager.get_fund_flow(normalized_symbol)
        news_result = self.provider_manager.get_news(normalized_symbol)
        index_result = self.provider_manager.get_index_quotes()

        sector = sector_result.data or SectorContext()
        concepts = list(concepts_result.data or sector.concepts or [])
        quote = quote_result.data or _empty_quote(normalized_symbol)
        kline = list(kline_result.data or [])
        fund_flow = dict(fund_flow_result.data or {})
        news = list(news_result.data or [])
        index_change_pct = _index_change_pct(index_result.data or [])
        stock_info = StockInfo(
            symbol=normalized_symbol,
            name=quote.name or sector.name or normalized_symbol,
            industry=sector.industry,
            concepts=concepts,
        )

        results = {
            "quote": quote_result,
            "kline": kline_result,
            "sector": sector_result,
            "concepts": concepts_result,
            "fund_flow": fund_flow_result,
            "news": news_result,
            "index": index_result,
        }
        return MarketDataSnapshot(
            symbol=normalized_symbol,
            quote=quote,
            kline=kline,
            sector=sector,
            concepts=concepts,
            stock_info=stock_info,
            fund_flow=fund_flow,
            news=news,
            index_change_pct=index_change_pct,
            source_map={
                key: _source_entry(result)
                for key, result in results.items()
            },
            warnings=[
                result.message
                for result in results.values()
                if result.message and not result.ok
            ],
        )


def _empty_quote(symbol: str) -> QuoteSnapshot:
    now = shanghai_now().strftime("%Y-%m-%d %H:%M:%S")
    return QuoteSnapshot(
        symbol=symbol,
        name=symbol,
        price=0.0,
        change=0.0,
        change_pct=0.0,
        open_price=0.0,
        high_price=0.0,
        low_price=0.0,
        prev_close=0.0,
        volume=0.0,
        amount=0.0,
        amplitude_pct=0.0,
        turnover_rate=0.0,
        fetched_at=now,
        source="unavailable",
        timestamp="",
        failure_reason="provider_unavailable",
    )


def _source_entry(result: ProviderResult[Any]) -> dict[str, Any]:
    return {
        "source": result.source,
        "status": result.status.value,
        "message": result.message,
        "error_type": result.error_type,
        "cache_hit": bool(result.metadata.get("cache_hit")),
        "fallback": bool(result.metadata.get("fallback")),
    }


def _index_change_pct(quotes: list[QuoteSnapshot]) -> float:
    if not quotes:
        return 0.0
    return sum(quote.change_pct for quote in quotes) / len(quotes)


_aggregator: Optional[MarketDataAggregator] = None


def get_market_data_aggregator() -> MarketDataAggregator:
    """Return the process-wide MarketDataAggregator."""
    global _aggregator  # noqa: PLW0603
    if _aggregator is None:
        _aggregator = MarketDataAggregator()
    return _aggregator
