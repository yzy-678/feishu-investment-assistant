"""Provider adapters used by the Investment Rating Engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from src.market.akshare_source import HistoryBar
from src.market.service import DailyBar
from src.providers.base import KlineProvider, ProviderResult

if TYPE_CHECKING:
    from src.rating.sector_provider import SectorContext

logger = logging.getLogger(__name__)


class MarketDataHistorySource(Protocol):
    def get_history(self, symbol: str, period: int = 60) -> list[HistoryBar]:
        ...


class EastMoneyHistorySource(Protocol):
    def get_recent_bars(
        self,
        symbol: str,
        market: str = "CN",
        limit: int = 60,
    ) -> list[DailyBar]:
        ...


class SectorContextSource(Protocol):
    def get_sector_context(self, symbol: str) -> "SectorContext":
        ...


class MarketDataKlineProvider:
    """Wrap the existing market data service as a K-line provider."""

    def __init__(self, source: MarketDataHistorySource, source_name: str = "AkShare"):
        self.source = source
        self.source_name = source_name

    def get_history(
        self,
        symbol: str,
        period: int = 60,
    ) -> ProviderResult[list[HistoryBar]]:
        try:
            history = self.source.get_history(symbol, period=period)
        except TimeoutError as exc:
            logger.warning(
                "Kline provider timeout: source=%s symbol=%s error=%s",
                self.source_name,
                symbol,
                exc,
            )
            return ProviderResult.timeout(
                self.source_name,
                "历史K线不可用",
                metadata={"symbol": symbol, "period": period},
            )
        except Exception as exc:
            logger.warning(
                "Kline provider failed: source=%s symbol=%s error=%s",
                self.source_name,
                symbol,
                exc,
            )
            return ProviderResult.failed(
                self.source_name,
                "历史K线不可用",
                error_type=type(exc).__name__,
                metadata={"symbol": symbol, "period": period},
            )

        if not history:
            return ProviderResult.partial(
                [],
                self.source_name,
                "历史K线为空",
                metadata={"symbol": symbol, "period": period},
            )
        return ProviderResult.success(
            history,
            self.source_name,
            metadata={"symbol": symbol, "period": period, "rows": len(history)},
        )


class EastMoneyKlineProvider:
    """Read historical K-lines from EastMoney recent bars."""

    def __init__(
        self,
        source: EastMoneyHistorySource,
        source_name: str = "EastMoney",
    ) -> None:
        self.source = source
        self.source_name = source_name

    def get_history(
        self,
        symbol: str,
        period: int = 60,
    ) -> ProviderResult[list[HistoryBar]]:
        try:
            bars = self.source.get_recent_bars(symbol, market="CN", limit=period)
        except TimeoutError as exc:
            logger.warning(
                "EastMoney kline provider timeout: symbol=%s error=%s",
                symbol,
                exc,
            )
            return ProviderResult.timeout(
                self.source_name,
                "EastMoney历史K线不可用",
                metadata={"symbol": symbol, "period": period},
            )
        except Exception as exc:
            logger.warning(
                "EastMoney kline provider failed: symbol=%s error=%s",
                symbol,
                exc,
            )
            return ProviderResult.failed(
                self.source_name,
                "EastMoney历史K线不可用",
                error_type=type(exc).__name__,
                metadata={"symbol": symbol, "period": period},
            )

        history = [_daily_bar_to_history_bar(bar) for bar in bars]
        if not history:
            return ProviderResult.partial(
                [],
                self.source_name,
                "EastMoney历史K线为空",
                metadata={"symbol": symbol, "period": period},
            )
        return ProviderResult.success(
            history,
            self.source_name,
            metadata={"symbol": symbol, "period": period, "rows": len(history)},
        )


class FallbackKlineProvider:
    """Try K-line providers in order until one returns usable data."""

    def __init__(self, providers: list[KlineProvider]) -> None:
        self.providers = providers

    def get_history(
        self,
        symbol: str,
        period: int = 60,
    ) -> ProviderResult[list[HistoryBar]]:
        failures: list[ProviderResult[list[HistoryBar]]] = []
        for provider in self.providers:
            result = provider.get_history(symbol, period=period)
            if result.ok and result.data:
                if failures:
                    return ProviderResult.success(
                        result.data,
                        result.source,
                        metadata={
                            **result.metadata,
                            "fallback": True,
                            "fallback_from": [item.source for item in failures],
                        },
                    )
                return result
            failures.append(result)

        if not failures:
            return ProviderResult.failed("FallbackKlineProvider", "历史K线不可用")

        last = failures[-1]
        return ProviderResult.failed(
            last.source,
            last.message or "历史K线不可用",
            error_type=last.error_type,
            metadata={
                **last.metadata,
                "fallback": True,
                "fallback_attempts": [
                    {
                        "source": item.source,
                        "status": item.status.value,
                        "message": item.message,
                    }
                    for item in failures
                ],
            },
        )


class RatingSectorProvider:
    """Wrap the current sector provider as a unified provider."""

    def __init__(self, source: SectorContextSource, source_name: str = "SectorProvider"):
        self.source = source
        self.source_name = source_name

    def get_sector_context(self, symbol: str) -> ProviderResult["SectorContext"]:
        try:
            context = self.source.get_sector_context(symbol)
        except TimeoutError as exc:
            logger.warning(
                "Sector provider timeout: source=%s symbol=%s error=%s",
                self.source_name,
                symbol,
                exc,
            )
            return ProviderResult.timeout(
                self.source_name,
                "行业/概念数据暂不可用",
                metadata={"symbol": symbol},
            )
        except Exception as exc:
            logger.warning(
                "Sector provider failed: source=%s symbol=%s error=%s",
                self.source_name,
                symbol,
                exc,
            )
            return ProviderResult.failed(
                self.source_name,
                "行业/概念数据暂不可用",
                error_type=type(exc).__name__,
                metadata={"symbol": symbol},
            )

        if not context.available:
            return ProviderResult.partial(
                context,
                context.data_source or self.source_name,
                context.warning,
                warnings=[context.warning] if context.warning else [],
                metadata={"symbol": symbol, "sector_status": context.sector_status},
            )
        return ProviderResult.success(
            context,
            context.data_source or self.source_name,
            metadata={"symbol": symbol, "sector_status": context.sector_status},
        )


def _daily_bar_to_history_bar(bar: DailyBar) -> HistoryBar:
    return HistoryBar(
        date=bar.trade_date,
        open=bar.open_price,
        high=bar.high_price,
        low=bar.low_price,
        close=bar.close_price,
        volume=bar.volume,
        amount=bar.amount,
    )
