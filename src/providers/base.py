"""Provider contracts shared by market data consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Optional, Protocol, TypeVar

from src.market.akshare_source import HistoryBar
from src.market.service import QuoteSnapshot

if TYPE_CHECKING:
    from src.rating.sector_provider import SectorContext

T = TypeVar("T")


class ProviderStatus(str, Enum):
    """Common provider result states."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DEGRADED = "degraded"
    CACHE = "cache"


@dataclass(frozen=True)
class ProviderConfig:
    """Runtime options every market data provider must expose."""

    data_source: str
    timeout: float = 8.0
    cache_enabled: bool = True
    fallback_priority: int = 100
    cache_ttl_seconds: float = 60.0


@dataclass(frozen=True)
class ProviderResult(Generic[T]):
    """Result wrapper for all data providers."""

    status: ProviderStatus
    source: str
    data: Optional[T] = None
    message: str = ""
    error_type: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {
            ProviderStatus.SUCCESS,
            ProviderStatus.PARTIAL,
            ProviderStatus.CACHE,
        }

    @classmethod
    def success(
        cls,
        data: T,
        source: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ProviderResult[T]":
        return cls(
            status=ProviderStatus.SUCCESS,
            source=source,
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def partial(
        cls,
        data: T,
        source: str,
        message: str,
        *,
        warnings: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ProviderResult[T]":
        return cls(
            status=ProviderStatus.PARTIAL,
            source=source,
            data=data,
            message=message,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        source: str,
        message: str,
        *,
        error_type: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ProviderResult[T]":
        return cls(
            status=ProviderStatus.FAILED,
            source=source,
            message=message,
            error_type=error_type,
            metadata=metadata or {},
        )

    @classmethod
    def timeout(
        cls,
        source: str,
        message: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ProviderResult[T]":
        return cls(
            status=ProviderStatus.TIMEOUT,
            source=source,
            message=message,
            error_type="timeout",
            metadata=metadata or {},
        )


class BaseProvider(Protocol):
    """Standard provider interface used by ProviderManager."""

    config: ProviderConfig
    data_source: str
    fallback_priority: int
    timeout: float
    cache_enabled: bool

    def get_quote(self, symbol: str) -> ProviderResult[Any]:
        ...

    def get_kline(self, symbol: str, period: int = 60) -> ProviderResult[Any]:
        ...

    def get_sector(self, symbol: str) -> ProviderResult[Any]:
        ...

    def get_concepts(self, symbol: str) -> ProviderResult[list[str]]:
        ...

    def get_fund_flow(self, symbol: str) -> ProviderResult[Any]:
        ...

    def get_news(self, symbol: str) -> ProviderResult[list[Any]]:
        ...

    def health_check(self) -> ProviderResult[dict[str, Any]]:
        ...


class KlineProvider(Protocol):
    """Historical K-line provider."""

    def get_history(
        self,
        symbol: str,
        period: int = 60,
    ) -> ProviderResult[list[HistoryBar]]:
        ...


class RealtimeQuoteProvider(Protocol):
    """Realtime quote provider."""

    def get_quote(
        self,
        symbol: str,
        market: str = "CN",
    ) -> ProviderResult[QuoteSnapshot]:
        ...


class SectorProvider(Protocol):
    """Sector context provider."""

    def get_sector_context(self, symbol: str) -> ProviderResult["SectorContext"]:
        ...
