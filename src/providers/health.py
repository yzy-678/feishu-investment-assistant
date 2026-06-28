"""Provider health monitoring."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol

from src.market.akshare_source import AkShareSource
from src.market.service import MarketDataService
from src.providers.cache import CacheManager, get_cache_manager
from src.time_utils import shanghai_now

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health states for providers and external dependencies."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HealthCheckResult:
    """Single health check result."""

    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    checked_at: str
    metadata: dict[str, object] = field(default_factory=dict)


class HealthCheck(Protocol):
    """Callable health check contract."""

    def __call__(self) -> HealthCheckResult:
        ...


class HealthMonitor:
    """Run health checks and summarize dependency status."""

    def __init__(
        self,
        checks: Optional[list[HealthCheck]] = None,
    ) -> None:
        self.checks = checks or default_health_checks()

    def run_all(self) -> list[HealthCheckResult]:
        results: list[HealthCheckResult] = []
        for check in self.checks:
            try:
                results.append(check())
            except Exception as exc:
                logger.warning("Health check crashed: error=%s", exc)
                results.append(
                    HealthCheckResult(
                        name=getattr(check, "__name__", check.__class__.__name__),
                        status=HealthStatus.UNKNOWN,
                        message="健康检查执行异常",
                        latency_ms=0.0,
                        checked_at=_checked_at(),
                        metadata={"error_type": type(exc).__name__},
                    )
                )
        return results

    def summarize(self) -> dict[str, object]:
        results = self.run_all()
        return {
            "status": overall_status(results).value,
            "checked_at": _checked_at(),
            "checks": results,
        }


def overall_status(results: list[HealthCheckResult]) -> HealthStatus:
    """Aggregate dependency health into a single status."""
    if not results:
        return HealthStatus.UNKNOWN
    statuses = {item.status for item in results}
    if HealthStatus.UNHEALTHY in statuses:
        return HealthStatus.UNHEALTHY
    if HealthStatus.DEGRADED in statuses or HealthStatus.UNKNOWN in statuses:
        return HealthStatus.DEGRADED
    return HealthStatus.HEALTHY


def default_health_checks() -> list[HealthCheck]:
    """Return safe, no-side-effect default checks."""
    return [
        AkShareHistoryHealthCheck(),
        EastMoneyQuoteHealthCheck(),
        CacheHealthCheck(),
    ]


class AkShareHistoryHealthCheck:
    """Check AkShare historical K-line availability."""

    def __init__(
        self,
        source: Optional[AkShareSource] = None,
        symbol: str = "000001",
    ) -> None:
        self.source = source or AkShareSource(timeout=3.0)
        self.symbol = symbol

    def __call__(self) -> HealthCheckResult:
        started_at = time.perf_counter()
        try:
            history = self.source.get_history(self.symbol, period=1)
        except TimeoutError as exc:
            return _result(
                "akshare_history",
                HealthStatus.UNHEALTHY,
                "AkShare历史K线超时",
                started_at,
                {"symbol": self.symbol, "error_type": type(exc).__name__},
            )
        except Exception as exc:
            return _result(
                "akshare_history",
                HealthStatus.UNHEALTHY,
                "AkShare历史K线不可用",
                started_at,
                {"symbol": self.symbol, "error_type": type(exc).__name__},
            )

        if not history:
            return _result(
                "akshare_history",
                HealthStatus.DEGRADED,
                "AkShare历史K线为空",
                started_at,
                {"symbol": self.symbol, "rows": 0},
            )
        return _result(
            "akshare_history",
            HealthStatus.HEALTHY,
            "AkShare历史K线可用",
            started_at,
            {"symbol": self.symbol, "rows": len(history)},
        )


class EastMoneyQuoteHealthCheck:
    """Check EastMoney realtime quote availability."""

    def __init__(
        self,
        service: Optional[MarketDataService] = None,
        symbol: str = "000001",
    ) -> None:
        self.service = service or MarketDataService(timeout=3.0)
        self.symbol = symbol

    def __call__(self) -> HealthCheckResult:
        started_at = time.perf_counter()
        try:
            quote = self.service.get_quote(self.symbol, market="CN")
        except TimeoutError as exc:
            return _result(
                "eastmoney_quote",
                HealthStatus.UNHEALTHY,
                "EastMoney实时行情超时",
                started_at,
                {"symbol": self.symbol, "error_type": type(exc).__name__},
            )
        except Exception as exc:
            return _result(
                "eastmoney_quote",
                HealthStatus.UNHEALTHY,
                "EastMoney实时行情不可用",
                started_at,
                {"symbol": self.symbol, "error_type": type(exc).__name__},
            )

        if quote.failure_reason:
            return _result(
                "eastmoney_quote",
                HealthStatus.DEGRADED,
                "EastMoney实时行情可用但数据可能陈旧",
                started_at,
                {
                    "symbol": self.symbol,
                    "failure_reason": quote.failure_reason,
                    "timestamp": quote.timestamp,
                },
            )
        return _result(
            "eastmoney_quote",
            HealthStatus.HEALTHY,
            "EastMoney实时行情可用",
            started_at,
            {"symbol": self.symbol, "timestamp": quote.timestamp},
        )


class CacheHealthCheck:
    """Check in-process CacheManager read/write behavior."""

    def __init__(self, cache: Optional[CacheManager] = None) -> None:
        self.cache = cache or get_cache_manager()

    def __call__(self) -> HealthCheckResult:
        started_at = time.perf_counter()
        key = self.cache.make_key("health", "cache")
        try:
            self.cache.set(key, "ok", ttl_seconds=5)
            value = self.cache.get(key)
            self.cache.invalidate(key)
        except Exception as exc:
            return _result(
                "cache_manager",
                HealthStatus.UNHEALTHY,
                "CacheManager不可用",
                started_at,
                {"error_type": type(exc).__name__},
            )

        if value != "ok":
            return _result(
                "cache_manager",
                HealthStatus.UNHEALTHY,
                "CacheManager读写校验失败",
                started_at,
            )
        return _result(
            "cache_manager",
            HealthStatus.HEALTHY,
            "CacheManager可用",
            started_at,
        )


def placeholder_health_check(name: str, message: str) -> HealthCheck:
    """Create a non-invasive placeholder for future external checks."""

    def check() -> HealthCheckResult:
        return HealthCheckResult(
            name=name,
            status=HealthStatus.UNKNOWN,
            message=message,
            latency_ms=0.0,
            checked_at=_checked_at(),
            metadata={"placeholder": True},
        )

    check.__name__ = name
    return check


def _result(
    name: str,
    status: HealthStatus,
    message: str,
    started_at: float,
    metadata: Optional[dict[str, object]] = None,
) -> HealthCheckResult:
    return HealthCheckResult(
        name=name,
        status=status,
        message=message,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        checked_at=_checked_at(),
        metadata=metadata or {},
    )


def _checked_at() -> str:
    return shanghai_now().strftime("%Y-%m-%d %H:%M:%S")


_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Return the process-wide HealthMonitor."""
    global _health_monitor  # noqa: PLW0603
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
