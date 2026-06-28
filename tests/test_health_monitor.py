"""Provider health monitor tests."""

from types import SimpleNamespace

from src.market.akshare_source import HistoryBar
from src.market.service import QuoteSnapshot
from src.providers.cache import CacheManager
from src.providers.health import (
    AkShareHistoryHealthCheck,
    CacheHealthCheck,
    EastMoneyQuoteHealthCheck,
    HealthCheckResult,
    HealthMonitor,
    HealthStatus,
    overall_status,
    placeholder_health_check,
)


def make_bar():
    return HistoryBar(
        date="2026-06-29",
        open=10,
        high=11,
        low=9,
        close=10.5,
        volume=1000,
        amount=100000,
    )


def make_quote(failure_reason=""):
    return QuoteSnapshot(
        symbol="000001",
        name="平安银行",
        price=10.2,
        change=0.1,
        change_pct=1.0,
        open_price=10,
        high_price=10.5,
        low_price=9.9,
        prev_close=10.1,
        volume=1000,
        amount=100000,
        amplitude_pct=2,
        turnover_rate=1,
        fetched_at="2026-06-29 08:30:00",
        timestamp="2026-06-29 08:30:00",
        failure_reason=failure_reason,
    )


def test_overall_status_aggregates_results():
    healthy = HealthCheckResult(
        name="a",
        status=HealthStatus.HEALTHY,
        message="ok",
        latency_ms=1,
        checked_at="2026-06-29 08:30:00",
    )
    degraded = HealthCheckResult(
        name="b",
        status=HealthStatus.DEGRADED,
        message="slow",
        latency_ms=1,
        checked_at="2026-06-29 08:30:00",
    )
    unhealthy = HealthCheckResult(
        name="c",
        status=HealthStatus.UNHEALTHY,
        message="down",
        latency_ms=1,
        checked_at="2026-06-29 08:30:00",
    )

    assert overall_status([healthy]) == HealthStatus.HEALTHY
    assert overall_status([healthy, degraded]) == HealthStatus.DEGRADED
    assert overall_status([healthy, unhealthy]) == HealthStatus.UNHEALTHY
    assert overall_status([]) == HealthStatus.UNKNOWN


def test_akshare_history_health_check_reports_healthy_and_empty():
    healthy = AkShareHistoryHealthCheck(
        source=SimpleNamespace(get_history=lambda symbol, period=1: [make_bar()])
    )()
    empty = AkShareHistoryHealthCheck(
        source=SimpleNamespace(get_history=lambda symbol, period=1: [])
    )()

    assert healthy.status == HealthStatus.HEALTHY
    assert healthy.metadata["rows"] == 1
    assert empty.status == HealthStatus.DEGRADED


def test_eastmoney_quote_health_check_reports_degraded_when_stale():
    service = SimpleNamespace(
        get_quote=lambda symbol, market="CN": make_quote(failure_reason="stale_quote")
    )

    result = EastMoneyQuoteHealthCheck(service=service)()

    assert result.status == HealthStatus.DEGRADED
    assert result.metadata["failure_reason"] == "stale_quote"


def test_cache_health_check_validates_read_write():
    result = CacheHealthCheck(cache=CacheManager())()

    assert result.status == HealthStatus.HEALTHY
    assert result.message == "CacheManager可用"


def test_health_monitor_catches_crashed_checks_and_summarizes():
    def crash():
        raise RuntimeError("boom")

    monitor = HealthMonitor(checks=[crash])

    summary = monitor.summarize()
    results = summary["checks"]

    assert summary["status"] == HealthStatus.DEGRADED.value
    assert results[0].status == HealthStatus.UNKNOWN
    assert results[0].message == "健康检查执行异常"


def test_placeholder_health_check_is_non_invasive():
    result = placeholder_health_check(
        "deepseek",
        "DeepSeek健康检查预留，默认不产生AI调用。",
    )()

    assert result.status == HealthStatus.UNKNOWN
    assert result.metadata["placeholder"] is True
