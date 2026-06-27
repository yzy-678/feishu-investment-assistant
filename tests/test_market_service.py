"""MarketDataService diagnostics tests."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pytest

from src.market.service import (
    MAX_QUOTE_AGE_SECONDS,
    MarketDataError,
    MarketDataService,
)
from src.time_utils import SHANGHAI_TZ


def _install_transport(monkeypatch, handler) -> None:
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client_factory(**kwargs):
        return real_client(
            base_url=kwargs.get("base_url", ""),
            timeout=kwargs.get("timeout"),
            headers=kwargs.get("headers"),
            follow_redirects=kwargs.get("follow_redirects", False),
            transport=transport,
        )

    monkeypatch.setattr("src.market.service.httpx.Client", client_factory)
    monkeypatch.setattr("src.market.service.time.sleep", lambda _: None)


def _quote_data(
    quote_time: Optional[datetime],
    sessions: str = "",
) -> dict:
    return {
        "f57": "300136",
        "f58": "信维通信",
        "f43": 4210,
        "f44": 4300,
        "f45": 4100,
        "f46": 4150,
        "f47": 123456,
        "f48": 987654321,
        "f60": 4180,
        "f80": sessions,
        "f86": int(quote_time.timestamp()) if quote_time else 0,
        "f168": 321,
        "f169": 30,
        "f170": 72,
        "f171": 478,
    }


def test_quote_success_logs_request_response_and_fields(monkeypatch, caplog):
    now = datetime(2026, 6, 24, 10, 0, tzinfo=SHANGHAI_TZ)
    quote_time = now - timedelta(seconds=60)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"data": _quote_data(
                quote_time,
                '[{"b":202606240930,"e":202606241130}]',
            )},
        )

    _install_transport(monkeypatch, handler)
    monkeypatch.setattr("src.market.service.shanghai_now", lambda: now)
    caplog.set_level(logging.INFO, logger="src.market.service")

    quote = MarketDataService().get_quote("300136")

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "symbol=300136" in logs
    assert "url=https://push2delay.eastmoney.com/api/qt/stock/get?" in logs
    assert "status=200" in logs
    assert "elapsed_ms=" in logs
    assert "price=42.1" in logs
    assert "change_pct=0.72" in logs
    assert f"timestamp={quote.timestamp}" in logs
    assert "fetched_at=2026-06-24 10:00:00" in logs
    assert "data_age_seconds=60" in logs
    assert "source=EastMoney" in logs
    assert quote.timestamp == "2026-06-24 09:59:00"
    assert quote.fetched_at == "2026-06-24 10:00:00"
    assert quote.data_age_seconds == 60
    assert quote.failure_reason == ""


def test_quote_timeout_is_classified(monkeypatch, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("request timed out", request=request)

    _install_transport(monkeypatch, handler)
    caplog.set_level(logging.INFO, logger="src.market.service")

    with pytest.raises(MarketDataError) as exc_info:
        MarketDataService().get_quote("300136")

    assert exc_info.value.reason == "timeout"
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "reason=timeout" in logs
    assert "elapsed_ms=" in logs


def test_quote_symbol_not_found_is_classified(monkeypatch, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json={"data": None})

    _install_transport(monkeypatch, handler)
    caplog.set_level(logging.INFO, logger="src.market.service")

    with pytest.raises(MarketDataError) as exc_info:
        MarketDataService().get_quote("300136")

    assert exc_info.value.reason == "symbol_not_found"
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "reason=symbol_not_found" in logs


def test_quote_parse_error_is_classified(monkeypatch, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "data": {
                    "f57": "300136",
                    "f58": "信维通信",
                    "f43": "not-a-number",
                }
            },
        )

    _install_transport(monkeypatch, handler)
    caplog.set_level(logging.INFO, logger="src.market.service")

    with pytest.raises(MarketDataError) as exc_info:
        MarketDataService().get_quote("300136")

    assert exc_info.value.reason == "parse_error"
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "reason=parse_error" in logs


def test_fresh_quote_during_trading_session_is_valid(monkeypatch):
    now = datetime(2026, 6, 24, 10, 0, tzinfo=SHANGHAI_TZ)
    data = _quote_data(
        now - timedelta(seconds=MAX_QUOTE_AGE_SECONDS),
        '[{"b":202606240930,"e":202606241130}]',
    )
    monkeypatch.setattr("src.market.service.shanghai_now", lambda: now)

    quote = MarketDataService._parse_quote(data)

    assert quote.is_trading_session is True
    assert quote.data_age_seconds == MAX_QUOTE_AGE_SECONDS
    assert quote.failure_reason == ""


def test_stale_quote_during_trading_session_is_invalid(monkeypatch):
    now = datetime(2026, 6, 24, 10, 0, tzinfo=SHANGHAI_TZ)
    data = _quote_data(
        now - timedelta(seconds=MAX_QUOTE_AGE_SECONDS + 1),
        '[{"b":202606240930,"e":202606241130}]',
    )
    monkeypatch.setattr("src.market.service.shanghai_now", lambda: now)

    quote = MarketDataService._parse_quote(data)

    assert quote.is_trading_session is True
    assert quote.failure_reason == "stale_quote"


def test_old_quote_outside_trading_session_is_not_stale(monkeypatch):
    now = datetime(2026, 6, 24, 20, 0, tzinfo=SHANGHAI_TZ)
    data = _quote_data(
        now - timedelta(hours=5),
        (
            '[{"b":202606240930,"e":202606241130},'
            '{"b":202606241300,"e":202606241500}]'
        ),
    )
    monkeypatch.setattr("src.market.service.shanghai_now", lambda: now)

    quote = MarketDataService._parse_quote(data)

    assert quote.is_trading_session is False
    assert quote.data_age_seconds == 18000
    assert quote.failure_reason == ""


def test_market_holiday_schedule_is_not_treated_as_live_session(monkeypatch):
    now = datetime(2026, 6, 25, 10, 0, tzinfo=SHANGHAI_TZ)
    data = _quote_data(
        now - timedelta(hours=19),
        (
            '[{"b":202606240930,"e":202606241130},'
            '{"b":202606241300,"e":202606241500}]'
        ),
    )
    monkeypatch.setattr("src.market.service.shanghai_now", lambda: now)

    quote = MarketDataService._parse_quote(data)

    assert quote.is_trading_session is False
    assert quote.failure_reason == ""


def test_missing_provider_timestamp_is_not_replaced_by_fetched_at(monkeypatch):
    now = datetime(2026, 6, 24, 10, 0, tzinfo=SHANGHAI_TZ)
    data = _quote_data(
        None,
        '[{"b":202606240930,"e":202606241130}]',
    )
    monkeypatch.setattr("src.market.service.shanghai_now", lambda: now)

    quote = MarketDataService._parse_quote(data)

    assert quote.timestamp == ""
    assert quote.fetched_at == "2026-06-24 10:00:00"
    assert quote.data_age_seconds is None
