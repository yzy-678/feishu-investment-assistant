"""MarketDataService diagnostics tests."""

import logging

import httpx
import pytest

from src.market.service import MarketDataError, MarketDataService


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


def test_quote_success_logs_request_response_and_fields(monkeypatch, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "data": {
                    "f57": "300136",
                    "f58": "信维通信",
                    "f43": 4210,
                    "f44": 4300,
                    "f45": 4100,
                    "f46": 4150,
                    "f47": 123456,
                    "f48": 987654321,
                    "f60": 4180,
                    "f168": 321,
                    "f169": 30,
                    "f170": 72,
                    "f171": 478,
                }
            },
        )

    _install_transport(monkeypatch, handler)
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
    assert "source=EastMoney" in logs


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
