"""Unified provider interface tests."""

from types import SimpleNamespace

from src.market.akshare_source import HistoryBar
from src.market.service import DailyBar
from src.providers import (
    ProviderResult,
    ProviderStatus,
)
from src.providers.rating_adapters import (
    EastMoneyKlineProvider,
    FallbackKlineProvider,
    MarketDataKlineProvider,
    RatingSectorProvider,
)
from src.rating.rating_engine import InvestmentRatingEngine
from src.rating.sector_provider import SectorContext


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


def make_daily_bar(index=1):
    return DailyBar(
        trade_date=f"2026-06-{index:02d}",
        open_price=10,
        close_price=10.5,
        high_price=11,
        low_price=9,
        volume=1000,
        amount=100000,
        amplitude_pct=2,
        change_pct=1,
        change=0.1,
        turnover_rate=3,
    )


def test_provider_result_success_and_failed_states():
    ok = ProviderResult.success([1, 2, 3], "UnitTest")
    failed = ProviderResult.failed("UnitTest", "failed", error_type="RuntimeError")

    assert ok.ok is True
    assert ok.status == ProviderStatus.SUCCESS
    assert ok.data == [1, 2, 3]
    assert failed.ok is False
    assert failed.status == ProviderStatus.FAILED
    assert failed.error_type == "RuntimeError"


def test_market_data_kline_provider_wraps_success():
    source = SimpleNamespace(
        get_history=lambda symbol, period=60: [make_bar(1), make_bar(2)]
    )

    result = MarketDataKlineProvider(source).get_history("300001", period=2)

    assert result.ok is True
    assert result.status == ProviderStatus.SUCCESS
    assert len(result.data) == 2
    assert result.metadata["rows"] == 2


def test_market_data_kline_provider_wraps_failure_without_raw_exception():
    def boom(symbol, period=60):
        raise RuntimeError("upstream detail")

    result = MarketDataKlineProvider(SimpleNamespace(get_history=boom)).get_history(
        "300001"
    )

    assert result.ok is False
    assert result.status == ProviderStatus.FAILED
    assert result.message == "历史K线不可用"
    assert result.error_type == "RuntimeError"
    assert "upstream detail" not in result.message


def test_eastmoney_kline_provider_converts_recent_bars_to_history():
    source = SimpleNamespace(
        get_recent_bars=lambda symbol, market="CN", limit=60: [
            make_daily_bar(1),
            make_daily_bar(2),
        ]
    )

    result = EastMoneyKlineProvider(source).get_history("300001", period=2)

    assert result.ok is True
    assert result.source == "EastMoney"
    assert len(result.data) == 2
    assert result.data[0] == make_bar(1)


def test_fallback_kline_provider_uses_eastmoney_when_akshare_fails():
    akshare = SimpleNamespace(
        get_history=lambda symbol, period=60: ProviderResult.failed(
            "AkShare",
            "历史K线不可用",
            error_type="AkShareError",
        )
    )
    eastmoney = SimpleNamespace(
        get_history=lambda symbol, period=60: ProviderResult.success(
            [make_bar(1)],
            "EastMoney",
        )
    )

    result = FallbackKlineProvider([akshare, eastmoney]).get_history("300001")

    assert result.ok is True
    assert result.source == "EastMoney"
    assert result.metadata["fallback"] is True
    assert result.metadata["fallback_from"] == ["AkShare"]


def test_rating_sector_provider_wraps_partial_context():
    source = SimpleNamespace(
        get_sector_context=lambda symbol: SectorContext(
            industry="半导体",
            concepts=[],
            data_source="EastMoneyRaw",
            industry_score=10,
            warning="概念数据暂不可用，板块评分部分纳入。",
        )
    )

    result = RatingSectorProvider(source).get_sector_context("300001")

    assert result.ok is True
    assert result.status == ProviderStatus.SUCCESS
    assert result.data.sector_status == "部分纳入"
    assert result.source == "EastMoneyRaw"


def test_rating_engine_uses_unified_kline_provider():
    kline_provider = SimpleNamespace(
        get_history=lambda symbol, period=60: ProviderResult.success(
            [make_bar(1)],
            "TestKline",
        )
    )
    sector_provider = SimpleNamespace(
        get_sector_context=lambda symbol: ProviderResult.success(
            SectorContext(),
            "TestSector",
        )
    )

    rating = InvestmentRatingEngine(
        market_data=SimpleNamespace(
            get_quote=lambda symbol, market="CN": None,
            get_stock_info=lambda symbol: None,
            get_index_quotes=lambda market="CN": [],
        ),
        kline_provider=kline_provider,
        rating_sector_provider=sector_provider,
        persist_history=False,
    ).evaluate("300001")

    assert "历史K线为空" not in rating.warning
    assert "趋势评分数据不足" in rating.warning


def test_rating_engine_default_kline_provider_falls_back_to_eastmoney():
    market_data = SimpleNamespace(
        get_quote=lambda symbol, market="CN": None,
        get_history=lambda symbol, period=60: (_ for _ in ()).throw(
            RuntimeError("akshare down")
        ),
        get_recent_bars=lambda symbol, market="CN", limit=60: [make_daily_bar(1)],
        get_stock_info=lambda symbol: None,
        get_index_quotes=lambda market="CN": [],
    )
    sector_provider = SimpleNamespace(
        get_sector_context=lambda symbol: ProviderResult.success(
            SectorContext(),
            "TestSector",
        )
    )

    rating = InvestmentRatingEngine(
        market_data=market_data,
        rating_sector_provider=sector_provider,
        persist_history=False,
    ).evaluate("300001")

    assert "历史K线不可用" not in rating.warning
    assert "趋势评分数据不足" in rating.warning
