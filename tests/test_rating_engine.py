"""Investment Rating Engine tests."""

from datetime import date

from src.db import get_database, init_database
from src.market.akshare_source import HistoryBar, StockInfo
from src.market.service import QuoteSnapshot
from src.providers import ProviderResult, ProviderStatus
from src.rating.rating_engine import (
    InvestmentRatingEngine,
    RATING_DATA_SCOPE_WARNING,
    rating_level,
)
from src.rating.rating_models import RatingInputData, RatingLevel, ScoreBreakdown
from src.rating.rating_rules import (
    calculate_breakout_score,
    calculate_sector_score,
    calculate_strength_score,
    calculate_trend_score,
    calculate_volume_score,
)
from src.rating.score_calculator import InvestmentScoreCalculator
from src.rating.sector_provider import (
    EastMoneyRawSectorSource,
    SectorContext,
    SectorProvider,
    to_eastmoney_security_code,
)


class FakeMarketData:
    def __init__(
        self,
        quote=None,
        history=None,
        stock_info=None,
        index_quotes=None,
    ):
        self.quote = quote
        self.history = history or []
        self.stock_info = stock_info
        self.index_quotes = index_quotes or []

    def get_quote(self, symbol, market="CN"):
        if isinstance(self.quote, Exception):
            raise self.quote
        return self.quote

    def get_history(self, symbol, period=60):
        return self.history[-period:]

    def get_stock_info(self, symbol):
        if self.stock_info is None:
            raise RuntimeError("stock info missing")
        return self.stock_info

    def get_index_quotes(self, market="CN"):
        return self.index_quotes


class FakeStockInfoProvider:
    def __init__(self, stock_info=None, error=None):
        self.stock_info = stock_info
        self.error = error
        self.calls = 0

    def get_stock_info(self, symbol):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.stock_info


class FakeHTTPResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHTTPClient:
    def __init__(self, stock_payload=None, hot_payload=None, error=None):
        self.stock_payload = stock_payload or {"data": {}}
        self.hot_payload = hot_payload or {"data": []}
        self.error = error
        self.get_calls = []
        self.post_calls = []

    def get(self, url, params=None):
        self.get_calls.append((url, params or {}))
        if self.error is not None:
            raise self.error
        return FakeHTTPResponse(self.stock_payload)

    def post(self, url, json=None):
        self.post_calls.append((url, json or {}))
        if self.error is not None:
            raise self.error
        return FakeHTTPResponse(self.hot_payload)


class FakeSectorContextProvider:
    def __init__(self, context=None, error=None):
        self.context = context or SectorContext()
        self.error = error

    def get_sector_context(self, symbol):
        if self.error is not None:
            raise self.error
        return self.context


def make_bar(
    index,
    close,
    volume=1000,
    amount=100000,
    open_price=None,
    high=None,
    low=None,
):
    open_value = close - 0.2 if open_price is None else open_price
    return HistoryBar(
        date=f"2026-06-{index:02d}",
        open=open_value,
        high=high if high is not None else close + 0.2,
        low=low if low is not None else close - 0.4,
        close=close,
        volume=volume,
        amount=amount,
    )


def make_quote(symbol="300001", name="测试科技", change_pct=5.0):
    return QuoteSnapshot(
        symbol=symbol,
        name=name,
        price=12.8,
        change=0.6,
        change_pct=change_pct,
        open_price=12.0,
        high_price=13.0,
        low_price=11.8,
        prev_close=12.2,
        volume=2000,
        amount=1_500_000_000,
        amplitude_pct=5.0,
        turnover_rate=5.0,
        fetched_at="2026-06-28 08:30:00",
        source="EastMoney",
    )


def trend_history(count=25):
    return [
        make_bar(i + 1, 10 + i * 0.2, volume=1000 + i * 10, amount=100000 + i * 1000)
        for i in range(count)
    ]


def breakout_history():
    history = [
        make_bar(i + 1, 10.0 + (i % 3) * 0.05, open_price=10.1, high=10.3, low=9.8)
        for i in range(19)
    ]
    history.append(make_bar(20, 9.9, open_price=10.25, high=10.3, low=9.8, volume=1000))
    history.append(make_bar(21, 10.6, open_price=10.0, high=10.7, low=9.9, volume=1400))
    return history


def test_rating_level_thresholds():
    assert rating_level(95) == RatingLevel.S
    assert rating_level(90) == RatingLevel.A_PLUS
    assert rating_level(80) == RatingLevel.A
    assert rating_level(70) == RatingLevel.B_PLUS
    assert rating_level(60) == RatingLevel.B
    assert rating_level(50) == RatingLevel.C
    assert rating_level(49.99) == RatingLevel.D


def test_trend_score_is_rule_based_and_explainable():
    score, evidence, warnings = calculate_trend_score(
        RatingInputData(symbol="300001", history=trend_history())
    )

    assert score == 20
    assert warnings == []
    assert any("MA5 > MA10 > MA20" in item for item in evidence)
    assert any("站上 MA20" in item for item in evidence)


def test_volume_score_is_rule_based_and_explainable():
    history = [
        make_bar(1, 10, volume=1000, amount=100000),
        make_bar(2, 10.1, volume=1000, amount=100000),
        make_bar(3, 10.2, volume=1000, amount=100000),
        make_bar(4, 10.3, volume=1000, amount=100000),
        make_bar(5, 10.4, volume=1000, amount=100000),
        make_bar(6, 11.0, volume=1600, amount=170000),
    ]

    score, evidence, warnings = calculate_volume_score(
        RatingInputData(symbol="300001", history=history)
    )

    assert score == 20
    assert warnings == []
    assert any("成交量" in item for item in evidence)
    assert any("价升量增" in item for item in evidence)


def test_sector_score_uses_supplied_market_context_without_ai():
    score, evidence, warnings = calculate_sector_score(
        RatingInputData(
            symbol="300001",
            quote=make_quote(),
            sector_heat_score=100,
            sector_continuity_score=100,
            is_main_sector=True,
            sector_linkage_score=100,
            sector_available=True,
        )
    )

    assert score == 20
    assert warnings == []
    assert any("市场主线" in item for item in evidence)


def test_sector_score_is_none_when_sector_context_missing():
    score, evidence, warnings = calculate_sector_score(
        RatingInputData(symbol="300001", quote=make_quote())
    )

    assert score is None
    assert evidence == []
    assert any("板块评分暂未纳入" in item for item in warnings)


def test_dynamic_weight_excludes_missing_sector_score():
    breakdown = ScoreBreakdown(
        trend_score=20,
        volume_score=20,
        sector_score=None,
        breakout_score=20,
        strength_score=20,
    )

    assert breakdown.total_score == 100


def test_eastmoney_raw_sector_source_parses_f127_industry():
    client = FakeHTTPClient(
        stock_payload={
            "data": {
                "f58": "信维通信",
                "f127": "消费电子",
                "f128": "广东板块",
            }
        },
        hot_payload={"data": []},
    )

    context = EastMoneyRawSectorSource(client=client).get_sector_context("300136")

    assert context.name == "信维通信"
    assert context.industry == "消费电子"
    assert context.region_sector == "广东板块"
    assert context.available is True


def test_eastmoney_raw_sector_source_does_not_treat_f128_as_concept():
    client = FakeHTTPClient(
        stock_payload={
            "data": {
                "f58": "信维通信",
                "f127": "消费电子",
                "f128": "广东板块",
            }
        },
        hot_payload={"data": []},
    )

    context = EastMoneyRawSectorSource(client=client).get_sector_context("300136")

    assert context.region_sector == "广东板块"
    assert context.concepts == []


def test_bare_symbol_converts_to_eastmoney_prefixed_code():
    assert to_eastmoney_security_code("000001") == "SZ000001"
    assert to_eastmoney_security_code("300136") == "SZ300136"
    assert to_eastmoney_security_code("002594") == "SZ002594"
    assert to_eastmoney_security_code("003816") == "SZ003816"
    assert to_eastmoney_security_code("600519") == "SH600519"
    assert to_eastmoney_security_code("601318") == "SH601318"
    assert to_eastmoney_security_code("603777") == "SH603777"
    assert to_eastmoney_security_code("688981") == "SH688981"


def test_eastmoney_raw_sector_source_uses_prefixed_code_for_concepts():
    client = FakeHTTPClient(
        stock_payload={"data": {}},
        hot_payload={
            "data": [
                {"conceptName": "商业航天"},
                {"conceptName": "华为概念"},
            ]
        },
    )

    context = EastMoneyRawSectorSource(client=client).get_sector_context("300136")

    assert context.industry == ""
    assert context.concepts == ["商业航天", "华为概念"]
    assert client.post_calls[0][1]["srcSecurityCode"] == "SZ300136"


def test_sector_provider_uses_fallback_after_industry_fetch_failure():
    akshare = FakeStockInfoProvider(
        stock_info=StockInfo(
            symbol="300001",
            name="测试科技",
            industry="",
            concepts=["AI硬件"],
        )
    )
    eastmoney = FakeStockInfoProvider(
        stock_info=StockInfo(
            symbol="300001",
            name="测试科技",
            industry="半导体",
            concepts=["AI硬件"],
        )
    )

    context = SectorProvider(
        akshare_provider=akshare,
        eastmoney_provider=eastmoney,
    ).get_sector_context("300001")

    assert context.available is True
    assert context.industry == "半导体"
    assert context.concepts == ["AI硬件"]
    assert "EastMoney" in context.data_source
    assert akshare.calls == 1
    assert eastmoney.calls == 1


def test_sector_provider_returns_unavailable_when_concepts_fetch_failure():
    akshare = FakeStockInfoProvider(
        stock_info=StockInfo(
            symbol="300001",
            name="测试科技",
            industry="半导体",
            concepts=[],
        )
    )

    context = SectorProvider(akshare_provider=akshare).get_sector_context("300001")

    assert context.available is True
    assert context.industry == "半导体"
    assert context.concepts == []
    assert context.sector_status == "部分纳入"
    assert "板块评分部分纳入" in context.warning


def test_sector_provider_available_when_only_concepts_exist():
    akshare = FakeStockInfoProvider(
        stock_info=StockInfo(
            symbol="300001",
            name="测试科技",
            industry="",
            concepts=["AI硬件"],
        )
    )

    context = SectorProvider(akshare_provider=akshare).get_sector_context("300001")

    assert context.available is True
    assert context.industry == ""
    assert context.concepts == ["AI硬件"]
    assert context.sector_status == "部分纳入"


def test_sector_provider_unavailable_only_when_industry_and_concepts_missing():
    akshare = FakeStockInfoProvider(
        stock_info=StockInfo(
            symbol="300001",
            name="测试科技",
            industry="",
            concepts=[],
        )
    )

    context = SectorProvider(akshare_provider=akshare).get_sector_context("300001")

    assert context.available is False
    assert context.sector_status == "暂未纳入"
    assert "板块评分暂未纳入" in context.warning


def test_sector_score_supports_partial_weight():
    score, evidence, warnings = calculate_sector_score(
        RatingInputData(
            symbol="300001",
            industry_available=True,
            concepts_available=False,
            industry_score=10,
            concept_score=None,
            sector_available=True,
        )
    )

    assert score == 10
    assert any("行业数据可用" in item for item in evidence)
    assert any("概念数据暂不可用" in item for item in warnings)


def test_sector_http_error_is_logged_but_not_shown_to_user():
    market_data = FakeMarketData(
        quote=make_quote(),
        history=trend_history(25),
        stock_info=StockInfo(
            symbol="300001",
            name="测试科技",
            industry="半导体",
            concepts=[],
        ),
        index_quotes=[],
    )

    rating = InvestmentRatingEngine(
        market_data=market_data,
        sector_provider=FakeSectorContextProvider(
            error=RuntimeError("HTTP 500 upstream detail")
        ),
        persist_history=False,
    ).evaluate("300001")

    assert "板块评分暂未纳入" in rating.warning
    assert "HTTP 500" not in rating.warning
    assert "upstream detail" not in rating.warning


def test_breakout_score_is_rule_based_and_explainable():
    score, evidence, warnings = calculate_breakout_score(
        RatingInputData(symbol="300001", history=breakout_history())
    )

    assert score == 20
    assert warnings == []
    assert any("20日新高" in item for item in evidence)
    assert any("平台" in item for item in evidence)
    assert any("反包" in item for item in evidence)


def test_strength_score_uses_quote_index_and_industry_data():
    score, evidence, warnings = calculate_strength_score(
        RatingInputData(
            symbol="300001",
            quote=make_quote(change_pct=5.0),
            index_change_pct=1.0,
            industry_change_pct=2.0,
        )
    )

    assert score == 20
    assert warnings == []
    assert any("强于指数" in item for item in evidence)
    assert any("强于行业" in item for item in evidence)
    assert any("资金抱团" in item for item in evidence)


def test_score_calculator_preserves_future_extension_slots():
    data = RatingInputData(
        symbol="300001",
        quote=make_quote(),
        history=trend_history(25),
        sector_heat_score=100,
        sector_continuity_score=100,
        is_main_sector=True,
        sector_linkage_score=100,
        sector_available=True,
        industry_change_pct=2.0,
    )

    result = InvestmentScoreCalculator().calculate(data)

    assert result.total_score >= 70
    assert "fundamental" in result.evidence
    assert "news" in result.evidence
    assert "capital" in result.evidence
    assert "risk" in result.evidence


def test_rating_engine_evaluate_returns_unified_investment_rating():
    market_data = FakeMarketData(
        quote=make_quote(),
        history=trend_history(25),
        stock_info=StockInfo(
            symbol="300001",
            name="测试科技",
            industry="半导体",
            concepts=["AI硬件"],
        ),
        index_quotes=[
            make_quote(symbol="000001", name="上证指数", change_pct=1.0),
            make_quote(symbol="399001", name="深证成指", change_pct=1.2),
        ],
    )

    rating = InvestmentRatingEngine(
        market_data=market_data,
        sector_provider=FakeSectorContextProvider(
            context=SectorContext(
                industry="半导体",
                concepts=["AI硬件"],
                data_source="TestSector",
                industry_score=10,
                concept_score=10,
            )
        ),
        persist_history=False,
    ).evaluate("300001")

    assert rating.symbol == "300001"
    assert rating.name == "测试科技"
    assert rating.sector_score == 20
    assert rating.total_score == (
        rating.trend_score
        + rating.volume_score
        + rating.sector_score
        + rating.breakout_score
        + rating.strength_score
    )
    assert rating.rating_level in set(RatingLevel)
    assert rating.previous_score is None
    assert rating.score_change is None
    assert rating.change_direction == "new"
    assert rating.change_reasons == ["首次评级，暂无昨日评分对比。"]
    assert RATING_DATA_SCOPE_WARNING in rating.warning
    assert rating.data_source == "EastMoney, AkShare, TestSector"
    assert "evidence" in rating.reserved
    assert "data_quality" in rating.reserved
    assert rating.data_quality.summary == "数据质量正常"
    assert "future_extensions" in rating.reserved


def test_rating_engine_handles_missing_data_without_inventing_scores():
    market_data = FakeMarketData(
        quote=RuntimeError("quote unavailable"),
        history=[],
        stock_info=None,
        index_quotes=[],
    )

    rating = InvestmentRatingEngine(
        market_data=market_data,
        sector_provider=FakeSectorContextProvider(context=SectorContext()),
        persist_history=False,
    ).evaluate("300001")

    assert rating.total_score == 0
    assert rating.rating_level == RatingLevel.D
    assert rating.name == "300001"
    assert "不可用" in rating.warning or "不足" in rating.warning
    assert rating.data_source == "数据不足"
    assert "实时行情" in rating.data_quality.missing_dimensions
    assert "历史K线" in rating.data_quality.missing_dimensions
    assert "板块评分" in rating.data_quality.missing_dimensions


def test_rating_engine_records_cache_and_fallback_quality():
    market_data = FakeMarketData(
        quote=make_quote(),
        history=[],
        stock_info=StockInfo(symbol="300001", name="测试科技", industry="半导体"),
        index_quotes=[],
    )
    kline_provider = FakeSectorContextProvider()
    kline_provider.get_history = lambda symbol, period=60: ProviderResult(
        status=ProviderStatus.CACHE,
        source="EastMoney",
        data=trend_history(25),
        metadata={
            "cache_hit": True,
            "fallback": True,
            "source_status": "success",
        },
    )

    rating = InvestmentRatingEngine(
        market_data=market_data,
        kline_provider=kline_provider,
        sector_provider=FakeSectorContextProvider(context=SectorContext()),
        persist_history=False,
    ).evaluate("300001")

    assert rating.data_quality.has_cache is True
    assert rating.data_quality.has_fallback is True
    assert "使用缓存" in rating.data_quality.summary
    assert "使用fallback" in rating.data_quality.summary


def test_rating_engine_persists_history_and_reports_score_change(monkeypatch):
    init_database()
    conn = get_database().get_connection()
    conn.execute("DELETE FROM investment_rating_history WHERE symbol = ?", ("300001",))
    conn.commit()

    yesterday_history = trend_history(25)
    today_history = trend_history(25) + [
        make_bar(26, 16.0, volume=3000, amount=400000, high=16.2, low=15.4)
    ]
    stock_info = StockInfo(
        symbol="300001",
        name="测试科技",
        industry="半导体",
        concepts=[],
    )

    monkeypatch.setattr(
        "src.rating.rating_engine.shanghai_today",
        lambda: date(2026, 6, 27),
    )
    first_rating = InvestmentRatingEngine(
        market_data=FakeMarketData(
            quote=make_quote(change_pct=2.0),
            history=yesterday_history,
            stock_info=stock_info,
            index_quotes=[make_quote(symbol="000001", name="上证指数", change_pct=1.0)],
        )
    ).evaluate("300001")

    monkeypatch.setattr(
        "src.rating.rating_engine.shanghai_today",
        lambda: date(2026, 6, 28),
    )
    second_rating = InvestmentRatingEngine(
        market_data=FakeMarketData(
            quote=make_quote(change_pct=5.0),
            history=today_history,
            stock_info=stock_info,
            index_quotes=[make_quote(symbol="000001", name="上证指数", change_pct=1.0)],
        )
    ).evaluate("300001")

    assert second_rating.previous_score == first_rating.total_score
    assert second_rating.score_change == round(
        second_rating.total_score - first_rating.total_score,
        2,
    )
    assert second_rating.score_change > 0
    assert second_rating.change_direction == "⬆"
    assert any("提升" in reason for reason in second_rating.change_reasons)


def test_rating_engine_updates_same_day_without_using_it_as_previous(monkeypatch):
    init_database()
    conn = get_database().get_connection()
    conn.execute("DELETE FROM investment_rating_history WHERE symbol = ?", ("300002",))
    conn.commit()
    stock_info = StockInfo(symbol="300002", name="同日测试", industry="机器人")

    monkeypatch.setattr(
        "src.rating.rating_engine.shanghai_today",
        lambda: date(2026, 6, 28),
    )
    engine = InvestmentRatingEngine(
        market_data=FakeMarketData(
            quote=make_quote(symbol="300002", name="同日测试", change_pct=1.0),
            history=trend_history(25),
            stock_info=stock_info,
            index_quotes=[],
        )
    )

    first = engine.evaluate("300002")
    second = engine.evaluate("300002")

    assert first.previous_score is None
    assert second.previous_score is None
    rows = conn.execute(
        "SELECT COUNT(*) AS cnt FROM investment_rating_history WHERE symbol = ?",
        ("300002",),
    ).fetchone()
    assert rows["cnt"] == 1
