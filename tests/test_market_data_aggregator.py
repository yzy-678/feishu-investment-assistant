"""MarketDataAggregator and ProviderManager tests."""

from src.market.aggregator import MarketDataAggregator, MarketDataSnapshot
from src.market.akshare_source import HistoryBar, StockInfo
from src.market.service import QuoteSnapshot
from src.providers.base import ProviderConfig, ProviderResult
from src.providers.cache import CacheManager
from src.providers.provider_manager import ProviderManager
from src.rating.rating_engine import InvestmentRatingEngine
from src.rating.sector_provider import SectorContext


class FakeProvider:
    def __init__(
        self,
        source,
        priority,
        quote=None,
        kline=None,
        sector=None,
        concepts=None,
        fail_quote=False,
        fail_kline=False,
    ):
        self.config = ProviderConfig(
            data_source=source,
            fallback_priority=priority,
            cache_enabled=False,
        )
        self.quote = quote
        self.kline = kline
        self.sector = sector
        self.concepts = concepts or []
        self.fail_quote = fail_quote
        self.fail_kline = fail_kline

    @property
    def data_source(self):
        return self.config.data_source

    @property
    def fallback_priority(self):
        return self.config.fallback_priority

    @property
    def timeout(self):
        return self.config.timeout

    @property
    def cache_enabled(self):
        return self.config.cache_enabled

    def get_quote(self, symbol):
        if self.fail_quote or self.quote is None:
            return ProviderResult.failed(self.data_source, "quote failed")
        return ProviderResult.success(self.quote, self.data_source)

    def get_kline(self, symbol, period=60):
        if self.fail_kline or self.kline is None:
            return ProviderResult.failed(self.data_source, "kline failed")
        return ProviderResult.success(list(self.kline)[-period:], self.data_source)

    def get_sector(self, symbol):
        if self.sector is None:
            return ProviderResult.partial(SectorContext(), self.data_source, "sector empty")
        return ProviderResult.success(self.sector, self.data_source)

    def get_concepts(self, symbol):
        if self.concepts:
            return ProviderResult.success(list(self.concepts), self.data_source)
        return ProviderResult.partial([], self.data_source, "concepts empty")

    def get_fund_flow(self, symbol):
        return ProviderResult.partial({}, self.data_source, "fund flow reserved")

    def get_news(self, symbol):
        return ProviderResult.partial([], self.data_source, "news reserved")

    def get_index_quotes(self):
        return ProviderResult.success([make_quote("000001", "上证指数", 1.0)], self.data_source)

    def health_check(self):
        return ProviderResult.success({"ok": True}, self.data_source)


def make_quote(symbol="300001", name="测试科技", change_pct=3.0):
    return QuoteSnapshot(
        symbol=symbol,
        name=name,
        price=10.0,
        change=0.3,
        change_pct=change_pct,
        open_price=9.8,
        high_price=10.2,
        low_price=9.7,
        prev_close=9.7,
        volume=1000,
        amount=1000000,
        amplitude_pct=3.0,
        turnover_rate=2.0,
        fetched_at="2026-06-29 10:00:00",
        source="EastMoney",
    )


def make_history(count=25):
    return [
        HistoryBar(
            date=f"2026-06-{index + 1:02d}",
            open=10 + index * 0.1,
            high=10.2 + index * 0.1,
            low=9.8 + index * 0.1,
            close=10 + index * 0.1,
            volume=1000 + index * 10,
            amount=100000 + index * 1000,
        )
        for index in range(count)
    ]


def test_aggregator_mixes_quote_kline_sector_and_source_map():
    eastmoney = FakeProvider(
        "EastMoney",
        10,
        quote=make_quote(),
        sector=SectorContext(
            name="测试科技",
            industry="半导体",
            concepts=["先进封装"],
            data_source="EastMoneyRaw",
        ),
        concepts=["先进封装"],
    )
    akshare = FakeProvider(
        "AkShare",
        20,
        kline=make_history(),
        concepts=["机器人"],
    )
    aggregator = MarketDataAggregator(
        ProviderManager([eastmoney, akshare], cache_manager=CacheManager())
    )

    snapshot = aggregator.get_snapshot("300001")

    assert snapshot.quote.symbol == "300001"
    assert len(snapshot.kline) == 25
    assert snapshot.sector.industry == "半导体"
    assert snapshot.concepts == ["先进封装", "机器人"]
    assert snapshot.source_map["quote"]["source"] == "EastMoney"
    assert snapshot.source_map["kline"]["source"] == "AkShare"
    assert "EastMoneyRaw" in snapshot.source_map["sector"]["source"]
    assert "AkShare" in snapshot.source_map["sector"]["source"]


def test_aggregator_returns_fallback_structures_when_providers_fail():
    provider = FakeProvider(
        "Broken",
        10,
        fail_quote=True,
        fail_kline=True,
    )
    aggregator = MarketDataAggregator(
        ProviderManager([provider], cache_manager=CacheManager())
    )

    snapshot = aggregator.get_snapshot("300001")

    assert snapshot.quote is not None
    assert snapshot.quote.source == "unavailable"
    assert snapshot.kline == []
    assert snapshot.sector is not None
    assert snapshot.concepts == []
    assert snapshot.fund_flow == {}
    assert snapshot.news == []
    assert snapshot.source_map["quote"]["status"] == "failed"


class FakeAggregator:
    def get_snapshot(self, symbol, period=60):
        return MarketDataSnapshot(
            symbol=symbol,
            quote=make_quote(symbol),
            kline=make_history(),
            sector=SectorContext(),
            concepts=[],
            stock_info=StockInfo(symbol=symbol, name="测试科技"),
            index_change_pct=1.0,
            source_map={
                "quote": {"source": "EastMoney", "status": "success"},
                "kline": {"source": "AkShare", "status": "success"},
                "sector": {"source": "ProviderManager", "status": "partial"},
                "concepts": {"source": "ProviderManager", "status": "partial"},
                "fund_flow": {"source": "ProviderManager", "status": "partial"},
                "news": {"source": "ProviderManager", "status": "partial"},
                "index": {"source": "EastMoney", "status": "success"},
            },
        )


def test_rating_engine_uses_snapshot_and_does_not_fail_without_sector():
    engine = InvestmentRatingEngine(aggregator=FakeAggregator(), persist_history=False)

    rating = engine.evaluate("300001")

    assert rating.symbol == "300001"
    assert rating.total_score > 0
    assert rating.sector_score is None
    assert "板块评分" in rating.data_quality.summary
