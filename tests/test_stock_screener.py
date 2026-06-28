"""StrongStockScreener tests."""

import time
from types import SimpleNamespace

import pytest

from src.market.akshare_source import AkShareError, HistoryBar
from src.market.stock_screener import (
    AkShareProvider,
    RealtimeQuote,
    StrongStockScreener,
)


class FakeFrame:
    def __init__(self, records):
        self._records = list(records)
        self.empty = not self._records

    def tail(self, count):
        return FakeFrame(self._records[-count:])

    def to_dict(self, orient):
        assert orient == "records"
        return list(self._records)


class FakeProvider:
    def __init__(
        self,
        quotes,
        histories,
        hot_sectors=None,
        index_change_pct=0.0,
    ):
        self.quotes = quotes
        self.histories = histories
        self.hot_sectors = set(hot_sectors or [])
        self.index_change_pct = index_change_pct

    def get_realtime_quotes(self):
        return list(self.quotes)

    def get_history(self, symbol, period=60):
        value = self.histories.get(symbol, [])
        if isinstance(value, Exception):
            raise value
        return list(value)[-period:]

    def get_hot_sectors(self, limit=10):
        return set(list(self.hot_sectors)[:limit])

    def get_index_change_pct(self):
        return self.index_change_pct


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


def rising_history(count=25):
    return [
        make_bar(i + 1, 10 + i * 0.2, volume=1000 + i * 10, amount=100000 + i * 1000)
        for i in range(count)
    ]


def test_trend_score_requires_ma5_gt_ma10_gt_ma20():
    screener = StrongStockScreener(provider=FakeProvider([], {}))
    reasons = []

    score = screener.score_trend(rising_history(25), reasons)

    assert score == 25.0
    assert "趋势多头" in reasons[0]

    flat = [make_bar(i + 1, 10) for i in range(25)]
    assert screener.score_trend(flat) == 0.0


def test_volume_price_score_counts_volume_amount_and_price_confirmation():
    screener = StrongStockScreener(provider=FakeProvider([], {}))
    history = [
        make_bar(1, 10, volume=1000, amount=100000),
        make_bar(2, 10.1, volume=1000, amount=100000),
        make_bar(3, 10.2, volume=1000, amount=100000),
        make_bar(4, 10.3, volume=1000, amount=100000),
        make_bar(5, 10.4, volume=1000, amount=100000),
        make_bar(6, 11.0, volume=1500, amount=160000),
    ]
    reasons = []

    score = screener.score_volume_price(history, reasons=reasons)

    assert score == 25.0
    assert "成交量放大" in reasons
    assert "成交额增加" in reasons
    assert "价升量增" in reasons


def test_breakout_score_counts_platform_new_high_and_volume_reversal():
    screener = StrongStockScreener(provider=FakeProvider([], {}))
    history = [
        make_bar(i + 1, 10.0 + (i % 3) * 0.05, open_price=10.1, high=10.3, low=9.8)
        for i in range(19)
    ]
    history.append(make_bar(20, 9.9, open_price=10.25, high=10.3, low=9.8, volume=1000))
    history.append(make_bar(21, 10.6, open_price=10.0, high=10.7, low=9.9, volume=1400))
    reasons = []

    score = screener.score_breakout(history, reasons)

    assert score == 20.0
    assert "20日新高" in reasons
    assert "平台突破" in reasons
    assert "放量反包" in reasons


def test_sector_score_counts_hot_sector_and_linkage():
    screener = StrongStockScreener(provider=FakeProvider([], {}))
    quote = RealtimeQuote(
        symbol="300001",
        name="测试科技",
        industry="半导体",
        change_pct=3.2,
    )
    reasons = []

    score = screener.score_sector(quote, {"半导体", "机器人"}, reasons)

    assert score == 20.0
    assert "所属热点板块" in reasons
    assert "板块联动走强" in reasons


def test_screen_top_stocks_sorts_by_score_and_returns_top20():
    quotes = []
    histories = {}
    for index in range(25):
        symbol = f"300{index:03d}"
        change_pct = 25 - index
        quotes.append(
            RealtimeQuote(
                symbol=symbol,
                name=f"股票{index}",
                industry="半导体" if index < 5 else "其他",
                price=20 + index,
                change_pct=change_pct,
                volume=2000,
                amount=200000,
            )
        )
        histories[symbol] = rising_history(25)

    provider = FakeProvider(
        quotes=quotes,
        histories=histories,
        hot_sectors={"半导体"},
        index_change_pct=1.0,
    )
    screener = StrongStockScreener(provider=provider)

    result = screener.screen_top_stocks()

    assert len(result) == 20
    assert result[0].symbol == "300000"
    assert result[0].score >= result[1].score
    assert all(item.score >= next_item.score for item, next_item in zip(result, result[1:]))
    assert result[-1].symbol != "300024"


def test_screen_top_stocks_handles_missing_history():
    quotes = [
        RealtimeQuote(
            symbol="300001",
            name="缺数据",
            industry="未知",
            change_pct=0.5,
        )
    ]
    provider = FakeProvider(
        quotes=quotes,
        histories={"300001": RuntimeError("history missing")},
        hot_sectors=set(),
        index_change_pct=1.0,
    )
    screener = StrongStockScreener(provider=provider)

    result = screener.screen_top_stocks()

    assert len(result) == 1
    assert result[0].symbol == "300001"
    assert result[0].trend_score == 0.0
    assert result[0].volume_score == 0.0
    assert result[0].breakout_score == 0.0
    assert result[0].sector_score == 0.0
    assert result[0].score == 0.0
    assert "数据不足" in result[0].reason


def test_screen_top_stocks_degrades_when_realtime_quotes_unavailable():
    provider = FakeProvider([], {})
    provider.get_realtime_quotes = lambda: (_ for _ in ()).throw(
        RuntimeError("provider timeout")
    )
    screener = StrongStockScreener(provider=provider)

    assert screener.screen_top_stocks() == []


def test_akshare_provider_realtime_quotes_times_out_quickly():
    def slow_spot():
        time.sleep(0.2)
        return FakeFrame([])

    provider = AkShareProvider(
        ak_module=SimpleNamespace(stock_zh_a_spot_em=slow_spot),
        full_market_timeout=0.01,
    )

    with pytest.raises(AkShareError, match="超时"):
        provider.get_realtime_quotes()


def test_akshare_provider_parses_realtime_quotes_and_history():
    spot_frame = FakeFrame([
        {
            "代码": "300001",
            "名称": "测试科技",
            "所属行业": "半导体",
            "最新价": "12.30",
            "涨跌幅": "4.50",
            "成交量": "123456",
            "成交额": "987654321",
        }
    ])
    hist_frame = FakeFrame([
        {
            "日期": "2026-06-01",
            "开盘": "10",
            "最高": "11",
            "最低": "9",
            "收盘": "10.5",
            "成交量": "1000",
            "成交额": "100000",
        }
    ])
    ak = SimpleNamespace(
        stock_zh_a_spot_em=lambda: spot_frame,
        stock_zh_a_hist=lambda **kwargs: hist_frame,
    )

    provider = AkShareProvider(ak_module=ak)

    quotes = provider.get_realtime_quotes()
    history = provider.get_history("300001")

    assert quotes == [
        RealtimeQuote(
            symbol="300001",
            name="测试科技",
            industry="半导体",
            price=12.3,
            change_pct=4.5,
            volume=123456.0,
            amount=987654321.0,
        )
    ]
    assert history[0].close == pytest.approx(10.5)
