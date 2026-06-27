"""AkShareSource 数据层测试。

全部使用 mock/fake，不访问真实 AkShare 或网络。
"""

import logging
from types import SimpleNamespace

import pytest

from src.market.akshare_source import (
    AkShareError,
    AkShareSource,
    HistoryBar,
    MACDSnapshot,
    MASnapshot,
    StockInfo,
)
from src.market.service import MarketDataService


class FakeFrame:
    """最小 DataFrame 替身。"""

    def __init__(self, records):
        self._records = list(records)
        self.empty = not self._records

    def tail(self, count):
        return FakeFrame(self._records[-count:])

    def to_dict(self, orient):
        assert orient == "records"
        return list(self._records)


def make_history_records(count=60):
    return [
        {
            "日期": f"2026-06-{day:02d}",
            "开盘": 10 + index,
            "最高": 11 + index,
            "最低": 9 + index,
            "收盘": 10.5 + index,
            "成交量": 1000 + index,
            "成交额": 100000 + index,
        }
        for index, day in enumerate(range(1, count + 1))
    ]


class TestAkShareHistory:
    def test_get_history_returns_pydantic_models(self):
        ak = SimpleNamespace(
            stock_zh_a_hist=lambda **kwargs: FakeFrame(make_history_records(3))
        )
        source = AkShareSource(ak_module=ak)

        result = source.get_history("300136", period=2)

        assert len(result) == 2
        assert all(isinstance(item, HistoryBar) for item in result)
        assert result[0].date == "2026-06-02"
        assert result[0].open == 11
        assert result[0].high == 12
        assert result[0].low == 10
        assert result[0].close == 11.5
        assert result[0].volume == 1001
        assert result[0].amount == 100001

    def test_get_history_uses_akshare_daily_qfq(self):
        calls = []

        def stock_zh_a_hist(**kwargs):
            calls.append(kwargs)
            return FakeFrame(make_history_records(1))

        source = AkShareSource(
            ak_module=SimpleNamespace(stock_zh_a_hist=stock_zh_a_hist)
        )

        source.get_history("300136", period=60)

        assert calls == [
            {
                "symbol": "300136",
                "period": "daily",
                "adjust": "qfq",
            }
        ]

    def test_get_history_logs_request_success_and_failed(self, caplog):
        caplog.set_level(logging.INFO, logger="src.market.akshare_source")
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_zh_a_hist=lambda **kwargs: FakeFrame(make_history_records(1))
            )
        )

        source.get_history("300136")

        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert "AkShare request: function=stock_zh_a_hist symbol=300136" in logs
        assert "AkShare success: function=stock_zh_a_hist symbol=300136" in logs

        caplog.clear()

        def boom(**kwargs):
            raise RuntimeError("akshare unavailable")

        failed_source = AkShareSource(
            ak_module=SimpleNamespace(stock_zh_a_hist=boom)
        )
        with pytest.raises(AkShareError):
            failed_source.get_history("300136")

        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert "AkShare failed: function=stock_zh_a_hist symbol=300136" in logs
        assert "akshare unavailable" in logs


class TestAkShareIndicators:
    def test_get_ma_returns_ma_snapshot(self):
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_zh_a_hist=lambda **kwargs: FakeFrame(make_history_records(60))
            )
        )

        result = source.get_ma("300136")

        assert isinstance(result, MASnapshot)
        assert result.symbol == "300136"
        assert result.MA5 == pytest.approx(sum(65.5 + i for i in range(5)) / 5)
        assert result.MA10 is not None
        assert result.MA20 is not None
        assert result.MA60 is not None

    def test_get_ma_returns_none_when_not_enough_history(self):
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_zh_a_hist=lambda **kwargs: FakeFrame(make_history_records(8))
            )
        )

        result = source.get_ma("300136")

        assert result.MA5 is not None
        assert result.MA10 is None
        assert result.MA20 is None
        assert result.MA60 is None

    def test_get_macd_returns_macd_snapshot(self):
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_zh_a_hist=lambda **kwargs: FakeFrame(make_history_records(60))
            )
        )

        result = source.get_macd("300136")

        assert isinstance(result, MACDSnapshot)
        assert result.symbol == "300136"
        assert result.DIF is not None
        assert result.DEA is not None
        assert result.MACD is not None

    def test_get_macd_handles_empty_history(self):
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_zh_a_hist=lambda **kwargs: FakeFrame([])
            )
        )

        result = source.get_macd("300136")

        assert result == MACDSnapshot(symbol="300136")


class TestStockInfo:
    def test_get_stock_info_parses_name_industry_and_concepts(self):
        info_frame = FakeFrame([
            {"item": "股票简称", "value": "信维通信"},
            {"item": "行业", "value": "消费电子"},
            {"item": "概念板块", "value": "消费电子, 天线, 无线充电"},
        ])
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_individual_info_em=lambda **kwargs: info_frame
            )
        )

        result = source.get_stock_info("300136")

        assert isinstance(result, StockInfo)
        assert result.symbol == "300136"
        assert result.name == "信维通信"
        assert result.industry == "消费电子"
        assert result.concepts == ["消费电子", "天线", "无线充电"]

    def test_get_stock_info_falls_back_to_hot_keywords_for_concepts(self):
        info_frame = FakeFrame([
            {"item": "股票简称", "value": "信维通信"},
            {"item": "行业", "value": "消费电子"},
        ])
        keyword_frame = FakeFrame([
            {"概念名称": "商业航天"},
            {"概念名称": "卫星导航"},
        ])
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_individual_info_em=lambda **kwargs: info_frame,
                stock_hot_keyword_em=lambda **kwargs: keyword_frame,
            )
        )

        result = source.get_stock_info("300136")

        assert result.concepts == ["商业航天", "卫星导航"]

    def test_get_stock_info_returns_empty_concepts_when_unavailable(self):
        info_frame = FakeFrame([
            {"item": "股票简称", "value": "信维通信"},
            {"item": "行业", "value": "消费电子"},
        ])
        source = AkShareSource(
            ak_module=SimpleNamespace(
                stock_individual_info_em=lambda **kwargs: info_frame
            )
        )

        result = source.get_stock_info("300136")

        assert result.concepts == []


class TestMarketDataServiceAkShareDelegation:
    def test_market_data_service_delegates_history_to_akshare(self):
        fake_source = SimpleNamespace(
            get_history=lambda symbol, period=60: [HistoryBar(
                date="2026-06-01",
                open=1,
                high=2,
                low=0.5,
                close=1.5,
                volume=100,
                amount=1000,
            )]
        )
        service = MarketDataService(akshare_source=fake_source)

        result = service.get_history("300136", period=1)

        assert result[0].date == "2026-06-01"

    def test_market_data_service_delegates_indicators_and_info(self):
        fake_source = SimpleNamespace(
            get_history=lambda symbol, period=60: [],
            get_ma=lambda symbol: MASnapshot(symbol=symbol, MA5=10),
            get_macd=lambda symbol: MACDSnapshot(symbol=symbol, DIF=1, DEA=0.5, MACD=1),
            get_stock_info=lambda symbol: StockInfo(
                symbol=symbol,
                name="信维通信",
                industry="消费电子",
                concepts=["商业航天"],
            ),
        )
        service = MarketDataService(akshare_source=fake_source)

        assert service.get_ma("300136").MA5 == 10
        assert service.get_macd("300136").DIF == 1
        assert service.get_stock_info("300136").concepts == ["商业航天"]
