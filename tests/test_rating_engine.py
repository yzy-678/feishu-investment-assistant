"""Investment Rating Engine tests."""

from datetime import date

from src.db import get_database, init_database
from src.market.akshare_source import HistoryBar, StockInfo
from src.market.service import QuoteSnapshot
from src.rating.rating_engine import (
    InvestmentRatingEngine,
    RATING_DATA_SCOPE_WARNING,
    rating_level,
)
from src.rating.rating_models import RatingInputData, RatingLevel
from src.rating.rating_rules import (
    calculate_breakout_score,
    calculate_sector_score,
    calculate_strength_score,
    calculate_trend_score,
    calculate_volume_score,
)
from src.rating.score_calculator import InvestmentScoreCalculator


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
        )
    )

    assert score == 20
    assert warnings == []
    assert any("市场主线" in item for item in evidence)


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
        persist_history=False,
    ).evaluate("300001")

    assert rating.symbol == "300001"
    assert rating.name == "测试科技"
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
    assert rating.data_source == "EastMoney, AkShare"
    assert "evidence" in rating.reserved
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
        persist_history=False,
    ).evaluate("300001")

    assert rating.total_score == 0
    assert rating.rating_level == RatingLevel.D
    assert rating.name == "300001"
    assert "不可用" in rating.warning or "不足" in rating.warning
    assert rating.data_source == "数据不足"


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
