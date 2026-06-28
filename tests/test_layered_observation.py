"""Layered observation builder tests."""

from unittest.mock import MagicMock

from src.market.layered_observation import LayeredObservationBuilder
from src.market.stock_screener import StockCandidate
from src.market.strong_stock_analyzer import StrongStockPick


def make_candidate(
    symbol,
    score,
    industry="半导体",
    trend=25,
    volume=25,
    sector=20,
    breakout=20,
    strength=10,
    change_pct=3.0,
    amount=1000000000,
):
    return StockCandidate(
        symbol=symbol,
        name=f"股票{symbol[-2:]}",
        industry=industry,
        score=score,
        trend_score=trend,
        volume_score=volume,
        sector_score=sector,
        breakout_score=breakout,
        strength_score=strength,
        reason="平台突破；价升量增",
        reserved={
            "change_pct": change_pct,
            "amount": amount,
            "data_source": "AkShare",
        },
    )


def make_pick(candidate, rank):
    return StrongStockPick(
        symbol=candidate.symbol,
        name=candidate.name,
        industry=candidate.industry,
        score=candidate.score,
        rank=rank,
        reason="AI 只解释规则选出的股票",
        risk="波动风险",
        watch_points="观察量价延续",
        data_source="StockScreener",
        data_time="2026-06-28 08:30:00",
    )


def test_layered_builder_uses_rules_for_dragon_and_potential_pools():
    candidates = [
        make_candidate("300001", 95, amount=3000000000),
        make_candidate("300002", 93, amount=2500000000),
        make_candidate("300003", 91, amount=2000000000),
        make_candidate("300004", 82, breakout=20, volume=18, change_pct=2.0),
        make_candidate("300005", 80, breakout=18, volume=17, change_pct=1.5),
        make_candidate("300006", 78, breakout=14, volume=16, change_pct=2.5),
    ]
    screener = MagicMock()
    screener.screen_top_stocks.return_value = candidates
    analyzer = MagicMock()
    analyzer.analyze_candidates.return_value = [
        make_pick(candidate, index + 1)
        for index, candidate in enumerate(candidates[:3])
    ]

    report = LayeredObservationBuilder(
        stock_screener=screener,
        strong_stock_analyzer=analyzer,
    ).build()

    screener.screen_top_stocks.assert_called_once_with(limit=20)
    analyzer.analyze_candidates.assert_called_once_with(candidates[:3], limit=3)
    assert [pick.symbol for pick in report.dragon_pool] == [
        "300001",
        "300002",
        "300003",
    ]
    assert [pick.symbol for pick in report.potential_pool][:3] == [
        "300004",
        "300005",
        "300006",
    ]
    assert report.sectors[0].name == "半导体"
    assert len(report.observation_picks) == 6
