"""Layered morning observation system.

Rules select pools; AI only explains already-selected stocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from src.market.stock_screener import StockCandidate, get_stock_screener
from src.market.strong_stock_analyzer import (
    StrongStockAnalyzer,
    StrongStockPick,
    get_strong_stock_analyzer,
)


class StockScreenerProtocol(Protocol):
    def screen_top_stocks(self, limit: int = 20) -> list[StockCandidate]:
        ...


class StrongStockAnalyzerProtocol(Protocol):
    def analyze_candidates(
        self,
        candidates: list[StockCandidate],
        limit: int = 3,
    ) -> list[StrongStockPick]:
        ...


@dataclass(frozen=True)
class SectorObservation:
    name: str
    heat_score: float
    continuity_score: float
    capital_activity: str
    is_main_line: bool
    should_watch: bool


@dataclass(frozen=True)
class PotentialRelayPick:
    symbol: str
    name: str
    industry: str
    score: float
    stage: str
    reason: str
    data_source: str = "StockScreener"
    reserved: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LayeredObservationReport:
    sectors: list[SectorObservation]
    dragon_pool: list[StrongStockPick]
    potential_pool: list[PotentialRelayPick]
    observation_picks: list[StrongStockPick]


class AStockDataProvider(Protocol):
    """Reserved provider for announcements, research, flow, and estimates."""


class LayeredObservationBuilder:
    """Build the layered observation pools for the 08:30 morning push."""

    def __init__(
        self,
        stock_screener: Optional[StockScreenerProtocol] = None,
        strong_stock_analyzer: Optional[StrongStockAnalyzerProtocol] = None,
    ) -> None:
        self.stock_screener = stock_screener or get_stock_screener()
        self.strong_stock_analyzer = (
            strong_stock_analyzer or get_strong_stock_analyzer()
        )

    def build(self) -> LayeredObservationReport:
        candidates = self.stock_screener.screen_top_stocks(limit=20)
        sectors = self._build_sectors(candidates)
        dragon_candidates = self._select_dragon_candidates(candidates)
        dragon_pool = self.strong_stock_analyzer.analyze_candidates(
            dragon_candidates,
            limit=3,
        )
        potential_pool = self._select_potential_pool(
            candidates,
            excluded_symbols={pick.symbol for pick in dragon_pool},
        )
        observation_picks = dragon_pool + [
            _potential_to_observation_pick(pick, rank=index + 4)
            for index, pick in enumerate(potential_pool)
        ]
        return LayeredObservationReport(
            sectors=sectors,
            dragon_pool=dragon_pool,
            potential_pool=potential_pool,
            observation_picks=observation_picks,
        )

    @staticmethod
    def _build_sectors(
        candidates: list[StockCandidate],
        limit: int = 5,
    ) -> list[SectorObservation]:
        grouped: dict[str, list[StockCandidate]] = {}
        for candidate in candidates:
            industry = candidate.industry or "行业数据不足"
            grouped.setdefault(industry, []).append(candidate)

        sectors: list[SectorObservation] = []
        for industry, items in grouped.items():
            avg_score = _average([item.score for item in items])
            avg_sector = _average([item.sector_score for item in items])
            avg_volume = _average([item.volume_score for item in items])
            heat_score = round(min(100.0, avg_score), 1)
            continuity_score = round(min(100.0, avg_sector * 5), 1)
            capital_activity = _capital_activity(avg_volume)
            is_main_line = len(items) >= 2 or avg_sector >= 16
            should_watch = heat_score >= 60 or is_main_line
            sectors.append(
                SectorObservation(
                    name=industry,
                    heat_score=heat_score,
                    continuity_score=continuity_score,
                    capital_activity=capital_activity,
                    is_main_line=is_main_line,
                    should_watch=should_watch,
                )
            )

        sectors.sort(
            key=lambda item: (
                item.is_main_line,
                item.heat_score,
                item.continuity_score,
            ),
            reverse=True,
        )
        return sectors[:limit]

    @staticmethod
    def _select_dragon_candidates(
        candidates: list[StockCandidate],
        limit: int = 3,
    ) -> list[StockCandidate]:
        sorted_candidates = sorted(
            candidates,
            key=lambda item: (
                item.score,
                item.sector_score,
                item.volume_score,
                _amount(item),
                item.trend_score,
            ),
            reverse=True,
        )
        return sorted_candidates[:limit]

    @staticmethod
    def _select_potential_pool(
        candidates: list[StockCandidate],
        excluded_symbols: set[str],
        limit: int = 8,
    ) -> list[PotentialRelayPick]:
        pool: list[tuple[float, StockCandidate]] = []
        for candidate in candidates:
            if candidate.symbol in excluded_symbols:
                continue
            change_pct = float(candidate.reserved.get("change_pct") or 0.0)
            if change_pct >= 9.5:
                continue
            potential_score = (
                candidate.breakout_score * 1.5
                + candidate.trend_score
                + candidate.volume_score * 1.2
                + candidate.strength_score
            )
            if potential_score <= 0:
                continue
            pool.append((potential_score, candidate))

        pool.sort(key=lambda item: item[0], reverse=True)
        return [
            PotentialRelayPick(
                symbol=candidate.symbol,
                name=candidate.name,
                industry=candidate.industry,
                score=candidate.score,
                stage=_stage(candidate),
                reason=candidate.reason,
                reserved=candidate.reserved,
            )
            for _, candidate in pool[:limit]
        ]


def _potential_to_observation_pick(
    pick: PotentialRelayPick,
    rank: int,
) -> StrongStockPick:
    return StrongStockPick(
        symbol=pick.symbol,
        name=pick.name,
        industry=pick.industry,
        score=pick.score,
        rank=rank,
        reason=f"{pick.stage}阶段：{pick.reason}",
        risk="接力观察股仍需确认量价和板块持续性。",
        watch_points="观察是否继续放量、突破是否有效、板块是否延续。",
        data_source=pick.data_source,
        data_time=str(pick.reserved.get("data_time") or ""),
        reserved={"pool": "potential"},
    )


def _stage(candidate: StockCandidate) -> str:
    if candidate.breakout_score >= 14:
        return "突破"
    if candidate.trend_score >= 25 and candidate.volume_score >= 17:
        return "启动"
    if candidate.score >= 80:
        return "加速"
    return "整理"


def _capital_activity(avg_volume_score: float) -> str:
    if avg_volume_score >= 20:
        return "高"
    if avg_volume_score >= 12:
        return "中"
    return "低"


def _amount(candidate: StockCandidate) -> float:
    return float(candidate.reserved.get("amount") or 0.0)


def _average(values: list[float]) -> float:
    clean = list(values)
    if not clean:
        return 0.0
    return sum(clean) / len(clean)
