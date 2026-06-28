"""Investment rating data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.market.akshare_source import HistoryBar, StockInfo
from src.market.service import QuoteSnapshot


class RatingLevel(str, Enum):
    """Investment rating levels."""

    S = "S"
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class RatingInputData:
    """Truthful market data used by rule scoring."""

    symbol: str
    quote: Optional[QuoteSnapshot] = None
    history: list[HistoryBar] = field(default_factory=list)
    stock_info: Optional[StockInfo] = None
    index_change_pct: float = 0.0
    industry_change_pct: Optional[float] = None
    sector_heat_score: Optional[float] = None
    sector_continuity_score: Optional[float] = None
    is_main_sector: Optional[bool] = None
    sector_linkage_score: Optional[float] = None


@dataclass(frozen=True)
class ScoreBreakdown:
    """Component scores and explainable rule evidence."""

    trend_score: float
    volume_score: float
    sector_score: float
    breakout_score: float
    strength_score: float
    summary_parts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        return round(
            self.trend_score
            + self.volume_score
            + self.sector_score
            + self.breakout_score
            + self.strength_score,
            2,
        )


@dataclass(frozen=True)
class InvestmentRating:
    """Unified investment rating output shared by future agents and products."""

    symbol: str
    name: str
    total_score: float
    rating_level: RatingLevel
    trend_score: float
    volume_score: float
    sector_score: float
    breakout_score: float
    strength_score: float
    previous_score: Optional[float]
    score_change: Optional[float]
    change_direction: str
    change_reasons: list[str]
    summary: str
    warning: str
    timestamp: str
    data_source: str
    reserved: dict[str, Any] = field(default_factory=dict)
