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
    industry_score: Optional[float] = None
    concept_score: Optional[float] = None
    industry_available: bool = False
    concepts_available: bool = False
    sector_available: bool = False


@dataclass(frozen=True)
class ScoreBreakdown:
    """Component scores and explainable rule evidence."""

    trend_score: float
    volume_score: float
    sector_score: Optional[float]
    breakout_score: float
    strength_score: float
    industry_score: Optional[float] = None
    concept_score: Optional[float] = None
    summary_parts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        scores = [
            self.trend_score,
            self.volume_score,
            self.breakout_score,
            self.strength_score,
        ]
        if self.sector_score is not None:
            scores.append(self.sector_score)
        if not scores:
            return 0.0
        raw_score = sum(scores)
        max_score = len(scores) * 20
        return round(raw_score / max_score * 100, 2)


@dataclass(frozen=True)
class DataQualityItem:
    """Single data quality signal for a rating input."""

    name: str
    source: str
    status: str
    message: str = ""
    cache_hit: bool = False
    fallback: bool = False
    included: bool = True


@dataclass(frozen=True)
class DataQualityReport:
    """Structured data quality summary for user-facing trust signals."""

    items: list[DataQualityItem] = field(default_factory=list)
    missing_dimensions: list[str] = field(default_factory=list)

    @property
    def has_cache(self) -> bool:
        return any(item.cache_hit for item in self.items)

    @property
    def has_fallback(self) -> bool:
        return any(item.fallback for item in self.items)

    @property
    def summary(self) -> str:
        labels: list[str] = []
        if self.has_cache:
            labels.append("使用缓存")
        if self.has_fallback:
            labels.append("使用fallback")
        if self.missing_dimensions:
            labels.append(f"未纳入：{'、'.join(self.missing_dimensions)}")
        return "；".join(labels) if labels else "数据质量正常"

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "has_cache": self.has_cache,
            "has_fallback": self.has_fallback,
            "missing_dimensions": list(self.missing_dimensions),
            "items": [
                {
                    "name": item.name,
                    "source": item.source,
                    "status": item.status,
                    "message": item.message,
                    "cache_hit": item.cache_hit,
                    "fallback": item.fallback,
                    "included": item.included,
                }
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class InvestmentRating:
    """Unified investment rating output shared by future agents and products."""

    symbol: str
    name: str
    total_score: float
    rating_level: RatingLevel
    trend_score: float
    volume_score: float
    sector_score: Optional[float]
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
    data_quality: DataQualityReport = field(default_factory=DataQualityReport)
    reserved: dict[str, Any] = field(default_factory=dict)
