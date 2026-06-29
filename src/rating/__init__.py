"""Investment Rating Engine public API."""

from src.rating.rating_models import (
    DataQualityItem,
    DataQualityReport,
    InvestmentRating,
    RatingInputData,
    RatingLevel,
    ScoreBreakdown,
)
from src.rating.sector_provider import (
    EastMoneyRawSectorSource,
    SectorContext,
    SectorProvider,
)

__all__ = [
    "InvestmentRating",
    "InvestmentRatingEngine",
    "DataQualityItem",
    "DataQualityReport",
    "RatingInputData",
    "RatingLevel",
    "ScoreBreakdown",
    "EastMoneyRawSectorSource",
    "SectorContext",
    "SectorProvider",
    "get_rating_engine",
]


def __getattr__(name: str):
    if name in {"InvestmentRatingEngine", "get_rating_engine"}:
        from src.rating.rating_engine import InvestmentRatingEngine, get_rating_engine

        return {
            "InvestmentRatingEngine": InvestmentRatingEngine,
            "get_rating_engine": get_rating_engine,
        }[name]
    raise AttributeError(f"module 'src.rating' has no attribute {name!r}")
