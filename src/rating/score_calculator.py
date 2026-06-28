"""Investment rating score calculator."""

from __future__ import annotations

from src.rating.rating_models import RatingInputData, ScoreBreakdown
from src.rating.rating_rules import (
    calculate_breakout_score,
    calculate_sector_score,
    calculate_strength_score,
    calculate_trend_score,
    calculate_volume_score,
)


class InvestmentScoreCalculator:
    """Aggregate V1 technical scores into a 100-point rating."""

    def calculate(self, data: RatingInputData) -> ScoreBreakdown:
        trend_score, trend_evidence, trend_warnings = calculate_trend_score(data)
        volume_score, volume_evidence, volume_warnings = calculate_volume_score(data)
        sector_score, sector_evidence, sector_warnings = calculate_sector_score(data)
        breakout_score, breakout_evidence, breakout_warnings = calculate_breakout_score(
            data
        )
        strength_score, strength_evidence, strength_warnings = (
            calculate_strength_score(data)
        )

        evidence = {
            "trend": trend_evidence,
            "volume": volume_evidence,
            "sector": sector_evidence,
            "breakout": breakout_evidence,
            "strength": strength_evidence,
            "fundamental": ["预留：基本面评分尚未接入"],
            "news": ["预留：新闻/公告评分尚未接入"],
            "capital": ["预留：资金评分尚未接入"],
            "risk": ["预留：风险评分尚未接入"],
        }
        warnings = (
            trend_warnings
            + volume_warnings
            + sector_warnings
            + breakout_warnings
            + strength_warnings
        )

        summary_parts = [
            _main_signal("趋势", trend_score, trend_evidence),
            _main_signal("量价", volume_score, volume_evidence),
            _main_signal("板块", sector_score, sector_evidence),
            _main_signal("K线", breakout_score, breakout_evidence),
            _main_signal("强度", strength_score, strength_evidence),
        ]

        return ScoreBreakdown(
            trend_score=trend_score,
            volume_score=volume_score,
            sector_score=sector_score,
            breakout_score=breakout_score,
            strength_score=strength_score,
            summary_parts=[part for part in summary_parts if part],
            warnings=warnings,
            evidence=evidence,
        )


def _main_signal(
    label: str,
    score: float,
    evidence: list[str],
) -> str:
    if not evidence:
        return f"{label}{score:.1f}分，数据不足或信号不明显"
    return f"{label}{score:.1f}分：{evidence[0]}"
