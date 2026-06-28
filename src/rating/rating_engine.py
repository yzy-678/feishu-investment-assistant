"""Investment Rating Engine."""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional, Protocol

from src.db import get_database
from src.market.akshare_source import HistoryBar, StockInfo
from src.market.service import MarketDataError, QuoteSnapshot, get_market_data_service
from src.rating.rating_models import InvestmentRating, RatingInputData, RatingLevel
from src.rating.score_calculator import InvestmentScoreCalculator
from src.time_utils import shanghai_now, shanghai_today

logger = logging.getLogger(__name__)

RATING_DATA_SCOPE_WARNING = (
    "当前评级仅基于已接入的行情、技术和量价数据，"
    "不包含未接入的新闻、公告、财报和资金流数据。"
)


class MarketDataProvider(Protocol):
    def get_quote(self, symbol: str, market: str = "CN") -> QuoteSnapshot:
        ...

    def get_history(self, symbol: str, period: int = 60) -> list[HistoryBar]:
        ...

    def get_stock_info(self, symbol: str) -> StockInfo:
        ...

    def get_index_quotes(self, market: str = "CN") -> list[QuoteSnapshot]:
        ...


class InvestmentRatingEngine:
    """Unified rating engine for agents, reports, alerts, and future web UI."""

    def __init__(
        self,
        market_data: Optional[MarketDataProvider] = None,
        score_calculator: Optional[InvestmentScoreCalculator] = None,
        persist_history: bool = True,
    ) -> None:
        self.market_data = market_data or get_market_data_service()
        self.score_calculator = score_calculator or InvestmentScoreCalculator()
        self.persist_history = persist_history
        self._db = get_database() if persist_history else None
        if self._db is not None:
            self._db.init_db()

    def evaluate(self, symbol: str) -> InvestmentRating:
        """Evaluate a stock with truthful market data and deterministic rules."""
        normalized_symbol = symbol.strip().upper()
        quote, quote_warning = self._safe_get_quote(normalized_symbol)
        history, history_warning = self._safe_get_history(normalized_symbol)
        stock_info, info_warning = self._safe_get_stock_info(normalized_symbol)
        index_change_pct, index_warning = self._safe_get_index_change_pct()

        input_data = RatingInputData(
            symbol=normalized_symbol,
            quote=quote,
            history=history,
            stock_info=stock_info,
            index_change_pct=index_change_pct,
            industry_change_pct=None,
            sector_heat_score=None,
            sector_continuity_score=None,
            is_main_sector=None,
            sector_linkage_score=None,
        )
        breakdown = self.score_calculator.calculate(input_data)
        warnings = [
            item
            for item in [
                quote_warning,
                history_warning,
                info_warning,
                index_warning,
                *breakdown.warnings,
                RATING_DATA_SCOPE_WARNING,
            ]
            if item
        ]
        total_score = breakdown.total_score
        rating_date = shanghai_today().isoformat()
        previous = self._get_previous_rating(normalized_symbol, rating_date)
        score_change = _score_change(total_score, previous)
        change_direction = _change_direction(score_change)
        change_reasons = _change_reasons(breakdown.evidence, previous, breakdown)

        rating = InvestmentRating(
            symbol=normalized_symbol,
            name=_resolve_name(normalized_symbol, quote, stock_info),
            total_score=total_score,
            rating_level=rating_level(total_score),
            trend_score=breakdown.trend_score,
            volume_score=breakdown.volume_score,
            sector_score=breakdown.sector_score,
            breakout_score=breakdown.breakout_score,
            strength_score=breakdown.strength_score,
            previous_score=previous["total_score"] if previous else None,
            score_change=score_change,
            change_direction=change_direction,
            change_reasons=change_reasons,
            summary="；".join(breakdown.summary_parts),
            warning="；".join(warnings) if warnings else "暂无明显数据风险。",
            timestamp=shanghai_now().strftime("%Y-%m-%d %H:%M:%S"),
            data_source=_data_source(quote, history, stock_info),
            reserved={
                "evidence": breakdown.evidence,
                "future_extensions": [
                    "fundamental_score",
                    "news_score",
                    "capital_score",
                    "risk_score",
                ],
            },
        )
        self._save_rating_snapshot(rating, rating_date)
        return rating

    def _safe_get_quote(
        self,
        symbol: str,
    ) -> tuple[Optional[QuoteSnapshot], str]:
        try:
            return self.market_data.get_quote(symbol, market="CN"), ""
        except Exception as exc:
            logger.warning("Rating quote unavailable: symbol=%s error=%s", symbol, exc)
            return None, f"实时行情不可用：{exc}"

    def _safe_get_history(self, symbol: str) -> tuple[list[HistoryBar], str]:
        try:
            history = self.market_data.get_history(symbol, period=60)
            return history, "" if history else "历史K线为空"
        except Exception as exc:
            logger.warning("Rating history unavailable: symbol=%s error=%s", symbol, exc)
            return [], f"历史K线不可用：{exc}"

    def _safe_get_stock_info(
        self,
        symbol: str,
    ) -> tuple[Optional[StockInfo], str]:
        try:
            return self.market_data.get_stock_info(symbol), ""
        except Exception as exc:
            logger.warning("Rating stock info unavailable: symbol=%s error=%s", symbol, exc)
            return None, f"股票基础信息不可用：{exc}"

    def _safe_get_index_change_pct(self) -> tuple[float, str]:
        try:
            quotes = self.market_data.get_index_quotes(market="CN")
        except (MarketDataError, Exception) as exc:
            logger.warning("Rating index quote unavailable: %s", exc)
            return 0.0, f"指数行情不可用：{exc}"
        if not quotes:
            return 0.0, "指数行情为空"
        return sum(quote.change_pct for quote in quotes) / len(quotes), ""

    def _get_previous_rating(
        self,
        symbol: str,
        rating_date: str,
    ) -> Optional[sqlite3.Row]:
        if self._db is None:
            return None
        conn = self._db.get_connection()
        return conn.execute(
            """SELECT * FROM investment_rating_history
               WHERE symbol = ? AND rating_date < ?
               ORDER BY rating_date DESC
               LIMIT 1""",
            (symbol, rating_date),
        ).fetchone()

    def _save_rating_snapshot(
        self,
        rating: InvestmentRating,
        rating_date: str,
    ) -> None:
        if self._db is None:
            return
        conn = self._db.get_connection()
        conn.execute(
            """INSERT INTO investment_rating_history
               (symbol, rating_date, name, total_score, rating_level,
                trend_score, volume_score, sector_score, breakout_score,
                strength_score, summary, warning, data_source, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(symbol, rating_date) DO UPDATE SET
                   name = excluded.name,
                   total_score = excluded.total_score,
                   rating_level = excluded.rating_level,
                   trend_score = excluded.trend_score,
                   volume_score = excluded.volume_score,
                   sector_score = excluded.sector_score,
                   breakout_score = excluded.breakout_score,
                   strength_score = excluded.strength_score,
                   summary = excluded.summary,
                   warning = excluded.warning,
                   data_source = excluded.data_source,
                   updated_at = CURRENT_TIMESTAMP""",
            (
                rating.symbol,
                rating_date,
                rating.name,
                rating.total_score,
                rating.rating_level.value,
                rating.trend_score,
                rating.volume_score,
                rating.sector_score,
                rating.breakout_score,
                rating.strength_score,
                rating.summary,
                rating.warning,
                rating.data_source,
            ),
        )
        conn.commit()


def rating_level(total_score: float) -> RatingLevel:
    if total_score >= 95:
        return RatingLevel.S
    if total_score >= 90:
        return RatingLevel.A_PLUS
    if total_score >= 80:
        return RatingLevel.A
    if total_score >= 70:
        return RatingLevel.B_PLUS
    if total_score >= 60:
        return RatingLevel.B
    if total_score >= 50:
        return RatingLevel.C
    return RatingLevel.D


def _resolve_name(
    symbol: str,
    quote: Optional[QuoteSnapshot],
    stock_info: Optional[StockInfo],
) -> str:
    if quote and quote.name:
        return quote.name
    if stock_info and stock_info.name:
        return stock_info.name
    return symbol


def _data_source(
    quote: Optional[QuoteSnapshot],
    history: list[HistoryBar],
    stock_info: Optional[StockInfo],
) -> str:
    sources: list[str] = []
    if quote:
        sources.append(quote.source or "EastMoney")
    if history or stock_info:
        sources.append("AkShare")
    return ", ".join(dict.fromkeys(sources)) or "数据不足"


def _score_change(
    total_score: float,
    previous: Optional[sqlite3.Row],
) -> Optional[float]:
    if previous is None:
        return None
    return round(total_score - float(previous["total_score"] or 0.0), 2)


def _change_direction(score_change: Optional[float]) -> str:
    if score_change is None:
        return "new"
    if score_change > 0:
        return "⬆"
    if score_change < 0:
        return "⬇"
    return "→"


def _change_reasons(
    evidence: dict[str, list[str]],
    previous: Optional[sqlite3.Row],
    breakdown,
) -> list[str]:
    if previous is None:
        return ["首次评级，暂无昨日评分对比。"]

    comparisons = [
        ("trend_score", "趋势", breakdown.trend_score, evidence.get("trend", [])),
        ("volume_score", "量价", breakdown.volume_score, evidence.get("volume", [])),
        ("sector_score", "板块", breakdown.sector_score, evidence.get("sector", [])),
        (
            "breakout_score",
            "K线结构",
            breakdown.breakout_score,
            evidence.get("breakout", []),
        ),
        (
            "strength_score",
            "相对强度",
            breakdown.strength_score,
            evidence.get("strength", []),
        ),
    ]
    reasons: list[str] = []
    for field, label, current_score, current_evidence in comparisons:
        previous_score = float(previous[field] or 0.0)
        delta = round(current_score - previous_score, 2)
        if delta > 0:
            reasons.append(_positive_change_reason(label, delta, current_evidence))
        elif delta < 0:
            reasons.append(f"{label}评分回落 {delta:.1f} 分，需复核持续性。")

    if not reasons:
        return ["评分结构基本持平，暂无显著变化。"]
    return reasons[:5]


def _positive_change_reason(
    label: str,
    delta: float,
    evidence: list[str],
) -> str:
    joined = "；".join(evidence)
    if label == "K线结构" and any(marker in joined for marker in ("平台", "新高", "反包")):
        return f"放量突破/结构改善，{label}评分提升 +{delta:.1f}。"
    if label == "板块":
        return f"板块热度或主线联动提升，{label}评分提升 +{delta:.1f}。"
    if label == "量价":
        return f"成交量/成交额或价升量增改善，{label}评分提升 +{delta:.1f}。"
    if label == "相对强度":
        return f"强于指数/行业或资金抱团增强，{label}评分提升 +{delta:.1f}。"
    if label == "趋势":
        return f"均线结构或站上均线改善，{label}评分提升 +{delta:.1f}。"
    return f"{label}评分提升 +{delta:.1f}。"


_rating_engine: Optional[InvestmentRatingEngine] = None


def get_rating_engine() -> InvestmentRatingEngine:
    """Return the InvestmentRatingEngine singleton."""
    global _rating_engine  # noqa: PLW0603
    if _rating_engine is None:
        _rating_engine = InvestmentRatingEngine()
    return _rating_engine
