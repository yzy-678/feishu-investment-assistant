"""Investment Rating Engine."""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional, Protocol

from src.db import get_database
from src.market.aggregator import (
    MarketDataAggregator,
    MarketDataSnapshot,
    get_market_data_aggregator,
)
from src.market.akshare_source import HistoryBar, StockInfo
from src.market.service import MarketDataError, QuoteSnapshot, get_market_data_service
from src.providers import (
    CacheManager,
    CachedKlineProvider,
    CachedSectorProvider,
    KlineProvider,
    ProviderResult,
    SectorProvider as UnifiedSectorProvider,
    get_cache_manager,
)
from src.providers.rating_adapters import (
    EastMoneyKlineProvider,
    FallbackKlineProvider,
    MarketDataKlineProvider,
    RatingSectorProvider,
)
from src.rating.rating_models import (
    DataQualityItem,
    DataQualityReport,
    InvestmentRating,
    RatingInputData,
    RatingLevel,
)
from src.rating.score_calculator import InvestmentScoreCalculator
from src.rating.sector_provider import (
    EastMoneyRawSectorSource,
    SectorContext,
    SectorProvider as RatingSectorSource,
)
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
        sector_provider: Optional[RatingSectorSource] = None,
        kline_provider: Optional[KlineProvider] = None,
        rating_sector_provider: Optional[UnifiedSectorProvider] = None,
        aggregator: Optional[MarketDataAggregator] = None,
        cache_manager: Optional[CacheManager] = None,
        persist_history: bool = True,
    ) -> None:
        using_default_market_data = (
            market_data is None
            and sector_provider is None
            and kline_provider is None
            and rating_sector_provider is None
        )
        self.aggregator = aggregator or (
            get_market_data_aggregator() if using_default_market_data else None
        )
        self.market_data = market_data or get_market_data_service()
        self.score_calculator = score_calculator or InvestmentScoreCalculator()
        self.sector_provider = sector_provider or RatingSectorSource(
            eastmoney_raw_source=EastMoneyRawSectorSource(),
            akshare_provider=self.market_data,
        )
        self.cache_manager = cache_manager or (
            get_cache_manager() if using_default_market_data else None
        )
        base_kline_provider = kline_provider or FallbackKlineProvider(
            [
                MarketDataKlineProvider(self.market_data),
                EastMoneyKlineProvider(self.market_data),
            ]
        )
        base_sector_provider = rating_sector_provider or RatingSectorProvider(
            self.sector_provider
        )
        if self.cache_manager is not None and kline_provider is None:
            base_kline_provider = CachedKlineProvider(
                base_kline_provider,
                cache_manager=self.cache_manager,
            )
        if self.cache_manager is not None and rating_sector_provider is None:
            base_sector_provider = CachedSectorProvider(
                base_sector_provider,
                cache_manager=self.cache_manager,
            )
        self.kline_provider = base_kline_provider
        self.rating_sector_provider = base_sector_provider
        self.persist_history = persist_history
        self._db = get_database() if persist_history else None
        if self._db is not None:
            self._db.init_db()

    def evaluate(self, symbol: str) -> InvestmentRating:
        """Evaluate a stock with truthful market data and deterministic rules."""
        normalized_symbol = symbol.strip().upper()
        if self.aggregator is not None:
            snapshot = self.aggregator.get_snapshot(normalized_symbol, period=60)
            (
                quote,
                quote_warning,
                history,
                history_warning,
                history_quality,
                stock_info,
                info_warning,
                sector_context,
                sector_quality,
                index_change_pct,
                index_warning,
            ) = self._rating_data_from_snapshot(snapshot)
        else:
            quote, quote_warning = self._safe_get_quote(normalized_symbol)
            history, history_warning, history_quality = self._safe_get_history(
                normalized_symbol
            )
            stock_info, info_warning = self._safe_get_stock_info(normalized_symbol)
            sector_context, sector_quality = self._safe_get_sector_context(
                normalized_symbol
            )
            index_change_pct, index_warning = self._safe_get_index_change_pct()

        input_data = RatingInputData(
            symbol=normalized_symbol,
            quote=quote,
            history=history,
            stock_info=stock_info,
            index_change_pct=index_change_pct,
            industry_change_pct=None,
            sector_heat_score=sector_context.sector_heat_score,
            sector_continuity_score=sector_context.sector_continuity_score,
            is_main_sector=sector_context.is_main_sector,
            sector_linkage_score=sector_context.sector_linkage_score,
            industry_score=sector_context.industry_score,
            concept_score=sector_context.concept_score,
            industry_available=sector_context.industry_available,
            concepts_available=sector_context.concepts_available,
            sector_available=_sector_scoring_available(sector_context),
        )
        breakdown = self.score_calculator.calculate(input_data)
        warnings = [
            item
            for item in [
                quote_warning,
                history_warning,
                info_warning,
                sector_context.warning,
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
        data_quality = _data_quality_report(
            quote=quote,
            quote_warning=quote_warning,
            history=history,
            history_quality=history_quality,
            stock_info=stock_info,
            info_warning=info_warning,
            sector_context=sector_context,
            sector_quality=sector_quality,
            index_warning=index_warning,
            sector_score=breakdown.sector_score,
        )

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
            data_source=_data_source(quote, history, stock_info, sector_context),
            data_quality=data_quality,
            reserved={
                "evidence": breakdown.evidence,
                "data_quality": data_quality.as_dict(),
                "industry_score": breakdown.industry_score,
                "concept_score": breakdown.concept_score,
                "industry": sector_context.industry,
                "concepts": sector_context.concepts,
                "sector_status": sector_context.sector_status,
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

    def _rating_data_from_snapshot(
        self,
        snapshot: MarketDataSnapshot,
    ) -> tuple[
        Optional[QuoteSnapshot],
        str,
        list[HistoryBar],
        str,
        DataQualityItem,
        Optional[StockInfo],
        str,
        SectorContext,
        DataQualityItem,
        float,
        str,
    ]:
        source_map = snapshot.source_map
        quote_entry = source_map.get("quote", {})
        kline_entry = source_map.get("kline", {})
        sector_entry = source_map.get("sector", {})
        index_entry = source_map.get("index", {})

        quote = snapshot.quote if _entry_included(quote_entry) else None
        history = list(snapshot.kline)
        stock_info = snapshot.stock_info
        sector_context = snapshot.sector
        return (
            quote,
            "" if quote else quote_entry.get("message", "实时行情不可用"),
            history,
            "" if history else kline_entry.get("message", "历史K线为空"),
            _quality_item_from_source_entry("历史K线", kline_entry, included=bool(history)),
            stock_info,
            "" if (stock_info.name or stock_info.industry or stock_info.concepts) else "股票基础信息不完整",
            sector_context,
            _quality_item_from_source_entry(
                "板块上下文",
                sector_entry,
                included=sector_context.available,
            ),
            snapshot.index_change_pct,
            "" if _entry_included(index_entry) else index_entry.get("message", "指数行情不可用"),
        )

    def _safe_get_quote(
        self,
        symbol: str,
    ) -> tuple[Optional[QuoteSnapshot], str]:
        try:
            return self.market_data.get_quote(symbol, market="CN"), ""
        except Exception as exc:
            logger.warning("Rating quote unavailable: symbol=%s error=%s", symbol, exc)
            return None, f"实时行情不可用：{exc}"

    def _safe_get_history(
        self,
        symbol: str,
    ) -> tuple[list[HistoryBar], str, DataQualityItem]:
        result = self.kline_provider.get_history(symbol, period=60)
        history = result.data or []
        quality = _quality_item_from_result("历史K线", result, included=bool(history))
        if result.ok:
            return history, "" if history else (result.message or "历史K线为空"), quality
        logger.warning(
            "Rating history provider unavailable: symbol=%s source=%s "
            "status=%s error_type=%s message=%s",
            symbol,
            result.source,
            result.status.value,
            result.error_type,
            result.message,
        )
        return [], result.message or "历史K线不可用", quality

    def _safe_get_stock_info(
        self,
        symbol: str,
    ) -> tuple[Optional[StockInfo], str]:
        try:
            return self.market_data.get_stock_info(symbol), ""
        except Exception as exc:
            logger.warning("Rating stock info unavailable: symbol=%s error=%s", symbol, exc)
            return None, "股票基础信息不可用"

    def _safe_get_sector_context(
        self,
        symbol: str,
    ) -> tuple[SectorContext, DataQualityItem]:
        result = self.rating_sector_provider.get_sector_context(symbol)
        quality = _quality_item_from_result(
            "板块上下文",
            result,
            included=bool(result.data and result.data.available),
        )
        if result.data is not None:
            return result.data, quality
        logger.warning(
            "Rating sector provider unavailable: symbol=%s source=%s "
            "status=%s error_type=%s message=%s",
            symbol,
            result.source,
            result.status.value,
            result.error_type,
            result.message,
        )
        return SectorContext(), quality

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


def _quality_item_from_result(
    name: str,
    result: ProviderResult,
    *,
    included: bool,
) -> DataQualityItem:
    metadata = result.metadata or {}
    return DataQualityItem(
        name=name,
        source=result.source,
        status=result.status.value,
        message=result.message,
        cache_hit=bool(metadata.get("cache_hit")),
        fallback=bool(metadata.get("fallback")),
        included=included and result.ok,
    )


def _quality_item_from_source_entry(
    name: str,
    entry: dict,
    *,
    included: bool,
) -> DataQualityItem:
    return DataQualityItem(
        name=name,
        source=str(entry.get("source") or "ProviderManager"),
        status=str(entry.get("status") or "failed"),
        message=str(entry.get("message") or ""),
        cache_hit=bool(entry.get("cache_hit")),
        fallback=bool(entry.get("fallback")),
        included=included and _entry_included(entry),
    )


def _entry_included(entry: dict) -> bool:
    return str(entry.get("status") or "") in {"success", "partial", "cache"}


def _data_quality_report(
    *,
    quote: Optional[QuoteSnapshot],
    quote_warning: str,
    history: list[HistoryBar],
    history_quality: DataQualityItem,
    stock_info: Optional[StockInfo],
    info_warning: str,
    sector_context: SectorContext,
    sector_quality: DataQualityItem,
    index_warning: str,
    sector_score: Optional[float],
) -> DataQualityReport:
    items = [
        DataQualityItem(
            name="实时行情",
            source=(quote.source if quote else "EastMoney"),
            status="success" if quote else "failed",
            message=quote_warning,
            included=quote is not None,
        ),
        history_quality,
        DataQualityItem(
            name="股票基础信息",
            source="AkShare",
            status="success" if stock_info else "failed",
            message=info_warning,
            included=stock_info is not None,
        ),
        sector_quality,
        DataQualityItem(
            name="指数行情",
            source="EastMoney",
            status="success" if not index_warning else "partial",
            message=index_warning,
            included=not index_warning,
        ),
    ]

    missing: list[str] = []
    if quote is None:
        missing.append("实时行情")
    if not history:
        missing.append("历史K线")
    if stock_info is None:
        missing.append("股票基础信息")
    if sector_score is None:
        missing.append("板块评分")
    elif sector_context.sector_status == "部分纳入":
        missing.append("部分板块数据")
    if index_warning:
        missing.append("指数行情")

    return DataQualityReport(items=items, missing_dimensions=missing)


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
    sector_context: SectorContext,
) -> str:
    sources: list[str] = []
    def add_source(value: str) -> None:
        for item in str(value or "").split(","):
            source = item.strip()
            if source and source not in sources:
                sources.append(source)

    if quote:
        add_source(quote.source or "EastMoney")
    if history or stock_info:
        add_source("AkShare")
    if sector_context.data_source:
        add_source(sector_context.data_source)
    return ", ".join(sources) or "数据不足"


def _sector_scoring_available(context: SectorContext) -> bool:
    return context.available


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
        if current_score is None:
            if field == "sector_score":
                reasons.append("板块评分暂未纳入，行业/概念或板块统计数据暂不可用。")
            continue
        previous_value = previous[field]
        if previous_value is None:
            continue
        previous_score = float(previous_value)
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
