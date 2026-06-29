"""全市场强势股数据筛选器。

Sprint2 只做数据筛选：不接入 AI、不生成报告、不推送。
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from src.market.akshare_source import AkShareError, HistoryBar
from src.market.provider_utils import ProviderTimeoutError, run_with_timeout
from src.providers.provider_manager import ProviderManager, get_provider_manager

logger = logging.getLogger(__name__)

DEFAULT_AKSHARE_HISTORY_TIMEOUT_SECONDS = 5.0
DEFAULT_AKSHARE_FULL_MARKET_TIMEOUT_SECONDS = 20.0
DEFAULT_AKSHARE_BOARD_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class RealtimeQuote:
    """全市场实时行情中的单只股票快照。"""

    symbol: str
    name: str
    industry: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0
    amount: float = 0.0


@dataclass(frozen=True)
class StockCandidate:
    """强势股候选结果。"""

    symbol: str
    name: str
    industry: str
    score: float
    trend_score: float
    volume_score: float
    sector_score: float
    breakout_score: float
    strength_score: float
    reason: str
    reserved: dict[str, Any] = field(default_factory=dict)


class StockDataProvider(Protocol):
    """StrongStockScreener 依赖的数据接口。"""

    def get_realtime_quotes(self) -> list[RealtimeQuote]:
        ...

    def get_history(self, symbol: str, period: int = 60) -> list[HistoryBar]:
        ...

    def get_hot_sectors(self, limit: int = 10) -> set[str]:
        ...

    def get_index_change_pct(self) -> float:
        ...


class AkShareProvider:
    """AkShare 数据提供器。

    负责获取全市场实时行情、历史 K 线、成交量、成交额和热点板块。
    """

    def __init__(
        self,
        ak_module: Any = None,
        history_timeout: float = DEFAULT_AKSHARE_HISTORY_TIMEOUT_SECONDS,
        full_market_timeout: float = DEFAULT_AKSHARE_FULL_MARKET_TIMEOUT_SECONDS,
        board_timeout: float = DEFAULT_AKSHARE_BOARD_TIMEOUT_SECONDS,
    ) -> None:
        self._ak = ak_module
        self.history_timeout = history_timeout
        self.full_market_timeout = full_market_timeout
        self.board_timeout = board_timeout

    def get_realtime_quotes(self) -> list[RealtimeQuote]:
        """获取 A 股全市场实时行情。"""
        try:
            frame = run_with_timeout(
                lambda: self._akshare().stock_zh_a_spot_em(),
                self.full_market_timeout,
                "AkShare stock_zh_a_spot_em",
            )
        except ProviderTimeoutError as exc:
            logger.warning(
                "AkShare provider timeout: provider_failed=true degraded=true "
                "function=stock_zh_a_spot_em error=%s",
                exc,
            )
            raise AkShareError(f"AkShare 全市场实时行情获取超时: {exc}") from exc
        except Exception as exc:
            raise AkShareError(f"AkShare 全市场实时行情获取失败: {exc}") from exc

        quotes: list[RealtimeQuote] = []
        for row in _records(frame):
            symbol = _first_value(row, ("代码", "code", "symbol"))
            if not symbol:
                continue
            quotes.append(
                RealtimeQuote(
                    symbol=symbol,
                    name=_first_value(row, ("名称", "name")),
                    industry=_first_value(row, ("所属行业", "行业", "industry")),
                    price=_to_float(_first_value(row, ("最新价", "price", "最新"))),
                    change_pct=_to_float(_first_value(row, ("涨跌幅", "change_pct"))),
                    volume=_to_float(_first_value(row, ("成交量", "volume"))),
                    amount=_to_float(_first_value(row, ("成交额", "amount"))),
                )
            )
        logger.info("AkShare realtime quotes loaded: count=%d", len(quotes))
        return quotes

    def get_history(self, symbol: str, period: int = 60) -> list[HistoryBar]:
        """获取最近 period 条日 K，包含成交量和成交额。"""
        try:
            frame = run_with_timeout(
                lambda: self._akshare().stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    adjust="qfq",
                ),
                self.history_timeout,
                f"AkShare stock_zh_a_hist {symbol}",
            )
        except ProviderTimeoutError as exc:
            raise AkShareError(f"AkShare 历史 K 线获取超时: {exc}") from exc
        except Exception as exc:
            raise AkShareError(f"AkShare 历史 K 线获取失败: {exc}") from exc

        bars: list[HistoryBar] = []
        for row in _records(frame, tail=period):
            date = _first_value(row, ("日期", "date", "交易日期"))
            if not date:
                continue
            bars.append(
                HistoryBar(
                    date=date,
                    open=_to_float(_first_value(row, ("开盘", "open"))),
                    high=_to_float(_first_value(row, ("最高", "high"))),
                    low=_to_float(_first_value(row, ("最低", "low"))),
                    close=_to_float(_first_value(row, ("收盘", "close"))),
                    volume=_to_float(_first_value(row, ("成交量", "volume"))),
                    amount=_to_float(_first_value(row, ("成交额", "amount"))),
                )
            )
        return bars

    def get_hot_sectors(self, limit: int = 10) -> set[str]:
        """获取热点行业板块名称。"""
        ak = self._akshare()
        if not hasattr(ak, "stock_board_industry_name_em"):
            return set()

        try:
            frame = run_with_timeout(
                lambda: ak.stock_board_industry_name_em(),
                self.board_timeout,
                "AkShare stock_board_industry_name_em",
            )
        except ProviderTimeoutError as exc:
            logger.warning(
                "AkShare hot sectors timeout: provider_failed=true degraded=true "
                "error=%s",
                exc,
            )
            return set()
        except Exception as exc:
            logger.warning("AkShare hot sectors unavailable: %s", exc)
            return set()

        rows = sorted(
            _records(frame),
            key=lambda row: _to_float(_first_value(row, ("涨跌幅", "change_pct"))),
            reverse=True,
        )
        sectors: set[str] = set()
        for row in rows[:limit]:
            name = _first_value(row, ("板块名称", "名称", "行业", "name"))
            if name:
                sectors.add(name)
        return sectors

    def get_index_change_pct(self) -> float:
        """获取主要指数涨跌幅，失败时回退为 0。"""
        ak = self._akshare()
        if not hasattr(ak, "stock_zh_index_spot_em"):
            return 0.0

        try:
            for row in _records(ak.stock_zh_index_spot_em()):
                name = _first_value(row, ("名称", "name"))
                if name in ("上证指数", "沪深300", "深证成指"):
                    return _to_float(_first_value(row, ("涨跌幅", "change_pct")))
        except Exception as exc:
            logger.warning("AkShare index quote unavailable: %s", exc)
        return 0.0

    def _akshare(self) -> Any:
        if self._ak is None:
            try:
                self._ak = importlib.import_module("akshare")
            except ImportError as exc:
                raise AkShareError("未安装 akshare，请先安装依赖") from exc
        return self._ak


class ProviderManagerStockDataProvider:
    """StrongStockScreener adapter backed by ProviderManager."""

    def __init__(self, provider_manager: Optional[ProviderManager] = None) -> None:
        self.provider_manager = provider_manager or get_provider_manager()

    def get_realtime_quotes(self) -> list[RealtimeQuote]:
        for provider in self.provider_manager.providers:
            getter = getattr(provider, "get_realtime_quotes", None)
            if not callable(getter):
                continue
            result = getter()
            if result.ok and result.data:
                return [_quote_from_record(row) for row in result.data]
        return []

    def get_history(self, symbol: str, period: int = 60) -> list[HistoryBar]:
        result = self.provider_manager.get_kline(symbol, period=period)
        return list(result.data or [])

    def get_hot_sectors(self, limit: int = 10) -> set[str]:
        for provider in self.provider_manager.providers:
            getter = getattr(provider, "get_hot_sectors", None)
            if not callable(getter):
                continue
            result = getter(limit=limit)
            if result.ok and result.data:
                return set(result.data)
        return set()

    def get_index_change_pct(self) -> float:
        index_quotes = self.provider_manager.get_index_quotes()
        if index_quotes.ok and index_quotes.data:
            return _average([quote.change_pct for quote in index_quotes.data])
        for provider in self.provider_manager.providers:
            getter = getattr(provider, "get_index_change_pct", None)
            if not callable(getter):
                continue
            result = getter()
            if result.ok and result.data is not None:
                return float(result.data)
        return 0.0


class StrongStockScreener:
    """全市场强势股筛选器。"""

    def __init__(self, provider: Optional[StockDataProvider] = None) -> None:
        self.provider = provider or ProviderManagerStockDataProvider()

    def screen_top_stocks(self, limit: int = 20) -> list[StockCandidate]:
        """扫描并返回 Top20 强势股候选。

        Args:
            limit: 返回数量，默认 20。Sprint2 明确不返回 Top3。
        """
        try:
            quotes = self.provider.get_realtime_quotes()
        except Exception as exc:
            logger.warning(
                "Strong stock screener degraded: provider_failed=true "
                "degraded=true stage=realtime_quotes error=%s",
                exc,
            )
            return []

        try:
            hot_sectors = self.provider.get_hot_sectors(limit=10)
        except Exception as exc:
            logger.warning(
                "Strong stock screener degraded: provider_failed=true "
                "degraded=true stage=hot_sectors error=%s",
                exc,
            )
            hot_sectors = set()

        try:
            index_change_pct = self.provider.get_index_change_pct()
        except Exception as exc:
            logger.warning(
                "Strong stock screener degraded: provider_failed=true "
                "degraded=true stage=index_change error=%s",
                exc,
            )
            index_change_pct = 0.0
        industry_changes = _industry_change_map(quotes)

        candidates: list[StockCandidate] = []
        for quote in quotes:
            try:
                history = self.provider.get_history(quote.symbol, period=60)
            except Exception as exc:
                logger.warning("History unavailable for %s: %s", quote.symbol, exc)
                history = []

            candidates.append(
                self._build_candidate(
                    quote=quote,
                    history=history,
                    hot_sectors=hot_sectors,
                    index_change_pct=index_change_pct,
                    industry_change_pct=industry_changes.get(quote.industry, 0.0),
                )
            )

        candidates.sort(
            key=lambda item: (item.score, item.volume_score, item.sector_score),
            reverse=True,
        )
        return candidates[: max(limit, 0)]

    def _build_candidate(
        self,
        quote: RealtimeQuote,
        history: list[HistoryBar],
        hot_sectors: set[str],
        index_change_pct: float,
        industry_change_pct: float,
    ) -> StockCandidate:
        reasons: list[str] = []
        trend_score = self.score_trend(history, reasons)
        volume_score = self.score_volume_price(history, quote, reasons)
        breakout_score = self.score_breakout(history, reasons)
        sector_score = self.score_sector(
            quote,
            hot_sectors,
            reasons,
            industry_change_pct=industry_change_pct,
        )
        strength_score = self.score_relative_strength(
            quote,
            index_change_pct,
            industry_change_pct,
            reasons,
        )
        score = round(
            trend_score
            + volume_score
            + breakout_score
            + sector_score
            + strength_score,
            2,
        )

        return StockCandidate(
            symbol=quote.symbol,
            name=quote.name,
            industry=quote.industry,
            score=score,
            trend_score=trend_score,
            volume_score=volume_score,
            sector_score=sector_score,
            breakout_score=breakout_score,
            strength_score=strength_score,
            reason="；".join(reasons) if reasons else "数据不足，暂未形成强势信号",
            reserved={
                "price": quote.price,
                "change_pct": quote.change_pct,
                "volume": quote.volume,
                "amount": quote.amount,
                "data_source": "AkShare",
            },
        )

    def score_trend(
        self,
        history: list[HistoryBar],
        reasons: Optional[list[str]] = None,
    ) -> float:
        """趋势评分：MA5 > MA10 > MA20，满分 25。"""
        closes = [bar.close for bar in history if bar.close > 0]
        ma5 = _moving_average(closes, 5)
        ma10 = _moving_average(closes, 10)
        ma20 = _moving_average(closes, 20)
        if ma5 is None or ma10 is None or ma20 is None:
            return 0.0
        if ma5 > ma10 > ma20:
            if reasons is not None:
                reasons.append("MA5 > MA10 > MA20，趋势多头")
            return 25.0
        return 0.0

    def score_volume_price(
        self,
        history: list[HistoryBar],
        quote: Optional[RealtimeQuote] = None,
        reasons: Optional[list[str]] = None,
    ) -> float:
        """量价评分：成交量放大、成交额增加、价升量增，满分 25。"""
        if len(history) < 6:
            return 0.0

        latest = history[-1]
        previous = history[-2]
        baseline = history[-6:-1]
        avg_volume = _average([bar.volume for bar in baseline])
        avg_amount = _average([bar.amount for bar in baseline])
        score = 0.0

        if avg_volume > 0 and latest.volume >= avg_volume * 1.2:
            score += 8.0
            if reasons is not None:
                reasons.append("成交量放大")
        if avg_amount > 0 and latest.amount >= avg_amount * 1.2:
            score += 8.0
            if reasons is not None:
                reasons.append("成交额增加")

        price_up = latest.close > previous.close
        volume_up = latest.volume > previous.volume
        if quote is not None:
            price_up = price_up or quote.price > previous.close
            volume_up = volume_up or quote.volume > previous.volume
        if price_up and volume_up:
            score += 9.0
            if reasons is not None:
                reasons.append("价升量增")

        return score

    def score_breakout(
        self,
        history: list[HistoryBar],
        reasons: Optional[list[str]] = None,
    ) -> float:
        """K 线评分：平台突破、20 日新高、放量反包，满分 20。"""
        if len(history) < 21:
            return 0.0

        latest = history[-1]
        previous = history[-2]
        prior20 = history[-21:-1]
        prior10 = history[-11:-1]
        score = 0.0

        prior20_high = max(bar.high for bar in prior20)
        if latest.close >= prior20_high:
            score += 7.0
            if reasons is not None:
                reasons.append("20日新高")

        prior10_high = max(bar.high for bar in prior10)
        prior10_low = min(bar.low for bar in prior10)
        prior10_avg_close = _average([bar.close for bar in prior10])
        platform_range = (
            (prior10_high - prior10_low) / prior10_avg_close
            if prior10_avg_close > 0
            else 1.0
        )
        if platform_range <= 0.08 and latest.close > prior10_high:
            score += 7.0
            if reasons is not None:
                reasons.append("平台突破")

        reversal = (
            previous.close < previous.open
            and latest.close > latest.open
            and latest.close > previous.open
            and latest.volume >= previous.volume * 1.2
        )
        if reversal:
            score += 6.0
            if reasons is not None:
                reasons.append("放量反包")

        return score

    def score_sector(
        self,
        quote: RealtimeQuote,
        hot_sectors: set[str],
        reasons: Optional[list[str]] = None,
        industry_change_pct: float = 0.0,
    ) -> float:
        """板块评分：热点板块、板块联动，满分 20。"""
        if not quote.industry:
            return 0.0

        score = 0.0
        if quote.industry in hot_sectors:
            score += 12.0
            if reasons is not None:
                reasons.append("所属热点板块")

        if industry_change_pct >= 1.5 or quote.change_pct >= 2.0:
            score += 8.0
            if reasons is not None:
                reasons.append("板块联动走强")

        return score

    def score_relative_strength(
        self,
        quote: RealtimeQuote,
        index_change_pct: float,
        industry_change_pct: float,
        reasons: Optional[list[str]] = None,
    ) -> float:
        """相对强度评分：强于指数、强于行业，满分 10。"""
        score = 0.0
        if quote.change_pct > index_change_pct:
            score += 5.0
            if reasons is not None:
                reasons.append("强于指数")
        if quote.change_pct > industry_change_pct:
            score += 5.0
            if reasons is not None:
                reasons.append("强于行业")
        return score


def _industry_change_map(quotes: list[RealtimeQuote]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for quote in quotes:
        if quote.industry:
            grouped.setdefault(quote.industry, []).append(quote.change_pct)
    return {
        industry: _average(changes)
        for industry, changes in grouped.items()
    }


def _moving_average(values: list[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _average(values: list[float]) -> float:
    clean = list(values)
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def _records(frame: Any, tail: Optional[int] = None) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, list):
        records = frame
    elif isinstance(frame, dict):
        records = [frame]
    else:
        if getattr(frame, "empty", False):
            return []
        if tail is not None and hasattr(frame, "tail"):
            frame = frame.tail(tail)
        if not hasattr(frame, "to_dict"):
            return []
        records = frame.to_dict("records")
    return [dict(row) for row in records if isinstance(row, dict)]


def _quote_from_record(row: dict[str, Any]) -> RealtimeQuote:
    return RealtimeQuote(
        symbol=_first_value(row, ("代码", "code", "symbol")),
        name=_first_value(row, ("名称", "name")),
        industry=_first_value(row, ("所属行业", "行业", "industry")),
        price=_to_float(_first_value(row, ("最新价", "price", "最新"))),
        change_pct=_to_float(_first_value(row, ("涨跌幅", "change_pct"))),
        volume=_to_float(_first_value(row, ("成交量", "volume"))),
        amount=_to_float(_first_value(row, ("成交额", "amount"))),
    )


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key]).strip()
    return ""


def _to_float(raw: Any) -> float:
    if raw in (None, "", "-", "--"):
        return 0.0
    try:
        return float(str(raw).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return 0.0


_stock_screener: Optional[StrongStockScreener] = None


def get_stock_screener() -> StrongStockScreener:
    """Return the StrongStockScreener singleton."""
    global _stock_screener  # noqa: PLW0603
    if _stock_screener is None:
        _stock_screener = StrongStockScreener()
    return _stock_screener
