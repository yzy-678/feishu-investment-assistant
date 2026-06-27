"""
实时行情服务

当前仅接入 A 股 Eastmoney 快照，用于问答上下文、报告和盘中扫描。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Sequence, Union

import httpx

from src.db.models import WatchlistItem
from src.market.akshare_source import (
    AkShareSource,
    HistoryBar,
    MACDSnapshot,
    MASnapshot,
    StockInfo,
)
from src.time_utils import SHANGHAI_TZ, shanghai_now

logger = logging.getLogger(__name__)

EASTMONEY_BASE_URL = "https://push2.eastmoney.com"
EASTMONEY_HIS_BASE_URL = "https://push2his.eastmoney.com"
EASTMONEY_QUOTE_BASE_URLS: tuple[str, ...] = (
    "https://push2delay.eastmoney.com",
    EASTMONEY_BASE_URL,
)
EASTMONEY_HIS_BASE_URLS: tuple[str, ...] = (
    EASTMONEY_HIS_BASE_URL,
    "https://81.push2his.eastmoney.com",
    "https://82.push2his.eastmoney.com",
)
EASTMONEY_SEARCH_BASE_URL = "https://searchapi.eastmoney.com"
DEFAULT_TIMEOUT = 8.0
RETRY_STATUS_CODES = {502, 503, 504}
MAX_ATTEMPTS_PER_HOST = 2
MAX_QUOTE_AGE_SECONDS = 300

CN_INDEX_SECIDS: tuple[tuple[str, str], ...] = (
    ("1.000001", "上证指数"),
    ("0.399001", "深证成指"),
    ("0.399006", "创业板指"),
)


class MarketDataError(Exception):
    """实时行情获取失败。"""

    def __init__(self, message: str, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class QuoteSnapshot:
    """单只股票或指数快照。"""

    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    open_price: float
    high_price: float
    low_price: float
    prev_close: float
    volume: float
    amount: float
    amplitude_pct: float
    turnover_rate: float
    fetched_at: str
    source: str = "EastMoney"
    timestamp: str = ""
    data_age_seconds: Optional[int] = None
    is_trading_session: bool = False
    failure_reason: str = ""


@dataclass(frozen=True)
class DailyBar:
    """日线摘要。"""

    trade_date: str
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: float
    amount: float
    amplitude_pct: float
    change_pct: float
    change: float
    turnover_rate: float


class MarketDataService:
    """统一的实时行情访问入口。"""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        akshare_source: Optional[AkShareSource] = None,
    ) -> None:
        self.timeout = timeout
        self._akshare_source = akshare_source
        self._symbol_search_cache: dict[str, Optional[str]] = {}

    def supports_market(self, market: str) -> bool:
        return market.strip().upper() == "CN"

    def extract_symbol(self, text: str) -> Optional[str]:
        """从文本中提取 A 股代码。

        优先提取显式 6 位代码；没有代码时，使用 EastMoney 证券搜索把
        股票简称解析为代码，避免股票问题落入通用 LLM 兜底。
        """
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
        if match:
            return match.group(1)

        keyword = self._normalize_symbol_search_keyword(text)
        if not keyword:
            return None
        if keyword in self._symbol_search_cache:
            return self._symbol_search_cache[keyword]

        try:
            symbol = self._search_symbol_by_name(keyword)
        except MarketDataError as exc:
            logger.warning(
                "EastMoney symbol search failed: keyword=%s reason=%s error=%s",
                keyword,
                exc.reason,
                exc,
            )
            symbol = None

        self._symbol_search_cache[keyword] = symbol
        return symbol

    def get_quote(self, symbol: str, market: str = "CN") -> QuoteSnapshot:
        """获取单只 A 股实时快照。"""
        if not self.supports_market(market):
            raise MarketDataError(
                f"当前仅支持 A 股实时行情，收到市场: {market}",
                reason="invalid_response",
            )

        try:
            secid = self._to_secid(symbol)
            payload = self._request_json(
                EASTMONEY_QUOTE_BASE_URLS,
                "/api/qt/stock/get",
                params={
                    "secid": secid,
                    "fields": ",".join(
                        [
                            "f57", "f58", "f43", "f44", "f45", "f46", "f47",
                            "f48", "f60", "f80", "f86", "f168", "f169",
                            "f170", "f171",
                        ]
                    ),
                },
                symbol=symbol,
            )
        except MarketDataError as exc:
            logger.warning(
                "EastMoney quote failed: symbol=%s reason=%s error=%s",
                symbol,
                exc.reason,
                exc,
            )
            raise

        data = payload.get("data") or {}
        if not data or not data.get("f57"):
            logger.warning(
                "EastMoney quote failed: symbol=%s reason=symbol_not_found",
                symbol,
            )
            raise MarketDataError(
                f"未获取到股票 {symbol} 的实时行情",
                reason="symbol_not_found",
            )

        try:
            quote = self._parse_quote(data)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "EastMoney quote failed: symbol=%s reason=parse_error error=%s",
                symbol,
                exc,
            )
            raise MarketDataError(
                f"股票 {symbol} 行情解析失败: {exc}",
                reason="parse_error",
            ) from exc

        logger.info(
            "EastMoney quote success: symbol=%s price=%s change_pct=%s "
            "timestamp=%s fetched_at=%s data_age_seconds=%s source=%s "
            "failure_reason=%s",
            symbol,
            quote.price,
            quote.change_pct,
            quote.timestamp,
            quote.fetched_at,
            quote.data_age_seconds,
            quote.source,
            quote.failure_reason,
        )
        return quote

    def _search_symbol_by_name(self, keyword: str) -> Optional[str]:
        payload = self._request_json(
            EASTMONEY_SEARCH_BASE_URL,
            "/api/suggest/get",
            params={
                "input": keyword,
                "type": "14",
                "count": "5",
            },
            symbol=keyword,
        )
        rows = (
            payload.get("QuotationCodeTable", {}).get("Data", [])
            if isinstance(payload, dict)
            else []
        )
        if not isinstance(rows, list):
            return None

        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("Code") or row.get("UnifiedCode") or "").strip()
            name = str(row.get("Name") or "").strip()
            classify = str(row.get("Classify") or "").strip()
            if (
                re.fullmatch(r"\d{6}", code)
                and classify == "AStock"
                and (name == keyword or keyword in name or name in keyword)
            ):
                return code

        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("Code") or row.get("UnifiedCode") or "").strip()
            classify = str(row.get("Classify") or "").strip()
            if re.fullmatch(r"\d{6}", code) and classify == "AStock":
                return code
        return None

    @staticmethod
    def _normalize_symbol_search_keyword(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"[？?！!。,.，、:：；;（）()【】\[\]{}<>《》]", " ", cleaned)
        cleaned = re.sub(
            r"(帮我|请|麻烦|查一下|查下|查询|看一下|看看|看下|分析|点评|"
            r"怎么看|如何看|股票|个股|股价|行情|走势|今天|现在|一下|吗|呢)",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned or len(cleaned) > 12:
            return ""
        if not re.search(r"[\u4e00-\u9fff]{2,}", cleaned):
            return ""
        return cleaned

    def get_history(self, symbol: str, period: int = 60) -> list[HistoryBar]:
        """获取历史 K 线。数据源：AkShare。"""
        return self._akshare().get_history(symbol=symbol, period=period)

    def get_ma(self, symbol: str) -> MASnapshot:
        """获取 MA5 / MA10 / MA20 / MA60。数据源：AkShare。"""
        return self._akshare().get_ma(symbol=symbol)

    def get_macd(self, symbol: str) -> MACDSnapshot:
        """获取 DIF / DEA / MACD。数据源：AkShare。"""
        return self._akshare().get_macd(symbol=symbol)

    def get_stock_info(self, symbol: str) -> StockInfo:
        """获取股票名称、行业、概念板块。数据源：AkShare。"""
        return self._akshare().get_stock_info(symbol=symbol)

    def get_index_quotes(self, market: str = "CN") -> list[QuoteSnapshot]:
        """获取主要指数快照。"""
        if not self.supports_market(market):
            return []

        quotes: list[QuoteSnapshot] = []
        for secid, fallback_name in CN_INDEX_SECIDS:
            payload = self._request_json(
                EASTMONEY_QUOTE_BASE_URLS,
                "/api/qt/stock/get",
                params={
                    "secid": secid,
                    "fields": ",".join(
                        [
                            "f57", "f58", "f43", "f44", "f45", "f46", "f47",
                            "f48", "f60", "f80", "f86", "f168", "f169",
                            "f170", "f171",
                        ]
                    ),
                },
            )
            data = payload.get("data") or {}
            if not data:
                continue
            if not data.get("f58"):
                data["f58"] = fallback_name
            quotes.append(self._parse_quote(data))
        return quotes

    def get_recent_bars(
        self,
        symbol: str,
        market: str = "CN",
        limit: int = 5,
    ) -> list[DailyBar]:
        """获取最近 N 个交易日的日线摘要。"""
        if not self.supports_market(market):
            raise MarketDataError(f"当前仅支持 A 股日线数据，收到市场: {market}")

        try:
            payload = self._request_json(
                EASTMONEY_HIS_BASE_URLS,
                "/api/qt/stock/kline/get",
                params={
                    "secid": self._to_secid(symbol),
                    "klt": "101",
                    "fqt": "1",
                    "lmt": str(limit),
                    "end": "20500101",
                    "fields1": "f1,f2,f3",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                },
            )
        except MarketDataError as exc:
            logger.warning("EastMoney recent bars unavailable for %s: %s", symbol, exc)
            return []
        klines = (payload.get("data") or {}).get("klines") or []
        bars: list[DailyBar] = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 11:
                continue
            bars.append(
                DailyBar(
                    trade_date=parts[0],
                    open_price=self._to_float(parts[1]),
                    close_price=self._to_float(parts[2]),
                    high_price=self._to_float(parts[3]),
                    low_price=self._to_float(parts[4]),
                    volume=self._to_float(parts[5]),
                    amount=self._to_float(parts[6]),
                    amplitude_pct=self._to_float(parts[7]),
                    change_pct=self._to_float(parts[8]),
                    change=self._to_float(parts[9]),
                    turnover_rate=self._to_float(parts[10]),
                )
            )
        return bars

    def get_watchlist_quotes(
        self,
        items: Iterable[WatchlistItem],
        market: str = "CN",
    ) -> list[QuoteSnapshot]:
        """获取自选股实时快照。"""
        if not self.supports_market(market):
            return []

        quotes: list[QuoteSnapshot] = []
        for item in items:
            if item.market != "a":
                continue
            try:
                quotes.append(self.get_quote(item.symbol, market=market))
            except MarketDataError as exc:
                logger.warning("Failed to fetch quote for %s: %s", item.symbol, exc)
        return quotes

    def build_market_snapshot_text(
        self,
        market: str = "CN",
        watchlist_items: Optional[Iterable[WatchlistItem]] = None,
        focus_symbol: str = "",
    ) -> str:
        """构建适合注入给 AI 的市场快照文本。"""
        if not self.supports_market(market):
            return f"【实时行情】当前市场 {market} 暂未接入真实数据源。"

        lines: list[str] = [
            "【实时 A 股快照】",
            f"数据时间（Asia/Shanghai）：{shanghai_now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        index_quotes = self.get_index_quotes(market=market)
        if index_quotes:
            lines.append("主要指数：")
            for quote in index_quotes:
                lines.append(f"  - {self.format_quote_brief(quote, include_symbol=False)}")

        if focus_symbol:
            try:
                focus_quote = self.get_quote(focus_symbol, market=market)
                lines.append("关注个股：")
                lines.append(f"  - {self.format_quote_detail(focus_quote)}")
            except MarketDataError as exc:
                lines.append(f"关注个股：{focus_symbol} 行情暂不可用（{exc}）")

        if watchlist_items:
            watchlist_list = list(watchlist_items)[:8]
            watchlist_quotes = self.get_watchlist_quotes(watchlist_list, market=market)
            if watchlist_quotes:
                lines.append("自选股快照：")
                for quote in watchlist_quotes:
                    lines.append(f"  - {self.format_quote_detail(quote)}")

        return "\n".join(lines)

    @staticmethod
    def format_quote_brief(quote: QuoteSnapshot, include_symbol: bool = True) -> str:
        """格式化简短行情。"""
        prefix = f"{quote.symbol} {quote.name}" if include_symbol else quote.name
        return (
            f"{prefix} {quote.price:.2f} "
            f"({quote.change_pct:+.2f}%, {quote.change:+.2f})"
        )

    @staticmethod
    def format_quote_detail(quote: QuoteSnapshot) -> str:
        """格式化详细行情。"""
        return (
            f"{quote.symbol} {quote.name} "
            f"{quote.price:.2f} ({quote.change_pct:+.2f}%) "
            f"开/高/低 {quote.open_price:.2f}/{quote.high_price:.2f}/{quote.low_price:.2f} "
            f"振幅 {quote.amplitude_pct:.2f}% 换手 {quote.turnover_rate:.2f}% "
            f"成交额 {quote.amount / 100000000:.2f} 亿"
        )

    def _request_json(
        self,
        base_url: Union[str, Sequence[str]],
        path: str,
        params: dict[str, str],
        symbol: str = "",
    ) -> dict:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0.0.0 Safari/537.36"
            ),
            "Referer": "https://quote.eastmoney.com/",
            "Origin": "https://quote.eastmoney.com",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        base_urls = (base_url,) if isinstance(base_url, str) else tuple(base_url)
        last_exc: Optional[Exception] = None
        last_reason = "network_error"

        for current_base_url in base_urls:
            for attempt in range(1, MAX_ATTEMPTS_PER_HOST + 1):
                started_at = time.perf_counter()
                try:
                    with httpx.Client(
                        base_url=current_base_url,
                        timeout=self.timeout,
                        headers=headers,
                        follow_redirects=True,
                    ) as client:
                        request = client.build_request("GET", path, params=params)
                        logger.info(
                            "EastMoney request: symbol=%s url=%s",
                            symbol or "-",
                            request.url,
                        )
                        logger.debug("EastMoney request headers: %s", dict(request.headers))
                        logger.debug(
                            "EastMoney request host attempt: %s (%d/%d)",
                            current_base_url,
                            attempt,
                            MAX_ATTEMPTS_PER_HOST,
                        )

                        response = client.send(request)
                        elapsed_ms = (time.perf_counter() - started_at) * 1000
                        logger.info(
                            "EastMoney response: symbol=%s url=%s status=%s elapsed_ms=%.1f",
                            symbol or "-",
                            response.request.url,
                            response.status_code,
                            elapsed_ms,
                        )
                        logger.debug(
                            "EastMoney response location: %s",
                            response.headers.get("location", ""),
                        )
                        if response.history:
                            logger.debug(
                                "EastMoney redirect history: %s",
                                [
                                    {
                                        "status": item.status_code,
                                        "url": str(item.url),
                                        "location": item.headers.get("location", ""),
                                    }
                                    for item in response.history
                                ],
                            )
                        logger.debug(
                            "EastMoney response preview: %s",
                            response.text[:500],
                        )

                        if response.status_code in RETRY_STATUS_CODES:
                            last_reason = "invalid_response"
                            last_exc = httpx.HTTPStatusError(
                                f"EastMoney {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                            if attempt < MAX_ATTEMPTS_PER_HOST:
                                logger.warning(
                                    "EastMoney transient status: host=%s status=%d attempt=%d",
                                    current_base_url,
                                    response.status_code,
                                    attempt,
                                )
                                time.sleep(0.2 * attempt)
                                continue
                            logger.warning(
                                "EastMoney endpoint exhausted: host=%s status=%d",
                                current_base_url,
                                response.status_code,
                            )
                            break

                        response.raise_for_status()
                        try:
                            return response.json()
                        except ValueError as exc:
                            last_exc = exc
                            last_reason = "parse_error"
                            logger.warning(
                                "EastMoney request failed: symbol=%s reason=parse_error "
                                "url=%s status=%s elapsed_ms=%.1f error=%s",
                                symbol or "-",
                                response.request.url,
                                response.status_code,
                                elapsed_ms,
                                exc,
                            )
                            if attempt < MAX_ATTEMPTS_PER_HOST:
                                time.sleep(0.2 * attempt)
                                continue
                            break

                except httpx.TimeoutException as exc:
                    last_exc = exc
                    last_reason = "timeout"
                    elapsed_ms = (time.perf_counter() - started_at) * 1000
                    logger.warning(
                        "EastMoney request failed: symbol=%s reason=timeout "
                        "host=%s attempt=%d elapsed_ms=%.1f error=%s",
                        symbol or "-",
                        current_base_url,
                        attempt,
                        elapsed_ms,
                        exc,
                    )
                    if attempt < MAX_ATTEMPTS_PER_HOST:
                        time.sleep(0.2 * attempt)
                        continue
                    break
                except httpx.RequestError as exc:
                    last_exc = exc
                    last_reason = "network_error"
                    elapsed_ms = (time.perf_counter() - started_at) * 1000
                    logger.warning(
                        "EastMoney request failed: symbol=%s reason=network_error "
                        "host=%s attempt=%d elapsed_ms=%.1f error=%s",
                        symbol or "-",
                        current_base_url,
                        attempt,
                        elapsed_ms,
                        exc,
                    )
                    if attempt < MAX_ATTEMPTS_PER_HOST:
                        time.sleep(0.2 * attempt)
                        continue
                    break
                except httpx.HTTPStatusError as exc:
                    last_exc = exc
                    last_reason = "invalid_response"
                    elapsed_ms = (time.perf_counter() - started_at) * 1000
                    logger.warning(
                        "EastMoney request failed: symbol=%s reason=invalid_response "
                        "host=%s status=%s attempt=%d elapsed_ms=%.1f error=%s",
                        symbol or "-",
                        current_base_url,
                        exc.response.status_code,
                        attempt,
                        elapsed_ms,
                        exc,
                    )
                    if attempt < MAX_ATTEMPTS_PER_HOST:
                        time.sleep(0.2 * attempt)
                        continue
                    break

        raise MarketDataError(
            f"EastMoney all endpoints failed: {last_exc}",
            reason=last_reason,
        ) from last_exc

    def _akshare(self) -> AkShareSource:
        if self._akshare_source is None:
            self._akshare_source = AkShareSource()
        return self._akshare_source

    @staticmethod
    def _to_secid(symbol: str) -> str:
        normalized = symbol.strip()
        if not re.fullmatch(r"\d{6}", normalized):
            raise MarketDataError(
                f"仅支持 6 位 A 股代码，收到: {symbol}",
                reason="symbol_not_found",
            )
        market = "1" if normalized.startswith(("5", "6", "9")) else "0"
        return f"{market}.{normalized}"

    @staticmethod
    def _parse_quote(data: dict) -> QuoteSnapshot:
        fetched_datetime = shanghai_now()
        fetched_at = fetched_datetime.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_datetime = MarketDataService._parse_quote_timestamp(
            data.get("f86")
        )
        timestamp = (
            timestamp_datetime.strftime("%Y-%m-%d %H:%M:%S")
            if timestamp_datetime is not None
            else ""
        )
        data_age_seconds = (
            max(0, int((fetched_datetime - timestamp_datetime).total_seconds()))
            if timestamp_datetime is not None
            else None
        )
        is_trading_session = MarketDataService._is_trading_session(
            data.get("f80"),
            fetched_datetime,
        )
        failure_reason = (
            "stale_quote"
            if (
                is_trading_session
                and data_age_seconds is not None
                and data_age_seconds > MAX_QUOTE_AGE_SECONDS
            )
            else ""
        )
        return QuoteSnapshot(
            symbol=str(data.get("f57", "")),
            name=str(data.get("f58", "")),
            price=MarketDataService._scaled_price(data.get("f43")),
            change=MarketDataService._scaled_price(data.get("f169")),
            change_pct=MarketDataService._scaled_pct(data.get("f170")),
            open_price=MarketDataService._scaled_price(data.get("f46")),
            high_price=MarketDataService._scaled_price(data.get("f44")),
            low_price=MarketDataService._scaled_price(data.get("f45")),
            prev_close=MarketDataService._scaled_price(data.get("f60")),
            volume=MarketDataService._to_float(data.get("f47")),
            amount=MarketDataService._to_float(data.get("f48")),
            amplitude_pct=MarketDataService._scaled_pct(data.get("f171")),
            turnover_rate=MarketDataService._scaled_pct(data.get("f168")),
            fetched_at=fetched_at,
            source="EastMoney",
            timestamp=timestamp,
            data_age_seconds=data_age_seconds,
            is_trading_session=is_trading_session,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _parse_quote_timestamp(raw: object) -> Optional[datetime]:
        if raw in (None, "", "-", 0, "0"):
            return None
        try:
            timestamp = int(raw)
        except (TypeError, ValueError):
            return None
        if timestamp <= 0:
            return None
        try:
            return datetime.fromtimestamp(timestamp, SHANGHAI_TZ)
        except (OSError, OverflowError, ValueError):
            return None

    @staticmethod
    def _is_trading_session(raw_sessions: object, now: datetime) -> bool:
        if not raw_sessions:
            return False
        try:
            sessions = (
                json.loads(raw_sessions)
                if isinstance(raw_sessions, str)
                else raw_sessions
            )
        except (TypeError, ValueError):
            return False
        if not isinstance(sessions, list):
            return False

        now_key = int(now.strftime("%Y%m%d%H%M"))
        return any(
            isinstance(session, dict)
            and isinstance(session.get("b"), (int, float))
            and isinstance(session.get("e"), (int, float))
            and int(session["b"]) <= now_key <= int(session["e"])
            for session in sessions
        )

    @staticmethod
    def _scaled_price(raw: object) -> float:
        return round(MarketDataService._to_float(raw) / 100, 2)

    @staticmethod
    def _scaled_pct(raw: object) -> float:
        return round(MarketDataService._to_float(raw) / 100, 2)

    @staticmethod
    def _to_float(raw: object) -> float:
        if raw in (None, "-", ""):
            return 0.0
        return float(raw)


_market_data_service_instance: Optional[MarketDataService] = None


def get_market_data_service() -> MarketDataService:
    """获取实时行情服务单例。"""
    global _market_data_service_instance  # noqa: PLW0603
    if _market_data_service_instance is None:
        _market_data_service_instance = MarketDataService()
    return _market_data_service_instance
