"""股票名称/代码解析器。

把用户输入的股票名称、简称、股票代码统一解析为结构化对象。
解析失败或多匹配时不猜测，交由上层阻断 DeepSeek 分析。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


STOCK_UNRECOGNIZED_MESSAGE = "未能识别股票，请输入股票代码，例如 600206。"


class ResolvedStock(BaseModel):
    """已唯一识别的股票。"""

    symbol: str
    name: str
    exchange: str
    market: str = "CN"


@dataclass(frozen=True)
class StockResolveResult:
    """股票解析结果。"""

    stock: ResolvedStock | None = None
    candidates: tuple[ResolvedStock, ...] = ()
    cleaned_query: str = ""
    error: str = ""

    @property
    def resolved(self) -> bool:
        return self.stock is not None

    @property
    def ambiguous(self) -> bool:
        return bool(self.candidates)


KNOWN_STOCKS: tuple[ResolvedStock, ...] = (
    ResolvedStock(symbol="000001", name="平安银行", exchange="SZ", market="CN"),
    ResolvedStock(symbol="600206", name="有研新材", exchange="SH", market="CN"),
    ResolvedStock(symbol="003031", name="中瓷电子", exchange="SZ", market="CN"),
    ResolvedStock(symbol="300604", name="长川科技", exchange="SZ", market="CN"),
    ResolvedStock(symbol="301297", name="富乐德", exchange="SZ", market="CN"),
    ResolvedStock(symbol="301005", name="超捷股份", exchange="SZ", market="CN"),
)


class StockResolver:
    """稳定的 A 股名称/代码解析器。"""

    def __init__(self, stocks: Iterable[ResolvedStock] = KNOWN_STOCKS) -> None:
        self._stocks = tuple(stocks)
        self._by_symbol = {stock.symbol: stock for stock in self._stocks}

    def resolve(self, text: str) -> StockResolveResult:
        raw = str(text or "").strip()
        cleaned = self.normalize_query(raw)
        logger.info(
            "StockResolver request: raw=%r cleaned=%r",
            raw,
            cleaned,
        )

        code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", cleaned)
        if code_match:
            symbol = code_match.group(1)
            stock = self._by_symbol.get(symbol) or self._stock_from_code(symbol)
            logger.info(
                "StockResolver success: raw=%r symbol=%s name=%s",
                raw,
                stock.symbol,
                stock.name,
            )
            return StockResolveResult(stock=stock, cleaned_query=cleaned)

        if not cleaned:
            return StockResolveResult(cleaned_query=cleaned, error="empty_query")

        matches = self._match_by_name(cleaned)
        if len(matches) == 1:
            stock = matches[0]
            logger.info(
                "StockResolver success: raw=%r symbol=%s name=%s",
                raw,
                stock.symbol,
                stock.name,
            )
            return StockResolveResult(stock=stock, cleaned_query=cleaned)

        if len(matches) > 1:
            logger.warning(
                "StockResolver ambiguous: raw=%r cleaned=%r candidates=%s",
                raw,
                cleaned,
                [f"{item.symbol}:{item.name}" for item in matches],
            )
            return StockResolveResult(
                candidates=tuple(matches),
                cleaned_query=cleaned,
                error="ambiguous",
            )

        logger.info(
            "StockResolver failed: raw=%r cleaned=%r reason=not_found",
            raw,
            cleaned,
        )
        return StockResolveResult(cleaned_query=cleaned, error="not_found")

    def can_resolve(self, text: str) -> bool:
        result = self.resolve(text)
        return result.resolved or result.ambiguous

    def _match_by_name(self, cleaned: str) -> list[ResolvedStock]:
        exact = [stock for stock in self._stocks if cleaned == stock.name]
        if exact:
            return exact

        contained = [
            stock
            for stock in self._stocks
            if stock.name in cleaned or cleaned in stock.name
        ]
        return contained

    @staticmethod
    def normalize_query(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"<at\b[^>]*>.*?</at>", " ", cleaned, flags=re.I | re.S)
        cleaned = re.sub(r"[@#￥$%^&*+=|~`\"'“”‘’]", " ", cleaned)
        cleaned = re.sub(r"[？?！!。,.，、:：；;（）()【】\[\]{}<>《》]", " ", cleaned)
        cleaned = re.sub(
            r"(帮我|请|麻烦|查一下|查下|查询|看一下|看看|看下|分析一下|分析|"
            r"点评|看看|看|怎么看|如何看|股票|个股|股价|行情|走势|今天|现在|"
            r"一下|怎么样|咋样|如何|吗|呢|吧)",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _stock_from_code(symbol: str) -> ResolvedStock:
        exchange = "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
        return ResolvedStock(
            symbol=symbol,
            name=symbol,
            exchange=exchange,
            market="CN",
        )


_stock_resolver_instance: StockResolver | None = None


def get_stock_resolver() -> StockResolver:
    """获取 StockResolver 单例。"""
    global _stock_resolver_instance
    if _stock_resolver_instance is None:
        _stock_resolver_instance = StockResolver()
    return _stock_resolver_instance
