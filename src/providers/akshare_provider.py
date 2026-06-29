"""AkShare provider implementation for historical and scan data."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.market.akshare_source import (
    AkShareSource,
    HistoryBar,
    MACDSnapshot,
    MASnapshot,
    StockInfo,
)
from src.market.provider_utils import ProviderTimeoutError, run_with_timeout
from src.providers.base import ProviderConfig, ProviderResult
from src.rating.sector_provider import SectorContext

logger = logging.getLogger(__name__)


class AkShareProvider:
    """Provider facade for AkShare historical, indicator and scan data."""

    def __init__(
        self,
        source: Optional[AkShareSource] = None,
        config: Optional[ProviderConfig] = None,
    ) -> None:
        self.config = config or ProviderConfig(
            data_source="AkShare",
            timeout=8.0,
            cache_enabled=True,
            fallback_priority=20,
            cache_ttl_seconds=300.0,
        )
        self.source = source or AkShareSource(timeout=self.config.timeout)

    @property
    def data_source(self) -> str:
        return self.config.data_source

    @property
    def timeout(self) -> float:
        return self.config.timeout

    @property
    def cache_enabled(self) -> bool:
        return self.config.cache_enabled

    @property
    def fallback_priority(self) -> int:
        return self.config.fallback_priority

    def get_quote(self, symbol: str) -> ProviderResult[Any]:
        return ProviderResult.failed(
            self.data_source,
            "AkShare不作为实时行情主数据源",
            metadata={"symbol": symbol},
        )

    def get_kline(
        self,
        symbol: str,
        period: int = 60,
    ) -> ProviderResult[list[HistoryBar]]:
        try:
            history = self.source.get_history(symbol, period=period)
        except TimeoutError as exc:
            logger.warning("AkShare kline timeout: symbol=%s error=%s", symbol, exc)
            return ProviderResult.timeout(self.data_source, "历史K线超时")
        except Exception as exc:
            logger.warning("AkShare kline failed: symbol=%s error=%s", symbol, exc)
            return ProviderResult.failed(
                self.data_source,
                "历史K线不可用",
                error_type=type(exc).__name__,
            )
        if not history:
            return ProviderResult.partial([], self.data_source, "历史K线为空")
        return ProviderResult.success(
            history,
            self.data_source,
            metadata={"rows": len(history), "period": period},
        )

    def get_history(
        self,
        symbol: str,
        period: int = 60,
    ) -> ProviderResult[list[HistoryBar]]:
        return self.get_kline(symbol, period=period)

    def get_ma(self, symbol: str) -> ProviderResult[MASnapshot]:
        try:
            return ProviderResult.success(self.source.get_ma(symbol), self.data_source)
        except Exception as exc:
            logger.warning("AkShare MA failed: symbol=%s error=%s", symbol, exc)
            return ProviderResult.failed(
                self.data_source,
                "均线数据不可用",
                error_type=type(exc).__name__,
            )

    def get_macd(self, symbol: str) -> ProviderResult[MACDSnapshot]:
        try:
            return ProviderResult.success(self.source.get_macd(symbol), self.data_source)
        except Exception as exc:
            logger.warning("AkShare MACD failed: symbol=%s error=%s", symbol, exc)
            return ProviderResult.failed(
                self.data_source,
                "MACD数据不可用",
                error_type=type(exc).__name__,
            )

    def get_stock_info(self, symbol: str) -> ProviderResult[StockInfo]:
        try:
            info = self.source.get_stock_info(symbol)
        except Exception as exc:
            logger.warning("AkShare stock info failed: symbol=%s error=%s", symbol, exc)
            return ProviderResult.failed(
                self.data_source,
                "股票基础信息不可用",
                error_type=type(exc).__name__,
            )
        if info.industry or info.concepts or info.name:
            return ProviderResult.success(info, self.data_source)
        return ProviderResult.partial(info, self.data_source, "股票基础信息不完整")

    def get_sector(self, symbol: str) -> ProviderResult[SectorContext]:
        info = self.get_stock_info(symbol)
        if info.data is None:
            return ProviderResult.failed(
                self.data_source,
                info.message or "AkShare行业/概念数据不可用",
                error_type=info.error_type,
            )
        context = SectorContext(
            name=info.data.name,
            industry=info.data.industry,
            concepts=list(info.data.concepts),
            data_source=self.data_source if info.data.industry or info.data.concepts else "",
        )
        if context.available:
            return ProviderResult.success(
                context,
                self.data_source,
                metadata={"sector_status": context.sector_status},
            )
        return ProviderResult.partial(
            context,
            self.data_source,
            context.warning,
            warnings=[context.warning],
            metadata={"sector_status": context.sector_status},
        )

    def get_concepts(self, symbol: str) -> ProviderResult[list[str]]:
        info = self.get_stock_info(symbol)
        concepts = list(info.data.concepts) if info.data is not None else []
        if concepts:
            return ProviderResult.success(concepts, self.data_source)
        return ProviderResult.partial([], self.data_source, "AkShare概念数据暂不可用")

    def get_fund_flow(self, symbol: str) -> ProviderResult[dict[str, Any]]:
        return ProviderResult.partial(
            {},
            self.data_source,
            "AkShare资金流接口预留，暂未接入",
            metadata={"symbol": symbol},
        )

    def get_news(self, symbol: str) -> ProviderResult[list[Any]]:
        return ProviderResult.partial(
            [],
            self.data_source,
            "AkShare新闻接口预留，暂未接入",
            metadata={"symbol": symbol},
        )

    def get_realtime_quotes(self) -> ProviderResult[list[dict[str, Any]]]:
        try:
            frame = run_with_timeout(
                lambda: self.source._akshare().stock_zh_a_spot_em(),
                self.timeout,
                "AkShare stock_zh_a_spot_em",
            )
        except ProviderTimeoutError as exc:
            logger.warning("AkShare market scan timeout: error=%s", exc)
            return ProviderResult.timeout(self.data_source, "全市场实时行情超时")
        except Exception as exc:
            logger.warning("AkShare market scan failed: error=%s", exc)
            return ProviderResult.failed(
                self.data_source,
                "全市场实时行情不可用",
                error_type=type(exc).__name__,
            )
        rows = _records(frame)
        if not rows:
            return ProviderResult.partial([], self.data_source, "全市场实时行情为空")
        return ProviderResult.success(rows, self.data_source, metadata={"rows": len(rows)})

    def get_hot_sectors(self, limit: int = 10) -> ProviderResult[set[str]]:
        ak = self.source._akshare()
        if not hasattr(ak, "stock_board_industry_name_em"):
            return ProviderResult.partial(set(), self.data_source, "热点板块接口不可用")
        try:
            frame = run_with_timeout(
                lambda: ak.stock_board_industry_name_em(),
                self.timeout,
                "AkShare stock_board_industry_name_em",
            )
        except ProviderTimeoutError as exc:
            logger.warning("AkShare hot sectors timeout: error=%s", exc)
            return ProviderResult.timeout(self.data_source, "热点板块超时")
        except Exception as exc:
            logger.warning("AkShare hot sectors failed: error=%s", exc)
            return ProviderResult.failed(
                self.data_source,
                "热点板块不可用",
                error_type=type(exc).__name__,
            )

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
        if sectors:
            return ProviderResult.success(sectors, self.data_source)
        return ProviderResult.partial(set(), self.data_source, "热点板块为空")

    def get_index_change_pct(self) -> ProviderResult[float]:
        ak = self.source._akshare()
        if not hasattr(ak, "stock_zh_index_spot_em"):
            return ProviderResult.partial(0.0, self.data_source, "指数接口不可用")
        try:
            for row in _records(ak.stock_zh_index_spot_em()):
                name = _first_value(row, ("名称", "name"))
                if name in ("上证指数", "沪深300", "深证成指"):
                    return ProviderResult.success(
                        _to_float(_first_value(row, ("涨跌幅", "change_pct"))),
                        self.data_source,
                    )
        except Exception as exc:
            logger.warning("AkShare index quote failed: error=%s", exc)
            return ProviderResult.failed(
                self.data_source,
                "指数行情不可用",
                error_type=type(exc).__name__,
            )
        return ProviderResult.partial(0.0, self.data_source, "指数行情为空")

    def health_check(self) -> ProviderResult[dict[str, Any]]:
        history = self.get_kline("000001", period=1)
        if history.ok and history.data:
            return ProviderResult.success(
                {"history": "ok", "rows": len(history.data)},
                self.data_source,
            )
        return ProviderResult.failed(
            self.data_source,
            history.message or "AkShare健康检查失败",
            error_type=history.error_type,
        )


def _records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, list):
        records = frame
    elif isinstance(frame, dict):
        records = [frame]
    else:
        if getattr(frame, "empty", False):
            return []
        if not hasattr(frame, "to_dict"):
            return []
        records = frame.to_dict("records")
    return [dict(row) for row in records if isinstance(row, dict)]


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
